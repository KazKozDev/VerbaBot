import asyncio
import aiosqlite
import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import spacy
import aiofiles
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load NLP for entity extraction - use a smaller model for efficiency
try:
    nlp = spacy.load("en_core_web_sm")
except:
    logger.warning("Downloading spaCy model for entity extraction")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Constants
MEMORY_DB_FOLDER = 'memory_db'
os.makedirs(MEMORY_DB_FOLDER, exist_ok=True)

class ContextualMemory:
    """System for extracting, storing, and utilizing personal contextual information"""
    
    def __init__(self, embeddings_model=None):
        """Initialize the contextual memory system"""
        self.db_path = 'db/assistant.db'
        self.vectorstore = None
        
        # Use provided embeddings or initialize the default
        self.embeddings_model = embeddings_model or HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Memory categories and extraction patterns
        self.memory_categories = {
            'preference': {
                'patterns': [
                    r"I (?:like|love|enjoy|prefer|hate|dislike|don't like) (.+)",
                    r"My favorite (.+) is (.+)",
                    r"(?:I am|I'm) (?:interested in|passionate about|fond of) (.+)",
                    # Add English alternatives to previously Russian patterns
                    r"I (?:like|dislike) (.+)",
                    r"I (?:love|adore|hate) (.+)",
                ],
                'importance': 0.8,
                'entities': ['PRODUCT', 'EVENT', 'FAC', 'LOC', 'GPE']
            },
            'personal_fact': {
                'patterns': [
                    r"My (?:name|age|birthday|address|email|phone|job|title|role|company) is (.+)",
                    r"I (?:work at|live in|am from|was born in|graduated from) (.+)",
                    r"I am (?:a|an) (.+)",
                    # Add English alternatives to previously Russian patterns
                    r"My name is (.+)",
                    r"I (?:work in|live in|am from|was born in) (.+)",
                ],
                'importance': 0.9,
                'entities': ['PERSON', 'DATE', 'ORG', 'GPE', 'LOC']
            },
            'project': {
                'patterns': [
                    r"I'm working on (?:a|an|the) (.+)",
                    r"My current project is (?:about|on|related to) (.+)",
                    r"I'm building (?:a|an|the) (.+)",
                    # Add English alternatives to previously Russian patterns
                    r"I am working on (.+)",
                    r"My project is (.+)",
                ],
                'importance': 0.7,
                'entities': ['PRODUCT', 'ORG', 'EVENT']
            },
            'goal': {
                'patterns': [
                    r"I want to (.+)",
                    r"I need to (.+)",
                    r"I'm trying to (.+)",
                    r"My goal is to (.+)",
                    # Add English alternatives to previously Russian patterns
                    r"I want (.+)",
                    r"I need (.+)",
                    r"My goal is (.+)",
                ],
                'importance': 0.6,
                'entities': ['EVENT', 'FAC', 'LOC', 'GPE']
            }
        }
        
        # Load or create memory vector store
        self.load_or_create_vectorstore()
    
    async def setup_db(self):
        """Set up memory tables if they don't exist"""
        async with aiosqlite.connect(self.db_path) as db:
            # Create table for personal memory facts
            await db.execute('''
                CREATE TABLE IF NOT EXISTS personal_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    chat_id INTEGER,
                    importance REAL DEFAULT 0.5
                )
            ''')
            
            # Create table for memory settings
            await db.execute('''
                CREATE TABLE IF NOT EXISTS memory_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    setting_name TEXT UNIQUE NOT NULL,
                    setting_value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert default settings if they don't exist
            await db.execute('''
                INSERT OR IGNORE INTO memory_settings (setting_name, setting_value)
                VALUES (?, ?)
            ''', ('memory_enabled', 'true'))
            
            await db.execute('''
                INSERT OR IGNORE INTO memory_settings (setting_name, setting_value)
                VALUES (?, ?)
            ''', ('extraction_threshold', '0.7'))
            
            await db.commit()
            logger.info("Memory database tables initialized")
    
    def load_or_create_vectorstore(self):
        """Load existing memory vectorstore if available, otherwise create a new one"""
        try:
            if os.path.exists(f"{MEMORY_DB_FOLDER}/index.faiss"):
                self.vectorstore = FAISS.load_local(MEMORY_DB_FOLDER, self.embeddings_model, allow_dangerous_deserialization=True)
                logger.info("Loaded existing memory vector database")
            else:
                # Create empty vector store with initialization document
                self.vectorstore = FAISS.from_documents(
                    [Document(page_content="Memory initialization", metadata={"source": "init", "category": "system"})], 
                    self.embeddings_model
                )
                logger.info("Created new memory vector database")
        except Exception as e:
            logger.error(f"Error loading memory vector database: {e}")
            # Create a new one if loading fails
            self.vectorstore = FAISS.from_documents(
                [Document(page_content="Memory initialization", metadata={"source": "init", "category": "system"})], 
                self.embeddings_model
            )
    
    def save_vectorstore(self):
        """Save the current memory vectorstore to disk"""
        try:
            self.vectorstore.save_local(MEMORY_DB_FOLDER)
            logger.info("Memory vector database saved successfully")
        except Exception as e:
            logger.error(f"Error saving memory vector database: {e}")
    
    async def is_memory_enabled(self) -> bool:
        """Check if memory functionality is enabled in settings"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute('SELECT setting_value FROM memory_settings WHERE setting_name = ?', ('memory_enabled',))
                result = await cursor.fetchone()
                return result and result[0].lower() == 'true'
        except Exception as e:
            logger.error(f"Error checking if memory is enabled: {e}")
            return False
    
    async def process_message(self, message: str, chat_id: Optional[int] = None, is_user: bool = True):
        """Process a message to extract and store personal information"""
        if not is_user or not await self.is_memory_enabled():
            return  # Only process user messages when memory is enabled
        
        extracted_info = self._extract_personal_info(message)
        
        for info in extracted_info:
            await self._store_memory_fact(
                category=info['category'],
                content=info['content'],
                source_text=info['source_text'],
                confidence=info['confidence'],
                chat_id=chat_id,
                importance=info['importance']
            )
            
            # Add to vector store for semantic search
            doc = Document(
                page_content=info['content'],
                metadata={
                    "source": "user_message",
                    "category": info['category'],
                    "confidence": info['confidence'],
                    "chat_id": chat_id,
                    "importance": info['importance'],
                    "created_at": datetime.now().isoformat()
                }
            )
            self.vectorstore.add_documents([doc])
        
        # Save updated vectorstore if new info was added
        if extracted_info:
            self.save_vectorstore()
            
        return len(extracted_info)
    
    def _extract_personal_info(self, text: str) -> List[Dict[str, Any]]:
        """Extract personal information from text using patterns and NER"""
        extracted_info = []
        
        # Process with spaCy for entity recognition
        doc = nlp(text)
        recognized_entities = {}
        
        for ent in doc.ents:
            recognized_entities[ent.text] = ent.label_
        
        # Check each category's patterns
        for category, config in self.memory_categories.items():
            for pattern in config['patterns']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    captured_groups = match.groups()
                    if not captured_groups:
                        continue
                    
                    # Use the longest captured group as content
                    content = max(captured_groups, key=len).strip()
                    
                    # Calculate confidence based on presence of target entities
                    confidence = 0.7  # Base confidence for pattern match
                    
                    # Boost confidence if we find relevant entities
                    entity_found = False
                    for entity, entity_type in recognized_entities.items():
                        if entity_type in config['entities'] and entity in content:
                            confidence += 0.2
                            entity_found = True
                            break
                    
                    # Add to extracted info if confidence is high enough
                    if confidence >= 0.7:  # Threshold for extraction
                        extracted_info.append({
                            'category': category,
                            'content': content,
                            'source_text': text,
                            'confidence': confidence,
                            'importance': config['importance'],
                            'has_entity': entity_found
                        })
        
        return extracted_info
    
    async def _store_memory_fact(self, category: str, content: str, source_text: str, 
                                confidence: float, chat_id: Optional[int] = None,
                                importance: float = 0.5):
        """Store a memory fact in the database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Check if the same content already exists in this category
                cursor = await db.execute(
                    'SELECT id, confidence FROM personal_memory WHERE category = ? AND content = ?', 
                    (category, content)
                )
                existing = await cursor.fetchone()
                
                if existing:
                    # Update existing record if new confidence is higher
                    existing_id, existing_confidence = existing
                    if confidence > existing_confidence:
                        await db.execute(
                            'UPDATE personal_memory SET confidence = ?, last_accessed = ? WHERE id = ?',
                            (confidence, datetime.now(), existing_id)
                        )
                else:
                    # Insert new record
                    await db.execute('''
                        INSERT INTO personal_memory 
                        (category, content, source_text, confidence, created_at, last_accessed, chat_id, importance)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        category, content, source_text, confidence, 
                        datetime.now(), datetime.now(), chat_id, importance
                    ))
                
                await db.commit()
                logger.info(f"Stored memory fact: {category} - {content}")
                
        except Exception as e:
            logger.error(f"Error storing memory fact: {e}")
    
    async def get_relevant_memories(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for a query using hybrid retrieval"""
        # Semantic search from vector store
        semantic_results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Get exact keyword matches from database
        keyword_results = await self._keyword_search_memories(query)
        
        # Merge results, prioritizing higher importance and confidence
        all_results = []
        
        # Process semantic search results
        for doc, score in semantic_results:
            if doc.page_content == "Memory initialization":
                continue  # Skip initialization document
                
            all_results.append({
                'content': doc.page_content,
                'category': doc.metadata.get('category', 'unknown'),
                'confidence': doc.metadata.get('confidence', 0.5),
                'importance': doc.metadata.get('importance', 0.5),
                'score': float(1 - score),  # Convert FAISS score to 0-1 scale where 1 is best
                'source': 'semantic'
            })
        
        # Process keyword search results
        for item in keyword_results:
            all_results.append({
                'content': item['content'],
                'category': item['category'],
                'confidence': item['confidence'],
                'importance': item['importance'],
                'score': 0.8,  # Base score for exact matches
                'source': 'keyword'
            })
        
        # Sort by combined score (semantic relevance + importance + confidence)
        for item in all_results:
            item['combined_score'] = item['score'] * 0.5 + item['importance'] * 0.3 + item['confidence'] * 0.2
        
        all_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Remove duplicates (keep the one with higher score)
        unique_results = {}
        for item in all_results:
            content = item['content']
            if content not in unique_results or unique_results[content]['combined_score'] < item['combined_score']:
                unique_results[content] = item
        
        # Return top k unique results
        return list(unique_results.values())[:k]
    
    async def _keyword_search_memories(self, query: str) -> List[Dict[str, Any]]:
        """Search memories based on keyword matches"""
        try:
            # Extract keywords (simple approach - split and filter)
            words = set(query.lower().split())
            keywords = [word for word in words if len(word) > 3]  # Filter out short words
            
            if not keywords:
                return []
                
            # Prepare SQL with LIKE conditions for each keyword
            conditions = " OR ".join(["content LIKE ?" for _ in keywords])
            params = [f"%{keyword}%" for keyword in keywords]
            
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = sqlite3.Row
                cursor = await db.execute(f'''
                    SELECT id, category, content, confidence, importance
                    FROM personal_memory
                    WHERE {conditions}
                    ORDER BY importance DESC, confidence DESC
                    LIMIT 10
                ''', params)
                
                rows = await cursor.fetchall()
                
                # Update access counts for retrieved memories
                if rows:
                    ids = [row['id'] for row in rows]
                    placeholders = ','.join('?' for _ in ids)
                    await db.execute(f'''
                        UPDATE personal_memory
                        SET last_accessed = ?, access_count = access_count + 1
                        WHERE id IN ({placeholders})
                    ''', [datetime.now()] + ids)
                    await db.commit()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error during keyword memory search: {e}")
            return []
    
    def format_memories_for_context(self, memories: List[Dict[str, Any]]) -> str:
        """Format retrieved memories as context for the LLM"""
        if not memories:
            return ""
            
        context = "Personal context from previous conversations:\n"
        
        # Group by category
        by_category = {}
        for mem in memories:
            category = mem['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(mem)
        
        # Format each category
        for category, items in by_category.items():
            context += f"\n{category.capitalize()}:\n"
            for item in items:
                context += f"- {item['content']}\n"
        
        return context
    
    async def get_memory_management_commands(self) -> Dict[str, str]:
        """Get list of memory management commands for user reference"""
        return {
            "/memory show": "Display saved personal information",
            "/memory delete [fact]": "Remove a specific piece of information",
            "/memory add [category] [fact]": "Explicitly add information to memory",
            "/memory clear": "Clear all personal memory (use with caution)",
            "/memory disable": "Temporarily disable memory collection",
            "/memory enable": "Re-enable memory collection"
        }
    
    async def process_memory_command(self, command: str) -> str:
        """Process a memory management command from the user"""
        if command.startswith("/memory show"):
            return await self._show_memories()
            
        elif command.startswith("/memory delete"):
            fact = command.replace("/memory delete", "").strip()
            return await self._delete_memory(fact)
            
        elif command.startswith("/memory add"):
            parts = command.replace("/memory add", "").strip().split(" ", 1)
            if len(parts) >= 2:
                category, fact = parts
                return await self._add_memory(category, fact)
            else:
                return "Please specify both category and fact. Format: /memory add [category] [fact]"
                
        elif command.startswith("/memory clear"):
            return await self._clear_all_memories()
            
        elif command.startswith("/memory disable"):
            return await self._set_memory_enabled(False)
            
        elif command.startswith("/memory enable"):
            return await self._set_memory_enabled(True)
            
        else:
            commands = await self.get_memory_management_commands()
            response = "Available memory commands:\n\n"
            for cmd, desc in commands.items():
                response += f"{cmd} - {desc}\n"
            return response
    
    async def _show_memories(self) -> str:
        """Show saved memories to the user"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = sqlite3.Row
                cursor = await db.execute('''
                    SELECT category, content, confidence, importance
                    FROM personal_memory
                    ORDER BY category, importance DESC, confidence DESC
                ''')
                
                rows = await cursor.fetchall()
                
                if not rows:
                    return "No personal information has been saved yet."
                
                # Group by category
                by_category = {}
                for row in rows:
                    category = row['category']
                    if category not in by_category:
                        by_category[category] = []
                    by_category[category].append(dict(row))
                
                # Format response
                response = "Here's what I remember about you:\n\n"
                
                for category, items in by_category.items():
                    response += f"**{category.capitalize()}**:\n"
                    for item in items:
                        confidence_indicator = "✓" if item["confidence"] > 0.8 else "?"
                        response += f"- {item['content']} {confidence_indicator}\n"
                    response += "\n"
                
                return response
                
        except Exception as e:
            logger.error(f"Error showing memories: {e}")
            return "Sorry, I encountered an error while retrieving your personal information."
    
    async def _delete_memory(self, fact: str) -> str:
        """Delete a specific memory fact"""
        if not fact:
            return "Please specify the fact to delete."
            
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # First, check if the fact exists
                cursor = await db.execute('SELECT id FROM personal_memory WHERE content LIKE ?', (f"%{fact}%",))
                row = await cursor.fetchone()
                
                if not row:
                    return f"I couldn't find any saved information matching '{fact}'."
                
                # Delete the fact
                await db.execute('DELETE FROM personal_memory WHERE content LIKE ?', (f"%{fact}%",))
                await db.commit()
                
                # Also remove from vector database (rebuild)
                await self._rebuild_vector_memory()
                
                return f"I've removed information about '{fact}' from my memory."
                
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return "Sorry, I encountered an error while trying to delete that information."
    
    async def _add_memory(self, category: str, fact: str) -> str:
        """Explicitly add a memory fact"""
        valid_categories = list(self.memory_categories.keys())
        
        if category not in valid_categories:
            return f"Invalid category. Please use one of: {', '.join(valid_categories)}"
            
        importance = self.memory_categories[category]['importance']
        
        await self._store_memory_fact(
            category=category,
            content=fact,
            source_text=f"/memory add {category} {fact}",
            confidence=1.0,  # High confidence for explicit additions
            importance=importance
        )
        
        # Add to vector store
        doc = Document(
            page_content=fact,
            metadata={
                "source": "explicit_add",
                "category": category,
                "confidence": 1.0,
                "importance": importance,
                "created_at": datetime.now().isoformat()
            }
        )
        self.vectorstore.add_documents([doc])
        self.save_vectorstore()
        
        return f"I've added this information to my memory under '{category}'."
    
    async def _clear_all_memories(self) -> str:
        """Clear all memory (with confirmation)"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('DELETE FROM personal_memory')
                await db.commit()
            
            # Reset vector store
            self.vectorstore = FAISS.from_documents(
                [Document(page_content="Memory initialization", metadata={"source": "init", "category": "system"})], 
                self.embeddings_model
            )
            self.save_vectorstore()
            
            return "I've cleared all personal information from my memory."
            
        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
            return "Sorry, I encountered an error while trying to clear your personal information."
    
    async def _set_memory_enabled(self, enabled: bool) -> str:
        """Enable or disable memory functionality"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    'UPDATE memory_settings SET setting_value = ?, updated_at = ? WHERE setting_name = ?',
                    ('true' if enabled else 'false', datetime.now(), 'memory_enabled')
                )
                await db.commit()
                
            status = "enabled" if enabled else "disabled"
            return f"Personal memory collection is now {status}."
            
        except Exception as e:
            logger.error(f"Error setting memory enabled status: {e}")
            return "Sorry, I encountered an error while updating memory settings."
    
    async def _rebuild_vector_memory(self):
        """Rebuild vector memory database from SQL database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = sqlite3.Row
                cursor = await db.execute('''
                    SELECT category, content, confidence, importance, created_at
                    FROM personal_memory
                ''')
                
                rows = await cursor.fetchall()
                
                # Create documents from database records
                documents = [
                    Document(
                        page_content=row['content'],
                        metadata={
                            "source": "db_rebuild",
                            "category": row['category'],
                            "confidence": row['confidence'],
                            "importance": row['importance'],
                            "created_at": row['created_at']
                        }
                    ) for row in rows
                ]
                
                # Add initialization document
                documents.append(Document(
                    page_content="Memory initialization", 
                    metadata={"source": "init", "category": "system"}
                ))
                
                # Rebuild vector store
                self.vectorstore = FAISS.from_documents(documents, self.embeddings_model)
                self.save_vectorstore()
                
        except Exception as e:
            logger.error(f"Error rebuilding vector memory: {e}")

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get counts by category
                db.row_factory = sqlite3.Row
                cursor = await db.execute('''
                    SELECT category, COUNT(*) as count 
                    FROM personal_memory 
                    GROUP BY category
                ''')
                
                rows = await cursor.fetchall()
                category_counts = {row['category']: row['count'] for row in rows}
                
                # Get total count
                cursor = await db.execute('SELECT COUNT(*) as total FROM personal_memory')
                total = (await cursor.fetchone())['total']
                
                # Get most recently added
                cursor = await db.execute('''
                    SELECT category, content, created_at 
                    FROM personal_memory 
                    ORDER BY created_at DESC 
                    LIMIT 1
                ''')
                
                latest = await cursor.fetchone()
                latest_memory = dict(latest) if latest else None
                
                return {
                    "total_memories": total,
                    "by_category": category_counts,
                    "most_recent": latest_memory
                }
                
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {
                "total_memories": 0,
                "by_category": {},
                "most_recent": None,
                "error": str(e)
            } 