import asyncio
import aiohttp
from quart import Quart, request, jsonify, send_file
from quart_cors import cors
import aiosqlite
from datetime import datetime
import traceback
import os
import mimetypes
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
from duckduckgo_search import DDGS
from typing import List
import logging
import textwrap
from bs4 import BeautifulSoup
import urllib.parse
import re
import base64
import shutil
import pandas as pd
import docx
import openpyxl
import requests
import json
from quart import Quart
from hypercorn.config import Config
from hypercorn.asyncio import serve
import asyncio

app = Quart(__name__)
app = cors(app, allow_origin="*")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp3', 'mp4', 'wav', 'xlsx', 'xls', 'docx', 'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

llm_model = "mistral-nemo:latest"
image_recognition_model = "llava:13b"
system_prompt = "From this point on, write a concise response that contains complete and useful information on the topic of the question. Do not repeat the user's query in one way or another in your answers."

class RAGOllama:
    def __init__(self, model_name: str = "mistral-nemo:latest"):
        self.model_name = model_name
        self.documents = []
        self.document_embeddings = []
        logging.info(f"Using Ollama model: {model_name}")

    def add_documents_from_file(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            self.documents.append({"content": content, "source": file_path})
            embedding = self.get_embedding(content)
            self.document_embeddings.append(embedding)
            logging.info(f"Added documents from file: {file_path}")
        except Exception as e:
            logging.error(f"Error in add_documents_from_file: {e}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        try:
            url = "http://localhost:11434/api/embeddings"
            data = {
                "model": self.model_name,
                "prompt": text
            }
            response = requests.post(url, json=data)
            if response.status_code == 200:
                return json.loads(response.text)["embedding"]
            else:
                raise Exception(f"Ollama API error: {response.status_code}, {response.text}")
        except Exception as e:
            logging.error(f"Error in get_embedding: {e}")
            raise

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5
        return dot_product / (magnitude_a * magnitude_b)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        try:
            query_embedding = self.get_embedding(query)
            similarities = [self.cosine_similarity(query_embedding, doc_emb) for doc_emb in self.document_embeddings]
            top_k_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]
            return [self.documents[i]["content"] for i in top_k_indices]
        except Exception as e:
            logging.error(f"Error in retrieve: {e}")
            raise

    def query_ollama(self, prompt: str) -> str:
        try:
            url = "http://localhost:11434/api/generate"
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(url, json=data)
            if response.status_code == 200:
                return json.loads(response.text)["response"]
            else:
                raise Exception(f"Ollama API error: {response.status_code}, {response.text}")
        except Exception as e:
            logging.error(f"Error in query_ollama: {e}")
            raise

    def generate(self, query: str) -> str:
        try:
            retrieved_docs = self.retrieve(query)
            context = "\n".join(retrieved_docs)
            prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            return self.query_ollama(prompt)
        except Exception as e:
            logging.error(f"Error in generate: {e}")
            raise

rag = RAGOllama()

current_directory = os.path.dirname(os.path.abspath(__file__))
personal_info_path = os.path.join(current_directory, "RAG.txt")
rag.add_documents_from_file(personal_info_path)

async def init_db():
    async with aiosqlite.connect('assistant.db') as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP,
                title TEXT,
                "order" INTEGER,
                pinned BOOLEAN DEFAULT 0,
                manually_renamed BOOLEAN DEFAULT 0
            )
        ''')
        await db.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                content TEXT,
                is_user BOOLEAN,
                timestamp TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES chats (id)
            )
        ''')
        await db.commit()

async def create_chat():
    try:
        async with aiosqlite.connect('assistant.db') as db:
            cursor = await db.execute('SELECT COUNT(*) FROM chats')
            count = (await cursor.fetchone())[0]
            await db.execute('INSERT INTO chats (created_at, title, "order") VALUES (?, ?, ?)', (datetime.now(), 'New Chat', count))
            await db.commit()
            cursor = await db.execute('SELECT last_insert_rowid()')
            return (await cursor.fetchone())[0]
    except sqlite3.OperationalError as e:
        logger.error(f"Database error: {e}")
        await init_db()  # Attempt to initialize the DB
        return await create_chat()  # Retry creating the chat

async def store_message(chat_id, content, is_user):
    try:
        async with aiosqlite.connect('assistant.db') as db:
            cursor = await db.execute('''
                INSERT INTO messages (chat_id, content, is_user, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (chat_id, content, is_user, datetime.now()))
            await db.commit()
            return cursor.lastrowid
    except sqlite3.OperationalError as e:
        logger.error(f"Database error in store_message: {e}")
        await init_db()
        return await store_message(chat_id, content, is_user)

async def get_chat_history(chat_id, limit=20):
    try:
        async with aiosqlite.connect('assistant.db') as db:
            async with db.execute('''
                SELECT id, content, is_user FROM messages
                WHERE chat_id = ?
                ORDER BY timestamp DESC LIMIT ?
            ''', (chat_id, limit)) as cursor:
                return await cursor.fetchall()
    except sqlite3.OperationalError as e:
        logger.error(f"Database error in get_chat_history: {e}")
        await init_db()
        return await get_chat_history(chat_id, limit)

async def update_chat_title(chat_id, title, manually_renamed=False):
    try:
        async with aiosqlite.connect('assistant.db') as db:
            await db.execute('UPDATE chats SET title = ?, manually_renamed = ? WHERE id = ?', (title, manually_renamed, chat_id))
            await db.commit()
    except sqlite3.OperationalError as e:
        logger.error(f"Database error in update_chat_title: {e}")
        await init_db()
        await update_chat_title(chat_id, title, manually_renamed)

async def generate_chat_title(chat_id):
    try:
        async with aiosqlite.connect('assistant.db') as db:
            async with db.execute('SELECT manually_renamed FROM chats WHERE id = ?', (chat_id,)) as cursor:
                result = await cursor.fetchone()
                if result and result[0]:
                    return None

        chat_history = await get_chat_history(chat_id, limit=5)
        prompt = "Based on the following chat messages, generate a brief, descriptive title for this conversation:\n\n"
        for _, content, is_user in reversed(chat_history):
            role = "Human" if is_user else "Assistant"
            prompt += f"{role}: {content}\n"
        prompt += "\nTitle:"

        title = await get_llm_response(prompt)
        await update_chat_title(chat_id, title)
        return title
    except sqlite3.OperationalError as e:
        logger.error(f"Database error in generate_chat_title: {e}")
        await init_db()
        return await generate_chat_title(chat_id)

async def generate_prompt(message, chat_id):
    chat_history = await get_chat_history(chat_id)

    prompt = "You are a personal AI assistant.\n\nOur conversation in this chat:\n"
    for _, content, is_user in reversed(chat_history):
        prompt += f"{'Human' if is_user else 'Assistant'}: {content}\n"

    prompt += f"\nHuman: {message}\n"
    prompt += "Assistant: From this point on, write a concise response that contains complete and useful information on the topic of the question. Do not repeat the user's query in one way or another in your answers. Respond in the language in which you are being communicated with."

    return prompt

async def get_llm_response(prompt, model=None):
    if model is None:
        model = llm_model
    try:
        logger.info(f"Sending prompt to LLM: {prompt}")
        async with aiohttp.ClientSession() as session:
            async with session.post('http://localhost:11434/api/generate', 
                                    json={
                                        "model": model,
                                        "prompt": prompt,
                                        "stream": False
                                    }) as response:
                response.raise_for_status()
                response_data = await response.json()

        logger.info("Received response from LLM, processing...")
        logger.debug(f"Full response data: {response_data}")

        if 'response' in response_data:
            response = response_data['response']
            return response
        else:
            logger.error("No 'response' field in LLM response")
            return "I'm sorry, I couldn't generate a proper response."
    except aiohttp.ClientError as e:
        logger.error(f"Error communicating with LLM: {e}")
        return "I'm sorry, I encountered an error while processing your request."
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return "I'm sorry, an unexpected error occurred while processing your request."

def get_chat_upload_folder(chat_id):
    return os.path.join(UPLOAD_FOLDER, str(chat_id))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

async def save_uploaded_file(file, chat_id):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        chat_folder = get_chat_upload_folder(chat_id)
        os.makedirs(chat_folder, exist_ok=True)
        file_path = os.path.join(chat_folder, filename)
        await file.save(file_path)
        return file_path
    return None

async def process_pdf_content(file_path):
    doc = fitz.open(file_path)
    formatted_content = []
    
    for page_num, page in enumerate(doc, 1):
        formatted_content.append(f"\n## Page {page_num}\n")
        
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4].strip()
            if text:
                if len(text) < 100 and not text.endswith('.'):
                    formatted_content.append(f"\n### {text}\n")
                else:
                    formatted_content.append(f"{text}\n\n")
        
        formatted_content.append("\n---\n")
    
    return ''.join(formatted_content)

async def read_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    file_extension = os.path.splitext(file_path)[1].lower()

    if mime_type and mime_type.startswith('text') or file_extension == '.csv':
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            return await f.read()
    elif mime_type and mime_type.startswith('application/pdf'):
        return await process_pdf_content(file_path)
    elif mime_type and mime_type.startswith('image'):
        with Image.open(file_path) as img:
            return pytesseract.image_to_string(img)
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
        return df.to_string()
    elif file_extension == '.docx':
        doc = docx.Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    else:
        return "File content could not be read."

async def perform_duckduckgo_search(query, max_results=5):
    search = DDGS()
    results = search.text(query, max_results=max_results)
    return [{'title': result['title'], 'link': result['href'], 'snippet': result['body']} for result in results]

def generate_duckduckgo_summary(query, results):
    summary = f"I performed a DuckDuckGo search for '{query}'. Here's a summary of the top results:\n\n"
    for i, result in enumerate(results, 1):
        summary += f"{i}. {result['title']}\n"
        summary += f"   URL: {result['link']}\n"
        summary += f"   Description: {result['snippet']}\n\n"
    return summary

async def get_google_news(query):
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.text()
            soup = BeautifulSoup(content, 'xml')
            
            news_items = []
            for item in soup.findAll('item')[:5]:
                title = item.find('title').text if item.find('title') else 'No title'
                link = item.find('link').text if item.find('link') else 'No link'
                pub_date = item.find('pubDate').text if item.find('pubDate') else 'Date not specified'
                
                try:
                    date_obj = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
                    formatted_date = date_obj.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    formatted_date = pub_date
                    
                news_items.append({
                    "title": title,
                    "date": formatted_date,
                    "link": link
                })
                
            return news_items
        
async def process_image_with_llava(image_path):
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
    prompt = "Analyze this image and describe what you see in detail."
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post('http://localhost:11434/api/generate', 
                                    json={
                                        "model": image_recognition_model,
                                        "prompt": prompt,
                                        "images": [image_data],
                                        "stream": False
                                    }) as response:
                response.raise_for_status()
                response_data = await response.json()
            
        if 'response' in response_data:
            return response_data['response']
        else:
            return "I'm sorry, I couldn't generate a description for this image."
    except aiohttp.ClientError as e:
        logger.error(f"Error communicating with LLM for image recognition: {e}")
        return "I'm sorry, I encountered an error while processing the image."
    except Exception as e:
        logger.error(f"Unexpected error in image recognition: {e}")
        return "I'm sorry, an unexpected error occurred while processing the image."
    
async def get_text_from_url(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                text_content = []
                for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    if tag.name.startswith('h'):
                        text_content.append(f"<{tag.name}>{tag.get_text(strip=True)}</{tag.name}>")
                    else:
                        text_content.append(f"<p>{tag.get_text(strip=True)}</p>")
                        
                return "\n".join(text_content)
    except aiohttp.ClientError as e:
        return f"<p>Error fetching data: {e}</p>"
    except Exception as e:
        return f"<p>An unexpected error occurred: {e}</p>"
    
@app.route('/chat', methods=['POST'])
async def chat():
    global system_prompt
    try:
        data = await request.get_json()
        message = data.get('message')
        chat_id = data.get('chat_id')
        use_duckduckgo = data.get('use_duckduckgo', False)
        use_google_news = data.get('use_google_news', False)
        use_link = data.get('use_link', False)
        system_prompt = data.get('system_prompt', system_prompt)
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        if not chat_id:
            chat_id = await create_chat()
            
        prompt = await generate_prompt(message, chat_id)
        
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
            
        if use_duckduckgo:
            search_results = await perform_duckduckgo_search(message)
            search_summary = generate_duckduckgo_summary(message, search_results)
            prompt += f"\n\nHere are some DuckDuckGo search results that might be helpful:\n{search_summary}"
            
        if use_google_news:
            news_items = await get_google_news(message)
            news_summary = "Here are some recent news articles related to your query:\n\n"
            for item in news_items:
                news_summary += f"- {item['title']} ({item['date']})\n  {item['link']}\n\n"
            prompt += f"\n\nPlease summarize and provide insights on the following news:\n{news_summary}"
            
        if use_link:
            url_match = re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message)
            if url_match:
                url = url_match.group()
                extracted_text = await get_text_from_url(url)
                prompt += f"\n\nHere's the extracted text from the provided URL:\n{extracted_text}"
            else:
                prompt += "\n\nYou mentioned using a link, but I couldn't find a valid URL in your message."
                
        response = rag.generate(prompt)
        
        if use_google_news:
            response += "\n\nHere are the original news articles I based my summary on:\n\n"
            for item in news_items:
                response += f"- {item['title']} ({item['date']})\n  {item['link']}\n"
                
        if use_duckduckgo:
            response += "\n\nHere are the original DuckDuckGo search results I based my summary on:\n\n"
            for item in search_results:
                response += f"- {item['title']}\n  {item['link']}\n"
                
        if use_link and url_match:
            response += f"\n\nHere's the full extracted text from the provided URL:\n{extracted_text}"
            
        user_message_id = await store_message(chat_id, message, True)
        assistant_message_id = await store_message(chat_id, response, False)
        
        chat_title = await generate_chat_title(chat_id)
        
        return jsonify({
            'response': response, 
            'chat_id': chat_id, 
            'chat_title': chat_title,
            'user_message_id': user_message_id,
            'assistant_message_id': assistant_message_id
        })
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An error occurred while processing your request. Please try again.'}), 500
    
@app.route('/upload', methods=['POST'])
async def upload_file():
    try:
        form = await request.form
        files = await request.files
        if 'file' not in files:
            return jsonify({"error": "No file part"}), 400
        
        file = files['file']
        chat_id = form.get('chat_id')
        
        if not chat_id:
            chat_id = await create_chat()
            
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        file_path = await save_uploaded_file(file, chat_id)
        
        if file_path:
            mime_type, _ = mimetypes.guess_type(file_path)
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if mime_type and mime_type.startswith('image'):
                file_content = await process_image_with_llava(file_path)
            else:
                file_content = await read_file(file_path)
                
            message = f"I've uploaded a file named {file.filename} for this chat. Here's its content or description: {file_content}"
            
            prompt = await generate_prompt(message, chat_id)
            response = await get_llm_response(prompt)
            
            user_message_id = await store_message(chat_id, message, True)
            assistant_message_id = await store_message(chat_id, response, False)
            
            chat_title = await generate_chat_title(chat_id)
            return jsonify({
                "response": response, 
                "chat_id": chat_id, 
                "chat_title": chat_title,
                "user_message_id": user_message_id,
                "assistant_message_id": assistant_message_id,
                "file_path": f"/uploads/{chat_id}/{os.path.basename(file_path)}",
                "mime_type": mime_type,
                "file_content": file_content,
                "llm_comment": response
            })
        
        return jsonify({"error": "File type not allowed"}), 400
    
    except Exception as e:
        logger.error(f"Error processing file upload: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'An error occurred while processing your file upload. Please try again.'}), 500
    
@app.route('/uploads/<int:chat_id>/<path:filename>')
async def uploaded_file(chat_id, filename):
    return await send_file(os.path.join(get_chat_upload_folder(chat_id), filename))

@app.route('/chat_history', methods=['GET'])
async def get_chats():
    try:
        async with aiosqlite.connect('assistant.db') as db:
            async with db.execute('SELECT id, created_at, title, "order", pinned, manually_renamed FROM chats ORDER BY pinned DESC, "order" ASC, created_at DESC') as cursor:
                chats = [{'id': row[0], 'created_at': row[1], 'title': row[2], 'order': row[3], 'pinned': bool(row[4]), 'manually_renamed': bool(row[5])} for row in await cursor.fetchall()]
        return jsonify({'chats': chats})
    except Exception as e:
        logger.error(f"Error fetching chat history: {str(e)}")
        return jsonify({'error': 'An error occurred while fetching chat history'}), 500
    
@app.route('/chat/<int:chat_id>', methods=['GET'])
async def get_chat(chat_id):
    try:
        messages = await get_chat_history(chat_id, limit=100)
        formatted_messages = [{'id': msg[0], 'content': msg[1], 'is_user': msg[2]} for msg in messages]
        return jsonify({'messages': formatted_messages})
    except Exception as e:
        logger.error(f"Error fetching chat: {str(e)}")
        return jsonify({'error': 'An error occurred while fetching the chat'}), 500
    
@app.route('/reorder_chats', methods=['POST'])
async def reorder_chats():
    try:
        data = await request.get_json()
        new_order = data['new_order']
        
        async with aiosqlite.connect('assistant.db') as db:
            for index, chat_id in enumerate(new_order):
                await db.execute('UPDATE chats SET "order" = ? WHERE id = ?', (index, chat_id))
            await db.commit()
            
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error reordering chats: {str(e)}")
        return jsonify({'error': 'An error occurred while reordering chats'}), 500
    
@app.route('/toggle_pin', methods=['POST'])
async def toggle_pin():
    try:
        data = await request.get_json()
        chat_id = data['chat_id']
        
        async with aiosqlite.connect('assistant.db') as db:
            await db.execute('UPDATE chats SET pinned = NOT pinned WHERE id = ?', (chat_id,))
            await db.commit()
            
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error toggling pin: {str(e)}")
        return jsonify({'error': 'An error occurred while toggling pin status'}), 500
    
@app.route('/delete_chat/<int:chat_id>', methods=['DELETE'])
async def delete_chat(chat_id):
    try:
        async with aiosqlite.connect('assistant.db') as db:
            await db.execute('DELETE FROM messages WHERE chat_id = ?', (chat_id,))
            await db.execute('DELETE FROM chats WHERE id = ?', (chat_id,))
            await db.commit()
            
        chat_folder = get_chat_upload_folder(chat_id)
        if os.path.exists(chat_folder):
            shutil.rmtree(chat_folder)
            
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error deleting chat: {str(e)}")
        return jsonify({'error': 'An error occurred while deleting the chat'}), 500
    
@app.route('/delete_message', methods=['POST'])
async def delete_message():
    try:
        data = await request.get_json()
        message_id = data['message_id']
        
        async with aiosqlite.connect('assistant.db') as db:
            await db.execute('DELETE FROM messages WHERE id = ?', (message_id,))
            await db.commit()
            
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error deleting message: {str(e)}")
        return jsonify({'error': 'An error occurred while deleting the message'}), 500
    
@app.route('/google_news', methods=['POST'])
async def google_news():
    try:
        data = await request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        news_items = await get_google_news(query)
        return jsonify({'news_items': news_items})
    except Exception as e:
        logger.error(f"Error processing Google News request: {str(e)}")
        return jsonify({'error': 'An error occurred while fetching news'}), 500
    
@app.route('/duckduckgo_search', methods=['POST'])
async def duckduckgo_search():
    try:
        data = await request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        search_results = await perform_duckduckgo_search(query)
        return jsonify({'search_results': search_results})
    except Exception as e:
        logger.error(f"Error processing DuckDuckGo search request: {str(e)}")
        return jsonify({'error': 'An error occurred while fetching search results'}), 500
    
@app.route('/update_settings', methods=['POST'])
async def update_settings():
    global image_recognition_model, llm_model, system_prompt
    try:
        data = await request.get_json()
        image_recognition_model = data.get('image_model', image_recognition_model)
        llm_model = data.get('llm_model', llm_model)
        system_prompt = data.get('system_prompt', system_prompt)
        return jsonify({'success': True, 'message': 'Settings updated successfully'})
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        return jsonify({'error': 'An error occurred while updating settings'}), 500
    
@app.route('/rename_chat', methods=['POST'])
async def rename_chat():
    try:
        data = await request.get_json()
        chat_id = data.get('chat_id')
        new_title = data.get('new_title')
        if not chat_id or not new_title:
            return jsonify({'error': 'Chat ID and new title are required'}), 400
        
        await update_chat_title(chat_id, new_title, manually_renamed=True)
        return jsonify({'success': True, 'message': 'Chat renamed successfully'})
    except Exception as e:
        logger.error(f"Error renaming chat: {str(e)}")
        return jsonify({'error': 'An error occurred while renaming the chat'}), 500

async def init_app():
    await init_db()
    config = Config()
    config.bind = ["localhost:5000"]
    await serve(app, config)
    
if __name__ == '__main__':
    asyncio.run(init_app())