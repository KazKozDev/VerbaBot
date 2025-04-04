import traceback
import json
from datetime import datetime, timedelta
import logging
import re

class CalendarRAGIntegration:
    """Integration layer between Calendar functionality, RAG system, and LLM"""
    
    def __init__(self, calendar_manager, rag_system):
        self.calendar_manager = calendar_manager
        self.rag_system = rag_system
        self.logger = logging.getLogger(__name__)
    
    async def process_calendar_query(self, query, chat_id=None, selected_model=None, system_prompt=None):
        """Process a calendar-related query through the LLM and execute the appropriate action"""
        try:
            # Get chat history for context if chat_id is provided
            chat_history = []
            if chat_id:
                # Import get_chat_history from the global scope
                from server import get_chat_history
                chat_history = await get_chat_history(chat_id, limit=5)
                chat_history.reverse()  # Most recent last
            
            # Build a prompt that helps the LLM understand and extract calendar actions
            prompt = self._build_calendar_prompt(query, chat_history, system_prompt)
            
            # Send to LLM for understanding intent
            llm_response = self.rag_system.query_ollama(prompt, model=selected_model)
            
            # Log full LLM response for debugging
            self.logger.info(f"Raw LLM response: {llm_response}")
            
            # Try to parse the LLM's response as JSON
            try:
                # Clean up the response to handle markdown code blocks
                cleaned_response = self._clean_json_response(llm_response)
                self.logger.info(f"Cleaned JSON response: {cleaned_response}")
                
                result = json.loads(cleaned_response)
                
                # Process calendar action based on intent
                calendar_response = await self._execute_calendar_action(result)
                
                # Generate a natural language response based on the action result
                natural_response = await self._generate_natural_response(result, calendar_response, selected_model)
                
                return {
                    'success': True,
                    'intent': result.get('intent'),
                    'action_result': calendar_response,
                    'message': natural_response,
                    'update_calendar': True
                }
            except json.JSONDecodeError as e:
                # If we couldn't parse the LLM response as JSON, create a generic response
                self.logger.warning(f"Failed to parse LLM response as JSON: {llm_response}, Error: {str(e)}")
                return {
                    'success': False,
                    'message': f"I understood you're asking about calendar functionality, but I couldn't process your request precisely. Could you rephrase your request with specific details like date, time, and event name?",
                    'update_calendar': False
                }
        except Exception as e:
            self.logger.error(f"Error processing calendar query: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'message': "I encountered an error while processing your calendar request.",
                'update_calendar': False
            }
    
    def _build_calendar_prompt(self, query, chat_history=None, system_prompt=None):
        """Build a prompt to help the LLM understand calendar intentions"""
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Start with system instructions
        base_prompt = system_prompt or "You are a helpful assistant with calendar management capabilities."
        
        calendar_instructions = f"""
You are processing a calendar-related request. The current date and time is {current_date}.

TASK: Analyze the user's message to identify the calendar operation they want to perform.
Extract all relevant details like event title, date, time, location, and description.

IMPORTANT: The user may write in English or Russian. Make sure to correctly understand their intent regardless of language.
For Russian:
- "создай встречу" = create meeting/event
- "добавь событие" = add event
- "напомни мне" = remind me
- "сегодня" = today
- "завтра" = tomorrow
- "утром" = morning
- "днем" = afternoon
- "вечером" = evening
- "в X утра/дня/вечера" = at X am/pm
- "какие встречи" = what meetings
- "какие у меня встречи" = what meetings do I have
- "мои встречи" = my meetings
- "покажи события" = show events
- "покажи встречи" = show meetings
- "что в календаре" = what's in the calendar

EXAMPLES OF RUSSIAN REQUESTS:
1. "создай встречу с Иваном завтра в 10 утра" -> {{"intent": "create", "title": "Встреча с Иваном", "start_time": "2025-04-01T10:00:00"}}
2. "добавь событие тренировка сегодня вечером" -> {{"intent": "create", "title": "Тренировка", "start_time": "2025-03-31T18:00:00"}}
3. "какие встречи у меня сегодня" -> {{"intent": "query", "date": "{datetime.now().date().isoformat()}"}}
4. "покажи события на завтра" -> {{"intent": "query", "date": "{(datetime.now() + timedelta(days=1)).date().isoformat()}"}}
5. "какие у меня встречи" -> {{"intent": "query"}} // Defaults to today's events if no date specified
6. "мои встречи" -> {{"intent": "query"}} // Same as above, defaults to today

Output a JSON object with the following structure:
{{
  "intent": "create"|"update"|"delete"|"query",
  "title": "Event title",
  "date": "YYYY-MM-DD", // Optional, for specific date queries
  "start_time": "YYYY-MM-DDTHH:MM:SS", // ISO format with timezone if available
  "end_time": "YYYY-MM-DDTHH:MM:SS", // ISO format with timezone if available
  "description": "Event description", // Optional
  "location": "Event location", // Optional
  "category": "default"|"work"|"personal"|"holiday"|"other", // Optional
  
  // For update operations only:
  "original_title": "Original event title to update", // Only for update intent
  "new_title": "New event title", // Only for update intent when changing title
  
  // For query operations:
  "start_date": "YYYY-MM-DD", // Optional, for date range queries
  "end_date": "YYYY-MM-DD" // Optional, for date range queries
}}

IMPORTANT RULES:
1. Parse dates intelligently, converting relative dates like "tomorrow", "today", "завтра", "сегодня" to ISO dates.
2. Set default meeting duration to 1 hour if not specified.
3. For morning events without specific time, default to 9:00 AM.
4. For afternoon events without specific time, default to 2:00 PM.
5. For evening events without specific time, default to 6:00 PM.
6. Time mentions like "11 pm", "11 вечера", or "11:00" should be properly parsed.
7. Include only the fields that are relevant to the specific intent.
8. Return VALID JSON that can be parsed programmatically.
9. For Russian text, extract proper event titles following Russian grammar rules.
10. For query intent, use the date field for specific dates, or start_date/end_date for ranges.
11. VERY IMPORTANT: Simple queries like "какие у меня встречи" (what meetings do I have) or "мои встречи" (my meetings) should ALWAYS be interpreted as a query request with intent "query" without any date limitation. These are general requests for all meetings.
"""
        
        # Add chat history context if available
        context = ""
        if chat_history and len(chat_history) > 0:
            history_parts = []
            for entry in chat_history:
                role = "Human" if entry.get("is_user") else "Assistant"
                content = entry.get("content", "")
                history_parts.append(f"{role}: {content}")
            context = "Previous conversation:\n" + "\n".join(history_parts) + "\n\n"
        
        # Combine everything
        full_prompt = f"{base_prompt}\n\n{calendar_instructions}\n\n{context}Human calendar request: {query}\n\nJSON response:"
        
        return full_prompt
    
    async def _execute_calendar_action(self, intent_data):
        """Execute the calendar action based on the intent extracted by the LLM"""
        intent = intent_data.get('intent')
        
        if intent == 'create':
            return await self._create_event(intent_data)
        elif intent == 'update':
            return await self._update_event(intent_data)
        elif intent == 'delete':
            return await self._delete_event(intent_data)
        elif intent == 'query':
            return await self._query_events(intent_data)
        else:
            return {
                'success': False,
                'error': f"Unknown intent: {intent}"
            }
    
    async def _create_event(self, data):
        """Create a new calendar event"""
        try:
            if 'title' not in data or 'start_time' not in data:
                self.logger.error(f"Missing required fields in calendar data: {data}")
                return {
                    'success': False,
                    'error': 'Missing required fields: title or start_time'
                }
            
            # Log raw data from LLM
            self.logger.info(f"Raw calendar data from LLM: {data}")
            
            # Normalize data for event creation
            event_data = {
                'title': data['title'],
                'start_time': self._normalize_datetime(data['start_time']),
                'end_time': self._normalize_datetime(data.get('end_time')),
                'description': data.get('description', ''),
                'location': data.get('location', ''),
                'category': data.get('category', 'default')
            }
            
            # Log normalized data
            self.logger.info(f"Normalized calendar data: {event_data}")
            
            # If end_time is not provided, set it to one hour after start_time
            if not event_data['end_time']:
                start = datetime.fromisoformat(event_data['start_time'].replace('Z', '+00:00'))
                event_data['end_time'] = (start + timedelta(hours=1)).isoformat()
                self.logger.info(f"Set end_time to one hour after start_time: {event_data['end_time']}")
            
            event_id = await self.calendar_manager.create_event(event_data)
            
            if event_id:
                event = await self.calendar_manager.get_event(event_id)
                self.logger.info(f"Successfully created event with ID {event_id}: {event}")
                return {
                    'success': True,
                    'event': event,
                    'action': 'created'
                }
            else:
                self.logger.error("Failed to create event, event_id is None")
                return {
                    'success': False,
                    'error': 'Failed to create event'
                }
        except Exception as e:
            self.logger.error(f"Error creating event: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': f'Error creating event: {str(e)}'
            }
    
    async def _update_event(self, data):
        """Update an existing calendar event"""
        try:
            # Find the event to update
            search_term = data.get('original_title', data.get('title'))
            
            if not search_term:
                return {
                    'success': False,
                    'error': 'Missing search criteria for event update'
                }
            
            events = await self.calendar_manager.get_events(search_term=search_term)
            
            if not events:
                return {
                    'success': False,
                    'error': f'No events found matching "{search_term}"'
                }
            
            # Update the first matching event
            event_id = events[0]['id']
            
            # Prepare update data
            update_data = {
                'title': data.get('new_title', data.get('title', events[0]['title'])),
                'start_time': self._normalize_datetime(data.get('start_time', events[0]['start_time'])),
                'end_time': self._normalize_datetime(data.get('end_time', events[0]['end_time'])),
                'description': data.get('description', events[0]['description']),
                'location': data.get('location', events[0]['location']),
                'category': data.get('category', events[0]['category'])
            }
            
            success = await self.calendar_manager.update_event(event_id, update_data)
            
            if success:
                updated_event = await self.calendar_manager.get_event(event_id)
                return {
                    'success': True,
                    'event': updated_event,
                    'action': 'updated',
                    'original': events[0]
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to update event'
                }
        except Exception as e:
            self.logger.error(f"Error updating event: {str(e)}")
            return {
                'success': False,
                'error': f'Error updating event: {str(e)}'
            }
    
    async def _delete_event(self, data):
        """Delete a calendar event"""
        try:
            # Find the event to delete
            search_term = data.get('title')
            
            if not search_term:
                return {
                    'success': False,
                    'error': 'Missing search criteria for event deletion'
                }
            
            events = await self.calendar_manager.get_events(search_term=search_term)
            
            if not events:
                return {
                    'success': False,
                    'error': f'No events found matching "{search_term}"'
                }
            
            # Delete the first matching event
            event_id = events[0]['id']
            success = await self.calendar_manager.delete_event(event_id)
            
            if success:
                return {
                    'success': True,
                    'deleted_event': events[0],
                    'action': 'deleted'
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to delete event'
                }
        except Exception as e:
            self.logger.error(f"Error deleting event: {str(e)}")
            return {
                'success': False,
                'error': f'Error deleting event: {str(e)}'
            }
    
    async def _query_events(self, data):
        """Query calendar events"""
        try:
            # Log the incoming query data
            self.logger.info(f"Calendar query request: {data}")
            
            # Prepare query parameters
            start_date = None
            end_date = None
            search_term = None
            
            if 'date' in data:
                # If a specific date is given, use it for both start and end
                date_value = self._normalize_date(data['date'])
                if date_value:
                    start_date = f"{date_value}T00:00:00"
                    end_date = f"{date_value}T23:59:59"
                    self.logger.info(f"Using specific date filter: {date_value}")
            
            if 'start_date' in data:
                start_date = f"{self._normalize_date(data['start_date'])}T00:00:00"
                self.logger.info(f"Using start_date filter: {start_date}")
            
            if 'end_date' in data:
                end_date = f"{self._normalize_date(data['end_date'])}T23:59:59"
                self.logger.info(f"Using end_date filter: {end_date}")
            
            if 'title' in data:
                search_term = data['title']
                self.logger.info(f"Using title search term: {search_term}")
            
            # If no date specified and query is about "today" or general meetings, use today's date
            if not start_date and not end_date:
                # Check if the query is about today's events using common terms in Russian and English
                today_terms = ["today", "сегодня", "today's", "сегодняшние"]
                query_text = json.dumps(data).lower()
                
                # Check if it's a general query about meetings with no specific timeframe
                general_meeting_query = (
                    'meeting' in query_text or 
                    'meetings' in query_text or 
                    'встреча' in query_text or 
                    'встречи' in query_text or
                    # If query intent is provided but no date params, assume today
                    (data.get('intent') == 'query' and 'date' not in data and 'start_date' not in data and 'end_date' not in data)
                )
                
                if any(term in query_text for term in today_terms) or general_meeting_query:
                    today = datetime.now().date().isoformat()
                    start_date = f"{today}T00:00:00"
                    end_date = f"{today}T23:59:59"
                    self.logger.info(f"Detected today's events query or general meetings query, using date range: {start_date} to {end_date}")
            
            # Query events
            self.logger.info(f"Querying events with params: start_date={start_date}, end_date={end_date}, search_term={search_term}")
            events = await self.calendar_manager.get_events(start_date, end_date, search_term)
            
            self.logger.info(f"Found {len(events)} events matching the query")
            
            return {
                'success': True,
                'events': events,
                'count': len(events),
                'action': 'queried',
                'query_params': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'search_term': search_term
                }
            }
        except Exception as e:
            self.logger.error(f"Error querying events: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': f'Error querying events: {str(e)}'
            }
    
    def _normalize_datetime(self, datetime_str):
        """Normalize datetime strings to ISO format"""
        self.logger.info(f"Normalizing datetime: {datetime_str}")
        
        if not datetime_str:
            self.logger.info("No datetime provided, returning None")
            return None
            
        try:
            # If it's already a proper ISO string
            parsed_datetime = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            self.logger.info(f"Valid ISO datetime parsed: {parsed_datetime.isoformat()}")
            return datetime_str
        except (ValueError, TypeError) as e:
            self.logger.info(f"Not a valid ISO datetime, trying natural language parsing: {e}")
            # Try to parse natural language dates
            now = datetime.now()
            
            if isinstance(datetime_str, str):
                lower_str = datetime_str.lower()
                
                # Check for Russian and English version of "tomorrow"
                if any(word in lower_str for word in ["tomorrow", "завтра"]):
                    result_date = now + timedelta(days=1)
                    
                    # Check for time indicators in Russian and English
                    if any(word in lower_str for word in ["morning", "утра", "утром"]):
                        result_date = result_date.replace(hour=9, minute=0, second=0, microsecond=0)
                    elif any(word in lower_str for word in ["afternoon", "дня", "днем"]):
                        result_date = result_date.replace(hour=14, minute=0, second=0, microsecond=0)
                    elif any(word in lower_str for word in ["evening", "вечера", "вечером"]):
                        result_date = result_date.replace(hour=18, minute=0, second=0, microsecond=0)
                    else:
                        # Default to 9 AM if no time specified
                        result_date = result_date.replace(hour=9, minute=0, second=0, microsecond=0)
                    
                    self.logger.info(f"Parsed 'tomorrow/завтра' to: {result_date.isoformat()}")
                    return result_date.isoformat()
                    
                # Check for Russian and English version of "today"
                elif any(word in lower_str for word in ["today", "сегодня"]):
                    result_date = now
                    
                    # Check for time indicators in Russian and English
                    if any(word in lower_str for word in ["morning", "утра", "утром"]):
                        result_date = result_date.replace(hour=9, minute=0, second=0, microsecond=0)
                    elif any(word in lower_str for word in ["afternoon", "дня", "днем"]):
                        result_date = result_date.replace(hour=14, minute=0, second=0, microsecond=0)
                    elif any(word in lower_str for word in ["evening", "вечера", "вечером"]):
                        result_date = result_date.replace(hour=18, minute=0, second=0, microsecond=0)
                    else:
                        # Default to current time
                        pass
                    
                    self.logger.info(f"Parsed 'today/сегодня' to: {result_date.isoformat()}")
                    return result_date.isoformat()
                
                # Check for specific time mentions like "11 pm" or "11 вечера"
                # Pattern matches: "10 am", "11 pm", "10 утра", "11 вечера", "2 дня", etc.
                time_pattern = r'(\d+)(?:\s*)(am|pm|утра|вечера|дня)'
                time_match = re.search(time_pattern, lower_str, re.IGNORECASE)
                if time_match:
                    hour = int(time_match.group(1))
                    period = time_match.group(2).lower()
                    
                    # Adjust for 12-hour format
                    if period in ['pm', 'вечера'] and hour < 12:
                        hour += 12
                    elif period in ['дня'] and hour < 12:  # Russian afternoon
                        hour += 12 if hour < 5 else 0  # Different handling for Russian time expressions
                    elif period in ['am', 'утра'] and hour == 12:
                        hour = 0
                    
                    result_date = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                    self.logger.info(f"Parsed time '{time_match.group(0)}' to: {result_date.isoformat()}")
                    return result_date.isoformat()
                
                # Check for 24-hour time format like "14:30" or "14.30"
                time_24h_pattern = r'(\d{1,2})[:\.](\d{2})'
                time_24h_match = re.search(time_24h_pattern, lower_str)
                if time_24h_match:
                    hour = int(time_24h_match.group(1))
                    minute = int(time_24h_match.group(2))
                    
                    result_date = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    self.logger.info(f"Parsed 24h time '{time_24h_match.group(0)}' to: {result_date.isoformat()}")
                    return result_date.isoformat()
            
            # Return current time as fallback
            self.logger.info(f"Using current time as fallback: {now.isoformat()}")
            return now.isoformat()
        except Exception as e:
            self.logger.error(f"Error normalizing datetime: {str(e)}")
            return None
    
    def _normalize_date(self, date_str):
        """Normalize date strings to YYYY-MM-DD format"""
        if not date_str:
            return None
            
        try:
            # Try to extract just the date part from an ISO datetime string
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.date().isoformat()
        except (ValueError, TypeError):
            # Try to parse natural language dates
            now = datetime.now()
            
            if isinstance(date_str, str):
                lower_str = date_str.lower()
                
                if "tomorrow" in lower_str:
                    return (now + timedelta(days=1)).date().isoformat()
                elif "today" in lower_str:
                    return now.date().isoformat()
                elif "next week" in lower_str:
                    return (now + timedelta(days=7)).date().isoformat()
            
            # Return current date as fallback
            return now.date().isoformat()
    
    async def _generate_natural_response(self, intent_data, action_result, model=None):
        """Generate a natural language response based on the action result"""
        try:
            if not action_result.get('success'):
                # If the action failed, return the error
                return f"I couldn't complete that calendar operation. {action_result.get('error', 'Something went wrong.')}"
            
            intent = intent_data.get('intent')
            
            # For query intent, we'll create a more formatted response for events
            if intent == 'query' and action_result.get('success'):
                events = action_result.get('events', [])
                count = len(events)
                
                if count == 0:
                    return "I didn't find any events matching your query."
                
                # Format a response with event details
                if count == 1:
                    event = events[0]
                    event_title = event.get('title', 'Untitled event')
                    start_time = self._format_datetime_for_display(event.get('start_time'))
                    
                    return f"I found 1 event: \"{event_title}\" at {start_time}."
                else:
                    # For multiple events, list them in chronological order
                    response = f"I found {count} events:\n\n"
                    
                    for i, event in enumerate(sorted(events, key=lambda e: e.get('start_time', ''))):
                        event_title = event.get('title', 'Untitled event')
                        start_time = self._format_datetime_for_display(event.get('start_time'))
                        response += f"{i+1}. \"{event_title}\" at {start_time}\n"
                    
                    return response
            
            # For other intents, use the LLM to generate a response
            # Create a prompt for the LLM to generate a natural response
            prompt = f"""
Create a friendly, concise response to let the user know about the result of their calendar request.

User's intent: {intent}
Original request data: {json.dumps(intent_data)}
Action result: {json.dumps(action_result)}

Response should be conversational and helpful. If it was successful, confirm what was done.
If showing event details, format dates and times in a human-readable way.
Keep the response under 3 sentences unless listing multiple events.
"""
            # Use the RAG system to generate a response
            response = self.rag_system.query_ollama(prompt, model=model)
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating natural response: {str(e)}")
            
            # Fallback responses based on intent
            if intent_data.get('intent') == 'create':
                return f"I've added the event '{intent_data.get('title')}' to your calendar."
            elif intent_data.get('intent') == 'update':
                return f"I've updated the event '{intent_data.get('title')}' in your calendar."
            elif intent_data.get('intent') == 'delete':
                return f"I've deleted the event '{intent_data.get('title')}' from your calendar."
            elif intent_data.get('intent') == 'query':
                events_count = len(action_result.get('events', []))
                return f"I found {events_count} events in your calendar."
            else:
                return "Your calendar has been updated successfully."
                
    def _format_datetime_for_display(self, datetime_str):
        """Format a datetime string for display in a natural way"""
        if not datetime_str:
            return "unspecified time"
            
        try:
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            today = datetime.now().date()
            tomorrow = today + timedelta(days=1)
            
            # Format the date part based on proximity to today
            if dt.date() == today:
                date_part = "today"
            elif dt.date() == tomorrow:
                date_part = "tomorrow"
            else:
                # Use more formal format for other dates
                date_part = dt.strftime("%A, %B %d")
                
            # Format the time part
            time_part = dt.strftime("%I:%M %p").lstrip("0")
            
            return f"{date_part} at {time_part}"
        except Exception:
            # If parsing fails, return the original string
            return datetime_str
    
    def _clean_json_response(self, response):
        """Clean up LLM response to extract valid JSON"""
        # Check if the response is wrapped in a markdown code block
        if response.startswith("```") and "```" in response[3:]:
            # Extract content between markdown code block markers
            pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip()
        
        # Remove any other markdown formatting or leading/trailing whitespace
        return response.strip()
