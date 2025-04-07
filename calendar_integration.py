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
                
                # Store the original query in the result for language detection
                result['original_query'] = query
                
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
Extract all relevant details like event title, date, time, location, description, and category.

IMPORTANT: The user may write in English or Russian. Make sure to correctly understand their intent regardless of language.
VERY IMPORTANT: Preserve the original language of the user's query. If the user writes in English, respond with English field values.
If the user writes in Russian, respond with Russian field values.

Common Russian calendar commands and their meanings:
- "создай встречу" = create meeting/event
- "добавь событие" = add event
- "напомни мне" = remind me
- "удали встречу" = delete meeting
- "удали событие" = delete event
- "отмени встречу" = cancel meeting
- "перенеси встречу" = reschedule meeting
- "покажи мои встречи" = show my meetings
- "что у меня в календаре" = what's in my calendar
- "когда у меня встреча" = when is my meeting
- "какие встречи у меня завтра" = what meetings do I have tomorrow
- "добавь в календарь" = add to calendar
- "запланируй" = schedule
- "назначь" = appoint/schedule

Time and date expressions in Russian:
- "сегодня" = today
- "завтра" = tomorrow
- "вчера" = yesterday 
- "послезавтра" = day after tomorrow
- "на следующей неделе" = next week
- "через неделю" = in a week
- "утром" = morning
- "днем" = afternoon
- "вечером" = evening
- "в X утра/дня/вечера" = at X am/pm
- "в полдень" = at noon
- "в полночь" = at midnight

Event types in Russian:
- "встреча" = meeting
- "созвон" = call
- "совещание" = conference/meeting
- "лекция" = lecture
- "занятие" = class/lesson
- "тренировка" = training/workout
- "дедлайн" = deadline
- "праздник" = holiday
- "день рождения" = birthday

Categories for events (categories can be mentioned in both languages):
- "work"/"работа" = work-related meetings, appointments, deadlines
- "personal"/"личное" = personal appointments, family events, birthdays
- "holiday"/"праздник" = holidays, celebrations, special events
- "health"/"здоровье" = doctor appointments, medical check-ups, gym sessions
- "travel"/"поездка" = travel plans, flights, hotel bookings
- "other"/"другое" = used when no specific category is mentioned

EXAMPLES:
1. For English input "Meeting with John tomorrow at 3pm" -> {{"intent": "create", "title": "Meeting with John", "start_time": "2025-04-01T15:00:00", "category": "work"}}
2. For English input "Doctor appointment tomorrow at 10am" -> {{"intent": "create", "title": "Doctor appointment", "start_time": "2025-04-01T10:00:00", "category": "health"}}
3. For English input "Delete the team meeting on Friday" -> {{"intent": "delete", "title": "team meeting", "date": "2025-04-05"}}
4. For English input "Show all my meetings for next week" -> {{"intent": "query", "start_date": "2025-04-07", "end_date": "2025-04-13"}}
5. For Russian input "создай встречу с Иваном завтра в 10 утра" -> {{"intent": "create", "title": "Встреча с Иваном", "start_time": "2025-04-01T10:00:00", "category": "work"}}
6. For Russian input "добавь праздничный ужин завтра в 19:00" -> {{"intent": "create", "title": "Праздничный ужин", "start_time": "2025-04-01T19:00:00", "category": "holiday"}}
7. For Russian input "удали встречу с Алексеем на пятницу" -> {{"intent": "delete", "title": "встреча с Алексеем", "date": "2025-04-05"}}
8. For Russian input "какие встречи у меня завтра" -> {{"intent": "query", "date": "2025-04-01"}}
9. For Russian input "перенеси встречу с командой на среду в 15:00" -> {{"intent": "update", "original_title": "встреча с командой", "start_time": "2025-04-03T15:00:00"}}

Output a JSON object with the following structure:
{{
  "intent": "create"|"update"|"delete"|"query",
  "title": "Event title",
  "date": "YYYY-MM-DD", // Optional, for specific date queries
  "start_time": "YYYY-MM-DDTHH:MM:SS", // ISO format with timezone if available
  "end_time": "YYYY-MM-DDTHH:MM:SS", // ISO format with timezone if available
  "description": "Event description", // Optional
  "location": "Event location", // Optional
  "category": "work"|"personal"|"holiday"|"health"|"travel"|"other", // Optional, categorize the event based on context
  
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
11. VERY IMPORTANT: Always infer the most appropriate category from the event context. For example, "doctor appointment" should be "health", "team meeting" should be "work".
12. When the category isn't explicitly mentioned, infer it based on the event title or description.
13. For free-form text about calendar events, try to identify the most likely intent based on the context.
14. If query appears completely unrelated to calendar events, provide a helpful response explaining how to use calendar commands.
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
            
            # Check if original query is in English (for proper title handling)
            original_query = data.get('original_query', '')
            is_english_query = not any(ord(c) > 127 for c in original_query)
            
            if is_english_query:
                # For English commands, ensure meeting/appointment names are fully in English
                title = data['title']
                self.logger.info(f"Processing English title: {title}")
                
                # First, check for specific Russian patterns in the title
                english_patterns = {
                    'Встреча с': 'Meeting with',
                    'Встреча': 'Meeting',
                    'Событие': 'Event',
                    'Созвон с': 'Call with',
                    'Созвон': 'Call',
                    'Праздник': 'Holiday',
                    'Поездка': 'Trip',
                    'Приём у врача': 'Doctor appointment'
                }
                
                for rus_pattern, eng_pattern in english_patterns.items():
                    if title.startswith(rus_pattern):
                        # Replace Russian pattern with English equivalent
                        title = title.replace(rus_pattern, eng_pattern, 1)
                        break
                
                # Then check if the title has any Cyrillic characters and extract names from the original query
                if any(ord(c) > 127 for c in title):
                    self.logger.info(f"Detected Cyrillic characters in title: {title}")
                    
                    # Extract name parts from the original query
                    # Common pattern: "Meeting with [Name]" or similar
                    name_match = re.search(r'(?:with|and)\s+([A-Za-z]+)', original_query)
                    if name_match:
                        extracted_name = name_match.group(1)
                        
                        # Replace Russian name with English name
                        # First try exact pattern "with Романом" -> "with Roman"
                        rus_name_pattern = re.search(r'(?:with|and)\s+([А-Яа-яЁё]+)', title)
                        if rus_name_pattern:
                            title = title.replace(rus_name_pattern.group(0), f"with {extracted_name}")
                        else:
                            # Otherwise, try to find any Russian word and replace it
                            rus_words = [word for word in title.split() if any(ord(c) > 127 for c in word)]
                            for rus_word in rus_words:
                                title = title.replace(rus_word, extracted_name)
                    else:
                        # If we can't extract the name but there's Russian text, replace with generic English
                        title = re.sub(r'[А-Яа-яЁё]+', 'Meeting', title)
                    
                    self.logger.info(f"Updated title after Cyrillic character removal: {title}")
                
                # Update the title
                data['title'] = title
                self.logger.info(f"Final English title: {title}")
                
                # Make sure category is also appropriate in English
                category_translations = {
                    'работа': 'work',
                    'личное': 'personal',
                    'праздник': 'holiday',
                    'здоровье': 'health',
                    'поездка': 'travel',
                    'другое': 'other'
                }
                
                if 'category' in data and data['category'] in category_translations:
                    data['category'] = category_translations[data['category']]
                    self.logger.info(f"Translated category to English: {data['category']}")
            else:
                # For Russian commands, ensure category is in Russian if needed
                category_translations = {
                    'work': 'работа',
                    'personal': 'личное',
                    'holiday': 'праздник',
                    'health': 'здоровье',
                    'travel': 'поездка',
                    'other': 'другое'
                }
                
                # Only translate if the Russian query has English category
                if any(ord(c) > 127 for c in original_query) and 'category' in data and data['category'] in category_translations.keys():
                    # Check that the category isn't already in Russian
                    if not any(ord(c) > 127 for c in data['category']):
                        data['category'] = category_translations.get(data['category'], data['category'])
                        self.logger.info(f"Translated category to Russian: {data['category']}")
            
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
            # Try multiple possible fields for search criteria
            search_term = data.get('title') or data.get('original_title')
            
            # If no direct title is provided, try to construct a search term from other fields
            if not search_term:
                # Check for any description that might help identify the event
                if data.get('description'):
                    search_term = data.get('description')
                # If we have a date, we can use that to narrow down events
                elif data.get('date') or data.get('start_time') or data.get('start_date'):
                    # We'll still need something to search by
                    date_ref = data.get('date') or data.get('start_date') or data.get('start_time', '').split('T')[0]
                    self.logger.info(f"Using date reference for event search: {date_ref}")
                    
                    # Let's get all events for this date and let the user pick from them
                    if date_ref:
                        try:
                            date_str = self._normalize_date(date_ref)
                            start_date = f"{date_str}T00:00:00"
                            end_date = f"{date_str}T23:59:59"
                            events = await self.calendar_manager.get_events(start_date, end_date)
                            
                            if events:
                                # Use the first event found on this date
                                event_id = events[0]['id']
                                success = await self.calendar_manager.delete_event(event_id)
                                
                                if success:
                                    return {
                                        'success': True,
                                        'deleted_event': events[0],
                                        'action': 'deleted',
                                        'message': f"Deleted the first event found on {date_str}"
                                    }
                        except Exception as date_error:
                            self.logger.error(f"Error using date for event search: {str(date_error)}")
            
            # Still no search term, report the error
            if not search_term:
                # Try to provide a more helpful error message
                self.logger.error(f"Missing search criteria for event deletion from data: {data}")
                return {
                    'success': False,
                    'error': 'Missing search criteria for event deletion',
                    'message': 'I need more information to delete an event. Please specify the event title or date.'
                }
            
            events = await self.calendar_manager.get_events(search_term=search_term)
            
            if not events:
                return {
                    'success': False,
                    'error': f'No events found matching "{search_term}"',
                    'message': f'I couldn\'t find any events matching "{search_term}".'
                }
            
            # Delete the first matching event
            event_id = events[0]['id']
            success = await self.calendar_manager.delete_event(event_id)
            
            if success:
                return {
                    'success': True,
                    'deleted_event': events[0],
                    'action': 'deleted',
                    'message': f'Successfully deleted the event "{events[0]["title"]}".'
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to delete event',
                    'message': 'I encountered a technical issue while deleting the event. Please try again.'
                }
        except Exception as e:
            self.logger.error(f"Error deleting event: {str(e)}")
            return {
                'success': False,
                'error': f'Error deleting event: {str(e)}',
                'message': 'Sorry, I encountered an error while trying to delete the event.'
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
        self.logger.info(f"Normalizing date: {date_str}")
        
        if not date_str:
            self.logger.info("No date provided, returning None")
            return None
            
        try:
            # If it's already a proper ISO date or datetime string
            if isinstance(date_str, str) and 'T' in date_str:
                # This is likely an ISO datetime, extract just the date
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                self.logger.info(f"Parsed ISO datetime to date: {dt.date().isoformat()}")
                return dt.date().isoformat()
                
            if isinstance(date_str, str) and re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                # This is already an ISO date
                self.logger.info(f"Already in ISO date format: {date_str}")
                return date_str
        except (ValueError, TypeError) as e:
            self.logger.info(f"Not a valid ISO date/datetime: {e}")
            
        # Try to parse natural language dates
        now = datetime.now()
        
        if isinstance(date_str, str):
            lower_str = date_str.lower()
            
            # Check for Russian and English version of dates
            # Tomorrow
            if any(word in lower_str for word in ["tomorrow", "завтра"]):
                result = (now + timedelta(days=1)).date().isoformat()
                self.logger.info(f"Parsed 'tomorrow/завтра' to: {result}")
                return result
                
            # Today
            elif any(word in lower_str for word in ["today", "сегодня"]):
                result = now.date().isoformat()
                self.logger.info(f"Parsed 'today/сегодня' to: {result}")
                return result
                
            # Day after tomorrow
            elif any(pattern in lower_str for pattern in ["day after tomorrow", "послезавтра", "after tomorrow"]):
                result = (now + timedelta(days=2)).date().isoformat()
                self.logger.info(f"Parsed 'day after tomorrow/послезавтра' to: {result}")
                return result
                
            # Next week (assuming 7 days from now)
            elif any(pattern in lower_str for pattern in ["next week", "следующая неделя", "следующей неделе"]):
                result = (now + timedelta(days=7)).date().isoformat()
                self.logger.info(f"Parsed 'next week/следующая неделя' to: {result}")
                return result
                
            # This week (today until end of week - Sunday)
            elif any(pattern in lower_str for pattern in ["this week", "эта неделя", "этой неделе"]):
                # Keep today's date
                result = now.date().isoformat()
                self.logger.info(f"Parsed 'this week/эта неделя' to: {result}")
                return result
                
            # Weekend (nearest upcoming Saturday)
            elif any(pattern in lower_str for pattern in ["weekend", "выходные"]):
                # Find the next Saturday
                days_ahead = 5 - now.weekday()  # Saturday is 5
                if days_ahead < 0:  # If today is already Sunday
                    days_ahead += 7
                next_saturday = now + timedelta(days=days_ahead)
                result = next_saturday.date().isoformat()
                self.logger.info(f"Parsed 'weekend/выходные' to next Saturday: {result}")
                return result
                
            # Specific weekday (e.g., "on Monday", "в понедельник")
            weekdays_en = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            weekdays_ru = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
            
            for i, (en, ru) in enumerate(zip(weekdays_en, weekdays_ru)):
                if en in lower_str or ru in lower_str:
                    # Calculate days until next occurrence of this weekday
                    days_ahead = i - now.weekday()
                    if days_ahead <= 0:  # If today is the requested day or already passed
                        days_ahead += 7
                    next_day = now + timedelta(days=days_ahead)
                    result = next_day.date().isoformat()
                    self.logger.info(f"Parsed weekday '{en}/{ru}' to: {result}")
                    return result
                    
            # Date formats like MM/DD, MM-DD, DD.MM
            # Try to match various date formats
            # Format MM/DD or MM-DD (US format)
            us_date_match = re.search(r'(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?', lower_str)
            if us_date_match:
                month, day, year = us_date_match.groups()
                if year is None:
                    year = now.year
                else:
                    # Handle 2-digit years
                    if len(year) == 2:
                        year = f"20{year}" if int(year) < 50 else f"19{year}"
                
                try:
                    parsed_date = datetime(int(year), int(month), int(day)).date()
                    result = parsed_date.isoformat()
                    self.logger.info(f"Parsed date format MM/DD/YYYY to: {result}")
                    return result
                except ValueError as e:
                    self.logger.info(f"Invalid date in MM/DD/YYYY format: {e}")
            
            # Format DD.MM or DD,MM (European/Russian format)
            eu_date_match = re.search(r'(\d{1,2})[.,](\d{1,2})(?:[.,](\d{2,4}))?', lower_str)
            if eu_date_match:
                day, month, year = eu_date_match.groups()
                if year is None:
                    year = now.year
                else:
                    # Handle 2-digit years
                    if len(year) == 2:
                        year = f"20{year}" if int(year) < 50 else f"19{year}"
                
                try:
                    parsed_date = datetime(int(year), int(month), int(day)).date()
                    result = parsed_date.isoformat()
                    self.logger.info(f"Parsed date format DD.MM.YYYY to: {result}")
                    return result
                except ValueError as e:
                    self.logger.info(f"Invalid date in DD.MM.YYYY format: {e}")
        
        # Return current date as fallback
        self.logger.info(f"No date pattern matched, using current date as fallback: {now.date().isoformat()}")
        return now.date().isoformat()
    
    async def _generate_natural_response(self, intent_data, action_result, selected_model=None):
        """Generate a natural language response based on the action result"""
        try:
            # Better language detection for input - check original query first, then title
            original_query = intent_data.get('original_query', '')
            title = intent_data.get('title', '')
            
            # Default to English
            is_russian = False
            
            # First method: Look for Cyrillic characters in original query
            if any(ord(c) > 127 for c in original_query):
                is_russian = True
            
            # Second method: Check for Russian words in the original query
            russian_keywords = ['встреча', 'событие', 'создай', 'добавь', 'завтра', 'сегодня', 'календарь']
            for word in russian_keywords:
                if word in original_query.lower():
                    is_russian = True
                    break
            
            # English detection - if original query contains English time/date patterns
            # Look for English AM/PM, "meeting", etc.
            english_patterns = ['am', 'pm', 'meeting', 'event', 'appointment', 'tomorrow', 'today']
            for pattern in english_patterns:
                if f' {pattern} ' in f' {original_query.lower()} ' or original_query.lower().endswith(f' {pattern}'):
                    is_russian = False
                    break
            
            self.logger.info(f"Language detection: original_query='{original_query}', is_russian={is_russian}")
            
            # Category translations for display in responses
            category_display = {
                # English categories
                'work': 'Work',
                'personal': 'Personal',
                'holiday': 'Holiday',
                'health': 'Health',
                'travel': 'Travel',
                'other': 'Other',
                'default': 'Default',
                
                # Russian categories (for when category is already in Russian)
                'работа': 'Работа',
                'личное': 'Личное',
                'праздник': 'Праздник',
                'здоровье': 'Здоровье',
                'поездка': 'Поездка',
                'другое': 'Другое'
            }
            
            if action_result.get('success', False):
                if intent_data.get('intent') == 'create':
                    event = action_result.get('event', {})
                    start_time = event.get('start_time', '')
                    formatted_time = ''
                    
                    try:
                        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                        formatted_time = start_dt.strftime('%B %d, %Y at %I:%M %p')
                    except:
                        formatted_time = start_time
                    
                    # Include category in response if available
                    category_info = ""
                    if event.get('category') and event.get('category') != 'default':
                        display_category = category_display.get(event.get('category'), event.get('category'))
                        if is_russian:
                            category_info = f" (категория: {display_category})"
                        else:
                            category_info = f" (category: {display_category})"
                    
                    if is_russian:
                        return f"Событие \"{event.get('title', '')}\" добавлено в календарь на {formatted_time}{category_info}."
                    else:
                        return f"Event \"{event.get('title', '')}\" has been added to your calendar for {formatted_time}{category_info}."
                
                elif intent_data.get('intent') == 'update':
                    event = action_result.get('event', {})
                    original = action_result.get('original', {})
                    
                    # Include category in response if it was updated
                    category_info = ""
                    if event.get('category') != original.get('category') and event.get('category') != 'default':
                        display_category = category_display.get(event.get('category'), event.get('category'))
                        if is_russian:
                            category_info = f" Новая категория: {display_category}."
                        else:
                            category_info = f" New category: {display_category}."
                    
                    if is_russian:
                        return f"Событие \"{original.get('title', '')}\" было обновлено в календаре.{category_info}"
                    else:
                        return f"Event \"{original.get('title', '')}\" has been updated in your calendar.{category_info}"
                
                elif intent_data.get('intent') == 'delete':
                    deleted = action_result.get('deleted', {})
                    
                    if is_russian:
                        return f"Событие \"{deleted.get('title', '')}\" было удалено из календаря."
                    else:
                        return f"Event \"{deleted.get('title', '')}\" has been deleted from your calendar."
                
                elif intent_data.get('intent') == 'query':
                    events = action_result.get('events', [])
                    
                    if not events:
                        if is_russian:
                            return "На указанную дату нет запланированных событий."
                        else:
                            return "You don't have any scheduled events for the specified date."
                    
                    # Format the events list with category information
                    formatted_events = []
                    for event in events:
                        start_time = event.get('start_time', '')
                        try:
                            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                            time_str = start_dt.strftime('%B %d, %Y at %I:%M %p')
                        except:
                            time_str = start_time
                        
                        # Add category if available
                        category_str = ""
                        if event.get('category') and event.get('category') != 'default':
                            display_category = category_display.get(event.get('category'), event.get('category'))
                            if is_russian:
                                category_str = f" - {display_category}"
                            else:
                                category_str = f" - {display_category}"
                        
                        formatted_events.append(f"- {event.get('title', '')} ({time_str}){category_str}")
                    
                    events_text = '\n'.join(formatted_events)
                    
                    if is_russian:
                        return f"Вот ваши запланированные события:\n\n{events_text}"
                    else:
                        return f"Here are your scheduled events:\n\n{events_text}"
                
                else:
                    if is_russian:
                        return "Операция с календарем выполнена успешно."
                    else:
                        return "Calendar operation completed successfully."
            else:
                error = action_result.get('error', '')
                
                if is_russian:
                    return f"Не удалось выполнить операцию с календарем: {error}"
                else:
                    return f"Failed to complete the calendar operation: {error}"
        
        except Exception as e:
            self.logger.error(f"Error generating natural response: {e}")
            return "An error occurred while processing your calendar request."
    
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
