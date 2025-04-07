#!/bin/bash

echo "Starting calendar integration update..."

# Create calendar_integration.py file
cat > calendar_integration.py << 'EOF'
import traceback
import json
from datetime import datetime, timedelta
import logging

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
            
            # Try to parse the LLM's response as JSON
            try:
                result = json.loads(llm_response)
                
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
            except json.JSONDecodeError:
                # If we couldn't parse the LLM response as JSON, create a generic response
                self.logger.warning(f"Failed to parse LLM response as JSON: {llm_response}")
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
1. Parse dates intelligently, converting relative dates like "tomorrow" or "next Tuesday" to ISO dates.
2. Set default meeting duration to 1 hour if not specified.
3. For morning events without specific time, default to 9:00 AM.
4. For afternoon events without specific time, default to 2:00 PM.
5. For evening events without specific time, default to 6:00 PM.
6. Include only the fields that are relevant to the specific intent.
7. Return VALID JSON that can be parsed programmatically.
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
                return {
                    'success': False,
                    'error': 'Missing required fields: title or start_time'
                }
            
            # Normalize data for event creation
            event_data = {
                'title': data['title'],
                'start_time': self._normalize_datetime(data['start_time']),
                'end_time': self._normalize_datetime(data.get('end_time')),
                'description': data.get('description', ''),
                'location': data.get('location', ''),
                'category': data.get('category', 'default')
            }
            
            # If end_time is not provided, set it to one hour after start_time
            if not event_data['end_time']:
                start = datetime.fromisoformat(event_data['start_time'].replace('Z', '+00:00'))
                event_data['end_time'] = (start + timedelta(hours=1)).isoformat()
            
            event_id = await self.calendar_manager.create_event(event_data)
            
            if event_id:
                event = await self.calendar_manager.get_event(event_id)
                return {
                    'success': True,
                    'event': event,
                    'action': 'created'
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to create event'
                }
        except Exception as e:
            self.logger.error(f"Error creating event: {str(e)}")
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
            
            if 'start_date' in data:
                start_date = f"{self._normalize_date(data['start_date'])}T00:00:00"
            
            if 'end_date' in data:
                end_date = f"{self._normalize_date(data['end_date'])}T23:59:59"
            
            if 'title' in data:
                search_term = data['title']
            
            # Query events
            events = await self.calendar_manager.get_events(start_date, end_date, search_term)
            
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
            return {
                'success': False,
                'error': f'Error querying events: {str(e)}'
            }
    
    def _normalize_datetime(self, datetime_str):
        """Normalize datetime strings to ISO format"""
        if not datetime_str:
            return None
            
        try:
            # If it's already a proper ISO string
            datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            return datetime_str
        except (ValueError, TypeError):
            # Try to parse natural language dates
            now = datetime.now()
            
            if isinstance(datetime_str, str):
                lower_str = datetime_str.lower()
                
                if "tomorrow" in lower_str:
                    result_date = now + timedelta(days=1)
                    
                    # Check for time indicators
                    if "morning" in lower_str:
                        result_date = result_date.replace(hour=9, minute=0, second=0, microsecond=0)
                    elif "afternoon" in lower_str:
                        result_date = result_date.replace(hour=14, minute=0, second=0, microsecond=0)
                    elif "evening" in lower_str:
                        result_date = result_date.replace(hour=18, minute=0, second=0, microsecond=0)
                    else:
                        # Default to 9 AM if no time specified
                        result_date = result_date.replace(hour=9, minute=0, second=0, microsecond=0)
                        
                    return result_date.isoformat()
                    
                elif "today" in lower_str:
                    result_date = now
                    
                    # Check for time indicators
                    if "morning" in lower_str:
                        result_date = result_date.replace(hour=9, minute=0, second=0, microsecond=0)
                    elif "afternoon" in lower_str:
                        result_date = result_date.replace(hour=14, minute=0, second=0, microsecond=0)
                    elif "evening" in lower_str:
                        result_date = result_date.replace(hour=18, minute=0, second=0, microsecond=0)
                    else:
                        # Default to current time
                        pass
                        
                    return result_date.isoformat()
            
            # Return current time as fallback
            return now.isoformat()
    
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
EOF
echo "Created calendar_integration.py"

# Create a JavaScript file with the necessary client-side updates
cat > calendar_chat_integration.js << 'EOF'
// Enhanced version of the sendMessage function to handle calendar updates
async function sendMessage() {
    if (isProcessing) return;
    const message = userInput.value.trim();
    if (message === '') return;
    
    const useDuckDuckGo = document.getElementById('use-duckduckgo').checked;
    const useGoogleNews = document.getElementById('use-google-news').checked;
    const useLink = useLinkCheckbox.checked;
    const useRag = document.getElementById('use-rag').checked;
    const selectedModel = modelSelect.value;
    
    isProcessing = true;
    sendButton.disabled = true;
    const loadingIndicator = document.createElement('div');
    loadingIndicator.className = 'loading';
    chatContainer.appendChild(loadingIndicator);
    
    try {
        const response = await fetch('http://127.0.0.1:5001/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                chat_id: currentChatId,
                use_duckduckgo: useDuckDuckGo,
                use_google_news: useGoogleNews,
                use_link: useLink,
                use_rag: useRag,
                selected_model: selectedModel,
                system_prompt: systemPrompt
            }),
        });
        const data = await response.json();
        addMessage(message, true, data.user_message_id);
        addMessage(data.response, false, data.assistant_message_id);
        
        if (data.chat_id) {
            currentChatId = data.chat_id;
            await updateChatHistory();
        }
        
        if (data.chat_title) {
            updateChatTitle(currentChatId, data.chat_title);
        }
        
        // Check if we need to update the calendar display
        if (data.update_calendar) {
            updateCalendarDisplay();
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('Sorry, there was an error processing your request.', false, null);
    } finally {
        loadingIndicator.remove();
        userInput.value = '';
        isProcessing = false;
        sendButton.disabled = false;
    }
}

// Function to update calendar display
function updateCalendarDisplay() {
    // Check if calendar module is initialized
    if (window.calendarModule) {
        console.log("Updating calendar display after LLM-processed calendar operation");
        
        // Reload events from server
        window.calendarModule.loadEvents()
            .then(() => {
                // Render calendar views
                window.calendarModule.renderCalendar();
                window.calendarModule.renderUpcomingEvents();
                
                // Optionally, show a notification
                window.calendarModule.showNotification("Calendar updated successfully", false);
            })
            .catch(error => {
                console.error("Error refreshing calendar after LLM operation:", error);
                window.calendarModule.showNotification("Error updating calendar", true);
            });
        
        // If the calendar tab isn't currently visible, add a notification indicator
        const calendarTab = document.querySelector('.sidebar-tab[data-tab="calendar"]');
        if (calendarTab && !calendarTab.classList.contains('active')) {
            // Add notification dot or indicator
            if (!calendarTab.querySelector('.calendar-update-indicator')) {
                const indicator = document.createElement('span');
                indicator.className = 'calendar-update-indicator';
                indicator.style.display = 'inline-block';
                indicator.style.width = '8px';
                indicator.style.height = '8px';
                indicator.style.backgroundColor = '#4CAF50';
                indicator.style.borderRadius = '50%';
                indicator.style.marginLeft = '5px';
                calendarTab.appendChild(indicator);
                
                // Remove indicator when user clicks on calendar tab
                calendarTab.addEventListener('click', function() {
                    const indicator = this.querySelector('.calendar-update-indicator');
                    if (indicator) {
                        indicator.remove();
                    }
                });
            }
        }
    }
}

// Enhanced isCalendarCommand function in CalendarModule
// This determines if a message should be processed by the calendar module
if (typeof CalendarModule !== 'undefined') {
    CalendarModule.prototype.isCalendarCommand = function(message) {
        const calendarKeywords = [
            'schedule', 'appointment', 'meeting', 'event', 'calendar',
            'add to calendar', 'create event', 'remove event', 'delete event',
            'cancel meeting', 'reschedule', 'move meeting',
            'show calendar', 'list events', 'upcoming events',
            'tomorrow', 'next week', 'sync calendar', 'remind me'
        ];
        
        const lowerMessage = message.toLowerCase();
        
        // Check for calendar-related keywords
        return calendarKeywords.some(keyword => lowerMessage.includes(keyword));
    }

    // Modify the calendar module's processCalendarCommand to work with the LLM
    CalendarModule.prototype.processCalendarCommand = async function(message) {
        try {
            // Display message in chat
            this.addMessageToChat(message, true);
            
            // Process through LLM using the /calendar/natural endpoint
            const response = await fetch('http://127.0.0.1:5001/calendar/natural', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    chat_id: window.currentChatId || null
                })
            });
            
            const data = await response.json();
            
            // Check if successful
            if (data.success) {
                // Display response in chat
                this.addMessageToChat(data.message, false);
                
                // Update calendar display
                this.refreshCalendar();
                
                return true;
            } else {
                // Display error message
                this.addMessageToChat(data.message || "I couldn't process that calendar request. Could you try again with more details?", false);
                return false;
            }
        } catch (error) {
            console.error('Error processing calendar command:', error);
            this.addMessageToChat(`I encountered an error processing your calendar request: ${error.message}`, false);
            return false;
        }
    }

    // Extend the refreshCalendar method to make it more robust
    CalendarModule.prototype.refreshCalendar = function() {
        console.log("Refreshing calendar view");
        try {
            // Reload events from server first
            this.loadEvents().then(() => {
                console.log(`Loaded ${this.events.length} events, rendering calendar`);
                // Then render calendar with new data
                this.renderCalendar();
                this.renderUpcomingEvents();
            }).catch(error => {
                console.error("Error loading events during refresh:", error);
                // Try to render with existing data anyway
                this.renderCalendar();
                this.renderUpcomingEvents();
            });
        } catch (error) {
            console.error("Error refreshing calendar:", error);
        }
    }
}

// Load this script after the page has loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log("Calendar chat integration loaded");
    
    // Override the existing sendMessage function
    if (typeof window.sendMessage === 'function') {
        console.log("Replacing existing sendMessage function");
        window.sendMessage = sendMessage;
    }
    
    // Add event listener for tab switching
    const tabs = document.querySelectorAll('.sidebar-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            if (e.target.getAttribute('data-tab') === 'calendar') {
                // Remove notification indicator if present
                const indicator = e.target.querySelector('.calendar-update-indicator');
                if (indicator) {
                    indicator.remove();
                }
            }
        });
    });
});
EOF
echo "Created calendar_chat_integration.js"

# Update server.py
if [ -f "server.py" ]; then
    echo "Updating server.py..."
    
    # Add import at the top of the file
    sed -i '1s/^/# Import calendar integration\nimport calendar_integration\n\n/' server.py
    
    # Find the line where calendar_manager is initialized and add code after it
    LINE_NUM=$(grep -n "calendar_manager = CalendarManager()" server.py | cut -d: -f1)
    if [ -n "$LINE_NUM" ]; then
        LINE_NUM=$((LINE_NUM + 2))
        sed -i "${LINE_NUM}i# Initialize calendar-RAG integration\ncalendar_rag = CalendarRAGIntegration(calendar_manager, rag_system)" server.py
    else
        echo "WARNING: Could not find where calendar_manager is initialized in server.py"
    fi
    
    # Update the /calendar/natural endpoint
    if grep -q "@app.route('/calendar/natural'" server.py; then
        ENDPOINT_START=$(grep -n "@app.route('/calendar/natural'" server.py | cut -d: -f1)
        if [ -n "$ENDPOINT_START" ]; then
            # Find the end of the function
            ENDPOINT_END=$(tail -n +$ENDPOINT_START server.py | grep -n "^@app" | head -1 | cut -d: -f1)
            if [ -n "$ENDPOINT_END" ]; then
                ENDPOINT_END=$((ENDPOINT_START + ENDPOINT_END - 1))
                
                # Replace the entire function
                sed -i "${ENDPOINT_START},${ENDPOINT_END}c\\
@app.route('/calendar/natural', methods=['POST'])\\
async def process_natural_language_calendar():\\
    try:\\
        data = await request.get_json()\\
        message = data.get('message')\\
        chat_id = data.get('chat_id')\\
        selected_model = data.get('selected_model', llm_model)\\
        \\
        if not message:\\
            return jsonify({'error': 'No message provided'}), 400\\
        \\
        # Process with calendar-RAG integration\\
        result = await calendar_rag.process_calendar_query(\\
            message,\\
            chat_id=chat_id,\\
            selected_model=selected_model,\\
            system_prompt=system_prompt\\
        )\\
        \\
        # Store in chat history if chat_id is provided\\
        if chat_id and result.get('success'):\\
            chat_id = int(chat_id)\\
            await store_message(chat_id, message, True)\\
            await store_message(chat_id, result.get('message', ''), False)\\
        \\
        return jsonify(result)\\
            \\
    except Exception as e:\\
        logger.error(f\"Error processing natural language calendar command: {str(e)}\")\\
        logger.error(traceback.format_exc())\\
        return jsonify({'error': 'An error occurred while processing the calendar command'}), 500\\
\\
" server.py
                echo "Updated /calendar/natural endpoint"
            else
                echo "WARNING: Could not find the end of /calendar/natural function"
            fi
        fi
    else
        echo "WARNING: Could not find /calendar/natural endpoint in server.py"
    fi
    
    # Update the /chat endpoint to check for calendar-related messages
    if grep -q "@app.route('/chat'" server.py; then
        ENDPOINT_START=$(grep -n "@app.route('/chat'" server.py | cut -d: -f1)
        if [ -n "$ENDPOINT_START" ]; then
            # Find the part where the response is generated
            RESPONSE_LINE=$(tail -n +$ENDPOINT_START server.py | grep -n "response = rag_system.generate" | head -1 | cut -d: -f1)
            if [ -n "$RESPONSE_LINE" ]; then
                RESPONSE_LINE=$((ENDPOINT_START + RESPONSE_LINE - 10))
                
                # Add calendar check before generating the response
                sed -i "${RESPONSE_LINE}i\\
        # First, check if this is a calendar-related message\\
        if any(keyword in message.lower() for keyword in [\\
            'calendar', 'schedule', 'event', 'meeting', 'appointment', \\
            'remind', 'reminder', 'plan', 'agenda'\\
        ]):\\
            # Process as potential calendar request\\
            calendar_result = await calendar_rag.process_calendar_query(\\
                message, \\
                chat_id=chat_id,\\
                selected_model=selected_model,\\
                system_prompt=current_system_prompt\\
            )\\
            \\
            if calendar_result.get('success'):\\
                # Store messages in database\\
                user_message_id = await store_message(chat_id, message, True)\\
                assistant_message_id = await store_message(chat_id, calendar_result.get('message', ''), False)\\
                \\
                # Generate chat title if needed\\
                chat_title = await generate_chat_title(chat_id)\\
                \\
                response = calendar_result.get('message', '')\\
                \\
                # Add update_calendar flag to trigger frontend refresh\\
                return jsonify({\\
                    'response': response,\\
                    'chat_id': chat_id,\\
                    'chat_title': chat_title,\\
                    'user_message_id': user_message_id,\\
                    'assistant_message_id': assistant_message_id,\\
                    'update_calendar': True  # This signals the frontend to refresh calendar\\
                })\\
\\
        # If not a calendar request or calendar processing failed, proceed with normal chat flow\\
" server.py
                echo "Updated /chat endpoint with calendar check"
            else
                echo "WARNING: Could not find where the response is generated in /chat endpoint"
            fi
            
            # Add update_calendar flag to the final jsonify response
            RETURN_LINE=$(tail -n +$ENDPOINT_START server.py | grep -n "return jsonify" | tail -1 | cut -d: -f1)
            if [ -n "$RETURN_LINE" ]; then
                RETURN_LINE=$((ENDPOINT_START + RETURN_LINE - 1))
                sed -i "${RETURN_LINE}s/assistant_message_id/assistant_message_id,\n            'update_calendar': False  # Default to false for non-calendar requests/" server.py
                echo "Added update_calendar flag to final response"
            else
                echo "WARNING: Could not find return jsonify in /chat endpoint"
            fi
        fi
    else
        echo "WARNING: Could not find /chat endpoint in server.py"
    fi
else
    echo "ERROR: server.py not found in current directory"
fi

# Add script tag to chat.html
if [ -f "chat.html" ]; then
    echo "Updating chat.html..."
    
    # Check if there's already an app.js script tag
    if grep -q '<script src="app.js"></script>' chat.html; then
        # Add our script tag after the app.js script
        sed -i 's/<script src="app.js"><\/script>/<script src="app.js"><\/script>\n    <script src="calendar_chat_integration.js"><\/script>/' chat.html
        echo "Added calendar_chat_integration.js script to chat.html"
    else
        echo "WARNING: Could not find app.js script tag in chat.html"
    fi
else
    echo "WARNING: chat.html not found in current directory"
fi

echo "Update completed. Please check the modifications and restart your server."