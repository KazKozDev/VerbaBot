// Calendar Chat Integration
// This file handles integration between chat and calendar functionality

document.addEventListener('DOMContentLoaded', function() {
    // Enhanced version of the sendMessage function to handle calendar updates
    const originalSendMessage = window.sendMessage;
    
    if (originalSendMessage) {
        // Override the original sendMessage function
        window.sendMessage = async function() {
            if (window.isProcessing) return;
            const message = window.userInput.value.trim();
            if (message === '') return;
            
            const useDuckDuckGo = document.getElementById('use-duckduckgo').checked;
            const useGoogleNews = document.getElementById('use-google-news').checked;
            const useLink = window.useLinkCheckbox.checked;
            const useRag = document.getElementById('use-rag').checked;
            const selectedModel = window.modelSelect.value;
            
            window.isProcessing = true;
            window.sendButton.disabled = true;
            const loadingIndicator = document.createElement('div');
            loadingIndicator.className = 'loading';
            window.chatContainer.appendChild(loadingIndicator);
            
            try {
                const response = await fetch('http://127.0.0.1:5001/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        chat_id: window.currentChatId,
                        use_duckduckgo: useDuckDuckGo,
                        use_google_news: useGoogleNews,
                        use_link: useLink,
                        use_rag: useRag,
                        selected_model: selectedModel,
                        system_prompt: window.systemPrompt
                    }),
                });
                const data = await response.json();
                window.addMessage(message, true, data.user_message_id);
                window.addMessage(data.response, false, data.assistant_message_id);
                
                if (data.chat_id) {
                    window.currentChatId = data.chat_id;
                    await window.updateChatHistory();
                }
                
                if (data.chat_title) {
                    window.updateChatTitle(window.currentChatId, data.chat_title);
                }
                
                // Check if we need to update the calendar display
                if (data.update_calendar) {
                    updateCalendarDisplay();
                }
            } catch (error) {
                console.error('Error:', error);
                window.addMessage('Sorry, there was an error processing your request.', false, null);
            } finally {
                loadingIndicator.remove();
                window.userInput.value = '';
                window.isProcessing = false;
                window.sendButton.disabled = false;
            }
        };
    }
    
    // Function to update calendar display
    function updateCalendarDisplay() {
        console.log("Updating calendar display after LLM-processed calendar operation");
        
        // Check if calendar module is initialized
        if (window.calendarModule) {
            // Fetch events directly from the server first instead of using loadEvents
            fetch('http://127.0.0.1:5001/calendar/events')
                .then(response => {
                    console.log("Calendar API response status:", response.status);
                    return response.json();
                })
                .then(data => {
                    console.log(`Direct fetch: Loaded ${data.events.length} events from server`);
                    
                    // Directly update the events array
                    window.calendarModule.events = data.events.map(event => ({
                        id: event.id,
                        title: event.title,
                        start: event.start_time,
                        end: event.end_time,
                        description: event.description || '',
                        location: event.location || '',
                        category: event.category || 'default',
                        source: 'server'
                    }));
                    
                    // Force render calendar views
                    window.calendarModule.renderCalendar();
                    window.calendarModule.renderUpcomingEvents();
                    
                    // Show notification
                    window.calendarModule.showNotification("Calendar updated successfully", false);
                })
                .catch(error => {
                    console.error("Error directly fetching calendar events:", error);
                    
                    // Fall back to the loadEvents method
                    window.calendarModule.loadEvents()
                        .then(() => {
                            window.calendarModule.renderCalendar();
                            window.calendarModule.renderUpcomingEvents();
                            window.calendarModule.showNotification("Calendar updated successfully", false);
                        })
                        .catch(error => {
                            console.error("Error refreshing calendar after LLM operation:", error);
                            window.calendarModule.showNotification("Error updating calendar", true);
                        });
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
        } else {
            console.error("Calendar module not found");
        }
    }
    
    // Enhanced CalendarModule prototype methods - these will be attached to the calendar module when it's loaded
    
    // Add a method to check if a message is a calendar command
    if (window.CalendarModule && window.CalendarModule.prototype) {
        // Enhanced isCalendarCommand function
        window.CalendarModule.prototype.isCalendarCommand = function(message) {
            const calendarKeywords = [
                'schedule', 'appointment', 'meeting', 'event', 'calendar',
                'add to calendar', 'create event', 'remove event', 'delete event',
                'cancel meeting', 'reschedule', 'move meeting',
                'show calendar', 'list events', 'upcoming events',
                'tomorrow', 'next week', 'sync calendar', 'remind me',
                'plan', 'agenda', 'booking', 'reservation'
            ];
            
            const lowerMessage = message.toLowerCase();
            
            // Check for calendar-related keywords
            return calendarKeywords.some(keyword => lowerMessage.includes(keyword));
        };
        
        // Process calendar commands through LLM
        window.CalendarModule.prototype.processCalendarCommand = async function(message) {
            try {
                // Display message in chat
                if (this.addMessageToChat) {
                    this.addMessageToChat(message, true);
                }
                
                // Process through LLM using the /calendar/natural endpoint
                const response = await fetch('http://127.0.0.1:5001/calendar/natural', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        message: message,
                        chat_id: window.currentChatId || null,
                        selected_model: window.modelSelect ? window.modelSelect.value : null
                    })
                });
                
                const data = await response.json();
                
                // Check if successful
                if (data.success) {
                    // Display response in chat
                    if (this.addMessageToChat) {
                        this.addMessageToChat(data.message, false);
                    }
                    
                    // Update calendar display
                    this.refreshCalendar();
                    
                    return true;
                } else {
                    // Display error message
                    if (this.addMessageToChat) {
                        this.addMessageToChat(data.message || "I couldn't process that calendar request. Could you try again with more details?", false);
                    }
                    return false;
                }
            } catch (error) {
                console.error('Error processing calendar command:', error);
                if (this.addMessageToChat) {
                    this.addMessageToChat(`I encountered an error processing your calendar request: ${error.message}`, false);
                }
                return false;
            }
        };
        
        // Extend the refreshCalendar method to make it more robust
        window.CalendarModule.prototype.refreshCalendar = function() {
            console.log("Refreshing calendar view");
            try {
                // Directly fetch events from server for more reliable updates
                fetch('http://127.0.0.1:5001/calendar/events')
                    .then(response => {
                        console.log("Calendar API refresh response status:", response.status);
                        return response.json();
                    })
                    .then(data => {
                        console.log(`Direct fetch in refreshCalendar: Loaded ${data.events.length} events from server`);
                        
                        // Directly update the events array
                        this.events = data.events.map(event => ({
                            id: event.id,
                            title: event.title,
                            start: event.start_time,
                            end: event.end_time,
                            description: event.description || '',
                            location: event.location || '',
                            category: event.category || 'default',
                            source: 'server'
                        }));
                        
                        // Force render calendar views
                        this.renderCalendar();
                        this.renderUpcomingEvents();
                    })
                    .catch(error => {
                        console.error("Error directly fetching calendar events in refreshCalendar:", error);
                        
                        // Fall back to the loadEvents method
                        this.loadEvents().then(() => {
                            console.log(`Loaded ${this.events.length} events, rendering calendar`);
                            this.renderCalendar();
                            this.renderUpcomingEvents();
                        }).catch(error => {
                            console.error("Error loading events during refresh:", error);
                            // Try to render with existing data anyway
                            this.renderCalendar();
                            this.renderUpcomingEvents();
                        });
                    });
            } catch (error) {
                console.error("Error refreshing calendar:", error);
            }
        };
        
        // Add a notification method if not already present
        if (!window.CalendarModule.prototype.showNotification) {
            window.CalendarModule.prototype.showNotification = function(message, isError = false) {
                // Create a notification element
                const notification = document.createElement('div');
                notification.className = `calendar-notification ${isError ? 'error' : 'success'}`;
                notification.textContent = message;
                notification.style.position = 'fixed';
                notification.style.bottom = '20px';
                notification.style.right = '20px';
                notification.style.padding = '10px 20px';
                notification.style.borderRadius = '4px';
                notification.style.backgroundColor = isError ? '#f44336' : '#4CAF50';
                notification.style.color = 'white';
                notification.style.zIndex = '1000';
                notification.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
                
                document.body.appendChild(notification);
                
                // Auto-remove after 3 seconds
                setTimeout(() => {
                    notification.style.opacity = '0';
                    notification.style.transition = 'opacity 0.5s ease';
                    setTimeout(() => {
                        notification.remove();
                    }, 500);
                }, 3000);
            };
        }
    }
    
    // Modify tab switching to handle calendar update indicator
    const tabs = document.querySelectorAll('.sidebar-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            // Update active tab content
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            const tabId = e.target.getAttribute('data-tab');
            const tabContent = document.getElementById(`${tabId}-tab`);
            if (tabContent) {
                tabContent.classList.add('active');
                
                if (tabId === 'calendar') {
                    // Remove notification indicator if present
                    const indicator = e.target.querySelector('.calendar-update-indicator');
                    if (indicator) {
                        indicator.remove();
                    }
                    
                    // Refresh calendar when switching to tab
                    if (window.calendarModule) {
                        window.calendarModule.refreshCalendar();
                    }
                }
            }
            
            // Update active tab button
            tabs.forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
        });
    });
});
