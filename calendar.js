/**
 * Calendar Module for VerbaBot
 * 
 * This module adds calendar functionality with Google Calendar synchronization
 * to the VerbaBot application, and enables LLM to manipulate calendar events
 * through natural language in the chat.
 */

class CalendarModule {
    constructor() {
        this.events = [];
        this.currentDate = new Date();
        this.currentView = 'month'; // 'month', 'week', 'day'
        this.initialized = false;
        this.googleCalendarConnected = false;
        this.calendarColors = {
            'default': '#3174ad',
            'work': '#4285F4',
            'personal': '#0F9D58',
            'holiday': '#DB4437',
            'other': '#F4B400'
        };
        this.db = null;
    }

    /**
     * Initialize the calendar
     */
    initCalendar() {
        // Setup container
        this.setupCalendarContainer();
        
        // Set current date to today
        this.currentDate = new Date();
        
        // Initialize calendar
        this.currentView = 'month'; // Default view (month, week, day)
        this.events = [];  // Calendar events
        
        // Define calendar colors
        this.calendarColors = {
            default: '#4285F4', // Blue
            work: '#0F9D58',    // Green
            personal: '#DB4437', // Red
            holiday: '#F4B400',  // Yellow
            other: '#8b08a0'     // Purple
        };
        
        // Init database
        this.initDatabase().then(() => {
            console.log("Database initialized");
            
            // Load events from database and server
            this.loadEvents().then(() => {
                console.log("Events loaded, rendering calendar");
                
                // Render initial calendar
                this.renderCalendar();
                this.renderUpcomingEvents();
                
                // Setup event listeners
                this.initEventListeners();
                
                // Init LLM calendar processor
                this.initLLMCalendarProcessor();
            }).catch(error => {
                console.error("Error loading events:", error);
                // Render empty calendar
                this.renderCalendar();
            });
        }).catch(error => {
            console.error("Error initializing database:", error);
        });
    }
    
    /**
     * Refresh calendar with current events
     */
    refreshCalendar() {
        console.log("Refreshing calendar view");
        try {
            this.renderCalendar();
            this.renderUpcomingEvents();
        } catch (error) {
            console.error("Error refreshing calendar:", error);
        }
    }

    /**
     * Initialize IndexedDB for storing calendar events
     */
    async initDatabase() {
        return new Promise((resolve, reject) => {
            console.log("Initializing IndexedDB database");
            
            // Проверяем, не существует ли уже база данных
            if (this.db) {
                console.log("Database already initialized");
                resolve();
                return;
            }
            
            const request = indexedDB.open('CalendarDB', 1);
            
            request.onupgradeneeded = (event) => {
                console.log("Database upgrade needed");
                const db = event.target.result;
                
                // Create an object store for events
                if (!db.objectStoreNames.contains('events')) {
                    console.log("Creating events store");
                    const store = db.createObjectStore('events', { keyPath: 'id', autoIncrement: true });
                    store.createIndex('start', 'start', { unique: false });
                    store.createIndex('end', 'end', { unique: false });
                    store.createIndex('title', 'title', { unique: false });
                }
            };
            
            request.onsuccess = (event) => {
                console.log("Database opened successfully");
                this.db = event.target.result;
                
                // Add error handler for database
                this.db.onerror = (event) => {
                    console.error("Database error:", event.target.error);
                    reject(event.target.error);
                };
                
                resolve();
            };
            
            request.onerror = (event) => {
                console.error("Error opening database:", event.target.error);
                reject(event.target.error);
            };
        });
    }

    /**
     * Load events from the server
     */
    async loadEvents() {
        if (!this.db) {
            console.error("Database not initialized");
            return Promise.reject(new Error("Database not initialized"));
        }
        
        console.log("Loading events from server...");
        
        try {
            // Загружаем события с сервера, используя полный URL
            const response = await fetch('http://127.0.0.1:5001/calendar/events');
            if (!response.ok) {
                throw new Error(`Failed to load events: ${response.status} ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log(`Loaded ${data.events.length} events from server`);
            
            // Преобразуем события в нужный формат
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
            
            // Также загружаем события из IndexedDB, если они есть
            const dbEvents = await this.getAllEventsFromDb();
            console.log(`Loaded ${dbEvents.length} events from IndexedDB`);
            
            // Объединяем события, избегая дубликатов
            const serverIds = new Set(this.events.map(e => e.id));
            const uniqueDbEvents = dbEvents.filter(e => !serverIds.has(e.id));
            
            this.events = [...this.events, ...uniqueDbEvents];
            console.log(`Total events after merge: ${this.events.length}`);
            
            // Отладочная информация о событиях
            this.debugShowAllEvents();
            
            // Рендерим календарь с новыми событиями
            this.renderCalendar();
            this.renderUpcomingEvents();
            
            return this.events;
        } catch (error) {
            console.error("Error loading events:", error);
            // Если ошибка загрузки с сервера, пробуем загрузить только из базы данных
            try {
                const dbEvents = await this.getAllEventsFromDb();
                console.log(`Loaded ${dbEvents.length} events from IndexedDB as fallback`);
                this.events = dbEvents;
                
                // Рендерим календарь с событиями из базы данных
                this.renderCalendar();
                this.renderUpcomingEvents();
                
                return this.events;
            } catch (dbError) {
                console.error("Failed to load events from IndexedDB:", dbError);
                return [];
            }
        }
    }
    
    /**
     * Get all events from IndexedDB
     */
    async getAllEventsFromDb() {
        return new Promise((resolve, reject) => {
            if (!this.db) {
                console.error("Cannot get events: Database not initialized");
                return reject(new Error("Database not initialized"));
            }
            
            const transaction = this.db.transaction(['events'], 'readonly');
            const store = transaction.objectStore('events');
            const request = store.getAll();
            
            request.onsuccess = (event) => {
                resolve(event.target.result);
            };
            
            request.onerror = (event) => {
                console.error("Error getting events from IndexedDB:", event.target.error);
                reject(event.target.error);
            };
        });
    }

    /**
     * Save an event to the database
     */
    async saveEvent(event) {
        try {
            // First save to IndexedDB
            const localId = await new Promise((resolve, reject) => {
                if (!this.db) {
                    console.error("Database not initialized");
                    reject(new Error('Database not initialized'));
                    return;
                }
                
                console.log("Saving event to local database:", event);
                
                const transaction = this.db.transaction(['events'], 'readwrite');
                const store = transaction.objectStore('events');
                
                // Convert Date objects to ISO strings for storage
                const eventToSave = {
                    ...event,
                    start: event.start instanceof Date ? event.start.toISOString() : event.start,
                    end: event.end instanceof Date ? event.end.toISOString() : event.end,
                };
                
                console.log("Converted event for storage:", eventToSave);
                
                const request = store.add(eventToSave);
                
                request.onsuccess = (event) => {
                    const id = event.target.result;
                    console.log("Event saved successfully to IndexedDB with ID:", id);
                    resolve(id);
                };
                
                request.onerror = (event) => {
                    console.error("Error saving event to IndexedDB:", event.target.error);
                    reject(event.target.error);
                };
            });
            
            // Then save to server
            console.log("Saving event to server:", event);
            
            // Prepare the event data for the server
            const serverEvent = {
                title: event.title,
                start_time: event.start instanceof Date ? event.start.toISOString() : event.start,
                end_time: event.end instanceof Date ? event.end.toISOString() : event.end,
                description: event.description || '',
                location: event.location || '',
                category: event.category || 'default'
            };
            
            // Send to server
            const response = await fetch('http://127.0.0.1:5001/calendar/events', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(serverEvent)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                console.error('Server error saving event:', errorData);
                // We still return the local ID since we saved it to IndexedDB successfully
                return localId;
            }
            
            const responseData = await response.json();
            console.log("Event saved to server with ID:", responseData.event_id);
            
            // Return the server ID if available, otherwise the local ID
            return responseData.event_id || localId;
        } catch (error) {
            console.error("Error in saveEvent:", error);
            throw error;
        }
    }

    /**
     * Update an existing event in the database
     */
    async updateEvent(event) {
        try {
            // First update in IndexedDB
            await new Promise((resolve, reject) => {
                if (!this.db) {
                    reject(new Error('Database not initialized'));
                    return;
                }
                
                const transaction = this.db.transaction(['events'], 'readwrite');
                const store = transaction.objectStore('events');
                
                // Convert Date objects to ISO strings for storage
                const eventToUpdate = {
                    ...event,
                    start: event.start instanceof Date ? event.start.toISOString() : event.start,
                    end: event.end instanceof Date ? event.end.toISOString() : event.end,
                };
                
                const request = store.put(eventToUpdate);
                
                request.onsuccess = () => {
                    console.log(`Event with ID ${event.id} updated in IndexedDB`);
                    resolve();
                };
                
                request.onerror = (event) => {
                    console.error('Error updating event in IndexedDB:', event.target.error);
                    reject(event.target.error);
                };
            });
            
            // Then update on server
            console.log("Updating event on server:", event);
            
            // Prepare the event data for the server
            const serverEvent = {
                title: event.title,
                start_time: event.start instanceof Date ? event.start.toISOString() : event.start,
                end_time: event.end instanceof Date ? event.end.toISOString() : event.end,
                description: event.description || '',
                location: event.location || '',
                category: event.category || 'default'
            };
            
            // Send to server
            const response = await fetch(`http://127.0.0.1:5001/calendar/events/${event.id}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(serverEvent)
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                console.error('Server error updating event:', errorData);
                // We still consider it successful since we updated it in IndexedDB
                return true;
            }
            
            console.log(`Event with ID ${event.id} updated on server successfully`);
            return true;
        } catch (error) {
            console.error("Error in updateEvent:", error);
            // We still consider it successful if at least one update worked
            return true;
        }
    }

    /**
     * Delete an event from the database
     */
    async deleteEvent(eventId) {
        try {
            // First delete from IndexedDB
            await new Promise((resolve, reject) => {
                if (!this.db) {
                    reject(new Error('Database not initialized'));
                    return;
                }
                
                const transaction = this.db.transaction(['events'], 'readwrite');
                const store = transaction.objectStore('events');
                const request = store.delete(eventId);
                
                request.onsuccess = () => {
                    resolve();
                };
                
                request.onerror = (event) => {
                    console.error('Error deleting event from IndexedDB:', event.target.error);
                    reject(event.target.error);
                };
            });
            
            // Then delete from server
            const response = await fetch(`http://127.0.0.1:5001/calendar/events/${eventId}`, {
                method: 'DELETE',
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                console.error('Server error deleting event:', errorData);
            } else {
                console.log(`Event ${eventId} successfully deleted from server`);
            }
            
            return true;
        } catch (error) {
            console.error('Error in deleteEvent:', error);
            // Still return true if local deletion succeeded
            return true;
        }
    }

    /**
     * Create calendar tab content
     */
    createCalendarTabContent() {
        // Вместо создания нового контента, просто инициализируем нужные элементы
        // Все элементы DOM уже существуют в HTML
        
        // Инициализация стилей календаря
        this.addCalendarStyles();
    }

    /**
     * Add calendar CSS styles to the page
     */
    addCalendarStyles() {
        const styleElement = document.createElement('style');
        styleElement.textContent = `
            /* Calendar module styles */
            .calendar-section {
                padding: 10px 0;
                color: #ececf1;
            }
            
            .calendar-header {
                display: flex;
                flex-direction: column;
                margin-bottom: 15px;
            }
            
            .calendar-nav {
                display: flex;
                align-items: center;
                margin-bottom: 10px;
            }
            
            #current-date-display {
                margin: 0 10px;
                flex-grow: 1;
                text-align: center;
            }
            
            .calendar-nav-btn {
                background-color: #2a2b32;
                color: #ececf1;
                border: none;
                border-radius: 3px;
                padding: 5px 10px;
                cursor: pointer;
            }
            
            .calendar-nav-btn:hover {
                background-color: #3a3b42;
            }
            
            .calendar-view-options {
                display: flex;
                justify-content: center;
                gap: 10px;
                margin-bottom: 15px;
            }
            
            .calendar-view-btn {
                background-color: #2a2b32;
                color: #ececf1;
                border: none;
                border-radius: 3px;
                padding: 5px 15px;
                cursor: pointer;
            }
            
            .calendar-view-btn:hover {
                background-color: #3a3b42;
            }
            
            .calendar-view-btn.active {
                background-color: #565869;
            }
            
            #calendar-container {
                background-color: #2a2b32;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 20px;
            }
            
            /* Month view grid */
            .month-grid {
                display: grid;
                grid-template-columns: repeat(7, 1fr);
                gap: 2px;
            }
            
            .day-header {
                text-align: center;
                padding: 5px;
                font-weight: bold;
                background-color: #3a3b42;
            }
            
            .day-cell {
                min-height: 80px;
                padding: 5px;
                background-color: #3a3b42;
                position: relative;
                overflow: hidden;
            }
            
            .day-cell.other-month {
                background-color: #313238;
                color: #8e8e96;
            }
            
            .day-cell.today {
                background-color: #444654;
                border: 1px solid #61dafb;
            }
            
            .day-number {
                position: absolute;
                top: 5px;
                right: 5px;
                font-size: 0.8em;
                color: #ececf1;
            }
            
            .day-events {
                margin-top: 20px;
                display: flex;
                flex-direction: column;
                gap: 2px;
            }
            
            .day-event {
                padding: 2px 4px;
                border-radius: 3px;
                font-size: 0.7em;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                cursor: pointer;
            }
            
            /* Week and day view */
            .time-grid {
                display: grid;
                grid-template-columns: 50px repeat(7, 1fr);
                gap: 1px;
            }
            
            .day-view-grid {
                display: grid;
                grid-template-columns: 50px 1fr;
                gap: 1px;
            }
            
            .time-cell {
                background-color: #3a3b42;
                border-bottom: 1px solid #202123;
                height: 50px; /* Increased from 40px to 50px */
                position: relative;
            }
            
            .time-label {
                font-size: 0.7em;
                text-align: right;
                padding-right: 5px;
                height: 50px; /* Increased from 40px to 50px */
                display: flex;
                align-items: center;
                justify-content: flex-end;
            }
            
            .event-item {
                position: absolute;
                left: 0;
                right: 0;
                border-radius: 3px;
                padding: 4px 5px; /* Increased top/bottom padding from 2px to 4px */
                overflow: hidden;
                font-size: 0.8em;
                cursor: pointer;
                z-index: 2;
                line-height: 1.2; /* Added line height for better readability */
            }
            
            /* Calendar events list */
            .calendar-events {
                margin-bottom: 20px;
            }
            
            #event-list {
                background-color: #2a2b32;
                border-radius: 5px;
                padding: 10px;
                max-height: 200px;
                overflow-y: auto;
            }
            
            .event-list-item {
                margin-bottom: 8px;
                padding: 8px;
                background-color: #3a3b42;
                border-radius: 3px;
                border-left: 3px solid #3174ad;
                cursor: pointer;
            }
            
            .event-list-item:hover {
                background-color: #40414f;
            }
            
            .event-list-title {
                font-weight: bold;
                margin-bottom: 3px;
            }
            
            .event-list-time {
                font-size: 0.8em;
                color: #c0c0c6;
            }
            
            /* Add event form */
            .calendar-actions {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            
            .calendar-btn {
                background-color: #565869;
                color: #ececf1;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                cursor: pointer;
                flex-grow: 1;
                text-align: center;
            }
            
            .calendar-btn:hover {
                background-color: #6e7081;
            }
            
            #add-event-form, #google-calendar-sync {
                background-color: #2a2b32;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 20px;
            }
            
            .form-group {
                margin-bottom: 10px;
            }
            
            .form-group label {
                display: block;
                margin-bottom: 5px;
            }
            
            .form-group input, .form-group select, .form-group textarea {
                width: 100%;
                padding: 8px;
                background-color: #40414f;
                color: #ececf1;
                border: 1px solid #565869;
                border-radius: 3px;
            }
            
            .form-group textarea {
                min-height: 60px;
                resize: vertical;
            }
            
            .form-actions {
                display: flex;
                gap: 10px;
                margin-top: 15px;
            }
            
            .hidden {
                display: none;
            }
            
            /* Google Calendar section */
            #google-calendars-list {
                margin-top: 15px;
            }
            
            .calendar-checkbox {
                display: flex;
                align-items: center;
                margin-bottom: 8px;
                padding: 5px;
                background-color: #3a3b42;
                border-radius: 3px;
            }
            
            .calendar-checkbox input {
                margin-right: 10px;
            }
            
            .calendar-color-dot {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 5px;
            }
            
            #calendars-container {
                max-height: 200px;
                overflow-y: auto;
                margin: 10px 0;
            }
            
            /* Event details popup */
            #event-details-popup {
                position: fixed;
                z-index: 1000;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background-color: #2a2b32;
                border-radius: 5px;
                padding: 20px;
                width: 80%;
                max-width: 400px;
                box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
            }
            
            .event-details-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }
            
            .event-details-title {
                margin: 0;
                padding: 0;
            }
            
            .close-popup {
                background: none;
                border: none;
                color: #ececf1;
                font-size: 1.2em;
                cursor: pointer;
            }
            
            .event-details-content {
                margin-bottom: 20px;
            }
            
            .event-details-actions {
                display: flex;
                justify-content: flex-end;
                gap: 10px;
            }
        `;
        document.head.appendChild(styleElement);
    }

    /**
     * Initialize event listeners for calendar navigation and actions
     */
    initEventListeners() {
        console.log("Initializing calendar event listeners with completely new handlers");
        
        // Tab switching
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
                        this.renderCalendar();
                        this.renderUpcomingEvents();
                    }
                }
                
                // Update active tab button
                tabs.forEach(t => t.classList.remove('active'));
                e.target.classList.add('active');
            });
        });
        
        // "Add Event" button
        const addEventBtn = document.getElementById('add-event-btn');
        if (addEventBtn) {
            addEventBtn.addEventListener('click', () => {
                console.log("Add Event button clicked");
                this.showEventForm();
            });
        } else {
            console.error("Add Event button not found in DOM");
        }
        
        // "Cancel" button in event form
        const cancelEventBtn = document.getElementById('cancel-event-btn');
        if (cancelEventBtn) {
            cancelEventBtn.addEventListener('click', (e) => {
                console.log("Cancel Event button clicked");
                e.preventDefault();
                this.hideEventForm();
            });
        } else {
            console.error("Cancel Event button not found in DOM");
        }
        
        // "X" close button in event form
        const closeEventBtn = document.getElementById('event-form-close');
        if (closeEventBtn) {
            closeEventBtn.addEventListener('click', () => {
                console.log("Form close (X) button clicked");
                this.hideEventForm();
            });
        } else {
            console.error("Form close button not found in DOM");
        }
        
        // Event form submission
        const eventForm = document.getElementById('event-form');
        if (eventForm) {
            eventForm.addEventListener('submit', (e) => {
                console.log("Form submit event triggered");
                this.handleAddEventSubmit(e);
            });
        } else {
            console.error("Event form not found in DOM");
        }
        
        // Modal overlay click handler
        const modalOverlay = document.getElementById('modal-overlay');
        if (modalOverlay) {
            modalOverlay.addEventListener('click', (e) => {
                // Only close if clicked directly on the overlay, not on its children
                if (e.target === modalOverlay) {
                    console.log("Modal overlay clicked, closing form");
                    this.hideEventForm();
                }
            });
        } else {
            console.error("Modal overlay not found in DOM");
        }
        
        // Navigation buttons
        const prevDateBtn = document.getElementById('prev-date-btn');
        if (prevDateBtn) {
            prevDateBtn.addEventListener('click', () => this.navigateDate('prev'));
        }
        
        const nextDateBtn = document.getElementById('next-date-btn');
        if (nextDateBtn) {
            nextDateBtn.addEventListener('click', () => this.navigateDate('next'));
        }
        
        const todayBtn = document.getElementById('today-btn');
        if (todayBtn) {
            todayBtn.addEventListener('click', () => this.goToToday());
        }
        
        // View buttons
        const monthViewBtn = document.getElementById('month-view-btn');
        if (monthViewBtn) {
            monthViewBtn.addEventListener('click', () => this.changeView('month'));
        }
        
        const weekViewBtn = document.getElementById('week-view-btn');
        if (weekViewBtn) {
            weekViewBtn.addEventListener('click', () => this.changeView('week'));
        }
        
        const dayViewBtn = document.getElementById('day-view-btn');
        if (dayViewBtn) {
            dayViewBtn.addEventListener('click', () => this.changeView('day'));
        }
        
        // Google Calendar sync
        const syncCalendarBtn = document.getElementById('sync-calendar-btn');
        if (syncCalendarBtn) {
            syncCalendarBtn.addEventListener('click', () => this.toggleGoogleCalendarSync());
        }
        
        const googleAuthBtn = document.getElementById('google-auth-btn');
        if (googleAuthBtn) {
            googleAuthBtn.addEventListener('click', () => this.connectToGoogleCalendar());
        }
        
        const closeGoogleSyncBtn = document.getElementById('close-google-sync');
        if (closeGoogleSyncBtn) {
            closeGoogleSyncBtn.addEventListener('click', () => this.toggleGoogleCalendarSync());
        }
        
        const syncSelectedCalendarsBtn = document.getElementById('sync-selected-calendars');
        if (syncSelectedCalendarsBtn) {
            syncSelectedCalendarsBtn.addEventListener('click', () => this.syncSelectedCalendars());
        }
        
        console.log("Calendar event listeners initialized successfully");
    }

    /**
     * Initialize the natural language processor for calendar commands in chat
     */
    initLLMCalendarProcessor() {
        // Intercept chat messages to look for calendar-related commands
        const originalSendMessage = window.sendMessage;
        
        window.sendMessage = async () => {
            const message = document.getElementById('user-input').value.trim();
            
            // Check if this is a calendar-related command
            if (this.isCalendarCommand(message)) {
                const result = await this.processCalendarCommand(message);
                if (result) {
                    // Clear the input field
                    document.getElementById('user-input').value = '';
                    // Refresh calendar display
                    this.refreshCalendar();
                    return;
                }
            }
            
            // If not a calendar command, or processing failed, use the original function
            return originalSendMessage();
        };
    }

    /**
     * Check if a message appears to be a calendar-related command
     */
    isCalendarCommand(message) {
        const calendarKeywords = [
            'schedule', 'appointment', 'meeting', 'event', 'calendar',
            'add to calendar', 'create event', 'remove event', 'delete event',
            'cancel meeting', 'reschedule', 'move meeting',
            'show calendar', 'list events', 'upcoming events',
            'tomorrow', 'next week', 'sync calendar'
        ];
        
        const lowerMessage = message.toLowerCase();
        
        // Check for command-like structure
        return calendarKeywords.some(keyword => lowerMessage.includes(keyword));
    }

    /**
     * Process a calendar command from chat
     */
    async processCalendarCommand(message) {
        try {
            console.log("Processing calendar command through RAG system:", message);
            
            // Store current chat ID to add messages to the right chat history
            const chatId = window.currentChatId || null;
            
            // Process through server-side RAG using the /calendar/natural endpoint
            try {
                // Create request data
                const requestData = {
                    message: message,
                    chat_id: chatId,
                    selected_model: window.modelSelect ? window.modelSelect.value : null
                };
                
                // Send to server
                const response = await fetch('http://127.0.0.1:5001/calendar/natural', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestData)
                });
                
                // Process response
                if (!response.ok) {
                    throw new Error(`Server returned ${response.status} ${response.statusText}`);
                }
                
                const data = await response.json();
                console.log("RAG calendar processing result:", data);
                
                // Display response in chat
                // Use window.addMessage if available, otherwise fallback to this.addMessageToChat
                if (window.addMessage && typeof window.addMessage === 'function') {
                    window.addMessage(data.message || "I couldn't process that calendar request. Could you try again with more details?", false, data.assistant_message_id);
                } else {
                    this.addMessageToChat(data.message || "I couldn't process that calendar request. Could you try again with more details?", false);
                }
                
                // Update calendar display if needed
                if (data.update_calendar) {
                    console.log("Updating calendar display after LLM-processed calendar operation");
                    await this.loadEvents();
                    this.renderCalendar();
                    this.renderUpcomingEvents();
                }
                
                // Return success status and message IDs to the caller
                return {
                    success: data.success,
                    message_ids: {
                        user_id: data.user_message_id,
                        assistant_id: data.assistant_message_id
                    }
                };
            } catch (error) {
                console.error('Error processing calendar command:', error);
                // Use window.addMessage if available, otherwise fallback to this.addMessageToChat
                if (window.addMessage && typeof window.addMessage === 'function') {
                    window.addMessage(`I encountered an error processing your calendar request: ${error.message}`, false, null);
                } else {
                    this.addMessageToChat(`I encountered an error processing your calendar request: ${error.message}`, false);
                }
                return { success: false };
            }
        } catch (error) {
            console.error('Error processing calendar command:', error);
            // Use window.addMessage if available, otherwise fallback to this.addMessageToChat
            if (window.addMessage && typeof window.addMessage === 'function') {
                window.addMessage(`I encountered an error processing your calendar request: ${error.message}`, false, null);
            } else {
                this.addMessageToChat(`I encountered an error processing your calendar request: ${error.message}`, false);
            }
            return { success: false };
        }
    }

    /**
     * Handle a command to remove an event
     */
    async handleRemoveEventCommand(message) {
        const eventInfo = this.extractEventInfo(message);
        
        if (!eventInfo.title && !eventInfo.date) {
            return "I need more information to remove an event. Please specify the event title or date.";
        }
        
        // Find matching events
        const matchingEvents = this.findMatchingEvents(eventInfo);
        
        if (matchingEvents.length === 0) {
            return "I couldn't find any events matching your description.";
        }
        
        if (matchingEvents.length === 1) {
            // Delete the event
            await this.deleteEvent(matchingEvents[0].id);
            
            // Refresh events list
            await this.loadEvents();
            
            return `I've removed the event "${matchingEvents[0].title}" from your calendar.`;
        }
        
        // Multiple matches, delete the first one and inform the user
        const deletedEvent = matchingEvents[0];
        await this.deleteEvent(deletedEvent.id);
        
        // Refresh events list
        await this.loadEvents();
        
        return `I found multiple matching events and removed "${deletedEvent.title}" on ${new Date(deletedEvent.start).toLocaleDateString()}. If this wasn't the right one, please provide more specific details.`;
    }

    /**
     * Handle a command to reschedule an event
     */
    async handleRescheduleEventCommand(message) {
        const eventInfo = this.extractEventInfo(message);
        
        if (!eventInfo.title || !eventInfo.start) {
            return "I need more information to reschedule an event. Please specify the event title and the new date/time.";
        }
        
        // Find matching events
        const matchingEvents = this.findMatchingEvents({title: eventInfo.title});
        
        if (matchingEvents.length === 0) {
            return `I couldn't find any events with the title "${eventInfo.title}".`;
        }
        
        // Update the first matching event
        const eventToUpdate = matchingEvents[0];
        const oldDate = new Date(eventToUpdate.start);
        
        // Update the event
        eventToUpdate.start = eventInfo.start;
        eventToUpdate.end = eventInfo.end || new Date(eventInfo.start.getTime() + (eventToUpdate.end - eventToUpdate.start));
        
        // If there are other fields to update
        if (eventInfo.description) eventToUpdate.description = eventInfo.description;
        if (eventInfo.location) eventToUpdate.location = eventInfo.location;
        if (eventInfo.category) eventToUpdate.category = eventInfo.category;
        
        // Save the updated event
        await this.updateEvent(eventToUpdate);
        
        // Refresh events list
        await this.loadEvents();
        
        // Format dates for response
        const oldFormatted = `${oldDate.toLocaleDateString()} at ${oldDate.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;
        const newFormatted = `${eventToUpdate.start.toLocaleDateString()} at ${eventToUpdate.start.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;
        
        return `I've rescheduled "${eventToUpdate.title}" from ${oldFormatted} to ${newFormatted}.`;
    }

    /**
     * Handle a command to list events
     */
    async handleListEventsCommand(message) {
        // Determine date range based on message
        let startDate = new Date();
        let endDate = new Date();
        let rangeDescription = "today";
        
        if (/tomorrow/i.test(message)) {
            startDate = new Date(startDate.setDate(startDate.getDate() + 1));
            endDate = new Date(startDate);
            endDate.setHours(23, 59, 59);
            rangeDescription = "tomorrow";
        } else if (/this week/i.test(message)) {
            // Set to the end of the current week (Sunday)
            const dayOfWeek = startDate.getDay();
            endDate = new Date(startDate);
            endDate.setDate(startDate.getDate() + (7 - dayOfWeek));
            endDate.setHours(23, 59, 59);
            rangeDescription = "this week";
        } else if (/next week/i.test(message)) {
            // Set to next week
            startDate = new Date(startDate);
            const dayOfWeek = startDate.getDay();
            startDate.setDate(startDate.getDate() + (7 - dayOfWeek + 1)); // Next Monday
            endDate = new Date(startDate);
            endDate.setDate(startDate.getDate() + 6); // Next Sunday
            endDate.setHours(23, 59, 59);
            rangeDescription = "next week";
        } else if (/this month/i.test(message)) {
            // Set to the end of the current month
            endDate = new Date(startDate.getFullYear(), startDate.getMonth() + 1, 0, 23, 59, 59);
            rangeDescription = "this month";
        } else {
            // Default to today
            endDate.setHours(23, 59, 59);
        }
        
        // Filter events within the date range
        const filteredEvents = this.events.filter(event => {
            const eventStart = new Date(event.start);
            return eventStart >= startDate && eventStart <= endDate;
        }).sort((a, b) => new Date(a.start) - new Date(b.start));
        
        // Create the response
        if (filteredEvents.length === 0) {
            return `You don't have any events scheduled for ${rangeDescription}.`;
        }
        
        let response = `Here are your events for ${rangeDescription}:\n\n`;
        
        filteredEvents.forEach((event, index) => {
            const eventDate = new Date(event.start);
            const formattedDate = `${eventDate.toLocaleDateString()} at ${eventDate.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;
            response += `${index + 1}. ${event.title} - ${formattedDate}\n`;
            if (event.location) response += `   Location: ${event.location}\n`;
        });
        
        return response;
    }

    /**
     * Find events matching the given criteria
     */
    findMatchingEvents(criteria) {
        return this.events.filter(event => {
            // Match by title (partial match)
            if (criteria.title && event.title.toLowerCase().includes(criteria.title.toLowerCase())) {
                return true;
            }
            
            // Match by exact date (ignoring time)
            if (criteria.date) {
                const eventDate = new Date(event.start);
                eventDate.setHours(0, 0, 0, 0);
                
                if (eventDate.getTime() === criteria.date.getTime()) {
                    return true;
                }
            }
            
            // Match by date range
            if (criteria.start && criteria.end) {
                const eventStart = new Date(event.start);
                return eventStart >= criteria.start && eventStart <= criteria.end;
            }
            
            return false;
        });
    }

    /**
     * Show the event form with default values
     */
    showEventForm() {
        console.log("Showing event form");
        
        const form = document.getElementById('add-event-form');
        const overlay = document.getElementById('modal-overlay');
        
        if (!form) {
            console.error("Event form element not found");
            return;
        }
        
        // Show form
        form.classList.remove('hidden');
        
        // Show overlay
        if (overlay) {
            overlay.classList.add('visible');
        } else {
            console.warn("Modal overlay element not found");
        }
        
        // Set default values
        const now = new Date();
        const formattedNow = this.formatDateTimeForInput(now);
        const formattedLater = this.formatDateTimeForInput(new Date(now.getTime() + 60 * 60 * 1000));
        
        document.getElementById('event-title').value = '';
        document.getElementById('event-start').value = formattedNow;
        document.getElementById('event-end').value = formattedLater;
        document.getElementById('event-category').value = 'default';
        document.getElementById('event-location').value = '';
        document.getElementById('event-description').value = '';
        
        // Focus on title field
        setTimeout(() => {
            document.getElementById('event-title').focus();
        }, 100);
        
        console.log("Event form shown successfully");
    }
    
    /**
     * Hide the event form
     */
    hideEventForm() {
        console.log("Hiding event form");
        
        const form = document.getElementById('add-event-form');
        const overlay = document.getElementById('modal-overlay');
        
        if (!form) {
            console.error("Event form element not found");
            return;
        }
        
        // Hide form
        form.classList.add('hidden');
        
        // Hide overlay
        if (overlay) {
            overlay.classList.remove('visible');
        }
        
        console.log("Event form hidden successfully");
    }
    
    /**
     * Handle event form submission
     */
    async handleAddEventSubmit(e) {
        e.preventDefault();
        console.log("Handling event form submission");
        
        const title = document.getElementById('event-title').value;
        const start = new Date(document.getElementById('event-start').value);
        const end = new Date(document.getElementById('event-end').value);
        const category = document.getElementById('event-category').value;
        const location = document.getElementById('event-location').value;
        const description = document.getElementById('event-description').value;
        
        // Create event object
        const newEvent = {
            title,
            start,
            end,
            category,
            location,
            description,
            source: 'form'
        };
        
        try {
            // Show loading state
            const submitButton = document.querySelector('#event-form button[type="submit"]');
            const originalText = submitButton.textContent;
            submitButton.textContent = 'Saving...';
            submitButton.disabled = true;
            
            // Save event
            const eventId = await this.saveEvent(newEvent);
            
            // Hide form BEFORE refreshing UI for better performance
            this.hideEventForm();
            
            // Reload events and update UI
            await this.loadEvents();
            this.renderCalendar();
            this.renderUpcomingEvents();
            
            // Show success notification
            this.showNotification(`Event "${title}" added successfully!`);
            
            // Reset button state
            submitButton.textContent = originalText;
            submitButton.disabled = false;
            
            console.log("Event saved successfully:", eventId);
        } catch (error) {
            console.error('Error saving event:', error);
            alert(`Error saving event: ${error.message}`);
            
            // Reset button state on error
            const submitButton = document.querySelector('#event-form button[type="submit"]');
            if (submitButton) {
                submitButton.textContent = 'Save';
                submitButton.disabled = false;
            }
        }
    }
    
    /**
     * Legacy method for backwards compatibility
     */
    toggleAddEventForm() {
        console.log("Legacy toggleAddEventForm called, redirecting to newer methods");
        const form = document.getElementById('add-event-form');
        
        if (form && form.classList.contains('hidden')) {
            this.showEventForm();
        } else {
            this.hideEventForm();
        }
    }

    /**
     * Show notification message
     */
    showNotification(message, isError = false) {
        // Создаем элемент уведомления, если его нет
        let notification = document.getElementById('calendar-notification');
        if (!notification) {
            notification = document.createElement('div');
            notification.id = 'calendar-notification';
            notification.style.position = 'fixed';
            notification.style.bottom = '20px';
            notification.style.right = '20px';
            notification.style.padding = '10px 20px';
            notification.style.borderRadius = '5px';
            notification.style.zIndex = '1000';
            notification.style.transition = 'opacity 0.3s ease-in-out';
            document.body.appendChild(notification);
        }
        
        // Устанавливаем стиль в зависимости от типа
        notification.style.backgroundColor = isError ? '#F44336' : '#4CAF50';
        notification.style.color = 'white';
        
        // Устанавливаем сообщение
        notification.textContent = message;
        notification.style.opacity = '1';
        
        // Скрываем через 3 секунды
        setTimeout(() => {
            notification.style.opacity = '0';
        }, 3000);
    }

    /**
     * Format a Date object for datetime-local input
     */
    formatDateTimeForInput(date) {
        return date.toISOString().slice(0, 16);
    }

    /**
     * Toggle the Google Calendar sync panel
     */
    toggleGoogleCalendarSync() {
        const panel = document.getElementById('google-calendar-sync');
        panel.classList.toggle('hidden');
        
        // Hide event form if open
        document.getElementById('add-event-form').classList.add('hidden');
    }

    /**
     * Connect to Google Calendar
     */
    connectToGoogleCalendar() {
        // Load the Google API client library
        const script = document.createElement('script');
        script.src = 'https://apis.google.com/js/api.js';
        script.onload = this.initGoogleCalendarApi.bind(this);
        document.body.appendChild(script);
    }

    /**
     * Initialize the Google Calendar API
     */
    initGoogleCalendarApi() {
        console.log("Initializing Google Calendar API with environment settings");
        
        // Fetch API keys and settings from server
        fetch('http://127.0.0.1:5001/google_calendar_settings')
            .then(response => response.json())
            .then(data => {
                if (!data.success || !data.api_key || !data.client_id) {
                    console.error("Google Calendar API keys not configured");
                    document.getElementById('auth-status').textContent = 'API keys not configured';
                    document.getElementById('auth-status').style.color = '#F44336';
                    return;
                }
                
                const apiKey = data.api_key;
                const clientId = data.client_id;
                const discoveryDocs = ['https://www.googleapis.com/discovery/v1/apis/calendar/v3/rest'];
                const scopes = 'https://www.googleapis.com/auth/calendar.readonly';
                
                gapi.load('client:auth2', () => {
                    gapi.client.init({
                        apiKey: apiKey,
                        clientId: clientId,
                        discoveryDocs: discoveryDocs,
                        scope: scopes
                    }).then(() => {
                        // Listen for sign-in state changes
                        gapi.auth2.getAuthInstance().isSignedIn.listen(this.updateSigninStatus.bind(this));
                        
                        // Handle the initial sign-in state
                        this.updateSigninStatus(gapi.auth2.getAuthInstance().isSignedIn.get());
                        
                        // Attach click handlers
                        document.getElementById('google-auth-btn').onclick = () => {
                            if (gapi.auth2.getAuthInstance().isSignedIn.get()) {
                                gapi.auth2.getAuthInstance().signOut();
                            } else {
                                gapi.auth2.getAuthInstance().signIn();
                            }
                        };
                    }).catch(error => {
                        console.error('Error initializing Google Calendar API:', error);
                        document.getElementById('auth-status').textContent = 'Error initializing API';
                    });
                });
            })
            .catch(error => {
                console.error('Error fetching Google Calendar settings:', error);
                document.getElementById('auth-status').textContent = 'Error fetching API settings';
                document.getElementById('auth-status').style.color = '#F44336';
            });
    }

    /**
     * Update the Google Calendar sign-in status and UI
     */
    updateSigninStatus(isSignedIn) {
        const authStatus = document.getElementById('auth-status');
        const authButton = document.getElementById('google-auth-btn');
        const calendarsList = document.getElementById('google-calendars-list');
        
        if (isSignedIn) {
            authStatus.textContent = 'Connected';
            authStatus.style.color = '#4CAF50';
            authButton.textContent = 'Disconnect from Google Calendar';
            this.googleCalendarConnected = true;
            
            // List the user's calendars
            this.listCalendars();
            calendarsList.classList.remove('hidden');
        } else {
            authStatus.textContent = 'Not connected';
            authStatus.style.color = '#F44336';
            authButton.textContent = 'Connect to Google Calendar';
            this.googleCalendarConnected = false;
            calendarsList.classList.add('hidden');
        }
    }

    /**
     * List the user's Google Calendars
     */
    listCalendars() {
        gapi.client.calendar.calendarList.list().then(response => {
            const calendars = response.result.items;
            const container = document.getElementById('calendars-container');
            container.innerHTML = '';
            
            calendars.forEach(calendar => {
                const checkbox = document.createElement('div');
                checkbox.className = 'calendar-checkbox';
                checkbox.innerHTML = `
                    <input type="checkbox" id="cal-${calendar.id}" data-cal-id="${calendar.id}">
                    <span class="calendar-color-dot" style="background-color: ${calendar.backgroundColor}"></span>
                    <label for="cal-${calendar.id}">${calendar.summary}</label>
                `;
                container.appendChild(checkbox);
            });
        }).catch(error => {
            console.error('Error listing calendars:', error);
            document.getElementById('calendars-container').innerHTML = '<p>Error listing calendars</p>';
        });
    }

    /**
     * Sync selected Google Calendars
     */
    syncSelectedCalendars() {
        const checkboxes = document.querySelectorAll('#calendars-container input[type="checkbox"]:checked');
        const calendarIds = Array.from(checkboxes).map(cb => cb.getAttribute('data-cal-id'));
        
        if (calendarIds.length === 0) {
            alert('Please select at least one calendar to sync');
            return;
        }
        
        // Get events from each selected calendar
        Promise.all(calendarIds.map(id => this.getCalendarEvents(id)))
            .then(results => {
                let importedCount = 0;
                
                // Flatten the results and save each event
                results.flat().forEach(async event => {
                    // Skip events without a start date/time
                    if (!event.start || (!event.start.dateTime && !event.start.date)) return;
                    
                    // Create a new event object
                    const newEvent = {
                        title: event.summary || 'Untitled Event',
                        start: new Date(event.start.dateTime || event.start.date),
                        end: new Date(event.end.dateTime || event.end.date),
                        description: event.description || '',
                        location: event.location || '',
                        category: 'default',
                        googleId: event.id,
                        source: 'google'
                    };
                    
                    // Save the event
                    await this.saveEvent(newEvent);
                    importedCount++;
                });
                
                // Reload events and update the UI
                this.loadEvents().then(() => {
                    this.renderCalendar();
                    this.renderUpcomingEvents();
                    alert(`Successfully imported ${importedCount} events from Google Calendar`);
                });
            })
            .catch(error => {
                console.error('Error syncing calendars:', error);
                alert('Error syncing calendars: ' + error.message);
            });
    }

    /**
     * Get events from a Google Calendar
     */
    getCalendarEvents(calendarId) {
        const timeMin = new Date();
        timeMin.setDate(timeMin.getDate() - 7); // Include events from last week
        
        const timeMax = new Date();
        timeMax.setFullYear(timeMax.getFullYear() + 1); // Include events up to a year from now
        
        return gapi.client.calendar.events.list({
            'calendarId': calendarId,
            'timeMin': timeMin.toISOString(),
            'timeMax': timeMax.toISOString(),
            'showDeleted': false,
            'singleEvents': true,
            'maxResults': 250,
            'orderBy': 'startTime'
        }).then(response => {
            return response.result.items;
        });
    }

    /**
     * Navigate to previous/next time period
     */
    navigateDate(direction) {
        if (this.currentView === 'month') {
            // Navigate by month
            if (direction === 'prev') {
                this.currentDate.setMonth(this.currentDate.getMonth() - 1);
            } else {
                this.currentDate.setMonth(this.currentDate.getMonth() + 1);
            }
        } else if (this.currentView === 'week') {
            // Navigate by week
            const days = direction === 'prev' ? -7 : 7;
            this.currentDate.setDate(this.currentDate.getDate() + days);
        } else {
            // Navigate by day
            const days = direction === 'prev' ? -1 : 1;
            this.currentDate.setDate(this.currentDate.getDate() + days);
        }
        
        this.renderCalendar();
    }

    /**
     * Go to today
     */
    goToToday() {
        this.currentDate = new Date();
        this.renderCalendar();
    }

    /**
     * Change calendar view (month, week, day)
     */
    changeView(view) {
        this.currentView = view;
        
        // Update active button
        document.querySelectorAll('.calendar-view-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.getElementById(`${view}-view-btn`).classList.add('active');
        
        this.renderCalendar();
    }

    /**
     * Render the calendar based on current view
     */
    renderCalendar() {
        const container = document.getElementById('calendar-container');
        const dateDisplay = document.getElementById('current-date-display');
        
        if (!container || !dateDisplay) return;
        
        // Update the date display
        if (this.currentView === 'month') {
            dateDisplay.textContent = this.formatMonthYearHeader(this.currentDate);
        } else if (this.currentView === 'week') {
            dateDisplay.textContent = this.formatWeekHeader(this.currentDate);
        } else {
            dateDisplay.textContent = this.formatDayHeader(this.currentDate);
        }
        
        // Render the appropriate view
        if (this.currentView === 'month') {
            this.renderMonthView(container);
        } else if (this.currentView === 'week') {
            this.renderWeekView(container);
        } else {
            this.renderDayView(container);
        }
    }

    /**
     * Format a month/year header
     */
    formatMonthYearHeader(date) {
        const options = { month: 'long', year: 'numeric' };
        return date.toLocaleDateString(undefined, options);
    }

    /**
     * Format a week header
     */
    formatWeekHeader(date) {
        const start = new Date(date);
        const dayOfWeek = start.getDay();
        start.setDate(start.getDate() - dayOfWeek); // Start of week (Sunday)
        
        const end = new Date(start);
        end.setDate(start.getDate() + 6); // End of week (Saturday)
        
        return `${start.toLocaleDateString()} - ${end.toLocaleDateString()}`;
    }

    /**
     * Format a day header
     */
    formatDayHeader(date) {
        const options = { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' };
        return date.toLocaleDateString(undefined, options);
    }

    /**
     * Render the month view
     */
    renderMonthView(container) {
        const year = this.currentDate.getFullYear();
        const month = this.currentDate.getMonth();
        
        // Get the first day of the month
        const firstDay = new Date(year, month, 1);
        const startingDayOfWeek = firstDay.getDay(); // 0 (Sunday) to 6 (Saturday)
        
        // Get the number of days in the month
        const lastDay = new Date(year, month + 1, 0);
        const daysInMonth = lastDay.getDate();
        
        // Get the last day of the previous month
        const prevMonthLastDay = new Date(year, month, 0);
        const daysInPrevMonth = prevMonthLastDay.getDate();
        
        // Create the grid
        let html = '<div class="month-grid">';
        
        // Add day headers
        const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
        days.forEach(day => {
            html += `<div class="day-header">${day}</div>`;
        });
        
        // Add days from previous month
        let day = 1;
        let date = 1;
        let nextMonth = false;
        
        // Previous month days
        for (let i = 0; i < startingDayOfWeek; i++) {
            const prevDate = daysInPrevMonth - startingDayOfWeek + i + 1;
            html += `<div class="day-cell other-month">
                        <span class="day-number">${prevDate}</span>
                        <div class="day-events"></div>
                     </div>`;
        }
        
        // Current month days
        while (date <= daysInMonth) {
            const currentDate = new Date(year, month, date);
            const isToday = this.isToday(currentDate);
            
            html += `<div class="day-cell ${isToday ? 'today' : ''}">
                        <span class="day-number">${date}</span>
                        <div class="day-events">
                            ${this.getDayEvents(currentDate)}
                        </div>
                     </div>`;
            
            date++;
            day++;
            
            // Start a new row every 7 days
            if (day > 7) {
                day = 1;
            }
        }
        
        // Next month days to fill the grid
        const remainingCells = 7 - ((startingDayOfWeek + daysInMonth) % 7);
        if (remainingCells < 7) {
            for (let i = 1; i <= remainingCells; i++) {
                html += `<div class="day-cell other-month">
                            <span class="day-number">${i}</span>
                            <div class="day-events"></div>
                         </div>`;
            }
        }
        
        html += '</div>';
        container.innerHTML = html;
        
        // Add event listeners for day events
        const eventElements = container.querySelectorAll('.day-event');
        eventElements.forEach(element => {
            element.addEventListener('click', () => {
                const eventId = element.getAttribute('data-event-id');
                if (eventId) {
                    this.showEventDetails(parseInt(eventId));
                }
            });
        });
    }

    /**
     * Render the week view
     */
    renderWeekView(container) {
        // Get the start of the week (Sunday)
        const startDate = new Date(this.currentDate);
        const dayOfWeek = startDate.getDay();
        startDate.setDate(startDate.getDate() - dayOfWeek);
        
        // Create grid
        let html = '<div class="time-grid">';
        
        // Add header row with days
        html += '<div></div>'; // Empty corner cell
        
        for (let i = 0; i < 7; i++) {
            const date = new Date(startDate);
            date.setDate(date.getDate() + i);
            const isToday = this.isToday(date);
            
            const dayName = date.toLocaleDateString(undefined, { weekday: 'short' });
            const dayNum = date.getDate();
            
            html += `<div class="day-header ${isToday ? 'today' : ''}">${dayName} ${dayNum}</div>`;
        }
        
        // Add time rows
        for (let hour = 7; hour < 20; hour++) { // 7 AM to 7 PM
            // Time label
            const hourLabel = hour % 12 === 0 ? 12 : hour % 12;
            const ampm = hour < 12 ? 'AM' : 'PM';
            
            html += `<div class="time-label">${hourLabel} ${ampm}</div>`;
            
            // Day columns
            for (let day = 0; day < 7; day++) {
                const date = new Date(startDate);
                date.setDate(date.getDate() + day);
                date.setHours(hour, 0, 0, 0);
                
                const endTime = new Date(date);
                endTime.setHours(hour + 1, 0, 0, 0);
                
                const isToday = this.isToday(date);
                
                html += `<div class="time-cell ${isToday ? 'today' : ''}" data-date="${date.toISOString()}">
                            ${this.getTimeSlotEvents(date, endTime)}
                         </div>`;
            }
        }
        
        html += '</div>';
        container.innerHTML = html;
        
        // Add event listeners for events
        const eventElements = container.querySelectorAll('.event-item');
        eventElements.forEach(element => {
            element.addEventListener('click', () => {
                const eventId = element.getAttribute('data-event-id');
                if (eventId) {
                    this.showEventDetails(parseInt(eventId));
                }
            });
        });
    }

    /**
     * Render the day view
     */
    renderDayView(container) {
        // Create grid
        let html = '<div class="day-view-grid">';
        
        // Add header
        const isToday = this.isToday(this.currentDate);
        const dayName = this.currentDate.toLocaleDateString(undefined, { weekday: 'long' });
        const monthDay = this.currentDate.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
        
        html += `<div></div>`;
        html += `<div class="day-header ${isToday ? 'today' : ''}">${dayName}, ${monthDay}</div>`;
        
        // Add time rows
        for (let hour = 7; hour < 20; hour++) { // 7 AM to 7 PM
            // Time label
            const hourLabel = hour % 12 === 0 ? 12 : hour % 12;
            const ampm = hour < 12 ? 'AM' : 'PM';
            
            html += `<div class="time-label">${hourLabel} ${ampm}</div>`;
            
            // Time slot
            const date = new Date(this.currentDate);
            date.setHours(hour, 0, 0, 0);
            
            const endTime = new Date(date);
            endTime.setHours(hour + 1, 0, 0, 0);
            
            html += `<div class="time-cell" data-date="${date.toISOString()}">
                        ${this.getTimeSlotEvents(date, endTime)}
                     </div>`;
        }
        
        html += '</div>';
        container.innerHTML = html;
        
        // Add event listeners for events
        const eventElements = container.querySelectorAll('.event-item');
        eventElements.forEach(element => {
            element.addEventListener('click', () => {
                const eventId = element.getAttribute('data-event-id');
                if (eventId) {
                    this.showEventDetails(parseInt(eventId));
                }
            });
        });
    }

    /**
     * Check if a date is today
     */
    isToday(date) {
        const today = new Date();
        return date.getDate() === today.getDate() &&
               date.getMonth() === today.getMonth() &&
               date.getFullYear() === today.getFullYear();
    }

    /**
     * Get events for a specific day
     */
    getDayEvents(date) {
        // Фильтруем события для этого дня
        const dayEvents = this.events.filter(event => {
            const eventStart = new Date(event.start);
            const eventEnd = new Date(event.end);
            
            // Проверяем, что событие начинается или продолжается в этот день
            const startDate = new Date(eventStart.getFullYear(), eventStart.getMonth(), eventStart.getDate());
            const endDate = new Date(eventEnd.getFullYear(), eventEnd.getMonth(), eventEnd.getDate());
            const currentDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());
            
            // Событие начинается в этот день или продолжается в этот день
            return (
                // Событие начинается в этот день
                (startDate.getTime() === currentDate.getTime()) ||
                // Многодневное событие, которое продолжается в этот день
                (startDate < currentDate && endDate >= currentDate)
            );
        });
        
        // Сортируем события по времени начала
        const sortedEvents = dayEvents.sort((a, b) => new Date(a.start) - new Date(b.start));
        
        if (sortedEvents.length === 0) return '';
        
        // Создаем Map для группировки событий с одинаковыми ID
        // Нам нужно только одно событие для каждого ID
        const uniqueEvents = new Map();
        
        sortedEvents.forEach(event => {
            // Если это событие уже есть в Map, пропускаем его
            if (uniqueEvents.has(event.id)) {
                return;
            }
            
            // Добавляем событие в Map
            uniqueEvents.set(event.id, event);
        });
        
        let html = '';
        // Обрабатываем только уникальные события
        uniqueEvents.forEach(event => {
            const eventStart = new Date(event.start);
            const eventEnd = new Date(event.end);
            
            // Определяем классы для стилизации
            // Для однодневных и многодневных событий используем разные стили
            const eventStartDate = new Date(eventStart.getFullYear(), eventStart.getMonth(), eventStart.getDate());
            const currentDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());
            const isStartDay = eventStartDate.getTime() === currentDate.getTime();
            
            // Если это день начала события, показываем время
            // Иначе показываем "Продолжение"
            const timeLabel = isStartDay ? 
                eventStart.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : 
                'Продолжение';
            
            const eventClass = isStartDay ? 'day-event' : 'day-event multi-day-event';
            const color = this.calendarColors[event.category] || this.calendarColors.default;
            
            html += `<div class="${eventClass}" style="background-color: ${color}" data-event-id="${event.id}">
                        ${timeLabel} ${event.title}
                     </div>`;
        });
        
        return html;
    }

    /**
     * Get events for a specific time slot
     */
    getTimeSlotEvents(startTime, endTime) {
        // Filter events for this time slot
        const slotEvents = this.events.filter(event => {
            const eventStart = new Date(event.start);
            const eventEnd = new Date(event.end);
            
            // Check if event overlaps with this time slot
            return (eventStart < endTime && eventEnd > startTime);
        }).sort((a, b) => new Date(a.start) - new Date(b.start));
        
        if (slotEvents.length === 0) return '';
        
        let html = '';
        slotEvents.forEach(event => {
            const eventStart = new Date(event.start);
            const eventEnd = new Date(event.end);
            
            // Calculate position and height
            let top = 0;
            let height = 100;
            
            // If event starts within this hour
            if (eventStart >= startTime) {
                const minutesAfterHour = eventStart.getMinutes();
                top = (minutesAfterHour / 60) * 100;
            }
            
            // If event ends within this hour
            if (eventEnd <= endTime) {
                const minutesAfterHour = eventEnd.getMinutes();
                height = (minutesAfterHour / 60) * 100 - top;
            } else {
                height = 100 - top;
            }
            
            // Ensure minimum height
            height = Math.max(height, 20); // Increased from 10 to 20 to make events taller
            
            const color = this.calendarColors[event.category] || this.calendarColors.default;
            
            html += `<div class="event-item" style="background-color: ${color}; top: ${top}%; height: ${height}%" data-event-id="${event.id}">
                        ${event.title}
                     </div>`;
        });
        
        return html;
    }

    /**
     * Render the upcoming events list
     */
    renderUpcomingEvents() {
        const container = document.getElementById('event-list');
        if (!container) return;
        
        // Get upcoming events (next 7 days)
        const now = new Date();
        const nextWeek = new Date(now);
        nextWeek.setDate(nextWeek.getDate() + 7);
        
        const upcomingEvents = this.events.filter(event => {
            const eventDate = new Date(event.start);
            return eventDate >= now && eventDate <= nextWeek;
        }).sort((a, b) => new Date(a.start) - new Date(b.start));
        
        if (upcomingEvents.length === 0) {
            container.innerHTML = '<p>No upcoming events</p>';
            return;
        }
        
        let html = '';
        upcomingEvents.slice(0, 5).forEach(event => {
            const date = new Date(event.start);
            const formattedDate = date.toLocaleDateString();
            const formattedTime = date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            const color = this.calendarColors[event.category] || this.calendarColors.default;
            
            html += `<div class="event-list-item" style="border-left-color: ${color}" data-event-id="${event.id}">
                        <div class="event-list-title">${event.title}</div>
                        <div class="event-list-time">${formattedDate} at ${formattedTime}</div>
                     </div>`;
        });
        
        container.innerHTML = html;
        
        // Add event listeners
        const eventItems = container.querySelectorAll('.event-list-item');
        eventItems.forEach(item => {
            item.addEventListener('click', () => {
                const eventId = item.getAttribute('data-event-id');
                if (eventId) {
                    this.showEventDetails(parseInt(eventId));
                }
            });
        });
    }

    /**
     * Show event details in a popup
     */
    showEventDetails(eventId) {
        const event = this.events.find(e => e.id === eventId);
        if (!event) return;
        
        // Create popup if it doesn't exist
        let popup = document.getElementById('event-details-popup');
        if (!popup) {
            popup = document.createElement('div');
            popup.id = 'event-details-popup';
            document.body.appendChild(popup);
        }
        
        // Format dates
        const startDate = new Date(event.start);
        const endDate = new Date(event.end);
        const formattedStart = `${startDate.toLocaleDateString()} at ${startDate.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;
        const formattedEnd = `${endDate.toLocaleDateString()} at ${endDate.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;
        
        // Populate popup
        popup.innerHTML = `
            <div class="event-details-header">
                <h3 class="event-details-title">${event.title}</h3>
                <button class="close-popup">&times;</button>
            </div>
            <div class="event-details-content">
                <p><strong>Start:</strong> ${formattedStart}</p>
                <p><strong>End:</strong> ${formattedEnd}</p>
                ${event.location ? `<p><strong>Location:</strong> ${event.location}</p>` : ''}
                ${event.description ? `<p><strong>Description:</strong> ${event.description}</p>` : ''}
                <p><strong>Category:</strong> ${event.category || 'default'}</p>
            </div>
            <div class="event-details-actions">
                <button id="edit-event-btn" class="calendar-btn">Edit</button>
                <button id="delete-event-btn" class="calendar-btn">Delete</button>
            </div>
        `;
        
        // Show the popup
        popup.style.display = 'block';
        
        // Add event listeners
        popup.querySelector('.close-popup').addEventListener('click', () => {
            popup.style.display = 'none';
        });
        
        popup.querySelector('#edit-event-btn').addEventListener('click', () => {
            this.editEvent(event);
            popup.style.display = 'none';
        });
        
        popup.querySelector('#delete-event-btn').addEventListener('click', async () => {
            if (confirm(`Are you sure you want to delete "${event.title}"?`)) {
                await this.deleteEvent(event.id);
                await this.loadEvents();
                this.renderCalendar();
                this.renderUpcomingEvents();
                popup.style.display = 'none';
            }
        });
        
        // Close when clicking outside
        window.addEventListener('click', (e) => {
            if (e.target === popup) {
                popup.style.display = 'none';
            }
        });
    }

    /**
     * Edit an event
     */
    editEvent(event) {
        // Show the add event form
        const form = document.getElementById('add-event-form');
        form.classList.remove('hidden');
        
        // Fill in the form with event details
        document.getElementById('event-title').value = event.title;
        document.getElementById('event-start').value = this.formatDateTimeForInput(new Date(event.start));
        document.getElementById('event-end').value = this.formatDateTimeForInput(new Date(event.end));
        document.getElementById('event-category').value = event.category || 'default';
        document.getElementById('event-location').value = event.location || '';
        document.getElementById('event-description').value = event.description || '';
        
        // Replace the form submit handler to update instead of add
        const originalSubmitHandler = document.getElementById('event-form').onsubmit;
        
        document.getElementById('event-form').onsubmit = async (e) => {
            e.preventDefault();
            
            // Get updated values
            const updatedEvent = {
                id: event.id,
                title: document.getElementById('event-title').value,
                start: new Date(document.getElementById('event-start').value),
                end: new Date(document.getElementById('event-end').value),
                category: document.getElementById('event-category').value,
                location: document.getElementById('event-location').value,
                description: document.getElementById('event-description').value,
                source: event.source
            };
            
            try {
                await this.updateEvent(updatedEvent);
                await this.loadEvents();
                this.toggleAddEventForm();
                this.renderCalendar();
                this.renderUpcomingEvents();
                
                // Restore original handler
                document.getElementById('event-form').onsubmit = originalSubmitHandler;
            } catch (error) {
                console.error('Error updating event:', error);
                alert('Error updating event: ' + error.message);
            }
        };
    }

    /**
     * Initialize the calendar module
     */
    async init() {
        if (this.initialized) return;
            
        console.log("Calendar module init() started");
        
        // Initialize IndexedDB for local storage
        await this.initDatabase();
        
        // Load events from database
        await this.loadEvents();
        
        // Initialize calendar visuals and UI
        this.initCalendar();
        
        // Initialize LLM calendar processor for chat commands
        this.initLLMCalendarProcessor();
        
        // Mark as initialized
        this.initialized = true;
        
        // Add calendar styles
        this.addCalendarStyles();
        
        console.log("Calendar module init() completed");
    }

    /**
     * Setup the calendar container and structure
     */
    setupCalendarContainer() {
        console.log("Setting up calendar container");
        
        // Check if calendar container already exists
        const container = document.getElementById('calendar-container');
        if (!container) {
            console.error("Calendar container not found in DOM");
            return;
        }
        
        // Make sure calendar tab content exists
        const calendarTab = document.getElementById('calendar-tab');
        if (!calendarTab) {
            console.error("Calendar tab content element not found");
            return;
        }
        
        console.log("Calendar container setup complete");
    }

    /**
     * Debug method to log all events to console
     */
    debugShowAllEvents() {
        console.log("All events in calendar module:", this.events.length);
        if (this.events.length === 0) {
            console.log("No events found");
            return;
        }
        
        this.events.forEach((event, index) => {
            const start = new Date(event.start);
            const end = new Date(event.end);
            console.log(`Event ${index + 1}:`, {
                id: event.id,
                title: event.title,
                start: start.toLocaleString(),
                end: end.toLocaleString(),
                category: event.category,
                source: event.source
            });
        });
    }

    /**
     * Add a message to the chat interface
     */
    addMessageToChat(message, isUser) {
        const chatContainer = document.getElementById('chat-container');
        if (!chatContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
        messageDiv.textContent = message;
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;

        // Check if the message contains calendar-related content
        if (!isUser && (message.includes('calendar') || message.includes('event') || message.includes('meeting'))) {
            this.refreshCalendar();
        }
    }

    /**
     * Extract event information from message
     */
    extractEventInfo(message) {
        console.log("Extracting event info from message:", message);
        
        // Initialize result
        const eventInfo = {
            title: null,
            startTime: null,
            endTime: null,
            description: null,
            location: null,
            category: 'default'
        };
        
        try {
            // Clean up the message by removing common command phrases in both languages
            let cleanMessage = message
                .replace(/(добав(ь|ить)\s+событие\s+)|(add\s+event\s+)|(schedule\s+meeting\s+)|(create\s+event\s+)/i, '')
                .trim();
            
            // Extract event title - look for text before time/date markers in both languages
            const titleMatch = cleanMessage.match(/^(.*?)(?:\s+(?:в|на|at|on|tomorrow|today|завтра|сегодня|послезавтра|\d{1,2}[:.]\d{2}|\d{1,2}\s*(?:am|pm|AM|PM)|\d{1,2}\s+\w+))/i);
            if (titleMatch && titleMatch[1]) {
                eventInfo.title = titleMatch[1].trim();
            } else {
                // If not found, take text until first number or date
                const altTitleMatch = cleanMessage.match(/^(.*?)(?:\s+\d)/i);
                if (altTitleMatch && altTitleMatch[1]) {
                    eventInfo.title = altTitleMatch[1].trim();
                } else {
                    // Last resort - take first word
                    eventInfo.title = cleanMessage.split(/\s+/)[0];
                }
            }
            
            // First check for AM/PM format time (like 11 AM, 5PM)
            const ampmTimeMatch = message.match(/(\d{1,2})\s*(am|pm|AM|PM)/i);
            const timeMatch = message.match(/(\d{1,2})[:.](\d{2})(?:\s*-\s*(\d{1,2})[:.:](\d{2}))?/);
            const startDate = new Date();
            
            if (ampmTimeMatch) {
                let hours = parseInt(ampmTimeMatch[1]);
                const isPM = ampmTimeMatch[2].toLowerCase() === 'pm';
                
                // Convert to 24-hour format
                if (isPM && hours < 12) hours += 12;
                if (!isPM && hours === 12) hours = 0;
                
                // Handle date references in both languages
                if (message.toLowerCase().includes('tomorrow') || message.toLowerCase().includes('завтра')) {
                    startDate.setDate(startDate.getDate() + 1);
                } else if (message.toLowerCase().includes('after tomorrow') || message.toLowerCase().includes('послезавтра')) {
                    startDate.setDate(startDate.getDate() + 2);
                }
                
                startDate.setHours(hours, 0, 0, 0);
                eventInfo.startTime = startDate;
                
                // Default end time is 1 hour later
                const endDate = new Date(startDate);
                endDate.setHours(endDate.getHours() + 1);
                eventInfo.endTime = endDate;
            } else if (timeMatch) {
                const startHours = parseInt(timeMatch[1]);
                const startMinutes = parseInt(timeMatch[2]);
                
                // Handle date references in both languages
                if (message.toLowerCase().includes('tomorrow') || message.toLowerCase().includes('завтра')) {
                    startDate.setDate(startDate.getDate() + 1);
                } else if (message.toLowerCase().includes('after tomorrow') || message.toLowerCase().includes('послезавтра')) {
                    startDate.setDate(startDate.getDate() + 2);
                }
                
                startDate.setHours(startHours, startMinutes, 0, 0);
                eventInfo.startTime = startDate;
                
                // Handle end time if specified
                if (timeMatch[3] && timeMatch[4]) {
                    const endDate = new Date(startDate);
                    endDate.setHours(parseInt(timeMatch[3]), parseInt(timeMatch[4]), 0, 0);
                    eventInfo.endTime = endDate;
                } else {
                    // Default end time is 1 hour later
                    const endDate = new Date(startDate);
                    endDate.setHours(endDate.getHours() + 1);
                    eventInfo.endTime = endDate;
                }
            }
            
            // Find location if specified
            const locationMatch = message.match(/(?:at|в|на)\s+([^,]+)/i);
            if (locationMatch && locationMatch[1]) {
                eventInfo.location = locationMatch[1].trim();
            }
            
            return eventInfo;
        } catch (error) {
            console.error('Error extracting event info:', error);
            return null;
        }
    }

    /**
     * Handle a command to add an event
     */
    async handleAddEventCommand(message) {
        try {
            console.log("Processing add event command:", message);
            
            // Extract event info from the message
            const eventInfo = this.extractEventInfo(message);
            if (!eventInfo || !eventInfo.title || !eventInfo.startTime) {
                this.addMessageToChat('Не удалось извлечь необходимую информацию о событии. Пожалуйста, укажите название события и время.', false);
                return;
            }
            
            console.log("Extracted event info:", eventInfo);
            
            // Create a calendar event object
            const event = {
                title: eventInfo.title,
                start: eventInfo.startTime,
                end: eventInfo.endTime || new Date(eventInfo.startTime.getTime() + 60 * 60 * 1000), // 1 hour by default
                description: eventInfo.description || '',
                location: eventInfo.location || '',
                category: eventInfo.category || 'default'
            };
            
            console.log("Creating event:", event);
            
            // Save the event to the database
            const savedEvent = await this.saveEvent(event);
            
            if (savedEvent) {
                console.log("Event saved successfully:", savedEvent);
                
                // Добавляем событие в массив this.events вручную
                const newEvent = {
                    id: savedEvent.id,
                    title: savedEvent.title,
                    start: new Date(savedEvent.start),
                    end: new Date(savedEvent.end),
                    description: savedEvent.description || '',
                    location: savedEvent.location || '',
                    category: savedEvent.category || 'default',
                    source: 'server'
                };
                
                this.events.push(newEvent);
                console.log(`Added event to local array, total events: ${this.events.length}`);
                
                // Обновляем отображение календаря
                this.refreshCalendar();
                
                // Add confirmation message to chat
                const startTimeFormatted = new Date(savedEvent.start).toLocaleString();
                this.addMessageToChat(`Событие "${savedEvent.title}" добавлено в календарь на ${startTimeFormatted}.`, false);
            } else {
                console.error("Failed to save event");
                this.addMessageToChat('Произошла ошибка при сохранении события.', false);
            }
        } catch (error) {
            console.error("Error adding event:", error);
            this.addMessageToChat('Произошла ошибка при добавлении события в календарь.', false);
        }
    }
}

// Create and initialize the calendar module when DOM is fully loaded
document.addEventListener('DOMContentLoaded', async () => {
    console.log("DOM fully loaded, initializing calendar");
    
    // Проверяем, не инициализирован ли уже календарь
    if (window.calendarModule) {
        console.log("Calendar module already initialized, skipping");
        return;
    }
    
    // Create calendar instance
    const calendar = new CalendarModule();
    
    // Initialize the calendar
    await calendar.init();
    
    // Make calendar accessible globally
    window.calendarModule = calendar;
    
    console.log("Calendar module initialized and assigned to window.calendarModule");
});