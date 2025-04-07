// Voice Input Module for VerbaBot
// This script handles voice recording and transcription

class VoiceInputManager {
    constructor(options = {}) {
        // Configuration options with defaults
        this.options = {
            serverUrl: options.serverUrl || 'http://127.0.0.1:5001',
            audioFormat: options.audioFormat || 'audio/webm',
            language: options.language || null,
            autoStop: options.autoStop || true,
            autoStopTime: options.autoStopTime || 5000, // ms of silence before auto-stopping
            silenceThreshold: options.silenceThreshold || -45, // dB
            onTranscriptionStart: options.onTranscriptionStart || (() => {}),
            onTranscriptionEnd: options.onTranscriptionEnd || (() => {}),
            onTranscriptionResult: options.onTranscriptionResult || (() => {}),
            onError: options.onError || ((err) => console.error('Voice input error:', err))
        };

        // State variables
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.silenceDetector = null;
        this.silenceStart = null;
        this.autoStopTimer = null;
        this.stream = null;
        this.audioContext = null;
        this.analyser = null;
        this.whisperModelLoaded = false;
        
        // Initialize
        this.loadWhisperModel('base');
    }

    async loadWhisperModel(modelSize = 'base') {
        try {
            const response = await fetch(`${this.options.serverUrl}/voice/load_model`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ model_size: modelSize })
            });
            
            const data = await response.json();
            this.whisperModelLoaded = data.success;
            
            if (!data.success) {
                console.warn('Failed to load Whisper model:', data.error);
            } else {
                console.log(`Whisper model ${modelSize} loaded successfully`);
            }
            
            return data.success;
        } catch (error) {
            console.error('Error loading Whisper model:', error);
            this.whisperModelLoaded = false;
            return false;
        }
    }

    async getAvailableModels() {
        try {
            const response = await fetch(`${this.options.serverUrl}/voice/models`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching available models:', error);
            this.options.onError('Failed to fetch available models');
            return { models: [] };
        }
    }

    async startRecording() {
        if (this.isRecording) {
            return;
        }

        try {
            // Request microphone access
            this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Create media recorder
            this.mediaRecorder = new MediaRecorder(this.stream, {
                mimeType: this.getSupportedMimeType()
            });
            
            // Set up audio processing for silence detection if autoStop is enabled
            if (this.options.autoStop) {
                this.setupSilenceDetection();
            }
            
            // Reset audio chunks
            this.audioChunks = [];
            
            // Set up event handlers
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = async () => {
                // Clean up
                this.cleanupRecording();
                
                // Process the recorded audio
                await this.processRecording();
            };
            
            // Start recording
            this.mediaRecorder.start(100); // Collect data in 100ms chunks
            this.isRecording = true;
            
            // Notify listener
            this.options.onTranscriptionStart();
            
            return true;
        } catch (error) {
            console.error('Error starting recording:', error);
            this.options.onError('Failed to start recording: ' + error.message);
            return false;
        }
    }

    stopRecording() {
        if (!this.isRecording || !this.mediaRecorder) {
            return;
        }
        
        try {
            this.mediaRecorder.stop();
            this.isRecording = false;
        } catch (error) {
            console.error('Error stopping recording:', error);
            this.options.onError('Failed to stop recording');
            this.cleanupRecording();
        }
    }

    cleanupRecording() {
        // Stop all media tracks
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        // Clean up audio processing
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
            this.analyser = null;
        }
        
        // Clear any pending timers
        if (this.autoStopTimer) {
            clearTimeout(this.autoStopTimer);
            this.autoStopTimer = null;
        }
        
        this.isRecording = false;
    }

    async processRecording() {
        if (this.audioChunks.length === 0) {
            this.options.onTranscriptionEnd();
            this.options.onError('No audio recorded');
            return;
        }
        
        try {
            // Create blob from audio chunks
            const audioBlob = new Blob(this.audioChunks, { type: this.getSupportedMimeType() });
            
            // Convert to base64
            const base64Audio = await this.blobToBase64(audioBlob);
            
            // Send to server for transcription
            const response = await fetch(`${this.options.serverUrl}/voice/transcribe`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    audio: base64Audio,
                    language: this.options.language
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.options.onTranscriptionResult(result.text, result);
            } else {
                this.options.onError('Transcription failed: ' + (result.error || 'Unknown error'));
            }
        } catch (error) {
            console.error('Error processing recording:', error);
            this.options.onError('Failed to process recording: ' + error.message);
        } finally {
            this.options.onTranscriptionEnd();
        }
    }

    setupSilenceDetection() {
        try {
            // Create audio context and analyser
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.analyser = this.audioContext.createAnalyser();
            
            // Connect the stream to the analyser
            const source = this.audioContext.createMediaStreamSource(this.stream);
            source.connect(this.analyser);
            
            // Configure the analyser
            this.analyser.fftSize = 2048;
            this.analyser.smoothingTimeConstant = 0.8;
            
            // Create buffer to receive data
            const bufferLength = this.analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            
            // Start silence detection loop
            this.detectSilence(dataArray);
        } catch (error) {
            console.error('Error setting up silence detection:', error);
        }
    }

    detectSilence(dataArray) {
        if (!this.isRecording || !this.analyser) {
            return;
        }
        
        // Get volume data
        this.analyser.getByteFrequencyData(dataArray);
        
        // Calculate average volume
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
            sum += dataArray[i];
        }
        const average = sum / dataArray.length;
        
        // Convert to dB (rough approximation)
        const volume = average > 0 ? 20 * Math.log10(average / 255) : -100;
        
        // Check for silence
        if (volume < this.options.silenceThreshold) {
            if (!this.silenceStart) {
                this.silenceStart = Date.now();
            } else if (Date.now() - this.silenceStart > this.options.autoStopTime) {
                // Stop recording after silence threshold exceeded
                this.stopRecording();
                return;
            }
        } else {
            this.silenceStart = null;
        }
        
        // Continue detection
        requestAnimationFrame(() => this.detectSilence(dataArray));
    }

    getSupportedMimeType() {
        const mimeTypes = [
            'audio/webm',
            'audio/webm;codecs=opus',
            'audio/ogg;codecs=opus',
            'audio/mp4'
        ];
        
        for (const type of mimeTypes) {
            if (MediaRecorder.isTypeSupported(type)) {
                return type;
            }
        }
        
        return '';
    }

    blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VoiceInputManager;
}