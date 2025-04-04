import asyncio
import tempfile
import os
import time
import logging
import base64
import whisper
from quart import Blueprint, request, jsonify
from aiofiles import open as aio_open

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Whisper model with a configurable model size
# Available sizes: tiny, base, small, medium, large
# For better performance with non-English languages, use the multilingual models (without the .en suffix)
DEFAULT_MODEL_SIZE = "base"
model = None

# Temporary directory for audio files
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# Create Blueprint for voice input routes
voice_input_bp = Blueprint('voice_input', __name__)

def load_whisper_model(model_size=DEFAULT_MODEL_SIZE):
    """Load Whisper model with specified size."""
    global model
    try:
        logger.info(f"Loading Whisper model: {model_size}")
        model = whisper.load_model(model_size)
        logger.info(f"Whisper model {model_size} loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading Whisper model: {e}")
        return False

@voice_input_bp.route('/voice/models', methods=['GET'])
async def get_available_models():
    """Return available Whisper model sizes."""
    return jsonify({
        'models': [
            {'id': 'tiny', 'name': 'Tiny (39M parameters)', 'speed': 'Fastest'},
            {'id': 'base', 'name': 'Base (74M parameters)', 'speed': 'Fast'},
            {'id': 'small', 'name': 'Small (244M parameters)', 'speed': 'Medium'},
            {'id': 'medium', 'name': 'Medium (769M parameters)', 'speed': 'Slow'},
            {'id': 'large', 'name': 'Large (1.5B parameters)', 'speed': 'Slowest'}
        ]
    })

@voice_input_bp.route('/voice/load_model', methods=['POST'])
async def load_model_endpoint():
    """Endpoint to load a specific Whisper model."""
    try:
        data = await request.get_json()
        model_size = data.get('model_size', DEFAULT_MODEL_SIZE)
        
        success = load_whisper_model(model_size)
        if success:
            return jsonify({'success': True, 'message': f'Model {model_size} loaded successfully'})
        else:
            return jsonify({'success': False, 'error': f'Failed to load model {model_size}'}), 500
    except Exception as e:
        logger.error(f"Error in load_model endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@voice_input_bp.route('/voice/transcribe', methods=['POST'])
async def transcribe_audio():
    """Transcribe audio file using Whisper."""
    try:
        # Check if model is loaded
        global model
        if model is None:
            load_whisper_model()
            if model is None:
                return jsonify({'error': 'Failed to load Whisper model'}), 500
        
        data = await request.get_json()
        base64_audio = data.get('audio')
        language = data.get('language', None)  # Optional language hint
        
        if not base64_audio:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(base64_audio.split(',')[1] if ',' in base64_audio else base64_audio)
        
        # Save to temporary file
        timestamp = int(time.time())
        temp_path = os.path.join(TEMP_DIR, f"audio_{timestamp}.webm")
        
        async with aio_open(temp_path, 'wb') as f:
            await f.write(audio_bytes)
        
        # Define transcription options
        options = {}
        if language:
            options['language'] = language
        
        # Run transcription in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: model.transcribe(temp_path, **options)
        )
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary audio file: {e}")
        
        # Return transcription result
        return jsonify({
            'success': True,
            'text': result['text'],
            'language': result.get('language', 'unknown'),
            'segments': result.get('segments', [])
        })
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@voice_input_bp.route('/voice/detect_language', methods=['POST'])
async def detect_language():
    """Detect language in audio file using Whisper."""
    try:
        # Check if model is loaded
        global model
        if model is None:
            load_whisper_model()
            if model is None:
                return jsonify({'error': 'Failed to load Whisper model'}), 500
        
        data = await request.get_json()
        base64_audio = data.get('audio')
        
        if not base64_audio:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(base64_audio.split(',')[1] if ',' in base64_audio else base64_audio)
        
        # Save to temporary file
        timestamp = int(time.time())
        temp_path = os.path.join(TEMP_DIR, f"audio_{timestamp}.webm")
        
        async with aio_open(temp_path, 'wb') as f:
            await f.write(audio_bytes)
            
        # Load audio
        audio = whisper.load_audio(temp_path)
        audio = whisper.pad_or_trim(audio)
        
        # Make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        
        # Detect language
        loop = asyncio.get_event_loop()
        _, probs = await loop.run_in_executor(
            None, 
            lambda: model.detect_language(mel)
        )
        
        language = max(probs, key=probs.get)
        language_probs = {k: float(v) for k, v in probs.items()}
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary audio file: {e}")
        
        return jsonify({
            'success': True,
            'language': language,
            'probabilities': language_probs
        })
        
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def init_app(app, model_size=DEFAULT_MODEL_SIZE):
    """Initialize the voice input module and register blueprint."""
    app.register_blueprint(voice_input_bp)
    
    # Load model in background to avoid blocking app startup
    @app.before_serving
    async def setup_whisper():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: load_whisper_model(model_size))
        
    return app