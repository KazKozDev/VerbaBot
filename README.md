<div align="center">
  <img src="https://github.com/user-attachments/assets/63bc3058-bace-4234-a473-7d92e6bd1c84" alt="VerbaBot Logo">
</div>
<h1 align="center">VerbaBot</h1>
<p align="center">A powerful multimodal LLM assistant with advanced RAG.</p>
<p align="center">Local Control. Global Capabilities.</p>

VerbaBot is a comprehensive AI assistant designed to run locally, ensuring complete privacy of your data. By integrating powerful language models with a sophisticated Retrieval-Augmented Generation (RAG) pipeline, VerbaBot offers intelligent conversations, information retrieval, and personal knowledge management - all within a user-friendly web interface.

<div align="center">
  <img src="https://github.com/user-attachments/assets/e20930d6-f29f-47f7-a8a0-39fefbdb3538" alt="VerbaBot Demo">
</div>

## Key Features

- **Modern Web Interface**: Clean, responsive UI for seamless interaction with the assistant
- **Calendar Integration**: Manage your schedule through natural language - create events, set reminders, and organize meetings
- **Personal Memory System**: The assistant remembers important personal information across conversations
- **Advanced RAG Pipeline**: Process and understand documents in multiple formats (PDF, DOCX, TXT, CSV, etc.)
- **Internet Access**: Retrieve real-time information using DuckDuckGo search and deep web exploration
- **Model Selection**: Switch between different Ollama models in real-time to optimize for specific tasks
- **Voice Input Capability**: Speak directly to your assistant (in development)
- **Image Analysis**: Analyze and describe images through multimodal models
- **Complete Privacy**: All data processing happens locally on your device

![VerbaBot Interface](https://github.com/user-attachments/assets/c8c5e8b6-2cb1-48d1-94c1-13bac9759e12)

## Architecture

VerbaBot consists of several integrated components:

- **Core Assistant**: Python-based server handling all interactions and intelligence
- **RAG System**: Document processing, vectorization, and retrieval system
- **Memory System**: Contextual memory for personalized interactions
- **Calendar System**: Natural language calendar management
- **Web Interface**: Responsive UI for interacting with the assistant

## Installation

```bash
# Clone the repository
git clone https://github.com/username/VerbaBot.git
cd VerbaBot

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running locally
# Visit https://ollama.ai/ to install if needed
```

## Usage

```bash
# Start the server
python server.py
```

The web interface will be available at http://localhost:5001/

## Configuration

### Models

VerbaBot uses local Ollama models. Default models:
- Main LLM: `gemma3:12b`
- Image Recognition: `gemma3:12b`

You can change these in the settings menu or directly in the server.py file.

### Optional Features

To enable voice input capability:
1. Uncomment the corresponding dependencies in the `requirements.txt` file
2. Install them with `pip install -r requirements.txt`
3. Ensure you have the required audio libraries for your system

### Environment Variables

Create a `.env` file in the root directory with these optional settings:

```
# Google Calendar API (optional)
GOOGLE_API_KEY=your_api_key
GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_client_secret

# Gmail API (optional)
GMAIL_API_KEY=your_api_key
GMAIL_CLIENT_ID=your_client_id
GMAIL_CLIENT_SECRET=your_client_secret
```

## Project Structure

- `server.py` - Main server application
- `memory.py` - Personal memory system
- `calendar_integration.py` - Calendar functionality
- `search.py` - Web search capabilities
- `chat.html` - Main web interface
- `calendar.js` - Calendar front-end
- `voice_input.py` - Voice input processing
- `requirements.txt` - Project dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT](LICENSE) 
