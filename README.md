<div align="center">
  <img src="https://github.com/user-attachments/assets/63bc3058-bace-4234-a473-7d92e6bd1c84" alt="VerbaBot Logo">
</div>
<h2 align="center">VerbaBot</h2>


<p align="center">A powerful multimodal LLM assistant with advanced RAG.</p>
<p align="center">Local Control. Global Capabilities.</p>

VerbaBot is an AI assistant designed to run locally, ensuring complete privacy of your data. By integrating language models with a Retrieval-Augmented Generation (RAG) pipeline, the chatbot offers intelligent conversations, information retrieval, and personal knowledge management - all within a user-friendly web interface.

<div align="center">
  <img src="https://github.com/user-attachments/assets/e20930d6-f29f-47f7-a8a0-39fefbdb3538" alt="VerbaBot Demo">
</div>

### Features

- **Calendar Integration**: Manage your schedule through natural language
- **Personal Memory System**: The assistant remembers important personal information across conversations
- **Advanced RAG Pipeline**: Process and understand documents in multiple formats (PDF, DOCX, TXT, CSV, etc.)
- **Image Analysis**: Analyze and describe images
- **Internet Access**: Retrieve real-time information using DuckDuckGo search and deep web exploration
- **Model Selection**: Switch between different Ollama models in real-time to optimize for specific tasks
- **Complete Privacy**: All data processing happens locally on your device

![VerbaBot Interface](https://github.com/user-attachments/assets/c8c5e8b6-2cb1-48d1-94c1-13bac9759e12)

### Architecture

VerbaBot consists of several integrated components:

- **Core Assistant**: Python-based server handling all interactions and intelligence
- **RAG System**: Document processing, vectorization, and retrieval system
- **Memory System**: Contextual memory for personalized interactions
- **Calendar System**: Natural language calendar management
- **Web Interface**: Responsive UI for interacting with the assistant

### Installation

```bash
# Clone the repository
git clone https://github.com/KazKozDev/VerbaBot.git
cd VerbaBot

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running locally
# Visit https://ollama.ai/ to install if needed
```

### Usage

```bash
# Start the server
python3 server.py
```

The web interface will be available at file:///.../VerbaBot/wavy-animation.html

### Configuration

VerbaBot uses local Ollama models. Default models:
- Main LLM: `gemma3:12b`
- Image Recognition: `gemma3:12b`

You can change these in the settings menu or directly in the server.py file.

### Environment Variables

Create a `.env` file in the root directory with these optional settings:

```
# Google Calendar API (optional)
GOOGLE_API_KEY=your_api_key
GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_client_secret
```

### Project Structure

- `server.py` - Main server application
- `memory.py` - Personal memory system
- `calendar_integration.py` - Calendar functionality
- `search.py` - Web search capabilities
- `chat.html` - Main web interface
- `calendar.js` - Calendar front-end
- `voice_input.py` - Voice input processing
- `requirements.txt` - Project dependencies

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

License - [MIT](LICENSE) 

---
If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[Artem KK](https://www.linkedin.com/in/kazkozdev/) | [Issue Tracker](https://github.com/KazKozDev/VerbaBot/issues) 
