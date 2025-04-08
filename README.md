<div align="center">
  <img src="https://github.com/user-attachments/assets/63bc3058-bace-4234-a473-7d92e6bd1c84" alt="VerbaBot Logo">
</div>
<h2 align="center">VerbaBot</h2>

<p align="center">
  <a href="https://github.com/KazKozDev/VerbaBot/actions"><img src="https://img.shields.io/github/actions/workflow/status/KazKozDev/VerbaBot/main.yml?branch=main" alt="Build Status"></a>
  <a href="https://github.com/KazKozDev/VerbaBot/releases"><img src="https://img.shields.io/github/v/release/KazKozDev/VerbaBot" alt="Version"></a>
  <a href="https://github.com/KazKozDev/VerbaBot/blob/main/LICENSE"><img src="https://img.shields.io/github/license/KazKozDev/VerbaBot" alt="License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-45.3%25-blue" alt="Python"></a>
  <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript"><img src="https://img.shields.io/badge/JavaScript-24.5%25-yellow" alt="JavaScript"></a>
  <a href="https://developer.mozilla.org/en-US/docs/Web/HTML"><img src="https://img.shields.io/badge/HTML-23.1%25-orange" alt="HTML"></a>
  <a href="https://www.gnu.org/software/bash/"><img src="https://img.shields.io/badge/Shell-7.1%25-lightgrey" alt="Shell"></a>
  <a href="https://github.com/KazKozDev/VerbaBot"><img src="https://img.shields.io/badge/Platforms-Windows%20%7C%20Linux%20%7C%20macOS-brightgreen" alt="Platforms"></a>
  <a href="https://libraries.io/github/KazKozDev/VerbaBot"><img src="https://img.shields.io/librariesio/github/KazKozDev/VerbaBot" alt="Dependencies"></a>
  <a href="https://github.com/KazKozDev/VerbaBot/stargazers"><img src="https://img.shields.io/github/stars/KazKozDev/VerbaBot" alt="GitHub Stars"></a>
  <a href="https://github.com/KazKozDev/VerbaBot/issues"><img src="https://img.shields.io/github/issues/KazKozDev/VerbaBot" alt="GitHub Issues"></a>
  <a href="https://github.com/KazKozDev/VerbaBot/commits/main"><img src="https://img.shields.io/github/last-commit/KazKozDev/VerbaBot" alt="Last Commit"></a>
  <a href="https://github.com/KazKozDev/VerbaBot"><img src="https://img.shields.io/badge/Documentation-Complete-brightgreen" alt="Documentation"></a>
  <a href="https://github.com/sponsors/KazKozDev"><img src="https://img.shields.io/badge/Sponsor-Support%20this%20project-blue" alt="Sponsor"></a>
</p>

<p align="center">A powerful multimodal LLM assistant with advanced RAG.</p>
<p align="center">Local Control. Global Capabilities.</p>

VerbaBot is a comprehensive AI assistant designed to run locally, ensuring complete privacy of your data. By integrating powerful language models with a sophisticated Retrieval-Augmented Generation (RAG) pipeline, VerbaBot offers intelligent conversations, information retrieval, and personal knowledge management - all within a user-friendly web interface.

<div align="center">
  <img src="https://github.com/user-attachments/assets/e20930d6-f29f-47f7-a8a0-39fefbdb3538" alt="VerbaBot Demo">
</div>

### Features

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
