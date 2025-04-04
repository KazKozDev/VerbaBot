<div align="center"><img src="https://github.com/user-attachments/assets/63bc3058-bace-4234-a473-7d92e6bd1c84" alt="logo"></div>
<p align="center">VerbaBot</p>
<p align="center">Multimodal LLM assistant with chat, voice, and RAG.</p>
<p align="center">Local Control. Global Capabilities.</p>

VerbaBot is designed for those who need a personal digital assistant running locally. It ensures complete privacy of your data, as all processing takes place on the device. By integrating different language models and the RAG pipeline into a single interface, the assistant offers intelligent conversations and analytical features. Additionally, VerbaBot includes network access capabilities, allowing it to retrieve real-time information, perform web searche.

<p align="center">
  <img src="https://github.com/user-attachments/assets/af81aa6b-66b4-42f3-9fc8-73fbc36205ea" alt="demo">
</p>

## Features

- Web interface for interacting with the chat bot
- Natural language Calendar management
- RAG (Retrieval Augmented Generation) support for document processing
- Customizable system prompt
- Document upload for contextual responses
- Flexible model and parameter settings

## Installation

```bash
# Clone the repository
git clone https://github.com/KazKozDev/VerbaBot.git
cd VerbaBot

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python server.py
```

The server will start at http://127.0.0.1:5001.

## Project Structure

- `server.py` - Main server file 
- `chat.html` - Web interface
- `voice_input.py` - Voice input module
- `requirements.txt` - Required dependencies

## Configuration

To enable voice input capability, uncomment the corresponding dependencies in the `requirements.txt` file and install them.

## License

[MIT](LICENSE) 
