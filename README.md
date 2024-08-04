# AI Chatbot Application

This project is an AI-powered chat application with features like file upload, web search, and news retrieval. It uses the Ollama API to interact with local language models.

## Features

- Chat with an AI assistant using Ollama models
- File upload and processing (including images, PDFs, and more)
- DuckDuckGo search integration
- Google News retrieval
- Chat history management
- Customizable settings

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have installed [Ollama](https://ollama.ai/)
* You have pulled the following Ollama models:
  - mistral-nemo:latest (Used as the default language model)
  - llava:13b (Used for image recognition tasks)

To pull these models, run the following commands after installing Ollama:

```
ollama pull mistral-nemo:latest
ollama pull llava:13b
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ai-assistant-chat.git
   cd ai-assistant-chat
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure Ollama is running on your system. By default, it should be accessible at `http://localhost:11434`.

## Usage

1. Start the Ollama service if it's not already running.

2. Run the server:
   ```
   python SERVER.py
   ```

3. Open `chat.html` in your web browser to start chatting with the AI assistant.

## Configuration

You can modify the `SERVER.py` file to change default settings such as the LLM model, image recognition model, and system prompt. The default models are:

- LLM Model: "mistral-nemo:latest"
- Image Recognition Model: "llava:13b"

You can change these in the `SERVER.py` file or through the settings interface in the application.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.