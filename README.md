# YouTube Transcript Chatbot (RAG Pipeline)

A robust Retrieval-Augmented Generation (RAG) pipeline built with Python, LangChain, and Ollama. This project extracts transcripts from YouTube videos, processes them into vectorized chunks, and allows you to ask targeted conversational questions about the content of the video using a local LLM through Ollama.

## Features

- **Automated Transcript Retrieval**: Fetches transcripts directly via `youtube-transcript-api` using just a YouTube video ID.
- **Efficient Text Chunking**: Leverages `RecursiveCharacterTextSplitter` to optimally segment long transcripts while retaining meaningful context overlap.
- **Local Embedding & Vector Search**: Utilizes `OllamaEmbeddings` (nomic-embed-text) and `FAISS` to store and swiftly retrieve relevant document chunks.
- **Local LLM Integration**: Employs `ChatOllama` (gemma2:2b) to generate conversational and entirely localized responses based purely on the retrieved video context.
- **Privacy-First**: Thanks to Ollama, both embedding generation and LLM inference run entirely on your local machine.

## Prerequisites

Before running this project, ensure you have the following installed:

1. **Python 3.8+**
2. **Ollama**: You need [Ollama](https://ollama.com/) installed and running locally.
3. **Ollama Models**: Pull the required local models used in instructions:
   ```bash
   ollama run gemma2:2b
   ollama pull nomic-embed-text
   ```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *If `requirements.txt` is missing, you can install the necessary packages manually:*
   ```bash
   pip install langchain-core langchain-ollama langchain-community langchain-text-splitters youtube-transcript-api faiss-cpu python-dotenv
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory if any specific environment variables are needed by your local setup, though local Ollama usage generally does not require an API key.

## Usage

1. Open `main.py`.
2. Find the `video_id` variable and replace it with the YouTube video ID you want to query. 
   *(e.g., for `https://www.youtube.com/watch?v=Gfr50f6ZBvo`, the ID is `Gfr50f6ZBvo`)*.
3. Set your target question in the `question` variable:
   ```python
   question = "What are the main topics discussed in this video?"
   ```
4. Run the script:
   ```bash
   python main.py
   ```
5. The LLM will output a helpful answer directly sourced from the video's transcript!

## Project Structure (Pipeline Steps)

1. **Indexing (Document Ingestion)**: Downloads the YouTube video transcript text.
2. **Text Splitting**: Splits text into chunk sizes of 1000 characters with a 200-character overlap for context retention.
3. **Embeddings**: Represents text chunks densely using local `nomic-embed-text`.
4. **Vector Store**: Indexes embedded chunks using `FAISS` for rapid similarity search.
5. **Retrieval**: Fetches the top `k=4` chunks most similar to your query.
6. **Generation / RAG**: Formats retrieved chunks as raw text and pairs them with the system prompt to guide `gemma2:2b` to answer your query securely.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/Pradeep102005/YouTube-ChatBot-LangChain-/issues).

## License

This project is licensed under the MIT License.
