# YouTube Chatbot with LangChain

Chat with any YouTube video using its transcript. Built to escape tutorial hell — by actually building something.

---

## What it does

Paste a YouTube URL → ask questions about the video → get answers grounded in the transcript.

No hallucinations about content that isn't there. If it's not in the transcript, it says so.

---

## How it works

```
YouTube URL
    │
    ▼
YoutubeLoader          ← pulls transcript via youtube-transcript-api
    │
    ▼
RecursiveCharacterTextSplitter    ← chunks transcript into ~1000 token pieces
    │
    ▼
OpenAIEmbeddings       ← embeds each chunk
    │
    ▼
FAISS / Chroma         ← stores vectors locally
    │
    ▼
ConversationalRetrievalChain      ← retrieves relevant chunks + answers with GPT
    │
    ▼
Chat interface (CLI / Streamlit)
```

---

## Stack

| Layer | Library |
|---|---|
| Document loading | `langchain-community` — `YoutubeLoader` |
| Text splitting | `RecursiveCharacterTextSplitter` |
| Embeddings | `OpenAIEmbeddings` (or swap for `HuggingFaceEmbeddings`) |
| Vector store | `FAISS` (local, no server needed) |
| LLM | `ChatOpenAI` — GPT-4o-mini |
| Memory | `ConversationBufferMemory` |
| UI | Streamlit (optional) |

---

## Project structure

```
youtube-chatbot/
├── app.py                  # Streamlit UI
├── chatbot.py              # Core chain logic
├── loader.py               # YouTube loading + splitting
├── vectorstore.py          # Embed + store + retrieve
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

**1. Clone and install**

```bash
git clone https://github.com/your-username/youtube-chatbot.git
cd youtube-chatbot
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Set your API key**

```bash
cp .env.example .env
# Edit .env and add your OpenAI key
```

```env
OPENAI_API_KEY=sk-...
```

**3. Run**

```bash
# CLI mode
python chatbot.py

# Streamlit UI
streamlit run app.py
```

---

## Usage

```python
from loader import load_youtube_video
from vectorstore import build_vectorstore
from chatbot import build_chain

# Load and index a video
docs = load_youtube_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
vectorstore = build_vectorstore(docs)
chain = build_chain(vectorstore)

# Chat
response = chain.invoke({
    "question": "What is the main topic of this video?",
    "chat_history": []
})
print(response["answer"])
```

---

## Core code

**loader.py**

```python
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_youtube_video(url: str):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(docs)
```

**vectorstore.py**

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def build_vectorstore(docs):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)
```

**chatbot.py**

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

def build_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
    )
```

---

## Limitations

- Only works on videos that have a transcript (auto-generated or manual captions)
- Very long videos (3h+) may hit context limits — consider filtering chunks by timestamp
- Private or age-restricted videos will fail at the loader step
- FAISS index is in-memory; reloading requires re-embedding

---

## What I learned building this

- **Document loaders are step 1 of everything** — garbage in, garbage out. `YoutubeLoader` metadata (`source`, `title`, `author`) flows all the way to the final answer source citations.
- **`lazy_load()` matters for long videos** — buffering a 2h transcript into memory at once is wasteful.
- **Chunk overlap is not optional** — sentences at chunk boundaries get split mid-thought. 200 token overlap rescued a lot of bad retrievals.
- **Memory key naming is a footgun** — `ConversationBufferMemory` and `ConversationalRetrievalChain` must agree on `memory_key` and `output_key` or the chain silently breaks.
- **`k=4` in the retriever is a reasonable default** — more chunks = more noise, fewer = missed context.

---

## Possible next steps

- [ ] Persist FAISS index to disk so re-embedding is skipped on reload
- [ ] Swap `FAISS` for `Chroma` with a local server for multi-video search
- [ ] Add timestamp metadata to chunks and surface them in answers
- [ ] Support playlists via `YoutubeLoader` batch loading
- [ ] Replace `OpenAIEmbeddings` with `nomic-embed-text` for a free local option

---

## Requirements

```
langchain>=0.3.0
langchain-community>=0.3.0
langchain-openai>=0.2.0
faiss-cpu>=1.8.0
youtube-transcript-api>=0.6.0
pytube>=15.0.0
streamlit>=1.35.0
python-dotenv>=1.0.0
```

---

## License

MIT