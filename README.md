# 📚 RAG Chatbot — Document Q&A with Claude & ChromaDB

A production-style **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload a PDF and have a grounded, source-cited conversation with it. Built with **FastAPI**, **LangChain**, **ChromaDB**, **Voyage AI embeddings**, and **Anthropic's Claude** as the LLM, with a **Streamlit** chat UI on top.

> Ask questions about any PDF and get answers that are strictly grounded in the document — no hallucinated context, with page-level source attribution and full conversational memory.

---

## ✨ Features

- **PDF ingestion pipeline** — load → chunk → embed → persist, with per-document metadata (`doc_id`, page numbers, chunk index)
- **Semantic retrieval** via ChromaDB + Voyage AI embeddings, scoped per document
- **Conversational memory** — a question-condensing chain rewrites follow-up questions into standalone queries using chat history, so "what about page 3?" actually resolves correctly
- **Strictly grounded answers** — the LLM is prompted to answer only from retrieved context and explicitly say when it doesn't know
- **Source citations** — every answer returns the originating document, page, and chunk
- **Session-based history** with automatic windowing/trimming to control context size
- **Streamlit chat UI** — upload, process, and chat with a document in a clean full-screen interface
- **Stateless, swappable vector store layer** — CRUD-style module wrapping ChromaDB (add, search, delete-by-doc, stats)

---

## 🏗️ Architecture

```
                         ┌─────────────────────┐
                         │   Streamlit UI       │
                         │  (app.py / frontend) │
                         └──────────┬───────────┘
                                    │ REST (requests)
                                    ▼
                         ┌─────────────────────┐
                         │     FastAPI API      │
                         │       (main.py)      │
                         └──────────┬───────────┘
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
        ┌────────────────┐ ┌───────────────┐ ┌──────────────────┐
        │  ingestion.py   │ │   chat.py     │ │    history.py     │
        │ PDF → chunks    │ │ Condense Q →  │ │ Per-session       │
        │ (PyPDFLoader +  │ │ Retrieve →    │ │ windowed message   │
        │  text splitter) │ │ Answer (Claude)│ │ store             │
        └────────┬────────┘ └───────┬───────┘ └──────────────────┘
                  │                  │
                  ▼                  ▼
           ┌─────────────────────────────┐
           │      vector_store.py         │
           │ ChromaDB + Voyage AI embeds   │
           └───────────────────────────────┘
```

**Chat flow:**
1. User's question + chat history → **condense chain** rewrites it into a standalone question
2. Standalone question → **similarity search** against the document's chunks (scoped by `doc_id`)
3. Retrieved chunks + history + question → **QA prompt** → Claude generates a grounded answer
4. Answer, message history, and source metadata are persisted / returned

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude (via `langchain-anthropic`) |
| Embeddings | Voyage AI (via `langchain-voyageai`) |
| Vector Store | ChromaDB (via `langchain-chroma`) |
| Orchestration | LangChain / LangGraph |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| PDF Parsing | `pypdf` / `PyPDFLoader` |
| Validation | Pydantic v2 |

---

## 📁 Project Structure

```
.
├── main.py             # FastAPI app entrypoint / route definitions
├── config.py            # Settings (env-driven, via pydantic-settings)
├── ingestion.py         # PDF loading, metadata enrichment, chunking
├── vector_store.py      # ChromaDB CRUD: add / search / delete / stats
├── chat.py               # Condense-question + retrieval-augmented QA chain
├── history.py            # Per-session windowed chat history
├── schemas.py            # Pydantic request/response models
├── utils.py               # IDs, sanitization, validation, source formatting, timing
├── app.py / frontend.py  # Streamlit chat UI
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone & install

```bash
git clone https://github.com/anshul-20/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key
VOYAGE_API_KEY=your_voyage_api_key

LLM_MODEL=claude-sonnet-4-5
LLM_TEMPERATURE=0.2
MAX_TOKENS=1024

EMBEDDING_MODEL=voyage-3
CHROMA_COLLECTION_NAME=rag_chatbot
CHROMA_PERSIST_DIR=./chroma_db

CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TOP_K_RETRIEVAL=4
HISTORY_WINDOW_SIZE=5

API_HOST=0.0.0.0
API_PORT=8000
```

> Adjust the model names, chunking, and retrieval parameters in `config.py` to match your actual settings — the values above are sensible defaults.

### 3. Run the backend

```bash
python main.py
# or
uvicorn main:app --reload --port 8000
```

### 4. Run the frontend

```bash
streamlit run app.py
```

The UI will open at `http://localhost:8501`, talking to the API at `http://localhost:8000`.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/session/new` | Create a new chat session, returns `session_id` |
| `POST` | `/ingest` | Upload and process a PDF (multipart `file`) |
| `POST` | `/chat` | Ask a question (`session_id`, `question`, optional `doc_id`) |
| `POST` | `/session/clear` | Clear a session's chat history |
| `DELETE` | `/document/{doc_id}` | Delete all chunks for a document from the vector store |
| `GET` | `/health` | Health check + collection stats |
| `GET` | `/sessions` | List active session IDs |

**Example — chat request**
```json
POST /chat
{
  "session_id": "b3f1c9...",
  "question": "What are the key findings in section 3?",
  "doc_id": "report_a1b2c3d4"
}
```

**Example — chat response**
```json
{
  "session_id": "b3f1c9...",
  "answer": "The key findings are ...",
  "sources": [
    { "source": "report.pdf", "page": 4, "chunk": 12 }
  ]
}
```

---

## 🧩 Design Notes

- **Grounded-only answers**: the QA prompt explicitly instructs the model to decline when the context doesn't contain the answer, reducing hallucination risk.
- **Per-document scoping**: retrieval can be filtered by `doc_id`, so multiple documents can share one collection without cross-contamination.
- **Windowed memory**: chat history is capped at a configurable number of turns to keep prompts small and costs predictable.
- **Stateless vector layer**: `vector_store.py` is a thin, swappable interface — the rest of the app doesn't know it's ChromaDB specifically.

---

## 🗺️ Possible Extensions

- [ ] Multi-file upload and cross-document Q&A
- [ ] Streaming responses (SSE) in the Streamlit UI
- [ ] Persistent session store (Redis/Postgres) instead of in-memory
- [ ] Support for other file types (docx, txt, HTML)
- [ ] Re-ranking retrieved chunks before generation

---

## 👤 Author

**Anshul Pandey** — Data Scientist & AI Engineer, specializing in Generative AI, RAG pipelines, and scalable LLM systems.

- 🌐 Portfolio: [anshulpandey.online](https://www.anshulpandey.online/)
- 💼 LinkedIn: [linkedin.com/in/anshulpandeyyy](https://linkedin.com/in/anshulpandeyyy)
- 🐙 GitHub: [github.com/anshul-20](https://github.com/anshul-20)
- ✉️ Email: [anshulpandey0077@gmail.com](mailto:anshulpandey0077@gmail.com)

---

## 📄 License

This project is licensed under the MIT License — feel free to use and adapt it.
