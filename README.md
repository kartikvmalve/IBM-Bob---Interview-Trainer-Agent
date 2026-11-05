# IBM Interview Trainer Agent
# Problem Statement No. 22

> **AI-powered interview preparation agent** using **IBM Granite** (watsonx.ai) + **RAG** (ChromaDB vector store).  
> Generates tailored question sets, model answers, 30-day preparation strategies, and evaluates candidate responses.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│               IBM Interview Trainer Agent                │
│                                                         │
│  ┌──────────────┐    ┌──────────────────────────────┐  │
│  │  FastAPI UI  │───▶│     Agent Orchestrator       │  │
│  │  (Browser)   │    │        (app/agent.py)        │  │
│  └──────────────┘    └──────────┬───────────────────┘  │
│                                 │                        │
│               ┌─────────────────┴──────────────────┐    │
│               ▼                                    ▼    │
│  ┌────────────────────────┐  ┌────────────────────────┐ │
│  │   RAG Pipeline         │  │  IBM Granite Client    │ │
│  │   (ChromaDB + SBERT)   │  │  (watsonx.ai REST API) │ │
│  │   app/rag_pipeline.py  │  │  app/granite_client.py │ │
│  └────────────────────────┘  └────────────────────────┘ │
│          ▲ retrieve                  ▲ generate          │
│          │                          │                    │
│  ┌───────────────┐        ┌─────────────────────────┐   │
│  │ Vector Store  │        │  IBM watsonx.ai          │   │
│  │ (ChromaDB)    │        │  ibm/granite-13b-chat-v2 │   │
│  │ ./data/chroma │        │  (IBM Granite — mandatory)│  │
│  └───────────────┘        └─────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Features

| Feature | Description |
|---|---|
| **Question Generator** | 3–20 tailored interview questions with model answers & coaching tips |
| **30-Day Prep Strategy** | Weekly plans, resources, behavioral story templates, day-of checklist |
| **Answer Evaluator** | Scores answers 1–10, identifies gaps, rewrites to 9/10 standard |
| **Resume Ingestion** | Upload PDF/TXT resume → indexed into vector store for personalized prep |
| **RAG Pipeline** | ChromaDB + Sentence-BERT embeddings retrieve role-specific context |
| **IBM Granite LLM** | IBM `granite-13b-chat-v2` via watsonx.ai REST API (mandatory) |

## IBM Cloud Services Used

- **IBM watsonx.ai** — hosts IBM Granite foundation models
- **IBM Granite** (`ibm/granite-13b-chat-v2`) — primary LLM for generation (**mandatory**)
- **IBM Watson Discovery** (optional) — for enterprise-scale document retrieval
- **IBM Cloud Object Storage** (optional) — for resume/document persistence

## Quick Start

### 1. Prerequisites

- Python 3.10+
- IBM Cloud account (free Lite tier works)
- watsonx.ai project (free)

### 2. Get IBM Credentials

1. Go to [cloud.ibm.com](https://cloud.ibm.com)
2. Create a **watsonx.ai** instance (Lite tier is free)
3. Create a **Project** in watsonx.ai
4. Generate an **API Key**: Manage → Access (IAM) → API Keys
5. Copy your **Project ID** from the project settings

### 3. Configure Environment

```bash
# Copy the example env file
cp env.example .env

# Edit .env with your credentials
notepad .env   # Windows
```

Fill in these required values in `.env`:
```
IBM_WATSONX_API_KEY=your_actual_api_key
IBM_WATSONX_PROJECT_ID=your_actual_project_id
```

### 4. Install & Run

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py
```

Open your browser at **http://localhost:8000**

### 5. API Docs

Swagger UI: **http://localhost:8000/docs**  
ReDoc: **http://localhost:8000/redoc**

## API Reference

### `POST /api/questions`
Generate personalized interview questions with model answers.

```json
{
  "name": "Priya Sharma",
  "role": "Software Engineer",
  "experience_level": "mid",
  "resume_text": "5 years Python, AWS, microservices...",
  "num_questions": 10
}
```

### `POST /api/strategy`
Generate a 30-day preparation strategy.

```json
{
  "name": "Arjun Mehta",
  "role": "Data Scientist",
  "experience_level": "senior",
  "resume_text": ""
}
```

### `POST /api/evaluate`
Score and improve a candidate's answer.

```json
{
  "question": "Explain the CAP theorem.",
  "user_answer": "CAP means you can't have all three...",
  "role": "Software Engineer",
  "experience_level": "mid"
}
```

### `POST /api/upload-resume`
Upload a PDF or TXT resume (multipart/form-data).

```
file: <resume.pdf>
role: software_engineer
experience_level: mid
```

## Project Structure

```
IBM BOB Project/
├── app/
│   ├── __init__.py
│   ├── config.py          # Pydantic settings / env loading
│   ├── granite_client.py  # IBM Granite / watsonx.ai HTTP client
│   ├── rag_pipeline.py    # ChromaDB vector store + RAG retrieval
│   ├── agent.py           # Core agent: orchestrates RAG + LLM
│   └── main.py            # FastAPI app + endpoints
├── static/
│   └── index.html         # Full-stack UI (no framework needed)
├── tests/
│   └── test_agent.py      # Pytest test suite
├── data/                  # ChromaDB vector store (auto-created)
├── .env                   # Your credentials (git-ignored)
├── env.example            # Template for .env
├── requirements.txt
├── run.py                 # Application entry point
└── README.md
```

## Experience Levels

| Level | Years | Use |
|---|---|---|
| `fresher` | 0–1 yr | Entry level, campus hiring |
| `junior` | 1–3 yrs | Early career |
| `mid` | 3–6 yrs | Standard industry hire |
| `senior` | 6–10 yrs | Senior IC, tech lead |
| `lead` | 10+ yrs | Principal, architect, director |

## Supported Roles

Software Engineer · Data Scientist · Product Manager · DevOps Engineer · Cloud Engineer · ML Engineer · SRE · and any custom role.

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

## License

IBM Internal — Problem Statement No. 22 — Interview Trainer Agent
