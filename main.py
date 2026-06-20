"""
main.py

FastAPI aplikasi utama untuk Customer Service Chatbot.
Mengintegrasikan LangChain + LangGraph + LangSmith.

Endpoints:
- GET  /                → Halaman sambutan
- GET  /health          → Health check (termasuk status LangSmith)
- POST /chat            → Endpoint utama chatbot
- GET  /graph/visualize → Visualisasi struktur LangGraph
- GET  /docs            → Swagger UI (otomatis dari FastAPI)
"""

import os
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Load environment variables
load_dotenv()

from app.schemas.models import ChatRequest, ChatResponse, HealthResponse
from app.graph.chatbot_graph import get_graph, build_customer_service_graph
from app.agents.langsmith_tracer import (
    is_langsmith_enabled,
    traced_process_message,
    get_langsmith_callbacks,
)


# ─────────────────────────────────────────────
# LIFESPAN (startup & shutdown)
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inisialisasi saat server startup"""
    print("🚀 Customer Service Chatbot starting...")
    print(f"   LangSmith tracing: {'✅ Aktif' if is_langsmith_enabled() else '❌ Nonaktif (isi LANGCHAIN_API_KEY di .env)'}")

    try:
        get_graph()
        print("   LangGraph: ✅ Graph berhasil dikompilasi")
    except Exception as e:
        print(f"   LangGraph: ⚠️  {e}")

    yield

    print("👋 Customer Service Chatbot shutting down...")


# ─────────────────────────────────────────────
# INISIALISASI FASTAPI
# ─────────────────────────────────────────────
app = FastAPI(
    title="Customer Service Chatbot API",
    description="""
## 🤖 Customer Service Chatbot

Sistem chatbot berbasis **NLP/LLM** yang dibangun dengan:

| Library | Fungsi |
|---------|--------|
| **LangChain** | Chains untuk intent classification, product inquiry, complaint handling |
| **LangGraph** | Mengatur alur kerja (workflow) chatbot dengan state machine |
| **LangSmith** | Monitoring, tracing, dan evaluasi performa chatbot |

### LLM
- **Groq** (llama-3.3-70b-versatile) — gratis & sangat cepat
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
import pathlib
frontend_path = pathlib.Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    """Sajikan frontend chat UI atau halaman sambutan API"""
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "message": "🤖 Customer Service Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "chat": "/chat",
        "tech_stack": {
            "framework": "FastAPI",
            "llm": "Groq (llama-3.3-70b-versatile)",
            "llm_chains": "LangChain",
            "workflow": "LangGraph",
            "monitoring": "LangSmith"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint — status aplikasi dan LangSmith."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        langsmith_enabled=is_langsmith_enabled(),
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
async def chat(request: ChatRequest):
    """
    ## 💬 Endpoint Utama Chatbot

    Mengirim pesan ke chatbot dan mendapatkan respons.

    ### Alur Kerja (LangGraph):
    1. **classify_intent** → Klasifikasi intent menggunakan LangChain
    2. **route_by_intent** → LangGraph conditional edge menentukan handler
    3. **handler node**    → LangChain chain yang sesuai menghasilkan respons
    4. **LangSmith**       → Semua langkah di-trace secara otomatis

    ### Contoh Pesan:
    - `"Berapa harga laptop TechPro X1?"` → product_inquiry
    - `"Laptop saya rusak"`               → complaint
    - `"Jam berapa toko kalian buka?"`    → general_info
    - `"Saya mau bicara dengan manusia"`  → escalate_human
    """
    start_time = time.time()

    try:
        graph = get_graph()

        initial_state = {
            "session_id": request.session_id,
            "user_message": request.message,
            "history": [
                {"role": msg.role.value, "content": msg.content}
                for msg in (request.history or [])
            ],
            "intent": "",
            "confidence": 0.0,
            "intent_reasoning": "",
            "response": "",
            "needs_human": False,
            "error": None,
        }

        callbacks = get_langsmith_callbacks(
            session_id=request.session_id,
            run_name=f"chat-{request.session_id}"
        )

        config = {}
        if callbacks:
            config["callbacks"] = callbacks

        # ── INVOKE LANGGRAPH ─────────────────────────
        final_state = graph.invoke(initial_state, config=config)

        # ── LANGSMITH @traceable ─────────────────────
        if is_langsmith_enabled():
            traced_process_message(
                session_id=request.session_id,
                message=request.message,
                intent=final_state.get("intent", "unknown"),
                response=final_state.get("response", ""),
                needs_human=final_state.get("needs_human", False),
                metadata={
                    "confidence": final_state.get("confidence", 0.0),
                    "latency_ms": round((time.time() - start_time) * 1000, 2),
                    "history_length": len(request.history or []),
                }
            )

        return ChatResponse(
            session_id=request.session_id,
            response=final_state.get("response", "Maaf, terjadi kesalahan."),
            intent=final_state.get("intent", "unknown"),
            confidence=final_state.get("confidence", 0.0),
            needs_human=final_state.get("needs_human", False),
            langsmith_run_url=(
                f"https://smith.langchain.com/projects/{os.getenv('LANGCHAIN_PROJECT', 'customer-service-chatbot')}"
                if is_langsmith_enabled() else None
            ),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan internal: {str(e)}"
        )


@app.get("/graph/visualize", tags=["Debug"])
async def visualize_graph():
    """Visualisasi struktur LangGraph dalam format ASCII + JSON."""
    return {
        "nodes": [
            {"id": "START",            "type": "entry"},
            {"id": "classify_intent",  "type": "process",  "chain": "IntentClassificationChain (LangChain)"},
            {"id": "product_inquiry",  "type": "handler",  "chain": "ProductInquiryChain (LangChain)"},
            {"id": "handle_complaint", "type": "handler",  "chain": "ComplaintChain (LangChain)"},
            {"id": "general_info",     "type": "handler",  "chain": "GeneralInfoChain (LangChain)"},
            {"id": "escalate",         "type": "handler",  "chain": "EscalationChain (LangChain)"},
            {"id": "END",              "type": "exit"},
        ],
        "ascii_diagram": """
    START
      │
      ▼
  [classify_intent] (LangChain: IntentClassificationChain + Groq)
      │
      ├── product_inquiry  ───► [product_inquiry]  ──► END
      ├── complaint/order  ───► [handle_complaint] ──► END
      ├── general_info     ───► [general_info]     ──► END
      └── escalate_human   ───► [escalate]         ──► END
        """
    }
