"""
agents/langsmith_tracer.py

Konfigurasi dan utilities untuk LangSmith tracing.

LangSmith digunakan untuk:
- Monitoring setiap run chatbot secara real-time
- Debugging chain dan graph execution
- Evaluasi kualitas respons
- Analisis performa dan latensi
"""

import os
from typing import Optional, Dict, Any
from langsmith import Client
from langsmith.run_helpers import traceable
from langchain_core.tracers import LangChainTracer


# ─────────────────────────────────────────────
# LANGSMITH CLIENT SETUP
# ─────────────────────────────────────────────
def get_langsmith_client() -> Optional[Client]:
    """
    Inisialisasi LangSmith client.
    Mengembalikan None jika LangSmith tidak dikonfigurasi.
    """
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        return None

    try:
        client = Client(
            api_url=os.getenv(
                "LANGCHAIN_ENDPOINT",
                "https://api.smith.langchain.com"
            ),
            api_key=api_key
        )
        return client
    except Exception:
        return None


def is_langsmith_enabled() -> bool:
    """Cek apakah LangSmith tracing aktif"""
    return (
        os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        and bool(os.getenv("LANGCHAIN_API_KEY"))
    )


# ─────────────────────────────────────────────
# CALLBACK MANAGER UNTUK LANGSMITH
# ─────────────────────────────────────────────
def get_langsmith_callbacks(
    session_id: str,
    run_name: str = "customer-service-chat"
) -> Optional[list]:
    """
    Buat LangSmith callback untuk tracking satu sesi percakapan.
    Digunakan sebagai config pada pemanggilan LangChain chains.
    """
    if not is_langsmith_enabled():
        return None

    project = os.getenv("LANGCHAIN_PROJECT", "customer-service-chatbot")

    tracer = LangChainTracer(project_name=project)
    return [tracer]


# ─────────────────────────────────────────────
# DECORATOR @traceable (LangSmith)
# ─────────────────────────────────────────────
@traceable(name="process_customer_message", run_type="chain")
def traced_process_message(
    session_id: str,
    message: str,
    intent: str,
    response: str,
    needs_human: bool,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Fungsi yang di-trace oleh LangSmith menggunakan @traceable decorator.

    Setiap pemanggilan fungsi ini akan tercatat di LangSmith dashboard
    dengan metadata lengkap untuk monitoring dan debugging.
    """
    return {
        "session_id": session_id,
        "message": message,
        "intent": intent,
        "response": response,
        "needs_human": needs_human,
        "metadata": metadata or {}
    }


# ─────────────────────────────────────────────
# LANGSMITH DATASET & EVALUATION (opsional)
# ─────────────────────────────────────────────
def create_evaluation_dataset(dataset_name: str = "cs-chatbot-eval"):
    """
    Membuat dataset di LangSmith untuk evaluasi chatbot.
    Dataset berisi contoh input-output yang diharapkan.
    """
    client = get_langsmith_client()
    if not client:
        return None

    examples = [
        {
            "inputs": {"message": "Berapa harga laptop TechPro X1?"},
            "outputs": {"expected_intent": "product_inquiry"}
        },
        {
            "inputs": {"message": "Laptop saya rusak, layarnya tidak menyala"},
            "outputs": {"expected_intent": "complaint"}
        },
        {
            "inputs": {"message": "Jam berapa toko kalian buka?"},
            "outputs": {"expected_intent": "general_info"}
        },
        {
            "inputs": {"message": "Saya mau bicara dengan manusia"},
            "outputs": {"expected_intent": "escalate_human"}
        },
    ]

    try:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Dataset evaluasi Customer Service Chatbot"
        )
        for example in examples:
            client.create_example(
                inputs=example["inputs"],
                outputs=example["outputs"],
                dataset_id=dataset.id
            )
        return dataset
    except Exception as e:
        print(f"Dataset sudah ada atau error: {e}")
        return None
