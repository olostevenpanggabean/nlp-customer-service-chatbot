"""
graph/chatbot_graph.py

Implementasi LangGraph untuk mengatur alur kerja chatbot:

ALUR KERJA:
    START
      │
      ▼
  [classify_intent]  ← Klasifikasi intent dengan LangChain
      │
      ▼
  [route_intent]  ← Routing berdasarkan intent (conditional edges)
      │
   ┌──┴──────────────┬───────────────┬──────────────┐
   ▼                 ▼               ▼              ▼
[product_inquiry] [handle_complaint] [general_info] [escalate]
   └──────────────────┴───────────────┴──────────────┘
                              │
                              ▼
                            END
"""

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END, START

from app.chains.customer_service_chains import (
    create_intent_classification_chain,
    create_product_inquiry_chain,
    create_complaint_chain,
    create_general_info_chain,
    create_escalation_chain,
    format_history_to_messages,
)


# ─────────────────────────────────────────────
# STATE DEFINITION (LangGraph)
# ─────────────────────────────────────────────
class CustomerServiceState(TypedDict):
    """
    State yang mengalir melalui setiap node di LangGraph.
    Setiap field menyimpan informasi yang dibutuhkan antar node.
    """
    session_id: str
    user_message: str
    history: List[dict]
    intent: str
    confidence: float
    intent_reasoning: str
    response: str
    needs_human: bool
    error: Optional[str]


# ─────────────────────────────────────────────
# NODE 1: CLASSIFY INTENT
# ─────────────────────────────────────────────
def classify_intent_node(state: CustomerServiceState) -> CustomerServiceState:
    """
    Node LangGraph: Mengklasifikasikan intent pesan pengguna
    menggunakan Intent Classification Chain dari LangChain.
    """
    try:
        classify = create_intent_classification_chain()
        result = classify(state["user_message"])

        return {
            **state,
            "intent": result.get("intent", "general_info"),
            "confidence": result.get("confidence", 0.5),
            "intent_reasoning": result.get("reasoning", ""),
            "error": None,
        }
    except Exception as e:
        return {
            **state,
            "intent": "general_info",
            "confidence": 0.0,
            "intent_reasoning": "Error dalam klasifikasi",
            "error": str(e),
        }


# ─────────────────────────────────────────────
# NODE 2: HANDLE PRODUCT INQUIRY
# ─────────────────────────────────────────────
def product_inquiry_node(state: CustomerServiceState) -> CustomerServiceState:
    """
    Node LangGraph: Menangani pertanyaan seputar produk
    menggunakan Product Inquiry Chain dari LangChain.
    """
    try:
        chain = create_product_inquiry_chain()
        history_messages = format_history_to_messages(state["history"])

        response = chain.invoke({
            "message": state["user_message"],
            "history": history_messages
        })

        return {
            **state,
            "response": response,
            "needs_human": False,
        }
    except Exception as e:
        return {
            **state,
            "response": "Maaf, terjadi kesalahan. Silakan coba lagi.",
            "needs_human": True,
            "error": str(e),
        }


# ─────────────────────────────────────────────
# NODE 3: HANDLE COMPLAINT
# ─────────────────────────────────────────────
def complaint_node(state: CustomerServiceState) -> CustomerServiceState:
    """
    Node LangGraph: Menangani keluhan pelanggan
    menggunakan Complaint Handling Chain dari LangChain.
    """
    try:
        chain = create_complaint_chain()
        history_messages = format_history_to_messages(state["history"])

        response = chain.invoke({
            "message": state["user_message"],
            "history": history_messages
        })

        # Keluhan dengan confidence tinggi → butuh perhatian manusia
        needs_human = state.get("confidence", 0) > 0.85

        return {
            **state,
            "response": response,
            "needs_human": needs_human,
        }
    except Exception as e:
        return {
            **state,
            "response": "Maaf atas ketidaknyamanannya. Tim kami akan segera menghubungi Anda.",
            "needs_human": True,
            "error": str(e),
        }


# ─────────────────────────────────────────────
# NODE 4: GENERAL INFO
# ─────────────────────────────────────────────
def general_info_node(state: CustomerServiceState) -> CustomerServiceState:
    """
    Node LangGraph: Memberikan informasi umum
    menggunakan General Info Chain dari LangChain.
    """
    try:
        chain = create_general_info_chain()
        history_messages = format_history_to_messages(state["history"])

        response = chain.invoke({
            "message": state["user_message"],
            "history": history_messages
        })

        return {
            **state,
            "response": response,
            "needs_human": False,
        }
    except Exception as e:
        return {
            **state,
            "response": "Maaf, terjadi kesalahan sistem.",
            "needs_human": True,
            "error": str(e),
        }


# ─────────────────────────────────────────────
# NODE 5: ESCALATE TO HUMAN
# ─────────────────────────────────────────────
def escalation_node(state: CustomerServiceState) -> CustomerServiceState:
    """
    Node LangGraph: Eskalasi ke agen manusia
    menggunakan Escalation Chain dari LangChain.
    """
    try:
        chain = create_escalation_chain()
        response = chain.invoke({"message": state["user_message"]})

        return {
            **state,
            "response": response,
            "needs_human": True,
        }
    except Exception as e:
        return {
            **state,
            "response": (
                "Permintaan Anda sedang kami proses. "
                "Tim kami akan menghubungi Anda segera. "
                "Hubungi kami di 0800-123-456."
            ),
            "needs_human": True,
            "error": str(e),
        }


# ─────────────────────────────────────────────
# CONDITIONAL EDGE: ROUTE BERDASARKAN INTENT
# ─────────────────────────────────────────────
def route_by_intent(state: CustomerServiceState) -> str:
    """
    LangGraph Conditional Edge:
    Menentukan node selanjutnya berdasarkan intent yang terklasifikasi.

    Returns nama node tujuan sebagai string.
    """
    intent = state.get("intent", "general_info")

    routing_map = {
        "product_inquiry":  "product_inquiry",
        "complaint":        "handle_complaint",
        "order_status":     "handle_complaint",   # order status mirip complaint
        "warranty_claim":   "handle_complaint",   # warranty mirip complaint
        "general_info":     "general_info",
        "escalate_human":   "escalate",
    }

    return routing_map.get(intent, "general_info")


# ─────────────────────────────────────────────
# BUILD LANGGRAPH
# ─────────────────────────────────────────────
def build_customer_service_graph():
    """
    Membangun dan mengompilasi LangGraph untuk customer service chatbot.

    Graph ini mendefinisikan alur kerja lengkap dari:
    - Klasifikasi intent
    - Routing conditional
    - Penanganan berbeda per intent
    """
    # Inisialisasi StateGraph dengan state schema
    graph = StateGraph(CustomerServiceState)

    # ── Tambahkan Nodes ──────────────────────────────
    graph.add_node("classify_intent",  classify_intent_node)
    graph.add_node("product_inquiry",  product_inquiry_node)
    graph.add_node("handle_complaint", complaint_node)
    graph.add_node("general_info",     general_info_node)
    graph.add_node("escalate",         escalation_node)

    # ── Definisikan Edges ────────────────────────────
    # Entry point: START → classify_intent
    graph.add_edge(START, "classify_intent")

    # Conditional edge: classify_intent → (routing berdasarkan intent)
    graph.add_conditional_edges(
        source="classify_intent",
        path=route_by_intent,
        path_map={
            "product_inquiry":  "product_inquiry",
            "handle_complaint": "handle_complaint",
            "general_info":     "general_info",
            "escalate":         "escalate",
        }
    )

    # Semua handler nodes → END
    graph.add_edge("product_inquiry",  END)
    graph.add_edge("handle_complaint", END)
    graph.add_edge("general_info",     END)
    graph.add_edge("escalate",         END)

    # Kompilasi graph
    compiled_graph = graph.compile()
    return compiled_graph


# Singleton graph instance
_graph_instance = None

def get_graph():
    """Get atau buat instance graph (singleton pattern)"""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_customer_service_graph()
    return _graph_instance
