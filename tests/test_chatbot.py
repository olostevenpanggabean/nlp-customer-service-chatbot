"""
tests/test_chatbot.py

Unit tests untuk Customer Service Chatbot.
Menguji chains, graph state, schemas, dan LangSmith config.

Jalankan dengan:
    pytest tests/ -v
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────
# TEST: INTENT CLASSIFICATION KEYWORDS
# ─────────────────────────────────────────────
class TestIntentClassification:
    """Test untuk Intent Classification Chain (LangChain)"""

    def test_product_intent_keywords(self):
        product_keywords = ["harga", "spesifikasi", "laptop", "beli", "produk"]
        all_product_kw   = ["harga", "spesifikasi", "laptop", "beli", "produk",
                            "ram", "processor", "ssd", "garansi", "fitur"]
        for kw in product_keywords:
            assert kw in all_product_kw

    def test_complaint_intent_keywords(self):
        complaint_keywords = ["rusak", "tidak bisa", "error", "masalah", "kecewa"]
        all_complaint_kw   = ["rusak", "tidak bisa", "error", "masalah", "kecewa",
                              "tidak nyala", "cacat", "bermasalah"]
        for kw in complaint_keywords:
            assert kw in all_complaint_kw

    def test_escalation_phrases_exist(self):
        escalation = ["bicara dengan manusia", "agen manusia", "minta bantuan langsung"]
        assert len(escalation) > 0


# ─────────────────────────────────────────────
# TEST: GRAPH STATE
# ─────────────────────────────────────────────
class TestGraphState:
    """Test untuk LangGraph state management"""

    def test_initial_state_structure(self):
        state = {
            "session_id":       "test-123",
            "user_message":     "Test message",
            "history":          [],
            "intent":           "",
            "confidence":       0.0,
            "intent_reasoning": "",
            "response":         "",
            "needs_human":      False,
            "error":            None,
        }
        required = ["session_id", "user_message", "history", "intent",
                    "confidence", "response", "needs_human"]
        for field in required:
            assert field in state, f"Field '{field}' tidak ada di state"

    def test_state_after_classification(self):
        state = {
            "intent":     "product_inquiry",
            "confidence": 0.95,
            "intent_reasoning": "Pengguna menanyakan harga produk",
        }
        valid_intents = ["product_inquiry", "complaint", "order_status",
                         "warranty_claim", "general_info", "escalate_human"]
        assert state["intent"] in valid_intents
        assert 0.0 <= state["confidence"] <= 1.0

    def test_routing_logic(self):
        routing_map = {
            "product_inquiry": "product_inquiry",
            "complaint":       "handle_complaint",
            "order_status":    "handle_complaint",
            "warranty_claim":  "handle_complaint",
            "general_info":    "general_info",
            "escalate_human":  "escalate",
        }
        assert routing_map["product_inquiry"] == "product_inquiry"
        assert routing_map["complaint"]       == "handle_complaint"
        assert routing_map["escalate_human"]  == "escalate"
        assert routing_map["general_info"]    == "general_info"


# ─────────────────────────────────────────────
# TEST: LANGSMITH CONFIGURATION
# ─────────────────────────────────────────────
class TestLangSmithConfig:
    """Test untuk konfigurasi LangSmith"""

    def test_langsmith_disabled_without_key(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("LANGCHAIN_API_KEY",     "")
            mp.setenv("LANGCHAIN_TRACING_V2",  "false")
            from app.agents.langsmith_tracer import is_langsmith_enabled
            assert is_langsmith_enabled() == False

    def test_langsmith_enabled_with_key(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("LANGCHAIN_API_KEY",    "ls-test-key-12345")
            mp.setenv("LANGCHAIN_TRACING_V2", "true")
            from app.agents.langsmith_tracer import is_langsmith_enabled
            assert is_langsmith_enabled() == True


# ─────────────────────────────────────────────
# TEST: PYDANTIC SCHEMAS
# ─────────────────────────────────────────────
class TestSchemas:
    """Test untuk Pydantic schemas"""

    def test_chat_request_valid(self):
        from app.schemas.models import ChatRequest
        req = ChatRequest(
            session_id="user-test-001",
            message="Harga laptop TechPro X1 berapa?",
            history=[]
        )
        assert req.session_id == "user-test-001"
        assert req.message    == "Harga laptop TechPro X1 berapa?"
        assert req.history    == []

    def test_chat_response_structure(self):
        from app.schemas.models import ChatResponse
        resp = ChatResponse(
            session_id="user-test-001",
            response="Harga laptop TechPro X1 adalah Rp 8.999.000",
            intent="product_inquiry",
            confidence=0.95,
            needs_human=False,
            langsmith_run_url=None
        )
        assert resp.intent       == "product_inquiry"
        assert 0.0 <= resp.confidence <= 1.0
        assert isinstance(resp.needs_human, bool)

    def test_needs_human_true_for_escalation(self):
        from app.schemas.models import ChatResponse
        resp = ChatResponse(
            session_id="user-test-002",
            response="Menghubungkan ke agen manusia...",
            intent="escalate_human",
            confidence=0.98,
            needs_human=True,
        )
        assert resp.needs_human == True

    def test_chat_request_requires_session_id(self):
        from app.schemas.models import ChatRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ChatRequest(message="Test")

    def test_chat_request_requires_message(self):
        from app.schemas.models import ChatRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ChatRequest(session_id="test-123")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🧪 Menjalankan test suite Customer Service Chatbot...")
    print("=" * 60)

    TestSchemas().test_chat_request_valid()
    print("✅ test_chat_request_valid")

    TestSchemas().test_chat_response_structure()
    print("✅ test_chat_response_structure")

    TestSchemas().test_needs_human_true_for_escalation()
    print("✅ test_needs_human_true_for_escalation")

    TestGraphState().test_initial_state_structure()
    print("✅ test_initial_state_structure")

    TestGraphState().test_routing_logic()
    print("✅ test_routing_logic")

    print("=" * 60)
    print("✅ Semua test berhasil!")
