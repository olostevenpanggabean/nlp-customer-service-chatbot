"""
schemas/models.py
Pydantic schemas untuk request dan response API
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="ID unik untuk sesi percakapan")
    message: str = Field(..., description="Pesan dari pengguna")
    history: Optional[List[ChatMessage]] = Field(
        default=[], description="Riwayat percakapan sebelumnya"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user-123",
                "message": "Saya ingin tahu tentang produk laptop kalian",
                "history": []
            }
        }


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str = Field(..., description="Intent yang terdeteksi dari pesan pengguna")
    confidence: float = Field(..., description="Tingkat kepercayaan klasifikasi intent")
    needs_human: bool = Field(..., description="Apakah perlu dialihkan ke agen manusia")
    langsmith_run_url: Optional[str] = Field(
        default=None, description="URL trace di LangSmith"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user-123",
                "response": "Halo! Tentu saya bisa membantu...",
                "intent": "product_inquiry",
                "confidence": 0.95,
                "needs_human": False,
                "langsmith_run_url": "https://smith.langchain.com/..."
            }
        }


class HealthResponse(BaseModel):
    status: str
    version: str
    langsmith_enabled: bool


class IntentClassification(BaseModel):
    intent: str
    confidence: float
    reasoning: str
