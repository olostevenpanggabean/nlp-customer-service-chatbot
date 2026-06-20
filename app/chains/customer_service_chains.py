"""
chains/customer_service_chains.py

Implementasi LangChain chains untuk:
1. Intent Classification Chain
2. Product Inquiry Chain
3. Complaint Handling Chain
4. General Response Chain
5. Escalation Chain

Menggunakan Groq (llama-3.3-70b-versatile) sebagai LLM.
"""

import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List


# ─────────────────────────────────────────────
# INISIALISASI LLM (LangChain + Groq)
# ─────────────────────────────────────────────
def get_llm(temperature: float = 0.3):
    """Inisialisasi LLM menggunakan LangChain ChatGroq"""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


# ─────────────────────────────────────────────
# KNOWLEDGE BASE PRODUK
# ─────────────────────────────────────────────
PRODUCT_KNOWLEDGE = """
=== PRODUK KAMI ===

**Laptop TechPro X1**
- Harga: Rp 8.999.000
- Spesifikasi: Intel Core i5-1235U, RAM 16GB DDR4, SSD 512GB NVMe, Layar 14" FHD IPS
- Garansi: 2 tahun resmi

**Laptop TechPro X2 (Budget)**
- Harga: Rp 5.499.000
- Spesifikasi: Intel Core i3-1215U, RAM 8GB DDR4, SSD 256GB, Layar 14" HD
- Garansi: 1 tahun resmi

**Laptop TechPro Pro**
- Harga: Rp 18.999.000
- Spesifikasi: Intel Core i9-13900H, RAM 32GB DDR5, SSD 2TB NVMe, RTX 4060, Layar 16" 2K 165Hz
- Garansi: 3 tahun resmi + on-site support

=== KEBIJAKAN ===
- Garansi: Klaim melalui service center atau hubungi 0800-123-456
- Pengembalian: Produk dapat dikembalikan dalam 14 hari
- Pengiriman: Gratis untuk pembelian di atas Rp 5.000.000
- Pembayaran: Transfer bank, kartu kredit, cicilan 0%

=== JAM OPERASIONAL ===
- Senin-Jumat: 08:00-17:00 WIB
- Sabtu: 09:00-14:00 WIB
- Minggu: Tutup

=== KONTAK ===
- Telepon: 0800-123-456
- WhatsApp: 0812-3456-7890
- Email: support@techpro.id
- Alamat: Jl. Sudirman No. 88, Pekanbaru, Riau
"""


# ─────────────────────────────────────────────
# CHAIN 1: INTENT CLASSIFICATION CHAIN
# ─────────────────────────────────────────────
class IntentOutput(BaseModel):
    intent: str = Field(description="Intent dari pesan pengguna")
    confidence: float = Field(description="Tingkat kepercayaan 0.0-1.0")
    reasoning: str = Field(description="Alasan klasifikasi")


def create_intent_classification_chain():
    """
    LangChain Chain untuk mengklasifikasikan intent pesan pengguna.

    Intent yang tersedia:
    - product_inquiry : Tanya tentang produk, harga, spesifikasi
    - complaint       : Keluhan atau masalah pada produk
    - order_status    : Status pemesanan, pengiriman, tracking
    - warranty_claim  : Klaim garansi, perbaikan
    - general_info    : Informasi umum, jam buka, kebijakan
    - escalate_human  : Perlu agen manusia
    """
    llm = get_llm(temperature=0.1)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Kamu adalah sistem klasifikasi intent untuk customer service.
Klasifikasikan pesan pengguna ke salah satu intent berikut:
- product_inquiry: Pertanyaan tentang produk, harga, spesifikasi
- complaint: Keluhan, produk rusak, tidak puas
- order_status: Status pesanan, pengiriman, tracking
- warranty_claim: Klaim garansi, perbaikan
- general_info: Informasi umum, jam buka, kebijakan
- escalate_human: Meminta bicara dengan manusia, masalah kompleks

Respond HANYA dalam format JSON berikut (tanpa markdown/backtick):
{{
  "intent": "nama_intent",
  "confidence": 0.95,
  "reasoning": "alasan singkat"
}}"""),
        ("human", "{message}")
    ])

    # LCEL Chain dengan pipe operator (LangChain Expression Language)
    chain = prompt | llm | StrOutputParser()

    # Wrapper untuk parse JSON output
    def parse_intent(message: str) -> dict:
        raw = chain.invoke({"message": message})
        try:
            # Bersihkan jika ada markdown fence
            clean = raw.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(clean)
        except Exception:
            return {"intent": "general_info", "confidence": 0.5, "reasoning": "parse error"}

    return parse_intent


# ─────────────────────────────────────────────
# CHAIN 2: PRODUCT INQUIRY CHAIN
# ─────────────────────────────────────────────
def create_product_inquiry_chain():
    """LangChain Chain untuk menjawab pertanyaan produk"""
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Kamu adalah asisten customer service profesional untuk toko laptop TechPro.
Gunakan informasi produk berikut untuk menjawab pertanyaan pelanggan:

{PRODUCT_KNOWLEDGE}

Panduan menjawab:
- Gunakan bahasa yang ramah dan profesional
- Berikan informasi yang akurat berdasarkan knowledge base
- Jika tidak ada informasi, sarankan menghubungi tim kami
- Selalu tawarkan bantuan lebih lanjut di akhir respons
"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}")
    ])

    # LangChain LCEL pipe chain
    chain = prompt | llm | StrOutputParser()
    return chain


# ─────────────────────────────────────────────
# CHAIN 3: COMPLAINT HANDLING CHAIN
# ─────────────────────────────────────────────
def create_complaint_chain():
    """LangChain Chain untuk menangani keluhan pelanggan"""
    llm = get_llm(temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Kamu adalah spesialis penanganan keluhan customer service TechPro.

Prinsip penanganan keluhan:
1. Tunjukkan empati dan minta maaf atas ketidaknyamanan
2. Dengarkan dan pahami keluhan dengan baik
3. Berikan solusi yang konkret berdasarkan kebijakan kami
4. Eskalasi ke tim teknis jika diperlukan

Informasi yang tersedia:
{PRODUCT_KNOWLEDGE}
"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}")
    ])

    chain = prompt | llm | StrOutputParser()
    return chain


# ─────────────────────────────────────────────
# CHAIN 4: GENERAL INFO CHAIN
# ─────────────────────────────────────────────
def create_general_info_chain():
    """LangChain Chain untuk informasi umum"""
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Kamu adalah asisten informasi customer service yang ramah untuk TechPro.

Informasi yang tersedia:
{PRODUCT_KNOWLEDGE}

Berikan informasi yang jelas, singkat, dan helpful.
"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}")
    ])

    chain = prompt | llm | StrOutputParser()
    return chain


# ─────────────────────────────────────────────
# CHAIN 5: ESCALATION CHAIN
# ─────────────────────────────────────────────
def create_escalation_chain():
    """LangChain Chain untuk eskalasi ke agen manusia"""
    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Kamu adalah asisten yang membantu menghubungkan pelanggan ke agen manusia TechPro.
Sampaikan bahwa permintaan mereka akan segera ditangani oleh tim kami.
Berikan informasi kontak dan perkiraan waktu tunggu.

Kontak Langsung:
- Live Chat: tersedia di website kami
- Telepon: 0800-123-456 (Senin-Jumat 08:00-17:00 WIB)
- Email: support@techpro.id (respons dalam 1x24 jam)
- WhatsApp: 0812-3456-7890
"""),
        ("human", "{message}")
    ])

    chain = prompt | llm | StrOutputParser()
    return chain


# ─────────────────────────────────────────────
# HELPER: FORMAT CHAT HISTORY
# ─────────────────────────────────────────────
def format_history_to_messages(history: list) -> list:
    """Konversi history dict ke LangChain message objects"""
    messages = []
    for msg in history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages
