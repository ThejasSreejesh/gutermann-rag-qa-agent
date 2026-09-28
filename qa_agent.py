#!/usr/bin/env python3
"""
CLI Q&A Agent over a product knowledge base (RAG, fully local via Ollama).

    python qa_agent.py

Ingests product_overview.md once at startup (parse -> chunk -> embed, cached to
disk), then answers questions interactively. Answers are grounded strictly in the
document; when the document lacks the information, the agent says so instead of
guessing. Type `exit` or `quit` to leave.

Backend: Ollama running locally (http://localhost:11434).
  - embeddings: nomic-embed-text
  - chat:       qwen2.5:7b   (override with $QA_CHAT_MODEL)
"""

import json
import os
import pickle
import re
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass

import numpy as np

# ----------------------------- configuration --------------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
EMBED_MODEL = os.environ.get("QA_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("QA_CHAT_MODEL", "qwen2.5:7b")
DOC_PATH = os.environ.get("QA_DOC", "product_overview.md")
CACHE_PATH = os.environ.get("QA_CACHE", ".qa_index.pkl")
TOP_K = int(os.environ.get("QA_TOP_K", "4"))

HERE = os.path.dirname(os.path.abspath(__file__))


def _p(name: str) -> str:
    """Resolve a path relative to this script so it runs from any cwd."""
    return name if os.path.isabs(name) else os.path.join(HERE, name)


# ------------------------------ Ollama client -------------------------------

def _post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}{path}", data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def embed(text: str) -> np.ndarray:
    out = _post("/api/embeddings", {"model": EMBED_MODEL, "prompt": text})
    return np.asarray(out["embedding"], dtype=np.float32)


def chat(system: str, user: str) -> str:
    out = _post("/api/chat", {
        "model": CHAT_MODEL,
        "stream": False,
        "options": {"temperature": 0.0},  # deterministic, low-invention
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    })
    reply = out["message"]["content"].strip()
    if len(reply) >= 2 and reply[0] == '"' and reply[-1] == '"' and reply.count('"') == 2:
        reply = reply[1:-1].strip()   # unwrap a fully-quoted reply
    return reply


def check_ollama() -> None:
    """Fail early with an actionable message if Ollama or a model is missing."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=10) as r:
            tags = json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError:
        sys.exit(
            f"[!] Cannot reach Ollama at {OLLAMA_HOST}.\n"
            f"    Start it with `ollama serve` and make sure it is running."
        )
    have = {m["name"].split(":")[0] for m in tags.get("models", [])}
    have |= {m["name"] for m in tags.get("models", [])}
    for model in (EMBED_MODEL, CHAT_MODEL):
        if model not in have and model.split(":")[0] not in have:
            sys.exit(f"[!] Model '{model}' not found. Pull it with:\n    ollama pull {model}")


# ------------------------------- ingestion ----------------------------------

@dataclass
class Chunk:
    section: str      # e.g. "Permanent Leak Detection Monitoring > ZONESCAN AI"
    product: str      # e.g. "ZONESCAN AI"
    text: str         # embedded/retrieved text (category context + product body)


def parse_chunks(md: str) -> list[Chunk]:
    """
    One chunk per product (### heading). Each chunk is prefixed with its category
    (## heading) and the category's intro paragraph, so shared context — e.g. the
    'no drilling / underground chamber / NB-IoT' description that lives under the
    Permanent Monitoring category rather than under a single product — travels with
    every product it applies to. See README for the rationale.
    """
    lines = md.splitlines()
    category = ""
    category_intro: list[str] = []       # text under a ## before the first ###
    seen_product_in_cat = False
    product = ""
    body: list[str] = []
    chunks: list[Chunk] = []

    def flush():
        nonlocal product, body
        if product:
            intro = "\n".join(category_intro).strip()
            head = f"Category: {category}\n"
            if intro:
                head += f"{intro}\n"
            content = "\n".join(body).strip()
            text = f"{head}\nProduct: {product}\n{content}".strip()
            chunks.append(Chunk(
                section=f"{category} > {product}" if category else product,
                product=product, text=text,
            ))
        product, body = "", []

    for ln in lines:
        if ln.startswith("## ") and not ln.startswith("###"):
            flush()
            category = ln[3:].strip()
            category_intro = []
            seen_product_in_cat = False
        elif ln.startswith("### "):
            flush()
            product = ln[4:].strip()
            seen_product_in_cat = True
        else:
            if seen_product_in_cat:
                body.append(ln)
            elif category and ln.strip():
                category_intro.append(ln)
    flush()
    return chunks


def build_index(force: bool = False):
    doc = _p(DOC_PATH)
    if not os.path.exists(doc):
        sys.exit(f"[!] Knowledge base not found: {doc}")
    mtime = os.path.getmtime(doc)
    cache = _p(CACHE_PATH)

    if not force and os.path.exists(cache):
        with open(cache, "rb") as f:
            saved = pickle.load(f)
        if (saved.get("mtime") == mtime and saved.get("embed_model") == EMBED_MODEL
                and saved.get("version") == 2):
            return saved["chunks"], saved["matrix"]

    md = open(doc, encoding="utf-8").read()
    chunks = parse_chunks(md)
    print(f"[ingest] {len(chunks)} chunks -> embedding with {EMBED_MODEL} ...",
          file=sys.stderr)
    vecs = [embed(c.text) for c in chunks]
    matrix = np.vstack(vecs)
    matrix /= (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)  # unit norm

    with open(cache, "wb") as f:
        pickle.dump({"version": 2, "mtime": mtime, "embed_model": EMBED_MODEL,
                     "chunks": chunks, "matrix": matrix}, f)
    return chunks, matrix


# ------------------------------- retrieval ----------------------------------

def retrieve(query: str, chunks, matrix, k: int = TOP_K):
    q = embed(query)
    q /= (np.linalg.norm(q) + 1e-9)
    scores = matrix @ q                       # cosine (matrix is unit-normed)
    idx = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in idx]


# --------------------------- answer generation ------------------------------

SYSTEM_PROMPT = """You are a product-knowledge assistant for Gutermann water leak \
detection equipment. Answer the user's question using ONLY the CONTEXT passages \
provided below. The context is the single source of truth.

Rules:
- Ground every claim in the context. Do NOT use outside or general knowledge.
- Each passage names its Product. A fact stated for one product does NOT apply to \
another. Do not transfer features between products.
- If the context does not contain enough information to answer, reply exactly: \
"The document does not contain enough information to answer that." Do not guess.
- Only discuss availability, launch dates, pre-series status, ordering or pricing \
when the QUESTION actually asks about those things. For any other question, do NOT \
volunteer launch dates or ordering notes even if a passage contains them.
- When the question IS about availability/ordering/pricing/dates: answer using \
whatever that specific product's passage says — a launch date, "contact sales", or \
pre-series status all count and must be surfaced. Fall back to "not stated in the \
document" only when its passage says nothing about it. Never invent a price, a \
date, or an ability to order.
- Be concise. When a product is central to the answer, name it explicitly.

Worked example of using availability info (do the same for similar questions):
  CONTEXT passage: "Product: WIDGET X - Launching Q3 2027. For pre-series units \
please contact sales."
  QUESTION: "Can I order the WIDGET X today?"
  GOOD ANSWER: "Not for general order yet - the WIDGET X is launching Q3 2027. For \
pre-series units, the document says to contact sales."
This is grounded: a launch date plus a "contact sales" note ARE the answer to an \
ordering question, so use them rather than refusing.
IMPORTANT: availability facts are product-specific too. A launch date, pre-series \
note or "contact sales" stated for ONE product must NOT be repeated for any other \
product, even one in the same category. If a product's passage gives no launch or \
availability note, do not invent one for it."""


def answer(query: str, retrieved) -> str:
    context = "\n\n---\n\n".join(
        f"[{c.section}]\n{c.text}" for c, _ in retrieved
    )
    user = f"CONTEXT:\n{context}\n\nQUESTION: {query}"
    return chat(SYSTEM_PROMPT, user)


# --------------------------------- CLI --------------------------------------

def main() -> None:
    show_sources = "--sources" in sys.argv or os.environ.get("QA_SOURCES") == "1"
    check_ollama()
    chunks, matrix = build_index(force="--reindex" in sys.argv)

    print("Product Q&A agent (local Ollama). Ask about Gutermann equipment.")
    print("Type your question, or 'exit' / 'quit' to leave.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        retrieved = retrieve(query, chunks, matrix)
        try:
            reply = answer(query, retrieved)
        except urllib.error.URLError as e:
            print(f"Agent: [error talking to Ollama: {e}]\n")
            continue

        print(f"Agent: {reply}")
        if show_sources:
            srcs = ", ".join(f"{c.section} ({s:.2f})" for c, s in retrieved)
            print(f"       [sources: {srcs}]")
        print()


if __name__ == "__main__":
    main()
