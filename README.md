# CLI Q&A Agent — Product Knowledge (local RAG via Ollama)

A single-file, in-memory Retrieval-Augmented-Generation agent that answers
questions about Gutermann water-leak-detection equipment, grounded strictly in
`product_overview.md`. Runs fully offline on a local Ollama server — no cloud
services, no API keys, no vector DB.

## Quick start

```bash
# 1. Ollama (once)
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
ollama serve            # if not already running

# 2. Python deps
pip install -r requirements.txt

# 3. Run
python qa_agent.py
```

```
> What sensors does the AQUASCAN 760T use?
Agent: The AQUASCAN 760T uses True Sound Sensors (TSS) ...
> exit
```

On startup the agent preflights the Ollama connection and both models, exiting with
an actionable message (e.g. `ollama pull …`) if something is missing.

Flags:
- `python qa_agent.py --sources` — show which product section(s) each answer came
  from, with cosine scores (the optional bonus).
- `python qa_agent.py --reindex` — force re-embedding (otherwise the index is
  cached in `.qa_index.pkl` and reused while the doc is unchanged).

Environment overrides (all optional):

| Var | Default | Purpose |
|-----|---------|---------|
| `QA_CHAT_MODEL` | `qwen2.5:7b` | Chat/answer model |
| `QA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `QA_TOP_K` | `4` | Chunks retrieved per query |
| `QA_DOC` | `product_overview.md` | Knowledge-base path |
| `QA_CACHE` | `.qa_index.pkl` | Embedding-cache path |
| `QA_SOURCES` | unset | `1` = always show sources |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |

## How it works

| Stage | Choice |
|-------|--------|
| Parse | Markdown, stripped of images/comments at KB-build time |
| Chunk | **One chunk per product** (`###`), prefixed with its category (`##`) and the category intro paragraph |
| Embed | `nomic-embed-text` via Ollama, L2-normalized, cached to `.qa_index.pkl` |
| Retrieve | Cosine similarity (dot product of unit vectors), top-k = 4 |
| Generate | `qwen2.5:7b`, temperature 0, strict grounding system prompt |

---

## Be ready to discuss

### 1. How did you chunk the markdown? Why that boundary?

**One chunk per product** (each `###` heading), and each chunk is **prefixed with
its category heading (`##`) plus the category's intro paragraph.**

Rationale:
- The document is a **product catalog** — its natural semantic unit is the
  product. A user question is almost always *about a product* ("does the ZONESCAN
  AI…", "difference between the 610 and 760T"). Product-sized chunks keep every
  fact about a product together and keep facts about *different* products apart,
  which is exactly what grounding and de-confusion need.
- Fixed-size / sliding-window chunking would cut mid-product or merge two
  products into one window, blurring the product boundary and causing
  feature-bleed between products (the failure mode Query 7 probes).
- **Attaching the category intro** matters: shared facts sometimes live under the
  category, not the product. The "no drilling of holes… 95% connectivity from
  underground chamber… NB-IoT" text sits under **Permanent Leak Detection
  Monitoring**, above ZONESCAN AI/HYDRO. Attaching it to each product chunk in
  that section is what lets Query 5 ("underground chambers, no drilling") retrieve
  and answer correctly.
- Chunks are small (a product is a few lines), so we can afford top-k = 4 and
  still fit comfortably in context — good for the comparison / multi-product
  queries (2 and 3) that need several products at once.

Trade-off: a product with a very long description would be one large chunk. Here
descriptions are short, so per-product is the sweet spot. If descriptions grew,
I'd sub-split long products by bullet groups while keeping the product+category
header on each sub-chunk.

### 2. Why is Query 4 hard for naive cosine similarity?

> *"I need to find leaks on plastic pipes over long distances — what do you
> recommend?"* → correct answer is **AQUASCAN TM3**.

- **Vocabulary / framing gap.** The query is a first-person *need* ("I need to
  find… what do you recommend"). The TM3 text is declarative spec language
  ("designed specifically to find leaks on plastic and large diameter pipes over
  long distances"). The most *task-relevant* words ("recommend", "I need") carry
  no signal and can pull the embedding toward generic phrasing rather than the
  right product.
- **Distractor overlap.** "Plastic pipes" appears in **several** products —
  AQUASCOPE 550 ("plastic pipes"), ZONESCAN HYDRO ("leaks even on plastic
  pipes"), AQUASCAN 760T (distribution networks). Naive cosine can rank a
  partial keyword match (AQUASCOPE 550, which is about plastic pipes but *not*
  long distances) above the true best fit, because it matches on "plastic" alone
  and misses the conjunction **plastic AND long distance**.
- The correct answer requires satisfying **two constraints jointly** (plastic +
  long distance); bag-of-embedding similarity rewards overlap on either term and
  doesn't enforce the "AND".

Mitigations used here: product-level chunks (so "plastic + long distance" co-occur
in the TM3 chunk, boosting its combined similarity), top-k = 4 (so even if TM3 is
rank 2–3 the LLM still sees it), and a system prompt that names the central
product explicitly.

### 3. Queries 6 & 7 — where would a wrong answer go wrong?

Both are **hallucination probes**; a wrong answer is a *grounding* failure, not a
retrieval one.

- **Query 6 — "Can I order the ZONESCAN HYDRO today?"** The document says HYDRO is
  *"Launching Q2 2026. For pre-series units please contact sales."* A wrong
  ("yes, sure") answer comes from the LLM defaulting to helpful-assistant
  behaviour and answering from prior/general knowledge instead of the passage.
  Grounding fix: the system prompt says availability/ordering/date claims may use
  *only* what that specific product's passage states (a launch date and a "contact
  sales" note **are** the answer to an ordering question), and must fall back to
  "not stated in the document" otherwise — never inventing a price, date or the
  ability to order.
- **Query 7 — "Does the ZONESCAN AI use hydrophone technology?"** Answer is **no**
  — ZONESCAN AI uses an *internal accelerometer*; hydrophone tech belongs to
  ZONESCAN **HYDRO**. The failure mode is **cross-product feature bleed**:
  retrieval returns both the AI and HYDRO chunks (they share the category and the
  "ZONESCAN" token), and a careless model attributes HYDRO's hydrophone line to
  AI. Per-product chunking keeps the two products as distinct passages, and the
  system prompt explicitly says a fact stated for one product does **not** apply
  to another — so the model can contrast them rather than merge them.

### 4. How the grounding prompt got to its final form

The two grounding rules pull against each other on a 7B local model, and the final
prompt reflects two bugs that only surfaced from running the real CLI end-to-end
(not from testing functions in isolation):

1. **Q6 over-refused.** The first, strict grounding prompt made the model reply
   "not enough information" even though HYDRO's launch line was in the retrieved
   chunk. Fix: tell the prompt that a launch date + "contact sales" *is*
   availability information to be surfaced, reinforced with a one-shot example.
2. **That fix regressed Q5.** With availability emphasised, the model began
   **bleeding HYDRO's "Launching Q2 2026" onto ZONESCAN AI** (which has no launch
   date). Strengthening the "facts are product-specific" rule alone did **not**
   stop it. The robust fix was a **scoping rule**: *only discuss availability /
   launch / ordering when the question actually asks about it.* Q5 is a
   recommendation query, so a launch date should never appear there; Q6 still
   works because it genuinely is an ordering question.

Takeaway: with a small model, "be maximally helpful about availability" and "never
transfer facts between products" conflict, and the clean resolution is to scope
*when* a class of information is eligible to appear rather than to pile on ever
stronger prohibitions.

### Verified behaviour

All seven interviewer queries produce correct, grounded answers, and a control
question with genuinely absent info ("How much does the AQUASCAN 610 cost?")
correctly returns *"The document does not contain enough information to answer
that."* — confirming the grounding guard refuses rather than always-answering.
