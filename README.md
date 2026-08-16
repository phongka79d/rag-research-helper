# RAG Research Helper

A local Streamlit mentor for research papers. You upload a paper, the app compiles it once, then you can **Ask**, **Teach**, or inspect the **Graph**.

It is a small, readable pipeline — not an enterprise product and not a LangChain wrapper. Expensive work happens at ingest (AOT). Ask and Teach only retrieve and generate.

## What you get

| Tab | What it does |
| --- | --- |
| **Ask** | Retrieve at most two parent sections, attach nearby concept relations, and answer with source labels. Tables and display math in the answer are rendered (GFM + KaTeX). |
| **Teach** | Walk a stored section roadmap step by step, with one-hop prerequisite context from the graph. |
| **Graph** | Show concepts and relations extracted from a selected section as a table and a Mermaid diagram. |

## Pipeline

```text
PDF  →  MinerU Flash (required)  →  section split
Markdown (already extracted)     →  section split
                                      │
                                      ▼
                         AOT compile (per section)
                         • learning roadmap
                         • HyDE questions (0–5)
                         • grounded concept graph
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
              Qdrant                              Neo4j
              research_curriculum                 Concept nodes
              (section + roadmap)                 + research relations
              research_questions
              (child → parent_id)
                    │
                    ▼
         Ask / Teach / Graph  (no re-extract)
```

A PDF never starts AOT until MinerU returns a **complete** extract (`data/mineru/<name>.md` + manifest). Incomplete extraction is fail-closed. Markdown skips MinerU and uses the same compile path.

Bibliography / References headings are skipped; appendix-style sections after that are kept.

## How retrieval works

Ask does **not** search the raw paper text first. It searches the hypothetical questions stored at ingest, then resolves the original section.

1. Embed the user query.
2. Search up to **25** HyDE question hits in Qdrant (`research_questions`).
3. Collapse to at most **5** unique parent sections.
4. Optional **Jina** rerank of that pool. If Jina is unset or uncertain, fall back to an **LLM** reranker, then to vector order.
5. Fuse at most **2** parent sections.
6. Pull 1–2 hop concept context from Neo4j and generate a grounded answer.

Provenance is recorded as `jina`, `llm_fallback`, `llm`, or `vector`. The two-parent cap is intentional: more parents is not treated as better evidence.

## How the graph is grounded

Each compiled section is scanned for research relations and also sent to a graph model. A kept edge must:

- name two concept endpoints (not pronouns, clause fragments, or function words);
- cite a numbered evidence span (`eN`) that actually appears in the section;
- use a relation the quote can support.

**18 canonical relations:** `PART_OF`, `PREREQUISITE_OF`, `DESCRIBES`, `RELATES_TO`, `USES`, `EVALUATED_ON`, `TRAINED_ON`, `BASED_ON`, `PROPOSES`, `OUTPERFORMS`, `COMPARES_TO`, `ACHIEVES`, `REQUIRES`, `APPLIED_TO`, `IMPROVES`, `ENABLES`, `PRODUCES`, `HAS_FEATURE`.

A wording table maps paper phrasing onto that set. If nothing matches, a well-formed novel `UPPER_SNAKE` predicate may be kept. Unknown garbage is dropped — it is **not** silently remapped to `RELATES_TO`.

Thin sections (under 200 characters of body) still get a roadmap when possible; they do not get a graph or HyDE questions.

Empty graph context at Ask/Teach time is left empty. The model is told not to invent relations.

## Stack

| Piece | Role |
| --- | --- |
| Streamlit (`main.py`) | UI and direct object wiring |
| OpenAI-compatible Responses + Embeddings | AOT, HyDE, rerank, answers, lessons |
| Qdrant | Parent sections, roadmap steps, HyDE children |
| Neo4j | Shared `Concept` nodes and relations |
| MinerU Flash | Required PDF layout/OCR → Markdown |
| Optional Jina | Rerank the bounded question pool |

There is no API server, no repository/service layer, and no LangChain / LangGraph. `RuntimeEngine` calls Qdrant, Neo4j, and `LLMService` directly.

## Setup

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. The provider must implement both `/v1/responses` and `/v1/embeddings`. Changing `.env` requires a full Streamlit restart; a process-level environment variable wins over `.env`.

Optional live checks (they do not write to Qdrant or Neo4j):

```powershell
python scripts/live_test_responses.py
python scripts/live_test_embeddings.py
```

Start local databases without wiping existing Neo4j data:

```powershell
python setup_env.py --start
```

`setup_env.py` fills non-secret defaults, recovers a Neo4j password only when it can verify it, and **refuses** to reset graph volumes. Do not delete Neo4j volumes casually.

Or, after `.env` has a verified `NEO4J_PASSWORD`:

```powershell
docker compose up -d qdrant neo4j
```

Run the app:

```powershell
streamlit run main.py
```

## Use

1. Upload a PDF, `.md`, or `.markdown` file in the sidebar. PDFs go through MinerU Flash, then AOT. Demo Markdown papers live in `data/papers/`.
2. Wait for the compile summary (sections compiled, graph candidates / approvals / retained). A zero in any graph counter is valid and means the Graph tab will be empty for that paper.
3. Use **Ask** (optionally scoped to one paper), **Teach** (pick a section), or **Graph**.

When `OPENAI_GRAPH_MODEL` is set, roadmap and HyDE stay on `OPENAI_MODEL`; graph extract / verify / recover use the graph model. Both share `OPENAI_BASE_URL` and `OPENAI_API_KEY`.

### MinerU Flash (PDF)

The app uploads the local PDF to MinerU’s no-token signed-upload endpoint, submits page ranges (default 10, max 20), and writes Markdown plus a JSON manifest. Flash is IP-rate-limited, accepts files up to about 10 MB, and returns Markdown only. Do not send sensitive papers without reviewing that implication.

A failed batch sets `complete=false` and never starts AOT. You can also run extraction alone (no Qdrant/Neo4j writes):

```powershell
python scripts/mineru_flash.py data/papers/your-paper.pdf --output data/mineru --batch-size 10
```

`pypdf` remains only for callers that still invoke `ingest_document` on a PDF without a complete manifest. The Streamlit path does not use that fallback.

## Evaluate

```powershell
python -m pytest -q
```

Retrieval evaluation (needs running Qdrant + Neo4j, a provider key, and ingested sections):

```powershell
# Tracked demo cases
python evaluate.py --workers 4 --output eval_results.json

# Local three-paper benchmark (add data/eval_real_papers.json yourself)
python evaluate.py --dataset data/eval_real_papers.json --workers 1 --output eval_results_real_papers.json
```

Reported **Recall@5 / Recall@2 / MRR** are **parent-section retrieval** scores. They are **not** graph-edge precision, and they are not an LLM-as-judge quality score.

To list unique stored `Concept` edges (no judge):

```powershell
python evaluate.py --graph-sample --source attention.pdf --source qlora_paper.pdf
```

Omit `--source` to sample the full Neo4j concept graph. `--graph-sample` needs only Neo4j.

## Configuration

Never commit `.env`.

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_BASE_URL` | No | OpenAI-compatible `/v1` URL. Default: `https://api.shopaikey.com/v1`. |
| `OPENAI_API_KEY` | Yes for app and eval | Provider credential. |
| `OPENAI_MODEL` | No | Text model. Default: `gpt-4o-mini`. |
| `OPENAI_GRAPH_MODEL` | No | Graph extract / verify / recover. Blank → `OPENAI_MODEL`. |
| `OPENAI_EMBEDDING_MODEL` | No | Default: `text-embedding-3-small`. |
| `OPENAI_EMBEDDING_DIM` | No | Qdrant vector size. Default: `1536`. Must match the embedding model. |
| `QDRANT_URL` | No | Default: `http://localhost:6333`. |
| `QDRANT_SEARCH_LIMIT` | No | HyDE hits per query. Default 25, capped at 25. |
| `QDRANT_MAX_CANDIDATE_PARENTS` | No | Unique parents sent to rerank. Default 5, capped at 5. |
| `NEO4J_URI` | No | Default: `bolt://localhost:7687`. |
| `NEO4J_USER` | No | Default: `neo4j`. |
| `NEO4J_PASSWORD` | Yes | Created/recovered and verified by `setup_env.py`. |
| `JINA_API_KEY` | No | Optional rerank credential. |
| `JINA_RERANK_URL` | No | Default: `https://api.jina.ai/v1/rerank`. |
| `JINA_RERANK_MODEL` | No | Default: `jina-reranker-v2-base-multilingual`. |
| `JINA_RPM` | No | Local request cap. Invalid values → 100/min. |
| `JINA_RERANK_MARGIN` | No | Margin before LLM fallback. Default: `0.08`. |

## Repository layout

```text
config/                 Settings from .env
core/                   AOT ingest, relation vocabulary, schemas
database/               MinerU Markdown split, Qdrant, Neo4j
orchestrator/           Direct OpenAI-compatible SDK client
runtime/                Ask / Teach / Markdown + Mermaid helpers
scripts/                Live API checks, MinerU helper, 3-paper validation
tests/                  Behavior-focused pytest
main.py                 Streamlit entry
evaluate.py             Retrieval + graph-sample CLI
setup_env.py            Safe Neo4j / compose bootstrap
docker-compose.yml      Local Qdrant + Neo4j
```

`data/papers/` holds uploads (gitignored except two demo Markdown files). MinerU extracts land under `data/mineru/`.

## Deliberate limits

- PDF ingest uploads the file to MinerU Flash. There is no offline PDF path in the app.
- The Graph tab is a table plus Mermaid, not an interactive network editor.
- AOT quality and answers follow the configured model. Structure is validated; content is not claimed as SOTA.
- Retrieval scores measure whether the right **section** came back, not whether every graph edge is correct.
- Ask stays at two fused parents. Raising that cap is not the quality lever this project uses.

## Reference

Patterns for AOT ingest, Qdrant parent/child retrieval, Neo4j concept walk, and a direct runtime were adapted from [rag-expert-mentor](https://github.com/phongka79d/rag-expert-mentor). Unrelated product features were not copied. Check that repository’s license before reusing source verbatim.
