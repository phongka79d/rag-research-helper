# RAG Research Helper

## Overview

RAG Research Helper is a local Streamlit app for exploring research papers as a compact research mentor. It performs expensive analysis during ingestion, then uses the precompiled results at runtime for three direct workflows:

- **Ask:** retrieve relevant paper sections, add nearby concept relationships, and generate a grounded answer with sources.
- **Teach:** turn a stored section roadmap into step-by-step lessons with prerequisite context.
- **Graph:** inspect concepts and relations extracted from a selected paper section.

The implementation intentionally stays small: the OpenAI Python SDK sends requests to one configured OpenAI-compatible provider, Qdrant provides semantic retrieval, Neo4j stores concept relationships, and Python wires the flow together. It does not use LangChain, LangGraph, or production-style service/repository layers.

## What This Folder Does

This repository is the complete local application. The implementation plan in `RAG_Research_Helper_Simple_Reference_First_Rebuild_Plan.md` is the product and architecture source of truth. Runtime behavior is defined primarily by:

- `main.py` for the Streamlit UI and direct object wiring.
- `core/data_ingestion.py` for ahead-of-time (AOT) paper compilation.
- `database/structural_db.py` for Qdrant storage and retrieval.
- `database/semantic_dag.py` for the Neo4j concept graph.
- `orchestrator/llm_service.py` for direct OpenAI-compatible Responses and Embeddings API calls.
- `runtime/engine.py` for the direct Ask and Teach flows.

Do not casually reset or delete local Neo4j volumes: they can contain existing graph data. `setup_env.py` is the safe credential and connectivity utility.

## Repository Structure

```text
.
├── config/                 # Environment-backed Settings
├── core/                   # Small schemas and AOT ingestion loop
├── database/               # PDF/Markdown parsing, Qdrant, and Neo4j
├── orchestrator/           # Direct OpenAI SDK client
├── runtime/                # Ask and Teach orchestration
├── data/
│   ├── papers/             # Locally uploaded or sample papers
│   ├── eval.json           # Legacy/demo retrieval evaluation cases
│   └── eval_real_papers.json # Source-scoped benchmark for the three supplied PDFs
├── tests/                  # Behavior-focused pytest coverage
├── main.py                 # Streamlit entry point
├── evaluate.py             # Baseline / HyDE-rerank / graph-context evaluation
├── setup_env.py            # Safe local Neo4j setup and verification
├── docker-compose.yml      # Local Qdrant and Neo4j services
├── requirements.txt
└── RAG_Research_Helper_Simple_Reference_First_Rebuild_Plan.md
```

## Main Workflows

### 1. Environment and databases

1. `Settings` loads `.env` through `config/settings.py`.
2. `setup_env.py` fills missing non-secret defaults, safely recovers an existing Neo4j credential only when it can verify it, and stops rather than resetting existing graph data.
3. `docker-compose.yml` starts Qdrant and Neo4j with the password supplied through `.env`.
4. `main.py` validates settings and calls `Neo4jManager.verify_connection()` before rendering the app.

### 2. Ingest a PDF or Markdown paper

1. The sidebar in `main.py` saves the uploaded file under `data/papers/`.
2. A PDF is extracted with MinerU Flash first (`data/mineru/<name>.md` plus a complete manifest). Incomplete extraction never starts AOT. Markdown is treated as already-extracted text.
3. `DocumentProcessor.process_mineru_markdown()` (or the Markdown parser) splits the extracted text into ordered sections with source, section, and sequence metadata.
4. `ingest_document()` in `core/data_ingestion.py` runs the existing AOT path: a learning roadmap, grounded graph candidates (local span scan first, model graph only when needed), HyDE questions, and Neo4j/Qdrant persistence. The local matcher can retain direct whole-part edges; the verifier still reviews model-proposed candidates.
4. The graph is stored in Neo4j; roadmap steps and full parent sections are stored in `research_curriculum`; up to five directly answerable hypothetical question children are stored in `research_questions` (thin sections may have none).
5. Each child question keeps its `parent_id`, so retrieval can resolve the complete original section.
6. The completion result reports graph candidates, verifier approvals, and retained relationships. A zero in any counter is valid and makes an empty Graph tab diagnosable.

### 3. Ask a paper question

1. `RuntimeEngine.ask()` calls `QdrantVectorStore.search_candidates_and_fetch_parent()`.
2. The query embedding searches up to 25 hypothetical-question hits, optionally filtered to one paper, and keeps the first hit for at most five unique parent sections.
3. If `JINA_API_KEY` is configured, the optional Jina reranker receives the user query and candidate hypothetical questions. The cascade is Jina → OpenAI-compatible LLM reranker when Jina is unavailable/uncertain → vector order when no reranker selection is usable. The final evidence set remains at most two parents, and stores truthful provenance (`jina`, `llm_fallback`, `llm`, or `vector`).
4. The engine reads 1–2 hop concept context from Neo4j and sends the parent text plus graph context to `LLMService.answer()`.
5. The UI displays the answer, stored source labels, and optional graph context.

### 4. Teach a section

1. The user selects a paper and section in the Teach tab.
2. `RuntimeEngine.teach_section()` fetches the full stored section and its AOT roadmap.
3. For each roadmap step, it fetches one-hop prerequisite context and asks `LLMService.teach_step()` for a grounded lesson.
4. Streamlit renders each lesson in roadmap order.

### 5. Inspect the graph and evaluate retrieval

- The Graph tab queries `Neo4jManager.get_visual_graph(locator)` and renders concepts and relationships as tables.
- `evaluate.py` compares the direct parent-section vector baseline with the same hypothetical-question retrieval and two-parent fusion used by Ask. It reports baseline Recall@5, runtime Recall@2 and MRR, all-expected-sources coverage, retrieval latency, graph-context size, fixed four-way rerank provenance rates, and effective bounded retrieval capacities in `eval_results.json`.

## Architecture

```text
PDF
  → MinerU Flash extract (required)
  → DocumentProcessor (MinerU Markdown)
  → AOT extraction (OpenAI) + local graph scan
  → Neo4j Concept graph + Qdrant parent / roadmap / question points

Markdown (already extracted)
  → DocumentProcessor
  → same AOT path

Question
  → OpenAI embedding
  → Qdrant question search
  → optional Jina rerank → OpenAI-compatible LLM fallback → vector order
  → parent section(s) + Neo4j context
  → grounded OpenAI answer + source labels
```

The key storage model is deliberately narrow:

- Qdrant collection `research_curriculum`: `section_anchor` and `roadmap_step` points.
- Qdrant collection `research_questions`: hypothetical question points with a `parent_id`.
- Neo4j: only `Concept` nodes and `PREREQUISITE_OF`, `RELATES_TO`, `PART_OF`, or `DESCRIBES` edges.

## Frontend

`main.py` is a single Streamlit entry point with:

- a sidebar for upload/ingestion and current-paper filtering;
- **Ask**, **Teach**, and **Graph** tabs;
- no separate API server, controller layer, or client-side state system.

## Data, Storage, and External Services

| Component | Purpose | Defined by |
| --- | --- | --- |
| OpenAI-compatible Responses API | AOT extraction, hypothetical questions, reranking, answers, and lessons | `orchestrator/llm_service.py` |
| OpenAI-compatible Embeddings API | Query, section, roadmap, and question vectors | `orchestrator/llm_service.py` |
| Optional Jina rerank API | Reranks the bounded query/question candidate pool when `JINA_API_KEY` is set | `orchestrator/llm_service.py` |
| Qdrant | Parent sections, roadmap steps, and hypothetical question retrieval | `database/structural_db.py` |
| Neo4j | Shared research concepts and bounded graph context | `database/semantic_dag.py` |
| `data/papers/` | Uploaded/local paper files | `main.py` |
| `data/eval.json` | Legacy/demo evaluation cases | `evaluate.py` |
| `data/eval_real_papers.json` | Three-real-paper evaluation cases | `evaluate.py` |

## Configuration

Copy `.env.example` to `.env`; never commit `.env` or its values.

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_BASE_URL` | No | OpenAI-compatible `/v1` base URL; defaults to `https://api.shopaikey.com/v1`. |
| `OPENAI_API_KEY` | Yes for app and evaluation | Credential passed to the configured provider through the OpenAI SDK. |
| `OPENAI_MODEL` | No | Text model; defaults to `gpt-4o-mini`. |
| `OPENAI_GRAPH_MODEL` | No | Graph extraction/verifier/recovery model; blank falls back to `OPENAI_MODEL` (for example, `gpt-4.1-nano`). It uses the same compatible endpoint and key. |
| `OPENAI_EMBEDDING_MODEL` | No | Embedding model; defaults to `text-embedding-3-small`. |
| `OPENAI_EMBEDDING_DIM` | No | Qdrant vector size; defaults to `1536`. It must match the configured embedding model. |
| `QDRANT_URL` | No | Qdrant endpoint; defaults to `http://localhost:6333`. |
| `QDRANT_SEARCH_LIMIT` | No | Maximum question hits searched per query; defaults to 25 and is capped at 25. |
| `QDRANT_MAX_CANDIDATE_PARENTS` | No | Maximum unique parents sent to reranking; defaults to 5 and is capped at 5. |
| `NEO4J_URI` | No | Neo4j Bolt endpoint; defaults to `bolt://localhost:7687`. |
| `NEO4J_USER` | No | Neo4j user; defaults to `neo4j`. |
| `NEO4J_PASSWORD` | Yes | Local Neo4j password, created/recovered and verified by `setup_env.py`. |
| `JINA_API_KEY` | No | Optional Jina credential. When set, query text and candidate questions are sent to the configured Jina endpoint for reranking. |
| `JINA_RERANK_URL` | No | Jina-compatible rerank endpoint; defaults to `https://api.jina.ai/v1/rerank`. |
| `JINA_RERANK_MODEL` | No | Jina rerank model; defaults to `jina-reranker-v2-base-multilingual`. |
| `JINA_RPM` | No | Local Jina request-rate limit; invalid values use 100 requests/minute. |
| `JINA_RERANK_MARGIN` | No | Minimum top-score margin before LLM fallback; defaults to 0.08. |

## Setup

1. Create and activate a virtual environment.

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env`, then set `OPENAI_API_KEY`.

   The configured provider must support both the OpenAI Responses (`/v1/responses`) and Embeddings (`/v1/embeddings`) contracts used by this app. Changing `.env` values requires stopping and fully restarting Streamlit; refreshing the browser is not enough. A process-level environment variable takes precedence over `.env`, so inspect the running process when a changed model appears to be ignored.

   Before ingesting a paper, run the opt-in compatibility checks. The Responses check makes one request for the text model and one for a distinct graph model (one request total when the graph setting falls back); neither check writes to Qdrant or Neo4j:

   ```powershell
   python scripts/live_test_responses.py
   python scripts/live_test_embeddings.py
   ```

3. Run the safe local database setup. It does not print the Neo4j password and refuses to reset existing Neo4j data when the credential cannot be verified.

   ```powershell
   python setup_env.py --start
   ```

4. Run the app.

   ```powershell
   streamlit run main.py
   ```

The compose file can also be started directly after `.env` contains a verified `NEO4J_PASSWORD`:

```powershell
docker compose up -d qdrant neo4j
```

## Running the Project

```powershell
streamlit run main.py
```

Upload a PDF, `.md`, or `.markdown` file in the sidebar. PDFs are extracted with MinerU Flash first, then compiled through the same AOT graph pipeline. Markdown skips MinerU. Compilation uses the configured OpenAI-compatible endpoint and writes into the local Qdrant and Neo4j services.

When `OPENAI_GRAPH_MODEL` is set, roadmap and hypothetical-question generation stay on
`OPENAI_MODEL`; graph extraction, verification, and recovery use the graph model. The
plan, graph, and question requests for one section run in a small bounded group. Both
models still use the same `OPENAI_BASE_URL` and `OPENAI_API_KEY`. Restart the Streamlit
process after changing `.env`; a process environment variable takes precedence over
`.env`.

For an OCR/layout experiment, run the opt-in MinerU Agent Flash helper:

```powershell
python scripts/mineru_flash.py data/papers/your-paper.pdf --output data/mineru --batch-size 10
```

It uses the no-token signed-upload endpoint, submits contiguous page ranges (default
10, maximum 20), polls each task, and writes Markdown plus a JSON manifest. Flash is
IP-rate-limited, accepts files up to 10 MB and 20 pages per request, and returns
Markdown only. The local PDF is uploaded to MinerU; do not use it for sensitive papers
without reviewing that privacy implication. A failed batch produces `complete=false`
and is never silently treated as a complete document. The app PDF ingest path uses this
same client, then compiles automatically. Running the helper by hand still does not
write Qdrant/Neo4j.

For a controlled end-to-end comparison on the three supplied papers, run the opt-in
validation command after Qdrant, Neo4j, and the configured provider are available:

```powershell
python scripts/validate_mineru_three_papers.py `
  --input-dir data/papers `
  --output-dir data/mineru-validation `
  --batch-size 10 `
  --workers 1
```

It processes `attention.pdf`, `qlora_paper.pdf`, and `slm_paper.pdf`, requires every
MinerU manifest to be complete, stores derived sources such as `mineru_attention.md`
(default `--source-prefix mineru_`), and writes a comparison report under the selected
output directory. The original PDF sources are not replaced. The report includes
parser cleanup, sections compiled this run versus sections skipped as already current,
source-scoped graph counts and rejection diagnostics, a bounded retained-edge audit
sample, Qdrant counts, retrieval metrics, and representative Ask source previews. A
partial extraction or unavailable live dependency makes the overall report incomplete;
it is not silently treated as a successful comparison.

Optional `--source-prefix` selects the derived-source name prefix (default `mineru_`).
It must use only letters, digits, and underscore; start with a letter; and end with
`_`. A distinct prefix creates distinct derived source identities and does not replace
original PDFs or prior MinerU-derived sources. Example with an isolated evidence run:

```powershell
python scripts/validate_mineru_three_papers.py `
  --input-dir data/papers `
  --output-dir data/mineru-validation-evidence-v2 `
  --source-prefix mineru_evidence_ `
  --batch-size 10 `
  --workers 1
```

The command does not delete or clean up databases or old sources.

## Testing and Validation

Run the behavior-focused suite:

```powershell
python -m pytest -q
```

Run the evaluation against the ingested evaluation corpus:

```powershell
python evaluate.py --workers 4 --output eval_results.json
```

For a score covering only the supplied real papers, force-ingest those three
PDFs once and run the dedicated corpus separately:

```powershell
python evaluate.py --dataset data/eval_real_papers.json --workers 1 --output eval_results_real_papers.json
```

The legacy `data/eval.json` and `eval_results.json` remain available for
comparison; they include earlier demo-paper cases and are not the dedicated
three-real-paper score.

The evaluation requires reachable Qdrant and Neo4j services, a configured
OpenAI-compatible provider key, and the sections referenced by the selected
dataset to be ingested. It does not automatically retry failed cases; compare
the reported effective limits and provenance rates when interpreting quality
and latency changes.

## Development Notes for AI Agents

- Read `RAG_Research_Helper_Simple_Reference_First_Rebuild_Plan.md` before changing behavior; it intentionally forbids extra architecture.
- Before changing a major module, inspect the matching `rag-expert-mentor` reference file listed in the plan. Adapt the small relevant pattern; do not copy unrelated features wholesale.
- Preserve the direct flow: `RuntimeEngine` calls Qdrant, Neo4j, and `LLMService` directly. Do not introduce repositories, services, adapters, factories, dependency injection, LangChain, or LangGraph.
- Keep JSON-producing compatible-provider calls in `LLMService` in JSON mode. The rerank call has a bounded token allowance so it can return its required JSON selection.
- Keep `.env` private. Before any Neo4j-dependent change, verify the existing database safely; never delete/reset data or volumes automatically.
- When changing ingestion, coordinate `core/data_ingestion.py`, `database/structural_db.py`, `database/semantic_dag.py`, and their focused tests. Preserve the parent-child `parent_id` contract and up to five stored hypothetical questions per section.
- When changing retrieval behavior, validate `tests/test_qdrant.py`, `tests/test_runtime.py`, and `evaluate.py`; test a real Ask path when configured services are available.
- Ignore generated caches such as `__pycache__/` and `.pytest_cache/`. Treat `eval_results.json` as a generated evaluation artifact.

## Known Gaps or Deliberate Limits

- App PDF ingest uses MinerU Flash (layout/OCR Markdown) before AOT. The local PDF is uploaded to MinerU; Flash is IP-rate-limited and accepts files up to 10 MB. Direct `pypdf` remains available only for non-app callers that still call `ingest_document` on a PDF without a manifest.
- The graph view is a table/list rather than an interactive network visualization, matching the plan’s stated first-version fallback.
- The app depends on local Qdrant and Neo4j plus a reachable OpenAI-compatible endpoint; it has no offline fallback.
- AOT extraction quality and answer quality remain model-dependent. The application validates JSON structure and falls back through the optional Jina → LLM → vector rerank cascade when needed.

## Reference Attribution

This project uses [rag-expert-mentor](https://github.com/phongka79d/rag-expert-mentor) as a file-by-file architecture and implementation reference. Relevant patterns were adapted for research-paper AOT ingestion, Qdrant parent-child retrieval, Neo4j concept traversal, and direct runtime orchestration. The project does not copy unrelated product features wholesale. Consult the reference repository’s license and author permissions before reusing substantial source verbatim.
