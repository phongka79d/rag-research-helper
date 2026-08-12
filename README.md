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
│   └── eval.json           # Retrieval evaluation cases
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
2. `DocumentProcessor` splits PDF or Markdown content into ordered sections with source, section, page, and sequence metadata.
3. `ingest_document()` in `core/data_ingestion.py` asks `LLMService` for an AOT result: main concepts, a learning roadmap, and a small knowledge graph.
4. The graph is stored in Neo4j; roadmap steps and full parent sections are stored in `research_curriculum`; exactly five hypothetical question children are stored in `research_questions`.
5. Each child question keeps its `parent_id`, so retrieval can resolve the complete original section.

### 3. Ask a paper question

1. `RuntimeEngine.ask()` calls `QdrantVectorStore.search_candidates_and_fetch_parent()`.
2. The query embedding searches the top five hypothetical questions, optionally filtered to one paper.
3. `LLMService.rerank_candidate_questions()` selects one or two valid parent IDs; invalid model output falls back to the leading vector candidate.
4. The engine reads 1–2 hop concept context from Neo4j and sends the parent text plus graph context to `LLMService.answer()`.
5. The UI displays the answer, stored source labels, and optional graph context.

### 4. Teach a section

1. The user selects a paper and section in the Teach tab.
2. `RuntimeEngine.teach_section()` fetches the full stored section and its AOT roadmap.
3. For each roadmap step, it fetches one-hop prerequisite context and asks `LLMService.teach_step()` for a grounded lesson.
4. Streamlit renders each lesson in roadmap order.

### 5. Inspect the graph and evaluate retrieval

- The Graph tab queries `Neo4jManager.get_visual_graph(locator)` and renders concepts and relationships as tables.
- `evaluate.py` runs three retrieval views over `data/eval.json`: parent-section vector baseline, hypothetical-question retrieval plus rerank, and the same retrieval with reported graph-context size. It writes `eval_results.json` with Recall@5 and MRR.

## Architecture

```text
PDF / Markdown
  → DocumentProcessor
  → AOT extraction (OpenAI)
  → Neo4j Concept graph + Qdrant parent / roadmap / question points

Question
  → OpenAI embedding
  → Qdrant question search
  → OpenAI rerank
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
| Qdrant | Parent sections, roadmap steps, and hypothetical question retrieval | `database/structural_db.py` |
| Neo4j | Shared research concepts and bounded graph context | `database/semantic_dag.py` |
| `data/papers/` | Uploaded/local paper files | `main.py` |
| `data/eval.json` | Evaluation cases | `evaluate.py` |

## Configuration

Copy `.env.example` to `.env`; never commit `.env` or its values.

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_BASE_URL` | No | OpenAI-compatible `/v1` base URL; defaults to `https://api.shopaikey.com/v1`. |
| `OPENAI_API_KEY` | Yes for app and evaluation | Credential passed to the configured provider through the OpenAI SDK. |
| `OPENAI_MODEL` | No | Text model; defaults to `gpt-4o-mini`. |
| `OPENAI_EMBEDDING_MODEL` | No | Embedding model; defaults to `text-embedding-3-small`. |
| `OPENAI_EMBEDDING_DIM` | No | Qdrant vector size; defaults to `1536`. It must match the configured embedding model. |
| `QDRANT_URL` | No | Qdrant endpoint; defaults to `http://localhost:6333`. |
| `NEO4J_URI` | No | Neo4j Bolt endpoint; defaults to `bolt://localhost:7687`. |
| `NEO4J_USER` | No | Neo4j user; defaults to `neo4j`. |
| `NEO4J_PASSWORD` | Yes | Local Neo4j password, created/recovered and verified by `setup_env.py`. |

## Setup

1. Create and activate a virtual environment.

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env`, then set `OPENAI_API_KEY`.

   The configured provider must support both the OpenAI Responses (`/v1/responses`) and Embeddings (`/v1/embeddings`) contracts used by this app. Changing any `OPENAI_*` value requires stopping and fully restarting Streamlit; refreshing the browser is not enough. A process-level `OPENAI_*` variable takes precedence over `.env`.

   Before ingesting a paper, run the opt-in compatibility checks. Each makes one provider request and does not write to Qdrant or Neo4j:

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

Upload a PDF, `.md`, or `.markdown` file in the sidebar. Ingestion uses the configured OpenAI-compatible endpoint and writes the compiled material into the local Qdrant and Neo4j services.

## Testing and Validation

Run the behavior-focused suite:

```powershell
python -m pytest -q
```

Run the evaluation against the ingested evaluation corpus:

```powershell
python evaluate.py --workers 4 --output eval_results.json
```

The evaluation requires reachable Qdrant and Neo4j services, a configured OpenAI key, and the sections referenced by `data/eval.json` to be ingested.

## Development Notes for AI Agents

- Read `RAG_Research_Helper_Simple_Reference_First_Rebuild_Plan.md` before changing behavior; it intentionally forbids extra architecture.
- Before changing a major module, inspect the matching `rag-expert-mentor` reference file listed in the plan. Adapt the small relevant pattern; do not copy unrelated features wholesale.
- Preserve the direct flow: `RuntimeEngine` calls Qdrant, Neo4j, and `LLMService` directly. Do not introduce repositories, services, adapters, factories, dependency injection, LangChain, or LangGraph.
- Keep JSON-producing compatible-provider calls in `LLMService` in JSON mode. The rerank call has a bounded token allowance so it can return its required JSON selection.
- Keep `.env` private. Before any Neo4j-dependent change, verify the existing database safely; never delete/reset data or volumes automatically.
- When changing ingestion, coordinate `core/data_ingestion.py`, `database/structural_db.py`, `database/semantic_dag.py`, and their focused tests. Preserve the parent-child `parent_id` contract and exactly five stored hypothetical questions per section.
- When changing retrieval behavior, validate `tests/test_qdrant.py`, `tests/test_runtime.py`, and `evaluate.py`; test a real Ask path when configured services are available.
- Ignore generated caches such as `__pycache__/` and `.pytest_cache/`. Treat `eval_results.json` as a generated evaluation artifact.

## Known Gaps or Deliberate Limits

- PDF extraction is best-effort `pypdf` text extraction; there is no OCR, layout model, or document-intelligence pipeline.
- The graph view is a table/list rather than an interactive network visualization, matching the plan’s stated first-version fallback.
- The app depends on local Qdrant and Neo4j plus a reachable OpenAI-compatible endpoint; it has no offline fallback.
- AOT extraction quality and answer quality remain model-dependent. The application validates JSON structure and keeps a vector-order fallback only for invalid rerank output.

## Reference Attribution

This project uses [rag-expert-mentor](https://github.com/phongka79d/rag-expert-mentor) as a file-by-file architecture and implementation reference. Relevant patterns were adapted for research-paper AOT ingestion, Qdrant parent-child retrieval, Neo4j concept traversal, and direct runtime orchestration. The project does not copy unrelated product features wholesale. Consult the reference repository’s license and author permissions before reusing substantial source verbatim.
