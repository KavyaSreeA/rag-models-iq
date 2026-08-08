# NovaTech Operations Console

A multi-agent "digital employee assistant" built with **LangChain** +
**LangGraph**, served by **Groq** (Llama 3.3), and deployed with
**Streamlit** in a custom dark "ops console" theme. Instead of one chatbot,
a Coordinator Agent classifies each employee request and routes it to a
specialized agent (HR, Research, Email, Document, Python) or to a
sequential/parallel multi-agent workflow.

## Architecture

```
Employee
  ↓
Streamlit UI (app.py)
  ↓
Coordinator Agent (agents/coordinator.py)  — RunnableBranch conditional routing
  ↓
 ┌────────────┬─────────────────┬─────────────┬──────────────┬────────────┐
 HR Agent   Research Agent   Email Agent   Document Agent   Python Tool Agent
  ↓              ↓                ↓              ↓
Sequential Workflow (Research → Summarize → Email → Send → Store)
Parallel Workflow (RunnableParallel fan-out across N companies)
  ↓
Shared Memory Layer
  short-term (per-session) · SQLite (persistent + long-term profile)
  ↓
Company Knowledge Base — ChromaDB (`hr_policies`, `company_docs` collections)
```

| Handbook Module | File(s) |
|---|---|
| 1. Employee Chat Interface | [app.py](app.py) |
| 2. HR Agent | [agents/hr_agent.py](agents/hr_agent.py) |
| 3. Research Agent | [agents/research_agent.py](agents/research_agent.py), [tools/search_tools.py](tools/search_tools.py) |
| 4. Email Agent | [agents/email_agent.py](agents/email_agent.py), [tools/gmail_tools.py](tools/gmail_tools.py) |
| 5. Document Agent | [agents/document_agent.py](agents/document_agent.py) |
| 6. Coordinator Agent | [agents/coordinator.py](agents/coordinator.py) |
| 7. Sequential Workflow | [workflows/sequential.py](workflows/sequential.py) |
| 8. Parallel Processing | [workflows/parallel.py](workflows/parallel.py) |
| 9. Conditional Routing | `RunnableBranch` in [agents/coordinator.py](agents/coordinator.py) |
| 10. Company Knowledge Base | [knowledge_base/ingest.py](knowledge_base/ingest.py), [knowledge_base/retriever.py](knowledge_base/retriever.py) |
| 11. Memory | [memory/](memory/) (short_term, persistent, long_term) |
| 12. Structured Outputs | [schemas/structured_output.py](schemas/structured_output.py) |
| 13. Python Tool | [tools/python_tool.py](tools/python_tool.py), [agents/python_tool_agent.py](agents/python_tool_agent.py) |
| 14. Google Drive (Optional) | [tools/drive_tools.py](tools/drive_tools.py) |

## Setup

1. **Clone/open this folder**, create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**

   ```bash
   copy .env.example .env
   ```

   Then open `.env` and set at minimum:

   - `GROQ_API_KEY` — required (default provider). Get one at https://console.groq.com/keys
   - `HUGGINGFACE_API_TOKEN` — required (default embeddings provider). Free
     token at https://huggingface.co/settings/tokens. Embeddings default to
     HF's hosted Inference API (`EMBEDDING_PROVIDER=huggingface_api`) — a
     lightweight API call with no heavy local ML dependencies, which matters
     a lot on resource-limited free hosting.

   Everything else is optional (the app runs fully without them, using local
   stubs — see below):

   - `LLM_PROVIDER=openai` + `OPENAI_API_KEY` — switch the chat model to
     OpenAI instead of Groq.
   - `EMBEDDING_PROVIDER=huggingface_local` — run the embedding model
     locally instead of via the API (no token needed, but pulls in
     `torch`/`transformers`/`sentence-transformers`, ~1-2GB, and is much
     slower to install and cold-boot — not recommended on free hosting).
   - `TAVILY_API_KEY` — better web search for the Research Agent. Without it,
     the app uses free DuckDuckGo + Wikipedia search.
   - `GOOGLE_CREDENTIALS_PATH` (default `credentials.json`) — for real Gmail
     sending and Google Drive uploads. Without a real `credentials.json` file
     present, the Email Agent writes to a local `data/outbox.jsonl` file and
     the Drive tools write to `data/drive_storage/` instead of crashing.

4. **(Optional) Enable real Gmail/Drive:**

   - Go to https://console.cloud.google.com/ → create a project.
   - Enable the **Gmail API** (and **Google Drive API** if you want Module 14).
   - Create OAuth 2.0 credentials of type **Desktop app**, download the JSON,
     save it as `credentials.json` in this project's root folder.
   - First real Gmail/Drive call opens a browser window to complete the OAuth
     consent flow once; a `token.json`/`drive_token.json` is cached after that.

5. **Build the sample knowledge base and run the app:**

   ```bash
   streamlit run app.py
   ```

   In the sidebar, click **"(Re)build knowledge base from sample_docs"** once
   to index the included sample HR policies and company documents into
   ChromaDB. You can also upload your own PDFs/TXT files from the sidebar.

## Running tests

```bash
pytest tests/ -v
```

Tests are written to run **without** a live API key — memory, schema
validation, routing logic, and workflow helper functions are tested directly
or with LLM calls monkeypatched out. A few integration paths (actual agent
`.invoke()` calls) require a real `GROQ_API_KEY` (or `OPENAI_API_KEY`) and are
exercised manually per the scenarios below rather than in CI.

## Example scenarios to try

- `What is our work from home policy?` → **HR Agent**, cites the WFH Rules doc.
- `Research Google's AI products and compare them with Microsoft.` →
  **Research Agent** (or **Parallel workflow** if phrased as multiple
  distinct subjects, e.g. "Research Google, Microsoft, Amazon, and OpenAI").
- `Draft an email requesting three days leave.` → **Email Agent** (drafts only).
- `Summarize the employee handbook and email it to HR.` → **Sequential
  workflow**: Document Agent → Summarize → Generate Email → Send → Store.
- `Calculate employee attendance percentage for 18 out of 20 working days.` →
  **Python Tool Agent**.
- `Remember that I belong to the Finance Department.` → **long-term memory**
  write.
- `What department do I belong to?` → **long-term memory** recall.

## Project layout

See [`agents/`](agents/), [`tools/`](tools/), [`knowledge_base/`](knowledge_base/),
[`memory/`](memory/), [`workflows/`](workflows/), and [`schemas/`](schemas/) for
the module-by-module implementation, and [`data/sample_docs/`](data/sample_docs/)
for the seed HR policy and company documents used to build the demo knowledge
base.

## Deploying to Hugging Face Spaces

The code is deploy-ready; creating and connecting the Space is a manual step
that needs your Hugging Face account (I don't hold credentials for you):

1. Go to https://huggingface.co/new-space, choose SDK **Streamlit**, and
   create the Space (e.g. `novatech-operations-console`).
2. In the new Space's **Settings -> Variables and secrets**, add secrets
   named `GROQ_API_KEY` and `HUGGINGFACE_API_TOKEN`. Optionally add
   `LLM_PROVIDER`, `EMBEDDING_PROVIDER`, etc. if you want non-default values.
   Note: on Hugging Face Spaces specifically, a Space-scoped token may
   already grant Inference API access without a separate secret — check the
   Space's own token permissions before assuming you need a second one.
3. Push this repo's code to the Space's git remote (shown on the Space page,
   looks like `https://huggingface.co/spaces/<you>/<space-name>`):

   ```bash
   git remote add space https://huggingface.co/spaces/<you>/<space-name>
   git push space main
   ```

4. Replace the Space's `README.md` with the contents of
   [HF_SPACE_README.md](HF_SPACE_README.md) (it carries the YAML front matter
   Spaces needs to detect the Streamlit SDK/entrypoint) — either edit it in
   the Space's file UI, or locally: `cp HF_SPACE_README.md README.md` on a
   branch pushed to `space` only, so the GitHub repo keeps its own README.
5. The Space will build and boot automatically — no heavy local model
   download needed since embeddings default to the hosted Inference API.

## Notes / known limitations

- Default LLM provider is Groq (`llama-3.3-70b-versatile`); switch to OpenAI
  or change the model via `.env`.
- Embeddings default to the Hugging Face hosted Inference API since Groq has
  no embeddings endpoint of its own; switch via `EMBEDDING_PROVIDER` to
  `openai` or `huggingface_local`.
- Web search defaults to free DuckDuckGo + Wikipedia (no key required);
  results are noisier than a paid provider like Tavily.
- Gmail "send" only sends real email once you've completed the Google OAuth
  setup above — otherwise it safely no-ops into a local outbox file.
- ChromaDB is persisted to `data/chroma_db/` (gitignored) — delete that
  folder to force a clean rebuild.
