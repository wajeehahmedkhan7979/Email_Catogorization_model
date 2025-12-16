# Email Categorization Worker (Azure-friendly)

Low-cost, Azure-ready pipeline to ingest email JSON blobs from Storage Queue/Blob, validate via Pydantic, clean/merge threads, embed with local sentence-transformers, match to a human-approved taxonomy, and run a lightweight intent classifier (Level 3). Outputs enriched JSON to Blob with monitoring hooks and optional Azure OpenAI fallback for low-confidence audits.

## Structure

```
/src
  /config         # env loader, Key Vault helper
  /models         # Pydantic schemas
  /services       # blobs, queues, AI, preprocessing, taxonomy
  /workers        # worker entrypoint
/scripts          # training utilities
/tests            # unit tests
requirements.txt
version.py
.github/workflows/deploy.yml
```

## Quickstart (local dev)

1) Create and activate a Python 3.10 virtualenv.  
2) `pip install -r requirements.txt`  
3) Copy `.env.example` to `.env` (or export vars) and fill required settings.  
4) Run tests: `pytest`.  
5) Start worker locally: `python -m src.workers.main_worker`.

## Core environment variables

- `INPUT_CONTAINER`, `OUTPUT_CONTAINER`, `POISON_CONTAINER`, `QUEUE_NAME`
- `MODEL_VERSION`, `KEY_VAULT_URL`, `APP_INSIGHTS_INSTRUMENTATIONKEY`
- `ALLOWED_LANGUAGES` (e.g. `["en","es"]`)
- `SPAM_KEYWORDS_PATH` (blob or local path)

## CI/CD

GitHub Actions workflow runs lint (`flake8`), format check (`black --check`), tests (`pytest`), and (placeholder) golden-dataset evaluation before deploy. Fill in Azure credentials and app name to activate deployment.

