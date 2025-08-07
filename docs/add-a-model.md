### Adding a new model

When a new provider model is added (e.g., OpenAI GPT-5), update all three layers so it works end-to-end:

- **Provider model catalog (LiteLLM registration)**
  - Prefer: run the scraper to refresh bundled provider data.
    - Command: `make update-models` (updates files under `bulkllm/model_registration/data/`, e.g. `openai.json`, `openai_detailed.json`).
  - If needed, add a manual registration in `bulkllm/model_registration/main.py` under `manual_model_registrations` so the model is available even if fetching fails. Include:
    - `litellm_provider`, `mode` (usually `chat`), token limits, and per-token costs when known.

- **App-visible configs (`bulkllm/llm_configs.py`)**
  - Add an `LLMConfig` for each new variant. Follow existing patterns for the family:
    - `slug`: kebab-case, ending with release date `YYYYMMDD` (or `-latest` for alias models).
    - `display_name`: human-friendly label shown in UI.
    - `company_name`: provider (e.g., `OpenAI`).
    - `litellm_model_name`: full provider-prefixed name (e.g., `openai/gpt-5-2025-08-07`).
    - `llm_family`: family without the dated suffix (e.g., `openai/gpt-5`). Used to group snapshots.
    - `temperature`, `max_tokens` or `max_completion_tokens` per model type, `thinking_config`, `system_prompt`, `release_date`.
    - Set `is_reasoning=True` where applicable.
  - If the model supersedes an older family, update `FAMILY_SUCCESSORS` so `current_model_configs()` selects the right snapshot.

- **Rate limits (`bulkllm/rate_limits.py`)**
  - Add a `ModelRateLimit` group for the family and snapshots. Use provider docs or mirrored limits from similar families.
  - Include aliases like `...-latest` where applicable.

### Sanity checklist
- Models appear in `model_info()` output with costs and rate limits.
- `pytest` passes.
- `cheap`, `current`, and `reasoning` groups include/exclude new models correctly.

### Tips
- Prefer adding snapshot-specific entries (e.g., `gpt-5-2025-08-07`) and the base family alias if present (e.g., `gpt-5`).
- Keep manual registrations minimal; provider JSON plus detailed JSON should carry most metadata.
