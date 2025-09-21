## Development

After making changes, always run `make checku` and resolve any issues that come up.

### Quick Commands
 - `make help` see available commands
 - `make autoformat` format code
 - `make autoformat-unsafe` format code - including 'unsafe' fixes
 - `make lint` run linter
 - `make typecheck` run type checker
 - `make test` run tests
 - `make coverage` run tests with coverage report
 - `make check` run all checks (format, lint, typecheck, test)
 - `make checku` run all checks  (format-unsafe, lint, typecheck, test)

### Code Conventions

#### Testing
- Use **pytest** (no test classes).
- Always set `match=` in `pytest.raises`.
- Prefer `monkeypatch` over other mocks.
- Mirror the source-tree layout in `tests/`.
- Always run `make checku` after making changes.

#### Exceptions
- Catch only specific exceptions—never blanket `except:` blocks.
- Don’t raise bare `Exception`.

#### Python
- Manage env/deps with **uv** (`uv add|remove`, `uv run -- …`).
- No logging config or side-effects at import time.
- Keep interfaces (CLI, web, etc.) thin; put logic elsewhere.
- Use `typer` for CLI interfaces, `fastapi` for web interfaces, and `pydantic` for data models.

### Frontend (Astro)
- The `frontend/` app is built with **Astro** and is pre-rendered at build time (SSG) for every route. Server-side code inside `.astro` files runs once during the build, *not* in the browser.
- **URL query parameters are *not* available to Astro at runtime.** Conditional sections that depend on `window.location.search` (e.g., `?details=1`) must be handled in a client island (React/Vue/solid) or by emitting all HTML and hiding/showing via client JS.
- Do *not* rely on `Astro.url.searchParams` for runtime interactivity—it only reflects the build-time URL.
- When you need query-driven interactivity:
  1. Render the section unconditionally (or behind `client:only`/`client:load` logic).
  2. In a client component, read `window.location.search` and toggle visibility.
- If you really need SSR with dynamic params, switch the page to `prerender: false` and use serverless/SSR deployment.
