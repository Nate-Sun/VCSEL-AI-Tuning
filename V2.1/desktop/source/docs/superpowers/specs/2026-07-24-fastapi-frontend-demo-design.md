# FastAPI Frontend Demo Design

## Goal

Run the existing VCSEL AI Tuning React dashboard and FastAPI API from one local process so the complete read-only demo is reachable from a single URL.

## Scope

- Track the existing `frontend/` project as source code.
- Build the frontend with Vite into `frontend/dist`.
- Let FastAPI serve files from that build directory.
- Return `index.html` for browser routes such as `/campaigns/<id>` so React Router works after a refresh.
- Preserve `/health`, `/capabilities`, `/api/*`, `/docs`, and `/openapi.json` as backend routes.
- Keep the existing read-only UI and hardware-safety boundaries.
- Do not change `PATH_HIERARCHICAL`, optimizer selection, hardware adapters, or campaign behavior.

## Runtime Behavior

FastAPI discovers `frontend/dist` relative to the repository root. `VCSEL_AI_FRONTEND_DIST` may override the directory for tests or an alternate local build. When no build exists and no explicit override is configured, FastAPI remains API-only so backend development and tests do not require Node.js.

When a valid build exists:

1. Existing API routes are registered first.
2. A final GET catch-all serves real frontend files when they exist.
3. Other non-API GET routes return `index.html`.
4. Unknown `/api/*` requests remain JSON 404 responses and are never converted into HTML.

## Verification

- A backend test proves root, static asset, SPA fallback, API precedence, and unknown API behavior.
- Frontend unit tests, type checking, linting, and production build pass.
- Focused API tests pass without running the actively modified hardware backend tests.
- A real FastAPI process answers `/`, `/health`, and a nested React route from one port.
