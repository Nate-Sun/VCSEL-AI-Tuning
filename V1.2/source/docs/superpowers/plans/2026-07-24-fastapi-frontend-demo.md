# FastAPI Frontend Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the existing React dashboard and FastAPI API from one local process for the fastest complete demo.

**Architecture:** Build React into `frontend/dist`, then register a final FastAPI GET catch-all after all API routers. The catch-all serves existing files, falls back to `index.html` for React routes, and preserves JSON 404 behavior for unknown API paths.

**Tech Stack:** Python 3.12+, FastAPI, Starlette responses, React 18, TypeScript, Vite, pytest, Vitest

---

### Task 1: Add static frontend regression coverage

**Files:**
- Create: `tests_py/ai_tuning/api/test_frontend_hosting.py`
- Modify: `src/vcsel_ai_tuning/api/app.py`

- [ ] **Step 1: Write the failing test**

Create a temporary Vite-style build containing `index.html` and `assets/app.js`. Construct the API with that directory and assert that `/`, `/assets/app.js`, and `/campaigns/demo` return frontend content, while `/health` remains JSON and `/api/not-a-route` remains a JSON 404.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv-ai-tuning\Scripts\python.exe -m pytest tests_py/ai_tuning/api/test_frontend_hosting.py -q`

Expected: FAIL because `create_app` does not accept or serve `frontend_dist`.

- [ ] **Step 3: Write the minimal implementation**

Add a repository-relative build-directory resolver, an optional `frontend_dist` argument to `create_app`, and a final GET catch-all that serves safe files or the SPA index. Reject unknown `api` paths with HTTP 404.

- [ ] **Step 4: Run the focused test**

Run: `.venv-ai-tuning\Scripts\python.exe -m pytest tests_py/ai_tuning/api/test_frontend_hosting.py tests_py/ai_tuning/api/test_routes.py tests_py/ai_tuning/api/test_runtime_routes.py -q`

Expected: all selected tests pass.

### Task 2: Document and verify the one-process demo

**Files:**
- Modify: `frontend/README.md`
- Track: `frontend/`

- [ ] **Step 1: Add exact demo commands**

Document `pnpm build` followed by `vcsel-ai-api`, with `http://127.0.0.1:8000/` as the single demo URL and FAKE runtime as the no-configuration default.

- [ ] **Step 2: Verify the frontend**

Run: `pnpm test`, `pnpm lint`, and `pnpm build` in `frontend/`.

Expected: all commands exit 0.

- [ ] **Step 3: Verify one-process serving**

Start `vcsel-ai-api`, request `/`, `/health`, and `/campaigns/demo`, then stop the process.

Expected: frontend HTML from `/` and `/campaigns/demo`; JSON runtime status from `/health`.

### Task 3: Create a clean commit

**Files:**
- Add only: `frontend/`, `src/vcsel_ai_tuning/api/app.py`, `tests_py/ai_tuning/api/test_frontend_hosting.py`, and the two FastAPI demo documents.

- [ ] **Step 1: Review the staged diff**

Run `git diff --cached --check` and `git diff --cached --stat`. Confirm that no hardware backend file, root `.gitignore`, generated `frontend/dist`, `node_modules`, or test output is staged.

- [ ] **Step 2: Commit**

Commit message: `feat: serve AI tuning dashboard from FastAPI`
