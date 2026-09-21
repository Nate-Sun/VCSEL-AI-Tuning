# Domain documentation

This repository uses a single-context domain layout.

## Discovery

Before domain-sensitive exploration:

1. Read root `CONTEXT.md` when it exists.
2. Read relevant decisions under `docs/adr/` when they exist.
3. If either location is absent, continue without creating placeholder documentation.

## Vocabulary and decisions

- Use the terms defined in `CONTEXT.md` in specifications, tickets, tests, and code.
- Treat an undefined concept as either a vocabulary gap to resolve or a sign that the proposed term does not belong in the model.
- Surface conflicts with an existing ADR explicitly; do not silently replace the recorded decision.
- Create or update domain documentation only when the work resolves a real term, invariant, boundary, or architectural decision.
