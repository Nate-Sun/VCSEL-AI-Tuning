# Issue tracker: Local Markdown

Specifications and implementation issues for this repository live under `.scratch/` and remain local unless the user explicitly requests publication elsewhere.

## Layout

- One effort per directory: `.scratch/<effort>/`.
- The approved specification is `.scratch/<effort>/spec.md`.
- The effort map is `.scratch/<effort>/map.md`.
- Each implementation ticket is a separate file at `.scratch/<effort>/issues/<NN>-<slug>.md`, numbered from `01`.
- Record workflow state with a `Status:` line near the top of each ticket.
- Append discussion under `## Comments`; preserve prior entries.

## Skill operations

- **Publish:** create or update the appropriate Markdown file under `.scratch/<effort>/`.
- **Fetch:** read the referenced local ticket or specification directly.
- **Claim:** change an unblocked ticket to `Status: claimed` before implementation.
- **Resolve:** add an `## Answer`, set `Status: resolved`, and add a concise decision pointer to `map.md`.
- **Block:** add `Blocked by: NN, NN`; a ticket becomes available only after every listed dependency is resolved.

Do not create external issues, pull requests, or tracker records without explicit user authorization.
