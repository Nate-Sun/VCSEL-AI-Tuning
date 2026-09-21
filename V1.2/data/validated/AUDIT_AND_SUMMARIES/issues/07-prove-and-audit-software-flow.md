# 07 — Prove and audit the complete software flow

**What to build:** A complete non-hardware acceptance run proves the real API, authorization, persistence, runner, staged executor, fake devices, spawned fake DCA worker, Dataset bundles, UI, and recovery semantics as one Native V1.2 system.

**Blocked by:** 06 — Wire the Native V1.2 runtime and truthful UI.

**Status:** resolved

- [x] A fake end-to-end run produces exactly 27 ordered candidate observations and completes at exact verified B0.
- [x] Every observation has requested/applied/readback equality, TDECQ status, ER/OMA evidence, measurement/acquisition IDs, and a complete Dataset bundle.
- [x] Every NPY loads, both PNGs decode, every SHA-256 matches, and database records match the corresponding files.
- [x] Invalid DCA, timeout, Dataset failure, stop, expired authorization, exact-readback mismatch, and restore failure stop later stages and retain durable evidence.
- [x] Complete Python, frontend test, build, lint, ValidateOnly, diff, lineage, and worktree-cleanliness checks pass without real hardware access.
- [x] The validation report records exact branch, Worktree, HEAD, source commit, test counts, and labels all real hardware capabilities `UNVERIFIED` until the separate hardware gates run.
