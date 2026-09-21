# 02 — Add the sequential planner and execution engine

**What to build:** Native V1.2 can produce and persist three observation-conditioned PILOT candidates, then drive the existing optimizer contract without any hardware dependency.

**Blocked by:** 01 — Freeze the staged policy and API contract.

**Status:** resolved

- [x] Candidate 2 is selected only after observation 1 is durable, and candidate 3 is selected only after observations 1 and 2 are durable.
- [x] Candidate 3 uses deterministic Bayesian acquisition and the same persisted history replays to the same result.
- [x] TDECQ is the only ranking objective; ER and OMA remain validation/audit evidence and never break a TDECQ tie.
- [x] Apply, exact readback, measurement, observation persistence, and next suggestion occur in that order; Dataset capture remains an injected plant responsibility for ticket 03/04 wiring.
- [x] Invalid DCA observations remain invalid, cannot be reused as numeric metrics, and stop later unsafe stages.
- [x] Cooperative stop, one restoration attempt, final guard, and close behavior are covered by RED/GREEN tests using only fake plants.

## Answer

Implemented by `8d83ea7` and `906f9e1`. The deterministic planner uses durable observation history, and the hardware-neutral executor proves exact readback, persist-before-next-suggestion, invalid-measurement stop, one restore attempt, final guard, and idempotent close. Focused engine tests: 8 passed. Planner, bridge, engine, and repository regression: 203 passed.
