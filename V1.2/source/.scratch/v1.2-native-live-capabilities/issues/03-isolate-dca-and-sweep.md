# 03 — Isolate DCA I/O and LIVE Sweep

**What to build:** Every LIVE DCA preflight, measurement, and Dataset capture runs in a spawned child process with durable command evidence and bounded reclamation, while I2C and BERT ownership remains in the controller process.

**Blocked by:** 01 — Freeze the staged policy and API contract.

**Status:** resolved

- [x] DCA client construction is connection-free and artifact-free.
- [x] Each SCPI operation journals task, item, round, command, result, and last in-flight state durably.
- [x] A 60-second production deadline triggers terminate, a 5-second cleanup grace, then kill if required; the worker is confirmed dead before control returns.
- [x] Timeout, invalid status, close failure, and incomplete evidence return null metrics and never reuse a previous value.
- [x] LIVE Sweep uses the same isolated DCA seam and preserves Native V1.2 same-acquisition Dataset completeness rules.
- [x] Software-only fake workers cover query hangs and lifecycle cleanup; no real device fault injection or hardware connection was used.

## Answer

Implemented in `31b9bca`. DCA VISA ownership now lives only in a request-scoped spawned worker, SCPI STARTED/COMPLETED/FAILED evidence is fsynced, timeout reclamation is bounded, terminal invalid metrics are nulled, and LIVE Sweep transports its same-worker Dataset capture without weakening Native V1.2 completeness checks. Hardware/LIVE/Dataset regression: 395 passed.
