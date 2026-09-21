# 05 — Orchestrate the PILOT gate, FAST24, and recovery

**What to build:** One pre-authorized campaign runs three safe PILOT observations, crosses a durable safety gate, automatically executes the 24-round FAST24 sequence, and either finishes at verified B0 or blocks further LIVE work in recovery.

**Blocked by:** 04 — Build the connection-free LIVE I2C/FIR plant.

**Status:** resolved

- [x] The PILOT gate requires three valid durable observations, complete Dataset evidence, unchanged identities, zero DCA faults, every per-candidate restore, and a fresh final B0 verification.
- [x] No FAST24 write occurs until the passing gate event is durably stored.
- [x] FAST24 executes exactly 6 coarse I2C, 16 fine I2C, 1 first formula FIR, and 1 iterative formula FIR round, retaining candidate state between rounds.
- [x] Authorization freezes all 27 rounds and revalidates the same policy at create, authorize, prepare, and start, including post-attestation expiry.
- [x] Stop or failure skips every unstarted round and performs one deterministic restoration attempt.
- [x] Failed restoration or identity drift enters `RECOVERY_REQUIRED`, preserves exact evidence, and blocks all later LIVE tasks.
