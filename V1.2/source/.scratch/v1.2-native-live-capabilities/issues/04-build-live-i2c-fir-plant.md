# 04 — Build the connection-free LIVE I2C/FIR plant

**What to build:** An authorized Native V1.2 execution can apply I2C or formula-generated FIR candidates, verify exact device readback, measure through isolated DCA, and restore the frozen B0 without opening hardware during construction.

**Blocked by:** 02 — Add the sequential planner and execution engine; 03 — Isolate DCA I/O and LIVE Sweep.

**Status:** resolved

- [x] Factory and executor construction import no optional hardware stack, open no connection, and write no artifact.
- [x] I2C writes are limited to the frozen staged ranges and require exact requested/applied/readback equality.
- [x] Formula FIR accepts seven integers with L1 at most 170 while Main, NLC, and pattern remain exactly at B0.
- [x] Non-Main taps have no B0 ±3 restriction; invalid formula output is rejected rather than clipped.
- [x] No rejected FIR candidate issues `setFir`, and every accepted FIR requires exact BERT readback before measurement.
- [x] Terminal restoration verifies I2C and BERT; one failed restore raises recovery-required evidence without an unbounded retry.

## Answer

Implemented in `9e0b0e3`. The inert factory opens drivers only inside run/recovery, PILOT exposes no FIR mutation, FAST24 accepts only formula FIR satisfying frozen Main/NLC/pattern and L1 <= 170, and all accepted I2C/FIR writes require exact readback. Restart recovery is anchored to persisted B0 rather than constructor-time drift. V1.1 module regression: 432 passed.
