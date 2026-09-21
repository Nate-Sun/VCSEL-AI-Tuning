# Native V1.2 staged LIVE capability map

Approved implementation tickets for the `PILOT_3_THEN_FAST24` effort.

| Ticket | Status | Blocked by | Outcome |
| --- | --- | --- | --- |
| 01 | resolved | None | Immutable staged policy and API contract (`c040d28`) |
| 02 | resolved | 01 | Sequential planner and hardware-independent engine (`8d83ea7`, `906f9e1`) |
| 03 | resolved | 01 | Killable DCA isolation and isolated LIVE Sweep (`31b9bca`) |
| 04 | resolved | 02, 03 | Connection-free LIVE I2C/FIR plant (`9e0b0e3`) |
| 05 | claimed | 04 | PILOT safety gate, FAST24, authorization, and recovery |
| 06 | ready-for-agent | 05 | Native V1.2 runtime and truthful UI |
| 07 | ready-for-agent | 06 | Complete fake integration regression and audit |

Frontier: ticket 05 (claimed).

Resolved: ticket 01 — immutable staged policy and API schema, `c040d28`.

Resolved: ticket 02 — deterministic sequential PILOT planner and hardware-neutral executor, `8d83ea7`, `906f9e1`.

Resolved: ticket 03 — spawned DCA isolation, durable SCPI evidence, and isolated LIVE Sweep, `31b9bca`.

Resolved: ticket 04 — connection-lazy LIVE I2C/FIR plants and persisted-B0 recovery, `9e0b0e3`.
