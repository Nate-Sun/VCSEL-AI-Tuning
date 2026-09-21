# 01 — Freeze the staged policy and API contract

**What to build:** Native V1.2 accepts one immutable `PILOT_3_THEN_FAST24` authorization shape whose stage budgets, parameter grids, Dataset choice, restore policy, and hardware authority are visible and validated before execution.

**Blocked by:** None — can start immediately.

**Status:** resolved

- [x] The profile fixes PILOT_3 at 3 candidates and FAST24 at 6 coarse I2C, 16 fine I2C, 1 first formula FIR, and 1 iterative formula FIR candidate.
- [x] PILOT_3 permits only `ibias_dac={69,71,73}` and `res_gain={4,8}`; FAST24 permits only `ibias_dac=60..80 step 1` and `res_gain=0..10 step 1`.
- [x] Dataset saving remains a strict boolean product option, while the formal acceptance profile requires it to be enabled.
- [x] Caller-supplied FIR taps, NLC, pattern, changed budgets, changed ranges, invalid settle time, and unknown staged fields fail closed before execution construction.
- [x] The compatibility API round-trips the exact staged authority without changing existing FAKE and READ_ONLY request shapes.
- [x] Focused policy and API schema tests demonstrate RED then GREEN with no hardware import or access.

## Answer

Implemented by `c040d28`. RED failed on the missing policy module and scope type; GREEN and compatibility verification finished with 183 passed. The policy module is hardware-neutral, the staged schema serializes exact authority, and legacy request serialization omits every new field.
