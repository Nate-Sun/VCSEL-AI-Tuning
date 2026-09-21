# V2.1 LIVE Preflight Bootstrap Design

**Status:** Approved by the hardware owner on 2026-08-12

## Goal

Allow a new V2.1 continuous-eye Campaign to perform the already-approved LIVE
read-only preflight without requiring a pre-existing mutating-stage
authorization. The preflight may contact DCA, I2C, and Kimi through the V2.1
program, but it must not write DUT parameters or contact BERT.

## Root cause

The original LIVE composition reused `ContinuousRequest` for both provider
preview and commissioning. `ContinuousRequest` requires `StageAuthorization`,
and the LIVE assembler also expected the Campaign to be in an active I2C stage
before capturing the initial eye. A new Campaign cannot satisfy either
condition until after preflight and operator approval, creating a circular
startup dependency.

The generic laboratory driver profile also intentionally has no
`safety_limits`. V2.1 continuous optimization needs its own reviewed profile
containing the nine approved encoding ranges and 15 percent per-round steps.

## Design

### Campaign bootstrap

The LIVE preflight creates or idempotently reuses one Campaign record using the
configured B0 hash, V2.1 driver-profile hash, and the diagnostic I2C-only tuning
path. Creating this record is local database work and does not construct device
adapters.

### I2C-only preflight capture

A dedicated orchestrator operation performs this sequence:

1. Require `DRAFT` or the post-preflight waiting state.
2. Query and verify DCA identity and measurement bindings.
3. Query and verify I2C device identity.
4. Read the current nine I2C fields.
5. Acquire exactly one fresh DCA observation and screen PNG.
6. Persist the observation, PNG, derived NPY, hashes, and raw read-only evidence.
7. Move a new Campaign from `DRAFT` through `PREFLIGHT` to the waiting state.

The operation never calls BERT, FIR, NLC, `setFir`, I2C `apply`, rollback, or
static B0 restore.

### Provider preview

Introduce a preflight request model containing only the captured `EyeSample`,
verified readback, V2.1 parameter policy, round count, and stopping policy. It
does not contain `StageAuthorization`.

The existing controller's provider-preview interface accepts this request,
builds the same bounded `ProviderContext`, calls Kimi through the existing
`KimiProvider`, validates the structured response, and returns the action. A
`PROPOSE` is recorded only as a preview; no candidate is materialized or
executed during preflight.

### Commissioning gate

`ContinuousRequest` and every execution method continue to require a matching
`StageAuthorization`. The CLI continues to require the separate commissioning
authorization token. This change does not create, approve, or execute a
commissioning authorization.

After the preflight report is reviewed, commissioning bootstrap and its exact
write actions will be presented to the hardware owner for separate approval.

### V2.1 driver and runtime configuration

Add a V2.1-only hardware profile with all nine encoding ranges and these
per-round maximum steps:

| Parameter | Range | Max step |
| --- | ---: | ---: |
| `ibias_dac` | 0..127 | 20 |
| `gain_ctrl` | 0..63 | 10 |
| `eq` | 0..31 | 5 |
| `efin_cur` | 0..15 | 3 |
| `vga_cur` | 0..15 | 3 |
| `res_gain` | 0..15 | 3 |
| `res_bw` | 0..15 | 3 |
| `vbias_ref` | 0..15 | 3 |
| `vref` | 0..15 | 3 |

The V2.1 runtime uses isolated SQLite and artifact locations. Its resource lease
contains DCA and I2C only. BERT endpoint fields may remain present for legacy
configuration-schema compatibility, but continuous assembly must not create a
BERT client.

## Verification

- Unit tests prove new-Campaign bootstrap is local-only until the explicit
  preflight call.
- Tests prove preflight does not need `StageAuthorization`.
- Tripwire adapters prove no I2C write, BERT call, rollback, or B0 restore.
- Tests prove provider preview never invokes the execution port.
- Configuration tests verify all nine ranges and steps exactly.
- The focused and full non-hardware suites pass before any real preflight.

## Out of scope

- Real device contact during implementation or automated verification.
- Any DUT write or three-round commissioning execution.
- FIR, NLC, BERT, and `setFir`.
- A new HTTP `/api/v2` interface.
