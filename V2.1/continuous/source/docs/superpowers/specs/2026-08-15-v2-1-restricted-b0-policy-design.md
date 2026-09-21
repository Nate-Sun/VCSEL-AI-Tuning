# V2.1 Restricted B0 Parameter Policy Design

**Status:** Approved in the design conversation; awaiting review of this written specification

**Date:** 2026-08-15

**Branch:** `codex/v2.1-continuous-eye-optimization`

**Worktree:** `C:\Users\NateS\Documents\VCSEL\.worktrees\v2-1-continuous-eye-optimization`

**Code baseline:** `89049b22bc3c97fb0cac1cae59e01c84e36f2c8e`

## 1. Goal

Replace the current B0 I2C values with the operator-approved degraded baseline and restrict Kimi to explicit campaign-level I2C tuning ranges. The same restricted policy must be shown to Kimi and enforced locally before any candidate reaches the existing authorization and hardware execution chain.

This change is deliberately narrow. It changes high-level B0 and continuous-campaign policy configuration without changing the I2C register map, I2C adapter, DCA adapter, BERT behavior, FIR behavior, NLC behavior, or `setFir` boundary.

The operator has explicitly chosen not to introduce a separate S0. The new B0 is therefore both the optimization starting point and the static I2C restore target on normal completion or safe failure.

## 2. Existing behavior

The current live campaign YAML does not contain a parameter policy. Instead:

- `DEFAULT_I2C_POLICY` exposes all nine supported I2C fields across their full encoding ranges.
- `continuous_live.py` inserts that policy into `ContinuousRequest`.
- `KimiApiClient` serializes the policy into the structured provider context.
- `I2cParameterPolicy.validate_proposal` checks provider output locally before candidate materialization.
- the existing campaign authorization independently checks campaign, baseline hash, stage, budget, tuning range, maximum step, and latest hardware readback before the write path proceeds.

Changing only the natural-language instruction would be insufficient because it would leave the local pre-execution policy at its original full range.

## 3. Approved B0

`configs/ai_tuning/b0.v2_1.continuous.yaml` is a campaign-specific `version: 2` baseline with these I2C values. The shared `b0.yaml` remains unchanged so older campaign and replay behavior is not silently altered:

| Parameter | B0 value |
|---|---:|
| `ibias_dac` | 60 |
| `efin_cur` | 15 |
| `vga_cur` | 13 |
| `eq` | 16 |
| `res_gain` | 6 |
| `res_bw` | 0 |
| `gain_ctrl` | 2 |
| `vbias_ref` | 15 |
| `vref` | 15 |

The existing FIR taps and NLC values remain unchanged. Continuous mode remains I2C-only and must not write FIR or NLC.

Changing B0 changes the baseline hash. The used campaign `v21-pf-20260814-07` and its authorization must never be reused. A live run requires a new unused campaign ID, a newly bound authorization, and a successful PRE-FLIGHT before COMMISSION can be considered.

## 4. Approved restricted policy

The V2.1 continuous live campaign configuration will contain:

```yaml
parameter_policy:
  version: i2c-restricted-v1
  max_changed_parameters: 3
  rules:
    ibias_dac: {minimum: 60, maximum: 80, max_step: 10}
    efin_cur:  {minimum: 10, maximum: 15, max_step: 3}
    vga_cur:   {minimum: 10, maximum: 15, max_step: 3}
    eq:        {minimum: 16, maximum: 25, max_step: 3}
    res_gain:  {minimum: 0,  maximum: 10, max_step: 3}
    res_bw:    {minimum: 0,  maximum: 5,  max_step: 1}
    gain_ctrl: {minimum: 0,  maximum: 10, max_step: 3}
    vbias_ref: {minimum: 10, maximum: 15, max_step: 3}
    vref:      {minimum: 10, maximum: 15, max_step: 3}
```

Every approved B0 value lies within its tuning range. Each tuning range lies within the existing hardware encoding envelope, and each campaign maximum step is no greater than the existing hardware safety maximum step.

`max_step` is a maximum absolute delta from the latest verified hardware readback. It is not a discrete grid interval. For example, when `ibias_dac` reads back as 60, any absolute integer value from 61 through 70 is eligible under the step rule and tuning range; the provider is not limited to 60, 70, and 80.

All nine fields remain available within their restricted ranges. A proposal may change at most three effective fields. Unchanged fields are omitted from the sparse proposal.

## 5. Configuration and policy model

The high-level live configuration becomes the single source of the campaign tuning ranges shown to Kimi and applied by the local pre-execution policy.

The policy model will distinguish:

- immutable hardware encoding bounds;
- campaign `tuning_min` and `tuning_max`;
- campaign `max_step`;
- maximum changed-parameter count.

The loader will build the restricted policy by combining the campaign YAML with the existing authorized hardware parameter specifications. It must fail closed when:

- a rule names an unknown or disabled parameter;
- a tuning bound lies outside the hardware encoding or authorization tuning range;
- `minimum` is greater than `maximum`;
- `max_step` is not a positive integer or exceeds the hardware-authorized maximum step;
- a B0 value lies outside the campaign tuning range;
- a required policy is missing from the V2.1 LIVE configuration.

The hardware profile remains unchanged and continues to represent the outer physical safety envelope. Campaign configuration may only narrow that envelope.

## 6. Provider boundary

The structured Kimi context will expose, for every enabled field:

- `tuning_min`;
- `tuning_max`;
- `max_step`;
- `integer_only: true`;
- `maximum_changed_parameters: 3` at policy level.

The system instruction will continue to require sparse absolute integer I2C values within the supplied policy. It will continue to prohibit FIR, BERT, NLC, and `setFir` requests.

The provider does not receive execution authority. A prompt-compliant value is still only a proposal.

## 7. Local enforcement and execution flow

The execution flow is:

1. Load and validate B0 version 2.
2. Load the explicit restricted campaign policy.
3. Cross-check every policy rule and B0 value against the existing hardware/authorization envelope.
4. Include the restricted policy in the provider context.
5. Receive a sparse absolute-integer proposal.
6. Reject unknown fields, non-integers, out-of-tuning-range values, excessive steps, empty changes, or more than three changed fields before candidate execution.
7. Pass the surviving immutable candidate through the existing authorization validation against campaign identity, baseline hash, stage, budget, and latest hardware readback.
8. Use the existing I2C-only apply, readback verification, DCA acquisition, artifact persistence, and recovery chain.
9. Restore and verify the new B0 on normal termination or safe failure.

Values are never clipped, rounded into range, or silently replaced. An invalid Kimi proposal follows the existing bounded correction behavior and is never written merely because the natural-language explanation appears reasonable.

## 8. Persistence and audit evidence

The continuous controller's existing `campaign.json` policy record will persist the effective restricted ranges and steps, not only the outer hardware encoding limits. The run evidence must preserve:

- exact branch, worktree, and HEAD;
- B0 version and baseline hash;
- effective restricted policy version and rules;
- campaign ID and authorization identity;
- requested, applied, and readback I2C values for every executed candidate;
- terminal state, stop reason, B0 restoration result, and provider error fields.

The report must distinguish B0 from AI-generated candidates. If B0 remains the lowest valid TDECQ, it is reported as the best baseline/reference, not as an AI-optimized result.

## 9. Tests

Tests will cover:

- parsing the approved YAML policy;
- rejecting a missing live policy;
- rejecting unknown fields and invalid range ordering;
- rejecting a range or step wider than the hardware authorization;
- rejecting a B0 value outside the restricted range;
- transmitting the exact restricted ranges and steps to Kimi;
- rejecting a Kimi proposal outside the restricted range before the execution port is called;
- rejecting a proposal that exceeds the campaign maximum step even when it remains inside the hardware envelope;
- rejecting more than three effective changes;
- accepting boundary values and intermediate integer values, confirming that `max_step` is not a grid quantum;
- preserving all existing FIR, BERT, NLC, `setFir`, readback, persistence, and B0-recovery prohibitions;
- persisting the effective restricted policy in campaign evidence.

Tests use fakes and temporary artifacts only. They do not connect to DCA, I2C, BERT, or any real hardware.

## 10. Acceptance criteria

The implementation is complete when:

- `b0.v2_1.continuous.yaml` contains the approved version 2 values and the V2.1 continuous runtime points to it;
- the live campaign YAML contains the approved restricted policy;
- Kimi receives exactly those tuning ranges, steps, and the three-field limit;
- local policy enforcement rejects any value outside those limits before execution;
- the existing hardware authorization remains an independent outer boundary;
- the hardware driver profile and low-level adapters are unchanged;
- all focused tests and the relevant existing test suite pass;
- no existing `artifacts/`, `outputs/`, or `state/` evidence is deleted or modified.

No PRE-FLIGHT, COMMISSION, I2C write, DCA acquisition, BERT access, FIR/NLC write, or `setFir` operation is authorized by this design or by its implementation alone.
