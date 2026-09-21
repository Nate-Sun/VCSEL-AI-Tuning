# V2.1 Continuous Eye Optimization Design

**Status:** Approved in the design conversation; awaiting review of this written specification  
**Date:** 2026-08-12  
**Code baseline:** `codex/v2.1-kimi-live` at `5d04fc1724b0622873990055ac6be6e1765b535d`  
**Implementation branch:** `codex/v2.1-continuous-eye-optimization`  
**Implementation worktree:** `C:\Users\NateS\Documents\VCSEL\.worktrees\v2-1-continuous-eye-optimization`

## 1. Goal

Build one reusable continuous eye-optimization entry point that can drive a real Kimi -> I2C -> hardware readback -> DCA measurement loop. The same entry point must later serve a Testbench, CLI, Dashboard, or backend without duplicating optimization or hardware logic.

The first live commissioning run uses `max_rounds: 3`, but round count is configuration, never business logic. The first run proves that the real closed loop is connected; it does not promise that the eye improves or reaches the target.

This is an I2C-only design. It must not adjust FIR or NLC, call `setFir`, expose BERT controls, or restore BERT/FIR state.

## 2. Existing modules to reuse

The implementation extends the existing V2.0/V2.1 path rather than creating a parallel stack:

- `KimiApiClient` remains the Moonshot transport and reads its key only from `MOONSHOT_API_KEY`.
- `KimiProvider` remains the Kimi-facing decision adapter and is generalized behind a provider-neutral decision interface.
- `KimiCampaignMemory` remains the compact, campaign-bound provider history.
- `BrainFactory` and `analyze_factory_gated` remain the provider creation and fail-closed assessment gates.
- `submit_kimi_step` / `execute_kimi_step` supply reusable one-step behavior; orchestration that belongs to continuous execution moves behind the new controller interface rather than into the Testbench.
- `CampaignRuntime` remains responsible for runtime execution and resource ownership, with an upper-layer extension for durable cross-request leases while awaiting a human.
- `StageAuthorization` and the V2.0 deterministic authorization checks remain the final authority for range, step, stage, budget, campaign, baseline, and latest-readback validation.
- The existing I2C executor, hardware readback, DCA measurement, rollback, and `RECOVERY_REQUIRED` behavior remain the real execution chain.

No second Kimi client, safety implementation, I2C execution path, or fake hardware workflow will be introduced.

## 3. Deliberate change from the original candidate-menu requirement

The approved design supersedes the earlier rule that Kimi must select from a locally generated menu of candidates.

Instead, every external provider receives the complete versioned parameter policy and proposes sparse absolute parameter values directly. The local program materializes that proposal as an immutable candidate, assigns its `candidate_id`, and then applies all existing deterministic V2.0 checks before any hardware action.

This reduces repeated prompt tokens and avoids large candidate lists. It does not give the provider execution authority. A prompt explains the policy, but only local code can authorize a proposal. Invalid values are never clipped, rounded into range, or silently corrected.

## 4. Parameter policy

All current regmap I2C parameters are available. The approved safe range equals each field's encoding range because the hardware owner has confirmed that any encoded value in these fields will not damage the VCSEL or driver.

The single-round step is `ceil(15% * encoding range width)`, measured from the latest pre-execution hardware readback. Values are integers with quantum 1.

| Parameter | Encoding range | First-run maximum absolute step |
|---|---:|---:|
| `ibias_dac` | 0..127 | 20 |
| `gain_ctrl` | 0..63 | 10 |
| `eq` | 0..31 | 5 |
| `efin_cur` | 0..15 | 3 |
| `vga_cur` | 0..15 | 3 |
| `res_gain` | 0..15 | 3 |
| `res_bw` | 0..15 | 3 |
| `vbias_ref` | 0..15 | 3 |
| `vref` | 0..15 | 3 |

The first live run may change at most three parameters in one proposal. This limit is configuration and may be enlarged in later applications. A proposal may change fewer parameters. Unchanged fields are not written merely to make a proposal dense.

All proposals still pass the existing stage authorization, current-readback, range, step, and execution-budget checks. Authorization is always evaluated against the last verified hardware readback, not a provider-reported or cached state.

## 5. Provider interface and prompt boundary

Kimi is the first adapter, using model `kimi-k3` and the official Moonshot base URL. Model and base URL are configuration. The provider-neutral interface applies to all later external model adapters.

### 5.1 Provider input

The bounded provider context contains only:

- the current DCA eye PNG and current instrument metrics;
- the latest verified I2C readback;
- the nine allowed parameter names, full encoding ranges, per-round step limits, integer rule, and maximum changed-parameter count;
- current hardware round, maximum hardware rounds, and remaining rounds;
- the target `TDECQ < 2.7 dB`;
- up to five recent valid execution summaries, including valid worse results, plus a separate best reference eye;
- provider policy version and allowed output schema.

Invalid/failed executions do not become normal learning history. A valid worse eye is retained as negative experience and optimization continues from the resulting verified hardware state.

### 5.2 Provider output

The provider returns exactly one structured action:

- `PROPOSE`: sparse absolute I2C parameter values;
- `HOLD`: reconsider the same observation without a hardware action;
- `STOP`: request local acceptance checking;
- `REQUEST_HUMAN`: request a human decision.

Each response contains bounded `reason_codes`, a configured confidence value, and a short summary. Reason codes are a versioned local enum; the initial vocabulary covers target status, expected improvement, exploration, learned negative result, exhausted search, low confidence, OOD, invalid observation, and need for a human. Unknown actions, fields, parameter names, or reason codes fail schema validation.

The provider never receives I2C, serial, VISA, DCA, BERT, FIR, `apply`, or `setFir` objects. It cannot execute hardware, declare global optimality, waive checks, choose rollback behavior, or count its assessment as execution evidence.

### 5.3 Action handling

- A valid `PROPOSE` is materialized locally as an immutable candidate. Its ID is derived from campaign, round, proposal attempt, and canonical requested parameters.
- An invalid `PROPOSE` is not executed. The provider receives compact deterministic error codes and may correct once against the same observation. A correction is not a hardware round. A second invalid proposal ends automatic execution.
- A first `HOLD` may trigger one re-query against the same observation only when the eye is valid, confidence passes the existing configured gate, and the result is not OOD. A second `HOLD` ends execution. Low confidence, invalid eye, OOD, or provider/API failure ends immediately without a HOLD retry.
- `STOP` requests acceptance; it never asserts that the current state is globally optimal.
- `REQUEST_HUMAN` stops automatic rounds and enters the durable human-wait flow below.

## 6. Deep controller module

The new `ContinuousOptimizationController` is the single deep module used by all callers. It hides continuous state transitions, provider retries, deterministic authorization, execution evidence, acceptance measurements, persistence, timing, and terminal recovery behind a small interface.

Conceptually:

```python
result = controller.start(request, dependencies)
controller.request_stop(campaign_id)
controller.resolve_human(campaign_id, decision)
status = controller.get_status(campaign_id)
```

`request` contains:

- `campaign_id`;
- current artifact-backed `EyeSample`;
- `StageAuthorization`;
- versioned I2C parameter policy;
- `max_rounds`;
- provider configuration;
- stopping and acceptance policy;
- static B0 configuration reference;
- result directory;
- explicit execution mode.

Injected dependencies include the provider adapter, `CampaignRuntime`, existing V2.0 authorization/execution path, I2C adapter, DCA adapter, clock, and artifact/state store. The controller accepts dependencies; it does not create hardware connections internally.

The Testbench, future Dashboard, and backend know this interface, not the orchestration implementation.

## 7. Round flow

Each hardware round follows this exact order:

1. Validate the current `EyeSample` and current verified readback.
2. Build the bounded provider context.
3. Receive and validate one provider action, including the permitted same-observation retry behavior.
4. Materialize a `PROPOSE` as a local immutable candidate.
5. Run V2.0 deterministic stage, range, step, parameter-count, latest-readback, and budget checks.
6. Execute through the existing real I2C chain.
7. Read back the hardware and require exact agreement with the selected proposal.
8. Ask the DCA adapter for a new observation and metrics.
9. Save the DCA model image and its derived NPY, hashes, display fingerprint, metrics, and execution evidence.
10. Atomically persist the round record.
11. Add a valid result, including a valid worse result, to compact history.
12. Continue, request acceptance, wait for a human, stop, or recover.

A hardware round increments `completed_rounds` only if all of these are true:

- the executor returned successfully;
- pre- and post-execution observation IDs are non-empty, campaign/round-bound, and different;
- applied values and hardware readback exactly match the immutable candidate;
- the DCA produced a new valid observation;
- the required structured record and artifacts were durably saved.

Provider assessment records alone are never hardware evidence.

## 8. DCA eye artifacts

For the first live implementation, the authoritative evidence image is the actual DCA graticule eye PNG captured for that observation. The provider image is a deterministic indexed-color PNG derived from that evidence image: it is fitted within a 1280 by 528 box with aspect ratio preserved using Pillow LANCZOS; smaller images retain their dimensions; the RGB result is then quantized to a fixed 256-color palette without dithering. This commissioning representation was selected from a side-by-side review of the real 2353 by 970 FlexDCA capture, where it becomes 1280 by 528 while preserving the eye openings and trace geometry and remains below the one-MiB provider boundary even for high-entropy RGB input.

For each before/after observation the DCA adapter stores:

- the exact original DCA screen PNG;
- the exact derived PNG sent to the provider;
- a deterministic NPY pixel array derived from the exact original DCA PNG;
- hashes linking the original PNG, provider PNG, NPY, metrics, and observation ID;
- a DCA display/acquisition fingerprint sufficient to detect incompatible scale or rendering changes;
- provenance `SCREEN_IMAGE_DERIVED` so the NPY is never represented as raw instrument eye-density data.

Instrument TDECQ remains authoritative and is stored separately from the image. Direct internal eye-density export is outside the first commissioning scope and can later be introduced behind the same eye-artifact interface.

## 9. Stop and acceptance policy

The only first-run optimization target is `TDECQ < 2.7 dB`. Outer ER is not an optimization target.

A local target hit or provider `STOP` starts the same acceptance procedure without a second provider call:

1. Keep the same verified I2C state.
2. Acquire one additional DCA observation.
3. Require both observations to be valid and to share the same I2C readback and compatible DCA fingerprint.
4. Require absolute TDECQ difference `<= 0.2 dB` and absolute Outer ER difference `<= 0.2 dB`.

A provider STOP may be accepted even when TDECQ remains at or above 2.7 dB; the result then records `target_reached: false` and `stop_reason: PROVIDER_STOP_ACCEPTED`. This is an accepted stopping decision, not proof of an optimum.

If the initial B0 observation already meets the target and the provider requests STOP, the campaign may finish with `BASELINE_ACCEPTED_WITHOUT_EXPLORATION`. It can pass acceptance but cannot satisfy first commissioning proof because no real write/readback/re-measure loop occurred.

Failure to reach the target before `max_rounds` is a normal completion with `target_reached: false` and `stop_reason: MAX_ROUNDS_REACHED`, not a system failure.

## 10. Human wait, resource ownership, and timeout

`REQUEST_HUMAN` transitions the campaign to `AWAITING_HUMAN`:

- keep the current verified non-B0 I2C state;
- perform no further automatic round;
- retain an exclusive bench lease across requests;
- persist enough state for status display and an explicit human decision;
- accept only resume, acceptance, or restore-B0 decisions through `resolve_human`.

The default wait timeout is 15 minutes and is configuration. Timeout triggers an attempted static B0 restore and terminal completion. A process crash, restart, lost lease, or inability to prove continuity while waiting results in `RECOVERY_REQUIRED`; automatic optimization does not resume from uncertain state.

This requires an upper-runtime durable lease/state extension. It does not require replacing the low-level I2C or DCA adapters.

## 11. States, failures, rollback, and B0

Public campaign states are:

- `RUNNING`;
- `AWAITING_HUMAN`;
- `COMPLETED`;
- `FAILED_SAFE`;
- `RECOVERY_REQUIRED`.

The result separately records `target_reached`, `acceptance_passed`, `closed_loop_proven`, and `stop_reason`. `COMPLETED` does not mean optimum or target reached.

Low confidence, invalid eye, OOD, API exception, exhausted invalid-proposal correction, failed I2C write, readback mismatch, DCA failure, or durable-persistence failure prevents the next automatic round.

Execution failure first uses the existing rollback path to restore and verify the pre-round snapshot. On normal terminal completion or confirmed safe failure, the controller then restores the fixed I2C values in `configs/ai_tuning/b0.yaml` and verifies exact readback. This is an I2C-only restore; FIR/NLC/BERT entries and `setFir` are never used.

If rollback or static B0 recovery cannot be confirmed, the final state is `RECOVERY_REQUIRED`. The controller must not continue merely because a write command was issued.

The one exception to immediate static B0 restoration is a live `AWAITING_HUMAN` campaign. It still restores B0 on timeout or an explicit restore/stop decision.

## 12. Timing

Durations use a monotonic clock; audit timestamps use timezone-aware wall-clock time.

Every round records:

- `started_at` and `finished_at`;
- provider request and correction/HOLD retry durations;
- local validation duration;
- I2C write duration;
- hardware readback duration;
- DCA acquisition duration;
- artifact persistence duration;
- total round duration;
- failure stage and all durations completed before failure.

Every campaign records start/finish timestamps, active optimization time, human-wait time, total elapsed time, individual round durations, and average completed-round duration. Human waiting is excluded from active time but included in elapsed time.

## 13. Persistence and evidence layout

Each campaign owns an isolated artifact directory:

```text
<result_dir>/<campaign_id>/
|-- campaign.json
|-- state.json
|-- events.jsonl
|-- rounds/
|   `-- 0001/
|       |-- round.json
|       |-- eye_before.png
|       |-- eye_after.png
|       |-- model_input_before.npy
|       |-- model_input_after.npy
|       `-- provider_exchange.json
|-- acceptance/
`-- recovery/
```

`state.json` is atomically replaced and represents the recoverable current state. `events.jsonl` is append-only and records state transitions, attempts, rollback, recovery, and timing. Acceptance remeasurement is stored under `acceptance/` and is not presented as a tuning round.

Each `round.json` includes at least:

- round, campaign, candidate, measurement, and observation IDs;
- provider action, proposal attempt, bounded reason codes, confidence, and summary;
- requested, applied, and readback parameters;
- before/after metrics and artifact references/hashes;
- execution status and execution evidence;
- DCA fingerprint and image provenance;
- failure/stop reason;
- timestamps and stage timings.

Prompt records, configuration snapshots, logs, and provider exchanges must be secret-scrubbed. `MOONSHOT_API_KEY` is read only from the environment and is never stored in code, YAML, prompts, logs, or artifacts.

## 14. Testbench and execution modes

The Testbench only:

1. loads configuration and `MOONSHOT_API_KEY` from the environment;
2. constructs official production dependencies through existing factories;
3. invokes the controller and its status/stop/human-decision interfaces;
4. displays live state, round/stage timing, and the final result.

It does not own the loop, generate proposals, judge safety, call I2C/DCA directly, or implement recovery.

Execution mode is explicit and persisted:

- `LIVE` uses the real Kimi, I2C, readback, and DCA adapters;
- `FAKE` replaces adapters only for deterministic non-hardware verification.

Both modes execute the same controller. There is no separate fake optimization flow. Only `LIVE` with complete real evidence may set `closed_loop_proven: true`.

Replay is intentionally omitted because no prior dataset exists. The first live campaign can later become replay material, but no replay adapter or replay acceptance gate is part of this implementation.

## 15. Verification and live authorization gate

The shortened first-run verification sequence is:

1. Focused unit tests for provider schema, complete encoding ranges, 15% step limits, maximum three changed parameters, retry rules, action handling, timing, evidence, states, and I2C-only B0 selection.
2. One FAKE end-to-end run through the production controller plus critical fault cases: readback mismatch, DCA failure, B0 recovery failure, and STOP acceptance remeasurement.
3. A LIVE read-only preflight that verifies configuration, adapter and device identities, Kimi connectivity, DCA PNG acquisition capability, and current readback without writing I2C.
4. A second explicit user authorization describing the exact proposed live action before any real I2C write or DCA optimization loop.
5. After authorization, one `LIVE` commissioning campaign configured with `max_rounds: 3`.
6. Inspection of structured evidence, stage/loop timing, and exact static B0 readback restoration.

No real I2C/DCA optimization loop is permitted before step 4 authorization. The read-only preflight also requires advance notice and must not silently contact instruments.

## 16. First commissioning acceptance

`closed_loop_proven: true` requires all of the following:

- execution mode is `LIVE` and adapter/device identities are recorded;
- at least one real proposal round completed;
- the provider returned a valid proposal;
- all V2.0 deterministic checks passed;
- the I2C executor returned, the write was applied, and exact hardware readback matched;
- the DCA produced a new valid observation and actual eye PNG;
- derived NPY, metrics, hashes, IDs, timestamps, and timings were durably saved;
- fixed static I2C B0 was restored and exact readback verified at campaign end.

Eye improvement and `TDECQ < 2.7 dB` are not required for commissioning proof. `BASELINE_ACCEPTED_WITHOUT_EXPLORATION` may pass normal acceptance but sets `closed_loop_proven: false` and cannot replace a later commissioning run with at least one real round.

## 17. Out of scope

- FIR, NLC, BERT, and `setFir` control;
- Dashboard or backend UI implementation;
- editing B0 from a frontend;
- database workflow infrastructure;
- replay dataset/adapters;
- multi-provider production qualification;
- exhaustive cross-process crash stress testing;
- live 5-round or 10-round campaigns;
- direct raw FlexDCA eye-density export.

These may be added later through the existing interfaces without moving orchestration into callers or weakening deterministic authorization.
