# V2.1 R46 Fast Continuation Design

## Purpose

Create a minimal derived LIVE test variant that starts from the best verified
readback of round 46 from campaign
`v21-b-20260817T125713321Z-b468337d`, while preserving the validated version B
worktree and tag unchanged.

This is a **fast continuation**, not a cross-campaign memory continuation. Kimi
starts a fresh campaign and receives no prior 50-round history. The only carried
state is the new B0/start-and-recovery I2C snapshot.

## Provenance and Isolation

- Source tag: `v2.1-continuous-eye-b-validated-20260817`
- Source commit: `4085b007546f820820dbdb7f7f0bba6ebc4b2049`
- Derived branch: `codex/v2.1-continuous-eye-r46-fast`
- Derived worktree:
  `C:\Users\NateS\Documents\VCSEL\.worktrees\v2-1-continuous-eye-r46-fast`
- Original B worktree remains:
  `C:\Users\NateS\Documents\VCSEL\.worktrees\v2-1-continuous-eye-optimization`

The original B worktree, tag, and its `artifacts/`, `outputs/`, and `state/`
directories must not be modified, moved, or cleaned.

## New B0

The derived variant uses the complete verified round-46 readback as B0:

| Parameter | Value |
|---|---:|
| `ibias_dac` | 68 |
| `efin_cur` | 15 |
| `vga_cur` | 11 |
| `eq` | 23 |
| `res_gain` | 9 |
| `res_bw` | 1 |
| `gain_ctrl` | 8 |
| `vbias_ref` | 15 |
| `vref` | 15 |

The FIR and NLC values remain identical to the validated version B baseline.
No FIR, BERT, or NLC tuning is added.

Because the current controller uses B0 as both the starting snapshot and the
recovery snapshot, normal completion and every failure path restore the round-46
parameter set above.

## Minimal Code and Configuration Scope

Production tuning logic is unchanged. The implementation changes only:

1. `configs/ai_tuning/b0.v2_1.continuous.yaml`
   - Replace the nine I2C B0 values with the verified round-46 readback.
   - Preserve all FIR/NLC values and file structure.
2. `scripts/run_v2_1_continuous_live.ps1`
   - Change only the expected branch from the validated B branch to the derived
     R46 branch.
   - Preserve the required implementation baseline, HEAD/session freeze,
     manifest validation, secret cleanup, PRE-FLIGHT/COMMISSION split, and all
     hardware safety gates.
3. Existing offline tests whose exact assertions encode the validated B0 or
   launcher branch.
   - Update expected values only.
   - Add or retain coverage proving the full R46 B0 and derived branch are
     required.

No changes are permitted to:

- DCA, I2C, BERT, FIR, or NLC adapters;
- continuous controller behavior;
- parameter ranges or per-round `max_step` values;
- maximum three changed parameters per proposal;
- Kimi schema, retry, raw-response journal, or failed-safe behavior;
- operator-entered maximum rounds and target TDECQ;
- automatic campaign-ID generation;
- evidence persistence.

## Runtime Flow

1. The operator manually writes and saves the complete R46 I2C snapshot on the
   DUT before starting the derived launcher.
2. The operator starts the launcher from the derived worktree.
3. The launcher asks only for maximum rounds, strict target TDECQ, and the
   Moonshot API key.
4. PRE-FLIGHT must verify DCA/I2C identity and show the complete R46 readback.
   It performs no DUT write and no BERT contact.
5. The operator separately approves COMMISSION.
6. The controller explores within the existing restricted parameter policy.
7. On completion or failure, the controller restores the R46 B0 and persists
   the recovery evidence.

If PRE-FLIGHT readback does not exactly match the intended R46 baseline, the
operator must not approve COMMISSION.

## Testing and Safety

All development verification is offline. It must include:

- exact R46 B0 load/readback expectations;
- launcher acceptance of only
  `codex/v2.1-continuous-eye-r46-fast`;
- rejection of the old branch in the derived launcher;
- unchanged restricted parameter policy;
- unchanged PRE-FLIGHT non-write behavior;
- unchanged Kimi truncation/invalid-JSON failed-safe behavior;
- unchanged controller B0 restoration behavior;
- full offline regression suite;
- `git diff --check` and an exact diff-scope review.

No PRE-FLIGHT, COMMISSION, Kimi network request, or real hardware operation is
run during implementation. A real test requires a separate operator decision
after offline verification completes.

## Acceptance Criteria

The derived variant is ready for operator testing only when:

1. The original validated B tag still resolves to commit `4085b007...`.
2. The original B worktree remains unchanged and immediately testable.
3. The derived branch contains only the approved B0, branch-gate, test, plan,
   and design-document changes.
4. All offline tests pass.
5. The provided launch command points only to the derived R46 worktree.
6. The handoff explicitly instructs the operator to verify the R46 readback in
   PRE-FLIGHT before approving COMMISSION.
