# Native V1.2 staged LIVE capabilities

Status: ready-for-agent

## Lineage

- Target product: Native V1.2, preserving its V1.2 identity, storage layout, continuous FIR convolution, and optional Dataset behavior.
- Target base: `0a7e35161f20effd9b56df1a5e81052f3298c1fa`.
- Reviewed migration source: committed V1.1.1 FAST24 behavior at `e7f08eb7f1e0c79eb17f94c037c6db6e23c5d1e9`.
- Compatibility route: `/api/v1.1` remains an intentional interface and does not determine product lineage.

## Problem Statement

Native V1.2 can run FAKE AI Optimization and has its own Dataset and FIR capabilities, but it cannot safely execute a real staged AI campaign. It lacks a real LIVE executor, immutable LIVE policy, killable DCA I/O isolation, and a watchdog-backed recovery path. Passing algorithm and FAKE tests therefore does not establish real hardware readiness.

The product must gain these capabilities without merging the V1.1 or V1.1.1 product lines, replacing Native V1.2 behavior, reusing their storage, or confusing their hardware evidence with Native V1.2 results.

## Solution

Add a pre-authorized staged LIVE campaign to Native V1.2. It runs a three-candidate PILOT_3 qualification and automatically enters FAST24 only after a strict safety gate. FAST24 runs six coarse I2C rounds, sixteen fine I2C rounds, one first-pass formula FIR round, and one iterative formula FIR round.

All DCA connection and acquisition work runs in a bounded child process. The main execution owns I2C and BERT mutation, exact readback, persistence ordering, and deterministic B0 restoration. Formal acceptance saves a complete Dataset for every round. Any invalid measurement, identity drift, readback mismatch, timeout, or unverified restoration fails closed.

## User Stories

1. As an operator, I want the complete PILOT_3 and FAST24 authority displayed before approval, so that no task expands its hardware permissions in flight.
2. As an operator, I want Native V1.2 identified by branch, Worktree, commit, execution policy, and product lineage, so that it cannot be confused with another product displaying 1.2.0.
3. As an operator, I want PILOT_3 to select three candidates sequentially from durable observations, so that it is a real observation-conditioned optimizer rather than a precomputed Sweep.
4. As an operator, I want every PILOT_3 candidate restored to B0 and read back, so that the narrow qualification does not leave a candidate active.
5. As an operator, I want FAST24 to start automatically only after every PILOT_3 safety check passes, so that automation remains bounded.
6. As an operator, I want FAST24 to use exactly six coarse, sixteen fine, one first FIR, and one iterative FIR round, so that the reviewed 24-round meaning is stable.
7. As an operator, I want PILOT_3 limited to `ibias_dac` values 69, 71, and 73 and `res_gain` values 4 and 8, so that its write authority remains narrow.
8. As an operator, I want FAST24 limited to `ibias_dac` 60 through 80 and `res_gain` 0 through 10 in unit steps, so that its search authority is explicit.
9. As an operator, I want every candidate to settle for 0.1 seconds before measurement, so that candidate timing is consistent.
10. As an analyst, I want only valid numeric TDECQ to rank candidates, so that ER movement cannot change the optimization objective.
11. As an analyst, I want ER, OMA, and other acquired metrics validated and recorded, so that non-objective evidence remains auditable.
12. As an operator, I want FAST24 I2C candidates to transition directly while retaining the current best state, so that the alternating optimizer keeps its intended state.
13. As an operator, I want formula FIR candidates derived from DCA Equalizer taps, so that no caller supplies arbitrary FIR values.
14. As an operator, I want FIR Main, NLC, and pattern frozen to B0, so that formula optimization cannot mutate those dimensions.
15. As an operator, I want non-Main formula taps free from the former B0 plus-or-minus-three limit while still satisfying device legality, seven integer taps, and FIR L1 no greater than 170, so that the reviewed formula is not silently clipped.
16. As an operator, I want an illegal FIR formula result rejected without a hardware write, so that validation never fabricates a nearby candidate.
17. As an operator, I want every I2C and FIR write followed by exact readback before DCA acquisition, so that measurements are tied to the state actually applied.
18. As an operator, I want DCA work isolated in a killable process with a 60-second operation timeout and a 5-second cleanup grace, so that blocked VISA I/O cannot hang the application indefinitely.
19. As an auditor, I want the last in-flight SCPI operation and worker cleanup evidence persisted after a timeout, so that the failure is diagnosable.
20. As an operator, I want a timed-out or invalid round to carry null metrics rather than a previous round's values, so that stale data cannot be optimized.
21. As an operator, I want normal completion, stop, timeout, and failure paths to restore and verify B0, so that terminal state is explicit.
22. As an operator, I want unverified restoration to enter `RECOVERY_REQUIRED` and block new LIVE work, so that a damaged state cannot be ignored.
23. As an operator, I want one deterministic automatic restore attempt rather than an unlimited retry loop, so that recovery does not create repeated writes.
24. As an auditor, I want every formal acceptance round to save density NPY, raw and equalized PNGs, requested/applied/readback settings, metrics, acquisition identities, and SHA-256 values, so that each result is independently verifiable.
25. As a user, I want Dataset saving to remain optional outside formal acceptance, so that ordinary runs can trade evidence volume for speed deliberately.
26. As an operator, I want the best parameters recorded but B0 left on the hardware after completion, so that optimization results do not silently become a new baseline.
27. As a maintainer, I want the V1.1.1 behavior migrated through a minimal reviewed dependency closure, so that Native V1.2 capabilities are not overwritten by a branch merge.
28. As a maintainer, I want the existing compatibility API retained, so that version correctness does not require an unrelated route migration.
29. As a tester, I want software-only timeout and process-reclamation tests, so that watchdog behavior is proven without disturbing laboratory hardware.
30. As an operator, I want the two-hour soak announced before it begins, so that a long hardware run is never started silently.

## Implementation Decisions

- Introduce one explicit staged LIVE profile rather than a collection of escalation flags.
- Freeze the complete two-stage authorization before opening mutation-capable hardware adapters.
- Keep the policy module pure and fail closed on unknown fields, ranges, budgets, profiles, or timing.
- Reuse the sequential PILOT_3 planner behavior: candidate two depends on durable observation one, and candidate three depends on durable observations one and two.
- Require all three PILOT_3 observations, three verified candidate restorations, complete Dataset evidence, unchanged device identities, and a verified final B0 before FAST24 starts.
- Preserve the FAST24 allocation of six coarse I2C, sixteen fine I2C, one first-pass FIR, and one iterative FIR round.
- Rank only valid numeric TDECQ. Treat other measurements as validity and audit evidence.
- Retain the current best state between FAST24 candidates. Always restore B0 at terminal exit.
- Generate FIR candidates only from the reviewed formula. Freeze Main, NLC, and pattern; require seven integer taps, device-legal values, and L1 no greater than 170. Do not impose a B0 plus-or-minus-three limit on non-Main taps.
- Place the primary execution seam at the task runner's AI execution factory. The service, persistence, lifecycle, and recovery code use the same interface in production and tests.
- Place DCA connection ownership behind a synchronous isolation-client interface implemented by a spawned child process. The child has no dependency capable of I2C or BERT mutation.
- Use one complete DCA connection lifecycle per isolated preflight or measurement. Bound each operation to 60 seconds and cleanup to 5 seconds.
- Journal DCA lifecycle and SCPI operation context before and after I/O so that a killed worker leaves durable evidence.
- Persist each observation and Dataset before requesting another candidate.
- Keep Dataset saving configurable, but require it for all formal hardware acceptance tasks.
- Execute one deterministic automatic B0 restore on failure. Block new LIVE work with `RECOVERY_REQUIRED` when exact restoration evidence is incomplete.
- Preserve Native V1.2 state and artifact roots, product identity, compatibility route, continuous FIR convolution, and current Dataset integrity rules.
- Selectively migrate reviewed behavior and its tests. Do not merge a source product branch or import its packaging and runtime outputs.

## Testing Decisions

- Prefer one high execution seam: task service through persistent runner, staged executor, fake I2C/BERT adapters, and a real spawned process using a fake DCA worker.
- Assert observable lifecycle, persistent state, candidate order, exact readbacks, Dataset records, terminal state, and B0 restoration rather than private call structure.
- Test immutable policy rejection before any hardware driver construction.
- Test the complete three-plus-twenty-four flow, including the exact 6/16/1/1 FAST24 stage allocation.
- Test that every PILOT_3 suggestion consumes only already-persisted observations.
- Test TDECQ-only ordering with ER moving in the opposite direction.
- Test formula FIR legality, frozen Main/NLC/pattern, lack of the former non-Main plus-or-minus-three constraint, and rejection without write.
- Test normal DCA child completion, worker exception, never-returning setup/query/write/close operations, terminate-then-kill cleanup, process disappearance, and null failure metrics using software fakes.
- Test I2C and FIR exact-readback mismatches, Dataset persistence failure, authorization expiry, identity drift, stop, and every B0 restoration outcome.
- Test that `RECOVERY_REQUIRED` blocks a subsequent LIVE start until recovery verifies B0.
- Run the full Python suite from the clean baseline of 1,878 passed and two Windows symlink skips, then run frontend test, build, lint, configuration validation, and `git diff --check`.
- Stage hardware acceptance as READ_ONLY preflight, B0 verification, one legal non-baseline formula FIR round and restore, standalone PILOT_3 qualification, the staged campaign, final B0 verification, and an announced two-hour soak.
- Do not inject faults into real hardware. Hardware anomalies end the current and all later hardware gates.

## Out of Scope

- Merging V1.1, V1.1.1, or the V1.1-derived V1.2 product branches.
- Replacing Native V1.2 launchers, product identity, state roots, artifact roots, compatibility routes, or continuous FIR convolution.
- Importing release candidates, runtime state, logs, artifacts, or user data from a source Worktree.
- Changing the objective from strict TDECQ to a weighted or multi-objective score.
- Allowing caller-supplied FIR dimensions, Main changes, NLC changes, pattern changes, illegal device values, or FIR L1 above 170.
- Leaving the best candidate applied after a task completes.
- Real-hardware fault injection, cable removal, forced instrument termination, or deliberate device disruption.
- Distributed locking against independent programs on other hosts. The hardware must be dedicated to the active acceptance run.
- A soak longer than two hours.
- Push, merge, or publication to an external issue tracker.

## Further Notes

- The source V1.1.1 Worktree was clean at the reviewed commit. A separate V1.1-derived V1.2 Worktree contained untracked release output and is not a migration source for runtime files.
- Automated test success cannot mark real hardware capabilities READY. Every hardware gate receives a separate result.
- The two-hour soak requires a user-visible start notice after all shorter hardware gates pass.
