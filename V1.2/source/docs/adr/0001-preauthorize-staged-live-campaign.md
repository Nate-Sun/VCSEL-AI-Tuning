# ADR-0001: Pre-authorize the staged LIVE campaign

Status: Accepted

Date: 2026-08-10

## Context

PILOT_3 writes within a narrow I2C policy, while FAST24 expands the I2C search and permits formula-generated FIR writes. Automatically starting FAST24 after authorizing only PILOT_3 would silently expand hardware authority during execution. Independent authorization between stages would preserve the boundary but would not provide the requested automatic transition.

## Decision

Expose one staged LIVE campaign whose complete PILOT_3 and FAST24 authority is frozen before the first hardware write.

The campaign enters FAST24 only when all three PILOT_3 observations are valid and durable, every candidate restoration is verified, the final B0 state is verified, Dataset evidence is complete, device identities are unchanged, and no DCA timeout or invalid result occurred.

Any failed gate ends the campaign without FAST24 execution. The terminal path always attempts one deterministic B0 restoration. An unverified restoration enters `RECOVERY_REQUIRED` and blocks new LIVE work.

## Consequences

- The operator sees the full I2C and FIR authority before approving the task.
- The PILOT_3-to-FAST24 transition requires no second confirmation when its safety gate passes.
- The policy interface gains one explicit staged profile rather than a collection of escalation flags.
- A standalone PILOT_3 qualification may still be used during staged hardware acceptance, but it does not authorize FAST24.
- Policy or gate changes require a new authorization contract rather than an in-flight expansion.

## Rejected alternatives

- Authorize PILOT_3 first and expand authority during the run: rejected because the operator would not have approved the eventual FIR and wider I2C writes.
- Require a second interactive authorization: rejected because it does not meet the requested automatic transition.
- Enter FAST24 after any normally returned PILOT_3 state: rejected because invalid measurements or incomplete restoration evidence must fail closed.
