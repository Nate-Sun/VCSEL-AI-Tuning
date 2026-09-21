# V1.2 LIVE Dataset Completeness Design

## Scope and lineage

This change applies only to the native V1.2 line in
`codex/v1.2-native-readonly-db-fix`. It does not merge the separate V1.1-derived
V1.2 implementation. It changes no I2C or BERT mutation behavior and does not
add a hardware test to the implementation workflow.

## Problem

The current LIVE dataset adapter always returns
`same_acquisition_verified=False`. Its acquisition guard also replays plain
measurement queries after TDECQ has selected `FUNC1`; on FlexDCA the relevant
measurement source selector is shared, so the subsequent Outer ER query can
silently read `FUNC1` instead of the primary `CHAN1A` measurement. OMA and RLM
are similarly discarded when their existing source is not `CHAN1A`.

The result is an honest but permanent `PARTIAL` dataset even when density and
PNG files are valid. Changing the boolean alone is forbidden because it would
remove the evidence gate rather than repair it.

## Design

### Source-bound acquisition guard

Both the primary guard and the before/after dataset guards will read Outer ER
and TDECQ as independent source-bound groups. Each group performs:

1. select the expected source;
2. read back and verify that source;
3. read status, measurement count, and value;
4. record every command response and error;
5. restore the original shared measurement source and verify the restoration.

The guard is consistent only when the primary, before, and after command maps
are identical, their sources/statuses are correct, their counts are positive,
and all source restorations succeeded.

### Source-bound optional metrics

RLM, OMA, and average power will be explicitly bound to the configured raw
source before their value is accepted. The adapter records original,
requested, verified, and restored sources. An unavailable or invalid optional
metric remains missing; a failed source restoration aborts dataset capture
because DCA configuration safety is more important than publishing a bundle.

### Exclusive acquisition window

A resource-scoped, re-entrant coordinator will cover the complete sequence
from `measure_single()` through dataset capture. The production dataset adapter
receives a verifier tied to that coordinator. A consistent snapshot without
exclusive ownership remains unverified.

This coordinator prevents two native V1.2 LIVE sweep executors in the same
service process from interleaving DCA commands. It does not claim to be a
cross-process instrument lock or the V1.1 watchdog/process-isolation feature.
That limitation remains explicit in dataset evidence.

### Same-acquisition decision

`same_acquisition_verified=True` requires all of the following:

- the resource-scoped exclusive acquisition window is owned by the caller;
- the primary acquisition guard matched the authoritative OER/TDECQ results;
- source-bound guard snapshots before and after dataset queries match primary;
- all guard and optional-measurement source restores succeeded;
- raw and equalized eye sources were read back correctly;
- the adapter issued no acquisition command.

The evidence records the verification method and the in-process isolation
limit. `DatasetWriter` remains strict: it may publish `COMPLETE` only when the
capture is verified and all required metrics/files are valid.

## Failure handling

- Source mismatch or changed count/value keeps the dataset `PARTIAL`.
- Missing OMA/RLM keeps the dataset `PARTIAL` without invalidating the primary
  measurement.
- Waveform or measurement-source restoration failure aborts capture and makes
  dataset persistence fail without changing Sweep restoration behavior.
- No new acquisition, I2C write, BERT write, FIR write, or retry is introduced.

## Tests

Tests will first reproduce the old failure, then prove:

- source-bound primary guards survive a shared FlexDCA selector;
- consistent guards under exclusive ownership become verified;
- the same evidence without exclusive ownership remains unverified;
- source drift, value/count changes, or restoration failure cannot verify;
- RLM/OMA/average power are selected from and restored to the expected source;
- `measure_single()` and dataset capture execute inside one coordinator scope;
- a complete verified capture produces `dataset_status=COMPLETE`;
- the complete Python and frontend suites remain green.
