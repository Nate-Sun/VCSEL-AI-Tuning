# Keysight FlexDCA FUNC1 / TDECQ equalizer SCPI research

Date: 2026-08-12
Scope: documentation-only research; no hardware connection, LIVE run, or device write was performed.

## Conclusion

Keysight's official FlexDCA Programmer's Guide and official example script support creating Function 1 as a TDECQ equalizer over Channel 1A with this core sequence:

```text
:FUNCtion1:FOPerator TEQualizer
:FUNCtion1:OPERand1 CHAN1A
:FUNCtion1:DISPlay ON
:MEASure:EYE:TDEQ:SOURce FUNCtion1
:MEASure:EYE:TDEQ
```

The first four lines modify only the FlexDCA waveform-processing/measurement configuration. They do not establish permission to write the DUT, I2C, BERT, FIR, NLC, or `setFir`. The final line installs/starts the TDECQ measurement; actual acquisition remains a separate operation.

This sequence is supported in principle for FlexDCA A.08.32.13 because the TDECQ equalizer and TDECQ measurement require A.05.60 or later. The official public pages reviewed do not explicitly list patch release A.08.32.13, so actual acceptance and exact query-return spelling must still be verified on the instrument under separate authorization.

## Official command evidence

| Purpose | Official syntax / expected behavior | Applicability and evidence |
|---|---|---|
| Create F1 as a TDECQ equalizer | `:FUNCtion1:FOPerator TEQualizer` | Keysight documents `:FUNCtion{1:64}:FOPerator` and the `TEQualizer` argument; TEQualizer creates a five-tap TDECQ equalizer and requires A.05.60+. [FOPerator command](https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/Commands/FUNCtion/FOPerator.htm) |
| Verify operator | `:FUNCtion1:FOPerator?` | The query syntax is official. The reviewed page does **not** explicitly guarantee whether the returned token is `TEQ`, `TEQualizer`, or another SCPI-normalized form. Do not hard-code one spelling without an authorized instrument observation. [FOPerator command](https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/Commands/FUNCtion/FOPerator.htm) |
| Bind F1 operand to Channel 1A | `:FUNCtion1:OPERand1 CHAN1A` | Official syntax accepts channel operands, and the official TDECQ script uses this exact form. [OPERand command](https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/Commands/FUNCtion/OPERand.htm), [official TDECQ Parallel script](https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/Python/TDECQ_parallel.htm) |
| Verify operand | `:FUNCtion1:OPERand1?` (or the single-operand query accepted by the target revision) | The command page documents `:FUNCtion{1:64}:OPERand?` and says `OPERand`/`OPERand1` are equivalent for a one-input function. It does not explicitly publish the exact query response spelling. Verify against normalized `CHAN1A` only after confirming the query form on A.08.32.13. [OPERand command](https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/Commands/FUNCtion/OPERand.htm) |
| Display F1 | `:FUNCtion1:DISPlay ON` | Official syntax is `ON|OFF|1|0`; query response is explicitly `1|0`. [DISPlay command](https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/Commands/FUNCtion/DISPlay.htm) |
| Verify F1 exists | `:FUNCtion1:STATus?` must return `AVA` | Keysight explicitly says a function becomes `AVA` after it is defined by `FOPerator`; display is not required for availability. [STATus query](https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/Commands/FUNCtion/STATus.htm) |
| Verify output is PAM4 | `:FUNCtion1:SIGNal:TYPE?` should report PAM4 | Function output signal type tracks its input by default (`AUTO`). Manual PAM4 setting requires tracking `MANual` and the applicable PAM license; this research does not recommend overriding a correctly tracked signal. [signal type](https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/Commands/FUNCtion/SIGNal_TYPE.htm), [tracking](https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/Commands/FUNCtion/SIGNal_TYPE_TRACking.htm) |
| Bind TDECQ to F1 | `:MEASure:EYE:TDEQ:SOURce FUNCtion1` | The official TDECQ page says the measurement source must be explicitly identified and its example uses a function source. The official TDECQ Parallel script uses `:MEAS:EYE:TDEQ:SOUR FUNC{n}` consistently. [TDECQ command](https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/Commands/MEASure/EYE_TDEQ.htm), [official script](https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/Python/TDECQ_parallel.htm) |
| Verify measurement result | `:MEASure:EYE:TDEQ:STATus?` must return `CORR` before reading `:MEASure:EYE:TDEQ?` | Keysight explicitly documents this readiness check and states TDECQ requires Eye Mode, PAM4, and pattern lock. [TDECQ command](https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/Commands/MEASure/EYE_TDEQ.htm) |
| Detect rejected SCPI | `:SYSTem:ERRor:COUNt?`, followed by `:SYSTem:ERRor:NEXT? DIAGnostic` | Keysight's official example reads the error buffer. `DIAGnostic` was added in A.08.30, so A.08.32.13 is new enough in version terms. [official error example](https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/Python/print_errors.htm) |

The Programmer's Guide describes FlexDCA as the GUI/application for N1010A and says programs can control N1010A FlexDCA and N1000A with little or no modification. [FlexDCA Programmer's Guide](https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/FlexDCA-PG-Home.htm)

## Recommended fail-closed verification order

This is a documentation-derived implementation recommendation, not a hardware-validated result:

1. Confirm the application is in Eye Mode and the input is a PAM4 signal with pattern lock; do not silently change unrelated acquisition configuration during a narrow FUNC1 repair.
2. Send `:FUNCtion1:FOPerator TEQualizer`.
3. Immediately inspect the SCPI error buffer; fail closed on any nonzero error.
4. Send `:FUNCtion1:OPERand1 CHAN1A` and inspect the error buffer again.
5. Send `:FUNCtion1:DISPlay ON` and inspect the error buffer again.
6. Query and require: F1 status `AVA`, operator semantically TEQualizer, operand semantically CHAN1A, display `1`, and signal type PAM4.
7. Send `:MEASure:EYE:TDEQ:SOURce FUNCtion1`; query the source and require normalized FUNC1 before acquisition.
8. Only after all checks pass may the existing TDECQ measurement/acquisition path continue. Any unsupported query, unexpected return token, or error-buffer entry must stop the operation.

`TIME1` is not part of Keysight's documented creation sequence for a TDECQ equalizer. It is a display-window/screenshot concern and must not be treated as proof that F1 is correctly defined.

## Important documentation ambiguity

The current official `:MEASure:EYE:TDEQ` page contains an internally inconsistent example: it creates/configures `FUNCtion2` but later selects `FUNCtion1` as the TDECQ source. That exact example must **not** be copied as-is. The official TDECQ Parallel script is internally consistent and supports the parameterized sequence `FUNC{n} -> CHAN{channel} -> TDEQ source FUNC{n}`.

Also, the public command pages do not prove the exact textual response returned by `FOPerator?`, `OPERand1?`, or `TDEQ:SOURce?` on A.08.32.13. Production verification should normalize documented SCPI short/long forms but must not map `CHAN1A` to `FUNC1` or otherwise relax the user's invariant that TDECQ source must be Function 1.

## What is not established by official material

- No official evidence was found that `:FUNCtion1:OPERand CHAN1A` is preferable to `:FUNCtion1:OPERand1 CHAN1A`; the official TDECQ script uses `OPERand1`, so that is the safer write form.
- No official evidence was found that `*OPC?` reliably synchronizes the three FUNCtion setup writes. Keysight limits reliable `*OPC?` use to a documented command list that does not include `FOPerator`, `OPERand`, or `DISPlay`. Use readback/error-buffer checks rather than assuming `*OPC?` proves completion. [*OPC](https://helpfiles.keysight.com/scopes/FlexDCA-PG/Content/Topics/Commands/Common/OPC.htm)
- No official source reviewed proves a hardware-validated V1.1/V1.2 implementation of this sequence.
- No real N1010A A.08.32.13 behavior was tested in this research task.

## Safety boundary

These commands alter FlexDCA software measurement/signal-processing state. They are not DUT-register operations. Nevertheless, they are real instrument-control writes and require explicit authorization before being sent to the laboratory DCA. This report does not authorize such execution.
