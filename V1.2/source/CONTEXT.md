# Domain glossary

## Product lineage

**Native V1.2** — The original V1.2 product line that owns the continuous FIR convolution, optional Dataset capture, V1.2 launch configuration, and V1.2 storage layout. It is distinct from every V1.1-derived build that also displays version 1.2.0.

**V1.1-derived V1.2** — A separate integration line that inherited the V1.1 LIVE executor and safety modules. Its results and readiness claims never establish the readiness of Native V1.2.

**V1.1.1 FAST24** — The independently packaged V1.1 descendant used only as a reviewed source of LIVE behavior for selective migration. It is not the target product.

## Experiment concepts

**B0** — The frozen, independently read-back hardware state to which a LIVE task must return. It includes I2C settings, FIR taps, NLC settings, pattern, channel mapping, DCA configuration fingerprint, and device identities.

**PILOT_3** — A three-candidate, observation-conditioned I2C optimization profile. Each candidate is selected only after the previous valid observation is durable, and each candidate is followed by verified B0 restoration.

**FAST24** — A 24-round staged optimization profile: six coarse I2C rounds, sixteen fine I2C rounds, one first-pass formula FIR round, and one iterative formula FIR round.

**Staged LIVE campaign** — One pre-authorized task that runs PILOT_3 and automatically enters FAST24 only after the PILOT_3 safety gate is fully satisfied.

**Candidate** — A proposed hardware state that has not yet produced a durable observation.

**Observation** — A candidate's validated measurement, exact hardware readback, acquisition identity, and decision evidence after durable persistence.

**Dataset capture** — The optional evidence bundle containing density data, eye images, metrics, acquisition identities, applied settings, readbacks, and integrity hashes. Formal LIVE acceptance requires it for every round.

## Safety concepts

**DCA I/O isolation** — Ownership of one complete DCA connection and acquisition lifecycle by a killable child process that has no I2C or BERT mutation capability.

**DCA watchdog** — The deadline controller that observes the DCA child process, records operation evidence, and reclaims the process after a bounded timeout.

**Terminal-invalid observation** — A DCA result that cannot be ranked or reused. It carries no measurement value and ends the active LIVE stage.

**RECOVERY_REQUIRED** — A terminal safety state entered when B0 restoration cannot be completely verified. New LIVE work remains blocked until a dedicated recovery operation verifies B0.
