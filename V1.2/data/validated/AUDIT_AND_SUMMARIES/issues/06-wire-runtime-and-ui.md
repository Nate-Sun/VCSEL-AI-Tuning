# 06 — Wire the Native V1.2 runtime and truthful UI

**What to build:** Native V1.2 advertises, authorizes, launches, and displays only the staged LIVE capability actually implemented, while validation and non-LIVE modes remain connection-free.

**Blocked by:** 05 — Orchestrate the PILOT gate, FAST24, and recovery.

**Status:** resolved

- [ ] LIVE runtime creates the staged executor lazily only after an authorized task is prepared.
- [ ] FAKE, READ_ONLY, and LIVE `ValidateOnly` perform no VISA, I2C, BERT, `setFir`, Preflight, or artifact mutation.
- [ ] State, database, and artifacts remain under Native V1.2 directories, and the `/api/v1.1` route remains explicitly a compatibility path.
- [ ] Capabilities show product `1.2.0`, 27 rounds, exact stage budgets, 60-second DCA timeout, 5-second cleanup grace, Dataset acceptance requirement, formula FIR authority, and terminal B0 restoration.
- [ ] The Wizard shows one `PILOT_3 → FAST24` profile and does not offer unsupported LIVE Sweep or caller-defined FIR authority.
- [ ] Existing FAKE, READ_ONLY, historical-task, frontend contract, lint, and build checks remain green.
