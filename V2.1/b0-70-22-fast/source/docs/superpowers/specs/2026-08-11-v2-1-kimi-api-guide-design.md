# V2.1 Kimi API Control and Collaboration Guide Design

## Goal

Create one concise Chinese Markdown manual for a beginner. The manual explains
how the existing V2.1 code lets Kimi, the local safety layer, DCA, and I2C work
together. It includes copyable code examples and plain-language explanations.

This task creates documentation only. It does not add a CLI, executable
controller, Python source module, Dashboard control, HTTP route, configuration
loader, or hardware path.

## Output

Create exactly one user-facing document:

```text
docs/v2.1-kimi-api-control-and-collaboration-guide.md
```

No production or test code is changed.

## Intended reader

The reader knows how to open PowerShell and set an environment variable but
does not need to understand Python architecture. Every technical term is
explained the first time it appears.

The manual keeps only the information needed to answer:

1. Kimi controls what?
2. The local program controls what?
3. How is the API Key supplied?
4. What data is sent to Kimi?
5. How do I ask Kimi to score one eye without touching hardware?
6. How will one real I2C adjustment be requested later?
7. Where are results and memory saved?
8. What should I check when something stops?

## Manual structure

### 1. One-page overview

Explain the collaboration in one short flow:

```text
DCA or saved sample
  -> local program prepares the eye and safe candidates
  -> Kimi scores the eye and chooses at most one candidate
  -> local safety checks accept or reject the choice
  -> V2.0 runner writes I2C, reads it back, and asks DCA to measure again
  -> local files remember the result for the next Kimi request
```

State prominently that Kimi never receives an I2C object and never calls
`apply`, serial, VISA, or DCA methods.

### 2. Roles in plain language

Use a four-row table:

- Kimi: look at the eye, score it, and select a supplied candidate;
- local V2.1 code: prepare the payload, remember history, and validate Kimi;
- V2.0 safety chain: validate range and step, execute, read back, remeasure,
  roll back, or enter `RECOVERY_REQUIRED`;
- operator: provide the API Key, real sample, approved ranges, and Campaign
  authorization.

### 3. Before starting

List only the required items:

- the `codex/v2.1-kimi-live` checkout;
- the project Python environment with the `kimi` optional dependency;
- `MOONSHOT_API_KEY` in the current PowerShell session;
- one valid saved `EyeSample` with its NPY artifact;
- locally prepared I2C-only candidates;
- for live use only: DCA, active Campaign, matching authorization, reviewed B0,
  and at least one explicit safety limit.

### 4. API Key

Show the existing environment-variable form:

```powershell
$env:MOONSHOT_API_KEY = "<your real key>"
```

Explain that the key is read by `KimiApiClient`, is not placed in a JSON file,
and lasts only for the current PowerShell process unless the user stores it by
some separate Windows mechanism. Never print or repeat a real key.

### 5. What is sent to Kimi

Explain that one request contains:

- the current eye PNG generated from the verified NPY;
- the best prior eye PNG when one exists;
- at most five compact prior summaries;
- current readback and DCA metrics;
- the instruction;
- local safe candidate IDs and values.

Explain that the full screen, API Key, complete image history, raw NPY text,
and hardware objects are not sent.

### 6. Score-only code example

Show a focused Python example that uses the existing:

- `KimiApiClient`;
- `KimiCampaignMemory`;
- `KimiProvider`;
- `MultimodalApiBrain`;
- `BrainFactory` with `PURE_API`.

The example assumes `sample`, `candidates`, and `artifact_root` have already
been loaded by the existing data path. It calls the factory-gated brain's
analysis interface only. It may append a successfully parsed result to a
separate `kimi_shadow_memory.jsonl` by using the existing
`KimiCampaignMemory.append` method. It does not import a live runtime or call
`execute_kimi_step`.

The guide explains that shadow memory and live execution memory must use
different paths. This prevents an exploratory score from being mistaken for a
hardware execution record or causing a duplicate-measurement collision later.

Below the code, explain every object in one sentence and explain how to read
`PROPOSE`, `HOLD`, confidence, quality score, and reason codes.

The manual must label this example clearly: it is real API scoring code, but it
is not a standalone sample-file loader.

### 7. Live single-step code example

Show the smallest accurate call to the existing `submit_kimi_step` function.
The example assumes the existing V2.0 workflow has already produced:

- `runtime`;
- active `campaign_id`;
- factory-gated `brain`;
- live Campaign `memory`;
- current `sample`;
- matching `authorization`;
- local `candidates`.

Explain each argument and show that the returned `Future` must be awaited with
`.result()`. State that `submit_kimi_step` is the only documented live entry in
the guide. Do not show direct I2C writes or adapter calls.

The guide must explicitly say that this code cannot run until DCA sample
binding, Campaign setup, stage authorization, and safety limits exist.

### 8. Results and memory

Explain:

- `v21_result.json`: what Kimi proposed and whether the selected candidate was
  held, executed, or failed;
- Campaign observation records: actual readback and new DCA measurement;
- `kimi_memory.jsonl`: compact local history used to prepare later API calls.

Clarify that a proposal record is not proof of hardware execution. Actual
execution requires an `EXECUTED` result plus matching readback and a new DCA
observation.

### 9. Stop and failure meanings

Use a small table for:

- `HOLD`;
- `PROVIDER_ERROR`;
- `LOW_CONFIDENCE`;
- `INVALID_EYE`;
- `OOD`;
- `FAILED`;
- `RECOVERY_REQUIRED`.

Each row gives one plain-language meaning and one next action. Do not expose
raw provider exceptions, API payloads, base64 data, or secrets.

### 10. Recommended test order

Give four short stages:

1. real Kimi API plus one saved eye, no hardware;
2. five to ten new DCA samples in scoring-only use;
3. one approved I2C parameter and one live step;
4. only after those pass, repeated optimization.

State that unresolved same-acquisition binding and raw/equalized-eye separation
allow plumbing tests but prevent claims about optimization effectiveness.

## Code-example rules

Every code block must match the current V2.1 interfaces. Code that depends on
objects produced elsewhere must say so immediately above the block. The guide
must not invent a nonexistent `vcsel-v21-kimi` command or imply that the current
branch has a one-click live launcher.

The examples must never:

- place an API Key literal in Python;
- construct a fake authorization for live use;
- call I2C, serial, DCA, BERT, FIR, or `setFir` directly;
- infer candidate ranges from register widths;
- claim that API scoring proves eye-quality improvement.

## Verification

Before committing the manual:

- compare every import and function signature with current V2.1 source;
- verify all referenced file links exist;
- scan for `TBD`, `TODO`, nonexistent commands, and secret-like strings;
- verify the live code calls only `submit_kimi_step` for execution;
- verify the score-only code has no live runtime or hardware call;
- proofread the document for beginner readability and remove repeated safety
  explanations.

## Out of scope

- implementing or changing any controller;
- adding a command-line program;
- calling the real Kimi API during document creation;
- connecting DCA, I2C, or BERT;
- changing Campaign, authorization, memory, or safety behavior;
- adding automatic optimization or model training.
