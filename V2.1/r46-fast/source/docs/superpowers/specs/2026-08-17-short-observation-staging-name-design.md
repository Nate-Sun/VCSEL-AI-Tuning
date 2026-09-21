# V2.1 Short Observation Staging Name Design

## Problem

On Windows with `LongPathsEnabled=0`, the first continuous candidate failed while publishing its observation. The final observation path was 240 characters, but `CampaignRepository` derived the temporary name from the complete final filename and appended a UUID, producing a 278-character path.

## Chosen design

Keep every public and durable name unchanged:

- campaign IDs remain unchanged;
- candidate and observation IDs remain unchanged;
- final artifact paths and SQLite records remain unchanged;
- existing campaigns remain readable.

Only the unpublished sibling temporary filename changes. Instead of:

```text
.<complete-final-name>.tmp-<32-hex-uuid>
```

use the fixed-length form:

```text
.stage-<32-hex-uuid>.tmp
```

The temporary remains in the final file's parent directory, preserving the repository's same-directory atomic promotion and cleanup invariants. Exclusive creation continues to protect against collisions.

## Error handling and safety

- Do not enable Windows long paths or change system configuration.
- Do not retry failed repository writes blindly.
- Preserve the original `RepositoryError` chaining and rollback behavior.
- Add an explicit pre-write path-budget check for both final and temporary paths when Windows long paths are disabled. A violation must fail before publication with a diagnostic containing the path length and limit.
- Do not contact DCA, I2C, BERT, Kimi, or run LIVE during verification.

## Tests

- Reproduce the exact long campaign/candidate observation name from campaign `v21-b-20260817T090138707Z-6a4329e1` in a repository test.
- Prove the old derived temporary name exceeds 260 characters.
- Prove the fixed-length sibling name remains below 260 characters and the observation publishes successfully.
- Verify temporary cleanup, immutable publication, and existing repository tests remain green.

## Non-goals

- Shortening final observation IDs or campaign IDs.
- Changing artifact layout or SQLite schema.
- Modifying optimization policy, B0, parameter ranges, prompts, or hardware behavior.
