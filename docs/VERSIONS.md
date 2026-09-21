# 版本清单 / Version manifest

每个目录代表独立冻结快照；禁止覆盖合并后继续沿用原始验证结论。
Each directory is a separate frozen snapshot. Combining them does not preserve their original validation claims.

| 目录 / Directory | 原始分支 / Original branch | 原始提交 / Original commit | 状态 / Status |
|---|---|---|---|
| [V1.2](../V1.2/README.md) | `codex/v1.2-native-pilot-fast24-isolation` | `c5bcca019d987abee08c055cf1480f62eb406da7` | HARDWARE_TESTED_CANDIDATE |
| [V2.0](../V2.0/README.md) | `codex/V2.0` | `eb102816175cd9083d4b3d718539f24fa40bf131` | SOFTWARE_FOUNDATION |
| [V2.1/continuous](../V2.1/continuous/README.md) | `codex/v2.1-continuous-eye-optimization` | `bb427a2f1b1ec687f2682e1acc3cc6d87a6e88d8` | EXPERIMENTAL |
| [V2.1/desktop](../V2.1/desktop/README.md) | `codex/v2.1-continuous-eye-desktop-app` | `5f6b70f35135256afe9c914a331fc82e89818fe6` | EXPERIMENTAL |
| [V2.1/b0-70-22-fast](../V2.1/b0-70-22-fast/README.md) | `codex/v2.1-continuous-eye-b0-70-22-fast` | `6c1a842f09a978d269bc77ef1c50c844a45bdfc0` | EXPERIMENTAL |
| [V2.1/r46-fast](../V2.1/r46-fast/README.md) | `codex/v2.1-continuous-eye-r46-fast` | `5a1f47dc670b3fd3683662951230c459bc9b08ce` | EXPERIMENTAL |
| [V2.1/kimi-live](../V2.1/kimi-live/README.md) | `codex/v2.1-kimi-live` | `5d04fc1724b0622873990055ac6be6e1765b535d` | EXPERIMENTAL |

原生 V1.2 与 V1.1 派生的 V1.2/V2.1 是不同谱系，不能仅凭 UI 版本号或 /api/v1.1 判断。原始 ZIP 中保留的代码字节未修改，除文档化排除的 UI 截图。
Native V1.2 is distinct from V1.1-derived V1.2/V2.1. UI version labels and API paths are insufficient identifiers. Retained source bytes match the original ZIPs; only documented UI screenshots were omitted.
