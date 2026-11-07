# VCSEL AI Tuning — 精简交付 / Compact handoff

整理日期 / Prepared: 2026-09-21。个人私有仓库；按 V1.2、V2.0、V2.1 分类保存源码与结构化实验结果。
Personal private repository containing versioned source snapshots and structured experimental results.

| 版本 / Version | 定位 / Status | 入口 / Start here |
|---|---|---|
| V1.2 原生分阶段 / Native staged | 最终硬件验证候选，有数据限制 / Hardware-tested candidate with evidence limitations | [V1.2](V1.2/README.md) |
| V2.0 AI Brain | 软件基础框架，FAKE provider / Software foundation, FAKE provider | [V2.0](V2.0/README.md) |
| V2.1 | 五个独立实验快照 / Five separate experimental snapshots | [V2.1](V2.1/README.md) |

**接手先读 / Start here:** [接手指南 / Handoff guide](docs/HANDOFF_GUIDE.md)（做什么、选哪个版本、参数、术语、换工位、已知问题 / what it does, which version, parameters, glossary, bench changes, known issues）→ [结果总结 / Results](docs/RESULTS.md)（关键数字、测量噪声、失败记录、重复性测试步骤 / key numbers, noise, failures, repeatability test）。

推荐阅读 / Read next: [版本与谱系 / Version lineage](docs/VERSIONS.md)、[安装使用 / Usage](docs/USAGE.md)、[数据范围 / Data scope](docs/DATA_SCOPE.md)、[验证与限制 / Verification](docs/VERIFICATION.md)。

源码已展开在各版本的 `source/` 下。历史源码文档按原样保存；本次新增的交付说明均为中英文双语。`main` 是精简交付目录的分支，各源码的原始 branch/commit 另见版本清单，不能用本仓库 main 的提交号冒充实验时的源码提交。
Sources are expanded under each version's `source/`. Historical source documentation is preserved verbatim; new handoff guides are bilingual. This repository's main branch packages snapshots; original experiment branch/commit identities are recorded separately.

不包含实验 PNG、NPY、大体积 JSONL 日志或原 Git bundle。源码内共 21 个小型 JSONL 测试夹具保留用于软件测试，它们不是被排除的实验日志。旧界面截图也已排除。
Experimental PNG/NPY, bulk JSONL journals and the original Git bundle are omitted. The 21 small tracked JSONL test fixtures remain for software tests; historical UI screenshots are omitted.

完整 10.69 GB 归档 `AI_TUNING_FINAL_2026-09-21` 继续保留在公司交付目录；本仓库不能完整重放视觉输入。数据中的旧路径和硬件地址仅用于溯源，不是新设备的即用配置。
The full 10.69 GB archive remains in the company handoff directory. This compact repository cannot fully replay visual inputs. Historical paths and bench addresses are provenance, not ready-to-use configuration for another bench.

V1.2 2h-r2 有 63/580 条、8h 有 284/2320 条记录缺少同次采集证明。V2.0 的只读边界针对 v2 模块，整个源码树仍含历史硬件入口。V2.1 保留为研发资产，不代表生产认证。
V1.2 lacks same-acquisition proof for 63/580 records in 2h-r2 and 284/2320 in 8h. V2.0's read-only boundary applies to the v2 module, not every inherited entry point. V2.1 is retained for research, not production certification.

下载后先运行 `python tools/verify_release.py`。在源码目录中安装依赖会改变本地文件集合；请在一份干净下载上核验交付内容。
After download, run `python tools/verify_release.py` first. Installation changes local file contents; verify a clean checkout before development.
