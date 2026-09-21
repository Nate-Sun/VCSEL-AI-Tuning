# 精简范围与恢复限制 / Compact scope and restoration limits

保留：七个版本快照中的源码、配置、测试及历史文档；JSON/SQLite 结构化实验结果；Excel/NDJSON 报告；审核、失败和早期运行的结构化资料。所有版本的原始身份见 VERSIONS.md。
Retained: source/configuration/tests/historical documents from seven snapshots, structured JSON/SQLite evidence, Excel/NDJSON reports, and structured audit/failure/early-run records. See VERSIONS.md for original identities.

排除：实验 PNG/NPY、采集与执行 JSONL 大日志、Git 历史 bundle、源码 ZIP 的重复副本、旧界面截图、缓存/运行锁/个人 .env 和中断写入的 .tmp-* 文件。唯一保留的 JSONL 为源码中的小型测试夹具。
Excluded: experimental PNG/NPY, bulk acquisition/execution JSONL, Git history bundle, duplicate source ZIPs, old UI screenshots, caches/runtime locks/personal .env and interrupted .tmp-* writes. Only small source test fixtures retain JSONL format.

精简仓库用于代码接手、结构化结果阅读和继续开发，不能完成原始视觉输入的完整重放。完整证据仍位于本地公司归档 AI_TUNING_FINAL_2026-09-21；本仓库不替代该备份。
This repository supports code handoff, structured-result review and development, but cannot fully replay original visual inputs. Complete evidence remains in the local company archive AI_TUNING_FINAL_2026-09-21; this repository does not replace it.

audit/PROVENANCE.csv 映射上传文件至原始归档路径或 ZIP 条目，记录保留字节的 SHA-256。audit/EXCLUSIONS.csv 列出被筛除的数据和 UI 截图。MANIFEST_SHA256.csv 覆盖全部交付文件（除自身），不会把 Git 数据库列入交付校验。
PROVENANCE.csv maps retained files to archive paths or ZIP entries and records their hashes. EXCLUSIONS.csv lists omitted evidence/UI screenshots. MANIFEST_SHA256.csv covers delivered files except itself and excludes the Git database.

新的双语说明是精简版权威入口；历史源码中的旧机器路径、设计计划和截图链接按原样保留，可能指向未上传内容。已有 API 密钥和个人凭据不应写入源码或数据；硬件地址与个人路径可能作为内部历史记录出现，仓库保持私有。
The new bilingual guides are the compact edition's primary entry points. Historical source paths, design plans and screenshot links remain unchanged and may reference omitted material. Do not add credentials to code/data. Bench addresses and local paths can appear in historical records; keep this repository private.
