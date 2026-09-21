# 验证与已知限制 / Verification and limitations

验证日期 / Validation date: 2026-09-21。本次只运行离线软件测试和文件核验，没有接触仪器或调用 Kimi/API。
Only offline software tests and file verification were performed; no equipment or Kimi/API was contacted.

## 精简源码复测 / Compact source retest

| 版本 / Version | 原始提交 / Source commit | 本次结果 / Current result |
|---|---|---|
| Native V1.2 | c5bcca019d987abee08c055cf1480f62eb406da7 | 2012 passed, 2 skipped in 155.99s |
| V2.0 v2 module | eb102816175cd9083d4b3d718539f24fa40bf131 | 344 passed, 1 skipped in 8.95s |

测试使用各版本 source/src 的显式 PYTHONPATH，Python 3.13.7；执行 `pytest tests_py/ai_tuning -q` 或 V2.0 的 `pytest tests_py/ai_tuning/v2 -q`。跳过项均是 Windows symlink 权限不足。日志见 audit/compact-v12-pytest.log 和 audit/compact-v20-pytest.log。
Tests explicitly bound PYTHONPATH to each snapshot's source/src under Python 3.13.7. V1.2 tested the AI tuning suite; V2.0 tested its v2 module. All skips are Windows symlink privilege limitations. Logs are in audit/.

## 文件与数据 / Files and data

- 上传的源文件及结构化证据与原始 ZIP/完整归档的 SHA-256 一致；源码没有算法改动，排除了历史 UI 截图。来源映射为 audit/PROVENANCE.csv。
  Retained source and structured evidence match their source ZIP/full-archive hashes. No algorithm was changed; old UI screenshots were omitted. See the provenance map.
- 64 个 SQLite 数据库通过只读 `PRAGMA quick_check`；4 个 Excel 工作簿可读取为 ZIP/XML。该检查不意味着实验物理结论全部正确。
  All 64 SQLite databases passed read-only quick_check; four Excel workbooks could be read as ZIP/XML. This does not certify every experimental conclusion.
- Gitleaks 8.30.1 扫描交付目录及数据库/Excel 提取文本，未发现规则匹配的密钥。它是有限规则扫描，不是不存在任何敏感信息的保证。内部硬件地址、配置和历史路径仍保留，仓库应保持私有。
  Gitleaks 8.30.1 found no matching credentials in the release directory and extracted database/Excel text. Rule-based scanning is not proof that no sensitive information exists. Internal bench addresses, configuration and historical paths remain; keep the repository private.
- `python tools/verify_release.py` 校验新的 MANIFEST_SHA256.csv。清单不包含自身或 .git，不继承完整包的旧清单。
  The verification script checks the new manifest, excluding itself and the Git database; it does not reuse the full archive's manifest.

## 实验验收边界 / Experimental acceptance limits

原生 V1.2 的单次最终记录为 29 条 COMPLETE、最终确认 TDECQ 2.38 dB；仅适用于该样品与配置。2h-r2 的 580 条记录中有 63 条 PARTIAL，8h 的 2320 条中有 284 条 PARTIAL，均缺少 same_acquisition_proof。初次 2h 运行 FAILED，后续 r2 PASSED；两者都保留，不能混淆。
The final single native V1.2 run contains 29 COMPLETE records and final TDECQ 2.38 dB for that sample/setup only. The 2h-r2 run has 63/580 PARTIAL records; the 8h run has 284/2320, all lacking same_acquisition_proof. The initial 2h attempt FAILED and the rerun PASSED; both histories remain distinct.

通用 STANDARD/DEEP/ADVANCED 不因 FAST24 结果获得认证。V2.0 的正式测试 provider 为 FAKE；V2.1 五个分支没有在本次统一、进行全量复测或获得生产认证。原始数据审阅摘要见 audit/dataset_record_audit.json。
FAST24 evidence does not certify STANDARD/DEEP/ADVANCED. V2.0's delivered test provider is FAKE. V2.1 variants were not integrated, fully retested or production-certified here. See the dataset record audit.

本次未重新构建前端、未验证全新机器安装，依赖仍需独立安装且未完全锁定。上传版本缺少原始 PNG/NPY/实验 JSONL 和旧 Git 历史，不能独立完成原始视觉实验的全量重放。
The frontend was not rebuilt and clean-machine installation was not validated. Dependencies need separate installation and are not fully locked. Without original images, arrays, experimental journals and old Git history, this repository cannot fully replay the visual experiments.
