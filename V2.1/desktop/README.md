# V2.1/desktop — 版本说明 / Version guide

实验快照，仅供追溯和继续研发，不是原生 V1.2 的硬件认证延续。
Experimental snapshot for traceability and further development; it does not inherit native V1.2 hardware acceptance.

- 原始分支 / Original branch: `codex/v2.1-continuous-eye-desktop-app`
- 原始提交 / Original commit: `5f6b70f35135256afe9c914a331fc82e89818fe6`
- 状态 / Status: **EXPERIMENTAL**
- 代码 / Code: [source](source/)

使用独立环境，在本目录的 source 下安装和测试：
Use a separate environment; install and test inside this version's source directory:

```powershell
Set-Location source
python -m venv .venv-ai-tuning
.\.venv-ai-tuning\Scripts\python.exe -m pip install -e '.[dev]'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
.\.venv-ai-tuning\Scripts\python.exe -c "import vcsel_ai_tuning; print(vcsel_ai_tuning.__file__)"
.\.venv-ai-tuning\Scripts\python.exe -m pytest tests_py/ai_tuning -q
```

本版本现有 data 目录（若存在）保存结构化证据，缺少原始图片/数组；旧记录中的路径不一定能在精简包中打开。不要把历史 SQLite state 用于新硬件运行。
The data directory, when present, preserves structured evidence without original images/arrays. Referenced historical paths may be absent. Do not reuse archived SQLite state for new hardware operation.

阅读 [安装使用 / Usage](../../docs/USAGE.md)、[数据范围 / Data scope](../../docs/DATA_SCOPE.md)、[验证限制 / Verification](../../docs/VERIFICATION.md)。
