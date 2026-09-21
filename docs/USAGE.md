# 安装与使用 / Installation and usage

Windows 建议克隆到较短路径，部分历史实验文件名较长。例如 `git -c core.longpaths=true clone https://github.com/Nate-Sun/VCSEL-AI-Tuning.git C:/VCSEL-AI-Tuning`。私有仓库需要先登录有权限的 GitHub 账号；不要把访问令牌写进 URL。
On Windows, clone to a short path using the command above; some historical evidence names are long. Authenticate with the authorized GitHub account for this private repository; do not embed a token in the URL.

需要 Python >=3.12；本次既有测试环境为 Python 3.13.7。Python 和前端依赖分别由每个 source 下的 pyproject.toml、frontend/package.json 声明。依赖未完全锁定；安装需要可用的软件源，精简仓库不包含虚拟环境或 node_modules。
Python >=3.12 is required; the reviewed environment used Python 3.13.7. Dependencies are declared in each snapshot's pyproject.toml and frontend/package.json. Dependencies are not fully locked; installation needs a package source. Environments and node_modules are excluded.

## V1.2：先运行 FAKE / Run FAKE first

从仓库根目录执行：
Run from the repository root:

```powershell
Set-Location V1.2/source
python -m venv .venv-ai-tuning
.\.venv-ai-tuning\Scripts\python.exe -m pip install -e '.[dev]'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:VCSEL_AI_PYTHON = (Resolve-Path .\.venv-ai-tuning\Scripts\python.exe).Path
.\.venv-ai-tuning\Scripts\python.exe -c "import vcsel_ai_tuning; print(vcsel_ai_tuning.__file__)"
.\.venv-ai-tuning\Scripts\python.exe -m pytest tests_py/ai_tuning -q
npm.cmd --prefix frontend install
npm.cmd --prefix frontend run build
.\start-v1.2.cmd -Mode FAKE -Port 8765 -ValidateOnly
.\start-v1.2.cmd -Mode FAKE -Port 8765
```

浏览器访问 http://127.0.0.1:8765；在终端 Ctrl+C 结束。确认 Python 导入位置属于本版本，避免其他 worktree 的 editable install 混入。
Open http://127.0.0.1:8765 and stop with Ctrl+C. Confirm the Python import belongs to this snapshot; avoid another worktree's editable installation.

真实实验必须重新核对仪器、COM/IP/VISA、通道映射、B0 和输出目录；安装 hardware extra 及现场驱动后，依原 SOP 完成 READ_ONLY preflight 和批准的写入/恢复检查。READ_ONLY 仍会连接仪器。使用策略 PILOT_3_THEN_FAST24，遇到 RECOVERY_REQUIRED 按恢复流程处理。
Before physical experiments, review instrument identity, COM/IP/VISA, channel mapping, B0 and storage. Install hardware extras/drivers and follow the original SOP for READ_ONLY preflight and approved write/restore checks. READ_ONLY still contacts equipment. Use PILOT_3_THEN_FAST24 and respect RECOVERY_REQUIRED.

## V2.0：离线模块 / Offline module

在 V2.0/source 内建立独立虚拟环境，安装 `.[dev]` 并设置本地 `src` 为 PYTHONPATH，然后运行 `python -m pytest tests_py/ai_tuning/v2 -q`。这里指对应虚拟环境的 python。
In V2.0/source, create an independent environment, install `.[dev]`, bind PYTHONPATH to its src, and run `python -m pytest tests_py/ai_tuning/v2 -q` using that environment's Python.

入口是 src/vcsel_ai_tuning/v2/ 及对应测试。V2.0 是基础框架，没有已认证的独立自动硬件调参应用；历史包版本显示 1.1.0 不影响原始分支/提交身份。
Start with the V2 module and its tests. V2.0 is a foundation, not a certified standalone automatic hardware-tuning application. Inherited 1.1.0 package labels do not identify the frozen branch/commit.

## V2.1：研发使用 / Research use

优先阅读 continuous/source/docs/v2-1-continuous-eye-testbench.md，再查看所选分支的设计说明。五个 source 目录独立安装，禁止覆盖混合。provider 凭据由使用者在本地配置，.env 文件被忽略。
Read continuous/source/docs/v2-1-continuous-eye-testbench.md and the selected variant's design notes. Install the five snapshots independently. Configure provider credentials locally; .env files are ignored.

## 数据阅读 / Reading evidence

JSON 提供参数、结果、状态和证据引用；SQLite 保存历史任务；Excel/NDJSON 保存结果导出。缺失的 PNG/NPY/JSONL 引用是精简策略造成的，不要据此伪造数据或修改原始记录。查看 SQLite 时使用只读连接并先复制到工作目录。
JSON stores parameters, results, status and evidence references; SQLite holds historical tasks; Excel/NDJSON provide exports. Missing PNG/NPY/JSONL references are intentional exclusions. Do not fabricate or rewrite evidence. Inspect SQLite through read-only connections on a working copy.
