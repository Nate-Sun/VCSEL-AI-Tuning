# DCA TDECQ → FIR Demo 快速开始

这版 Demo **只读取** FlexDCA 已经完成的 TDECQ 结果，计算并保存 7-tap FIR proposal。它没有 `--apply` 参数，不创建 BERT socket，也从不向 BERT 写入或发送命令；`setFir` 仅作为文本预览写入 JSON 和终端。

## 离线样例

在仓库根目录运行：

```powershell
py -3.13 scripts/run_fir_demo.py fixture --input configs/fir_demo_first.json --output artifacts/fir_demo_first
py -3.13 scripts/run_fir_demo.py fixture --input configs/fir_demo_iterative.json --output artifacts/fir_demo_iterative
```

结果保存在各输出目录的 `fir_proposal.json`。首轮样例应得到 `[1, 5, -20, 90, -6, 2, -1]`、L1 = 125；迭代样例应得到 `[0, 6, -26, 104, -7, -8, -2]`、L1 = 153。

## 只读 FlexDCA

先安装可选硬件依赖：

```powershell
py -3.13 -m pip install -e ".[hardware]"
```

首轮计算：

```powershell
py -3.13 scripts/run_fir_demo.py flexdca --visa-resource "TCPIP0::192.0.2.10::inst0::INSTR" --mode first --main 90 --upper-nlc 0 --lower-nlc 0 --function 1 --port port1 --channel 0 --output artifacts/fir_demo_live
```

迭代计算还必须显式提供恰好 7 个整数：

```powershell
py -3.13 scripts/run_fir_demo.py flexdca --visa-resource "TCPIP0::192.0.2.10::inst0::INSTR" --mode iterative --main 104 --upper-nlc 0 --lower-nlc 0 --current-fir "0,6,-28,112,-8,-10,0" --output artifacts/fir_demo_live_iterative
```

实机模式必须显式传入 `--upper-nlc` 和 `--lower-nlc`；示例中的 `0` 是人工输入，不是程序默认值。

退出码：`0` 表示 proposal 有效；proposal 无效时返回 `2`，仍保留 JSON，且 `command_preview` 为 `null`；argparse 检测到命令行语法错误时也返回 `2`，但不会生成 JSON；`1` 表示输入或仪器读取错误。

## English

This demo reads an already-completed FlexDCA TDECQ result and generates a JSON FIR proposal. It never writes to the BERT: there is no `--apply` option, no BERT socket, and `setFir` is text preview only. Live mode requires explicit upper and lower NLC values; iterative mode also requires exactly seven current FIR integers.
