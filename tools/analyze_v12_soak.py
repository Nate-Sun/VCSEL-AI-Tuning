"""Summarize native V1.2 FAST24 / soak evidence (read-only, standard library only).

汇总原生 V1.2 FAST24 与 soak 测试的结构化证据（只读，仅用标准库）。

Usage / 用法:
    python tools/analyze_v12_soak.py            # prints a Markdown report
    python tools/analyze_v12_soak.py --json     # prints machine-readable stats

The script only reads V1.2/data/validated/*/v1.2/*/iteration-*/record.json.
It never opens SQLite files for writing and never contacts hardware.
脚本只读取 record.json，不写任何文件，不连接硬件。
"""
from __future__ import annotations

import collections
import datetime as dt
import glob
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "V1.2" / "data" / "validated"
RUNS = ("FAST24_SINGLE_RUN", "SOAK_2H", "SOAK_8H")
SEARCH_PHASES = ("i2c_coarse", "i2c_fine", "fir_first", "fir_iterative")
I2C_PHASES = ("pilot", "b0", "i2c_coarse", "i2c_fine", "final_verification")


def phase_of(observation_id: str) -> str:
    if "-pilot-" in observation_id:
        return "pilot"
    return observation_id.rsplit(":", 1)[-1]


def key_of(params: dict) -> tuple:
    return (params.get("ibias_dac"), params.get("res_gain"))


def load_campaigns(run: str) -> list[dict]:
    campaigns = []
    for folder in sorted(glob.glob(str(ROOT / run / "v1.2" / "*") + "/")):
        records = []
        for path in sorted(glob.glob(folder + "iteration-*/record.json")):
            with open(path, encoding="utf-8") as f:
                r = json.load(f)
            records.append(
                {
                    "phase": phase_of(r["observation_id"]),
                    "tdecq": (r.get("metrics") or {}).get("tdecq"),
                    "params": key_of(r.get("applied_params") or {}),
                    "time": r["captured_at_utc"],
                    "status": r.get("dataset_status"),
                }
            )
        if not records:
            continue
        by_phase = collections.defaultdict(list)
        for x in records:
            by_phase[x["phase"]].append(x)
        search = [x for x in records if x["phase"] in SEARCH_PHASES and x["tdecq"] is not None]
        best = min(search, key=lambda x: x["tdecq"]) if search else None
        campaigns.append(
            {
                "id": Path(folder).name,
                "start": records[0]["time"],
                "b0": by_phase["b0"][0]["tdecq"] if by_phase["b0"] else None,
                "final": by_phase["final_verification"][0]["tdecq"] if by_phase["final_verification"] else None,
                "best": best["tdecq"] if best else None,
                "best_params": best["params"] if best else None,
                "best_phase": best["phase"] if best else None,
                "records": records,
            }
        )
    return campaigns


def summarize(campaigns: list[dict]) -> dict:
    full = [c for c in campaigns if None not in (c["b0"], c["final"], c["best"])]
    b0 = [c["b0"] for c in full]
    final = [c["final"] for c in full]
    best = [c["best"] for c in full]
    gain = [b - f for b, f in zip(b0, final)]
    optimism = [f - b for f, b in zip(final, best)]
    out: dict = {
        "campaigns": len(campaigns),
        "b0_mean": st.mean(b0),
        "final_mean": st.mean(final),
        "best_mean": st.mean(best),
        "best_min": min(best),
        "best_max": max(best),
        "records": sum(len(c["records"]) for c in campaigns),
        "records_without_tdecq": sum(x["tdecq"] is None for c in campaigns for x in c["records"]),
        "fir_round_won": sum(1 for c in full if c["best_phase"].startswith("fir")),
    }
    if len(full) > 1:
        out.update(
            {
                "gain_mean": st.mean(gain),
                "gain_sd": st.stdev(gain),
                "gain_t": st.mean(gain) / (st.stdev(gain) / math.sqrt(len(gain))),
                "improved": sum(g > 0 for g in gain),
                "unchanged": sum(g == 0 for g in gain),
                "worse": sum(g < 0 for g in gain),
                "best_optimism_mean": st.mean(optimism),
            }
        )
    # Short-term repeatability: identical I2C settings measured more than once
    # inside one campaign (FIR stays at B0 in these phases).
    sq, dof = 0.0, 0
    for c in campaigns:
        groups = collections.defaultdict(list)
        for x in c["records"]:
            if x["phase"] in I2C_PHASES and x["tdecq"] is not None:
                groups[x["params"]].append(x["tdecq"])
        for values in groups.values():
            if len(values) > 1:
                m = st.mean(values)
                sq += sum((v - m) ** 2 for v in values)
                dof += len(values) - 1
    out["repeat_sd"] = math.sqrt(sq / dof) if dof else None
    out["repeat_dof"] = dof
    # Drift of B0 across the run, by quarter.
    if len(full) >= 8:
        q = len(full) // 4
        out["b0_by_quarter"] = [st.mean(b0[i * q:(i + 1) * q]) for i in range(4)]
        out["final_by_quarter"] = [st.mean(final[i * q:(i + 1) * q]) for i in range(4)]
    out["winners"] = collections.Counter(
        f"ibias_dac={c['best_params'][0]}, res_gain={c['best_params'][1]}" for c in full
    ).most_common(5)
    # Where measurements fail (TDECQ missing) in the I2C search.
    miss = {"ibias_dac": collections.Counter(), "res_gain": collections.Counter()}
    total = {"ibias_dac": collections.Counter(), "res_gain": collections.Counter()}
    for c in campaigns:
        for x in c["records"]:
            if x["phase"] in ("i2c_coarse", "i2c_fine"):
                for i, name in enumerate(("ibias_dac", "res_gain")):
                    total[name][x["params"][i]] += 1
                    if x["tdecq"] is None:
                        miss[name][x["params"][i]] += 1
    out["missing_by_value"] = {
        name: {str(v): f"{miss[name][v]}/{total[name][v]}" for v in sorted(total[name])} for name in total
    }
    if campaigns:
        t0 = dt.datetime.fromisoformat(campaigns[0]["start"])
        t1 = dt.datetime.fromisoformat(campaigns[-1]["start"])
        out["span_hours"] = (t1 - t0).total_seconds() / 3600
    return out


def main() -> int:
    stats = {run: summarize(load_campaigns(run)) for run in RUNS}
    if "--json" in sys.argv:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0
    f = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    print("| Run | Campaigns | B0 mean | Final verification mean | Best mean (min–max) | B0−Final mean ± SD | Improved / same / worse | Repeat SD |")
    print("|---|---|---|---|---|---|---|---|")
    for run, s in stats.items():
        gain = f"{f(s.get('gain_mean'))} ± {f(s.get('gain_sd'))}" if "gain_mean" in s else "-"
        counts = f"{s['improved']}/{s['unchanged']}/{s['worse']}" if "improved" in s else "-"
        print(
            f"| {run} | {s['campaigns']} | {f(s['b0_mean'])} | {f(s['final_mean'])} | "
            f"{f(s['best_mean'])} ({f(s['best_min'])}–{f(s['best_max'])}) | {gain} | {counts} | {f(s['repeat_sd'])} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
