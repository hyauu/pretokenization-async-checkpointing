#!/usr/bin/env python3
"""Recompute the completed case study main comparisons from frozen run-summary CSVs.

Run from any directory: python /path/to/analysis/reproduce.py
Requires Python >= 3.9. Figure generation additionally requires matplotlib.
No original experiment directory, GPU, network, or training library is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

MAIN_COLUMNS = [
    "stage", "design", "workload", "asset_block", "repeat_block", "position",
    "condition", "input_mode", "save_mode", "save_interval_updates", "slot_id",
    "elapsed_s", "supervised_tokens", "seconds_per_million_tokens", "updates",
    "periodic_commits", "initial_commits", "fetch_s", "fetch_fraction",
    "replicate_scope", "source_csv", "source_row", "cost_source_csv",
    "cost_source_row", "original_source_path", "original_cost_source_path",
]

EFFECT_COLUMNS = [
    "stage", "design", "workload", "block", "replicate_scope", "tokens_per_run",
    "reference_setting", "target_setting", "cache_savings_reference_s",
    "cache_savings_target_s", "cache_savings_reference_s_per_mtoken",
    "cache_savings_target_s_per_mtoken", "cache_savings_reference_pct",
    "cache_savings_target_pct", "interaction_s", "interaction_s_per_mtoken",
    "interaction_original_name", "interaction_original_value",
    "interaction_original_unit", "interaction_normalized_pct",
    "interaction_normalization", "reported_B_target_pct", "source_rows",
    "reference_effect_csv", "reference_effect_row",
]

STAGES = {
    "finish_v3": {
        "design": "input_path_x_saving_enabled",
        "cost_file": "input_and_save_costs.csv",
        "expected_count": 24,
        "scope": "three seeded data-window/anchor blocks per workload",
    },
    "frequency": {
        "design": "input_path_x_save_frequency",
        "cost_file": "window_costs.csv",
        "expected_count": 16,
        "scope": "four temporal blocks of the same U_1 window and anchor",
    },
    "overlap": {
        "design": "input_path_x_save_scheduling",
        "cost_file": "window_costs.csv",
        "expected_count": 16,
        "scope": "four temporal blocks of the same U_1 window and anchor",
    },
}


def read_csv(relative: str) -> list[dict]:
    """Keep the original physical row number, with the header on line 1."""
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        previous_end = 1
        rows = []
        for row in reader:
            row["_source_row"] = previous_end + 1
            previous_end = reader.line_num
            rows.append(row)
        return rows


def finite(value: str | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric input: {value!r}")
    return result


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in columns} for row in rows)


def close(actual: float, expected: float, label: str, checks: list[dict]) -> None:
    error = abs(actual - expected)
    checks.append({"label": label, "absolute_error": error})
    if not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-8):
        raise ValueError(f"Mismatch at {label}: {actual!r} versus {expected!r}")


def original_paths() -> dict[str, str]:
    manifest = json.loads((DATA / "source_manifest.json").read_text())
    paths = {}
    for item in manifest:
        source = ROOT / item["package_path"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Frozen source changed: {item['package_path']}")
        paths[item["package_path"]] = item["original_source_path"]
    return paths


def make_runs(paths: dict[str, str], checks: list[dict]) -> list[dict]:
    result = []
    for stage, spec in STAGES.items():
        source_csv = f"data/original/{stage}/formal_runs.csv"
        cost_csv = f"data/original/{stage}/{spec['cost_file']}"
        original = read_csv(source_csv)
        costs = read_csv(cost_csv)
        key_name = "slot_id" if stage == "finish_v3" else "attempt_id"
        cost_by_id = {row[key_name]: row for row in costs}
        if len(cost_by_id) != len(costs):
            raise ValueError(f"Duplicate cost identifiers in {cost_csv}")
        if len(original) != spec["expected_count"]:
            raise ValueError(f"Unexpected number of frozen runs in {stage}")
        for row in original:
            is_finish = stage == "finish_v3"
            status = row["slot_status"] if is_finish else row["status"]
            if status != "VALID":
                raise ValueError(f"Unexpected non-valid frozen row: {row['slot_id']}")
            slot = row["slot_id"]
            condition = row["condition"] if is_finish else row["cell_id"]
            cost = cost_by_id[slot]
            elapsed = finite(row["primary_elapsed_s"] if is_finish else row["elapsed_s"])
            tokens = int(row["effective_supervised_tokens"] if is_finish else row["tokens"])
            fetch = finite(cost["fetch_sum_s"] if is_finish else cost["fetch_s"])
            y = elapsed / (tokens / 1e6)
            supplied_y = finite(row["seconds_per_million_tokens"] if is_finish else row["y"])
            close(y, supplied_y, f"{slot}: y", checks)
            close(elapsed, finite(cost["primary_elapsed_s"] if is_finish else cost["elapsed_s"]),
                  f"{slot}: main/cost elapsed", checks)
            work = "Alpaca" if row.get("workload") == "A" else "UltraChat"
            block = int(row["block"] if is_finish else row["repeat_block"])
            saves = int(row["periodic_commits"])
            if not saves:
                save_mode, interval = "none", ""
            else:
                save_mode = "wait_after_each_save" if condition.endswith("WAIT") else "async"
                interval = (68 if work == "Alpaca" else 47) if is_finish else int(row["interval"])
            result.append({
                "stage": stage, "design": spec["design"], "workload": work,
                "asset_block": block if is_finish else 1, "repeat_block": block,
                "position": int(row["position"]), "condition": condition,
                "input_mode": "online" if condition.startswith("O") else "pretokenized",
                "save_mode": save_mode, "save_interval_updates": interval,
                "slot_id": slot, "elapsed_s": elapsed, "supervised_tokens": tokens,
                "seconds_per_million_tokens": y,
                "updates": int(row["updates"]) if is_finish else 282,
                "periodic_commits": saves, "initial_commits": int(row["initial_commits"]),
                "fetch_s": fetch, "fetch_fraction": fetch / elapsed,
                "replicate_scope": spec["scope"], "source_csv": source_csv,
                "source_row": row["_source_row"], "cost_source_csv": cost_csv,
                "cost_source_row": cost["_source_row"],
                "original_source_path": paths[source_csv],
                "original_cost_source_path": paths[cost_csv],
            })
    if len(result) != 56 or len({row["slot_id"] for row in result}) != 56:
        raise ValueError("Expected 56 unique main comparison runs")
    return result


def make_effects(runs: list[dict], checks: list[dict]) -> list[dict]:
    groups = defaultdict(dict)
    for row in runs:
        key = (row["stage"], row["workload"], row["repeat_block"])
        if row["condition"] in groups[key]:
            raise ValueError(f"Duplicate condition in block {key}")
        groups[key][row["condition"]] = row

    references = {}
    for stage in STAGES:
        filename = f"data/original/{stage}/reported_effects.csv"
        for row in read_csv(filename):
            work = "Alpaca" if row.get("workload") == "A" else "UltraChat"
            block = int(row.get("block", row.get("repeat_block")))
            references[(stage, work, block)] = (filename, row)

    result = []
    for stage in STAGES:
        stage_keys = sorted(key for key in groups if key[0] == stage)
        for key in stage_keys:
            _, workload, block = key
            group = groups[key]
            if stage == "finish_v3":
                names = ["O0", "P0", "OA", "PA"]
                settings = ["no_saving", "async_saving"]
                metric, unit = "G", "percent_of_online_no_save_time"
            elif stage == "frequency":
                names = ["O94", "P94", "O47", "P47"]
                settings = ["K94", "K47"]
                metric, unit = "I", "seconds_per_million_supervised_tokens"
            else:
                names = ["O_WAIT", "P_WAIT", "O_ASYNC", "P_ASYNC"]
                settings = ["WAIT", "ASYNC"]
                metric, unit = "H", "seconds_per_million_supervised_tokens"
            if set(group) != set(names):
                raise ValueError(f"Incomplete/mismatched four-cell block: {key}")
            ro, rp, to, tp = (group[name] for name in names)
            token_set = {row["supervised_tokens"] for row in group.values()}
            if len(token_set) != 1:
                raise ValueError(f"Within-block workload mismatch: {key}")
            tokens = token_set.pop()
            ref_s = ro["elapsed_s"] - rp["elapsed_s"]
            target_s = to["elapsed_s"] - tp["elapsed_s"]
            ref_y = ro["seconds_per_million_tokens"] - rp["seconds_per_million_tokens"]
            target_y = to["seconds_per_million_tokens"] - tp["seconds_per_million_tokens"]
            interaction_y = target_y - ref_y
            interaction_s = target_s - ref_s
            ref_pct = 100 * ref_s / ro["elapsed_s"]
            target_pct = 100 * target_s / to["elapsed_s"]
            denominator_y = (ro["seconds_per_million_tokens"] if stage == "finish_v3"
                             else statistics.fmean(row["seconds_per_million_tokens"]
                                                   for row in group.values()))
            normalized = 100 * interaction_y / denominator_y
            original_metric = normalized if stage == "finish_v3" else interaction_y
            reference_csv, reference = references[key]
            original_B_key = {"finish_v3": "B_pct", "frequency": "B47", "overlap": "B_async"}[stage]
            original_metric_key = "G_pct" if stage == "finish_v3" else metric
            close(original_metric, finite(reference[original_metric_key]), f"{key}: {metric}", checks)
            close(-target_pct, finite(reference[original_B_key]), f"{key}: B target", checks)
            close(interaction_s, interaction_y * tokens / 1e6, f"{key}: seconds normalization", checks)
            if stage == "frequency":
                for actual, col in [(ref_y, "D94"), (target_y, "D47"),
                                    (-ref_pct, "B94"), (normalized, "J")]:
                    close(actual, finite(reference[col]), f"{key}: {col}", checks)
            elif stage == "overlap":
                for actual, col in [(ref_y, "D_wait"), (target_y, "D_async"),
                                    (-ref_pct, "B_wait"), (normalized, "Q"),
                                    (interaction_s, "H_seconds")]:
                    close(actual, finite(reference[col]), f"{key}: {col}", checks)
            result.append({
                "stage": stage, "design": STAGES[stage]["design"], "workload": workload,
                "block": block, "replicate_scope": STAGES[stage]["scope"],
                "tokens_per_run": tokens,
                "reference_setting": settings[0], "target_setting": settings[1],
                "cache_savings_reference_s": ref_s, "cache_savings_target_s": target_s,
                "cache_savings_reference_s_per_mtoken": ref_y,
                "cache_savings_target_s_per_mtoken": target_y,
                "cache_savings_reference_pct": ref_pct, "cache_savings_target_pct": target_pct,
                "interaction_s": interaction_s, "interaction_s_per_mtoken": interaction_y,
                "interaction_original_name": metric, "interaction_original_value": original_metric,
                "interaction_original_unit": unit, "interaction_normalized_pct": normalized,
                "interaction_normalization": "online_no_save_time" if stage == "finish_v3" else "four_cell_mean_time",
                "reported_B_target_pct": finite(reference[original_B_key]),
                "source_rows": ";".join(f"{group[name]['source_csv']}:{group[name]['source_row']}" for name in names),
                "reference_effect_csv": reference_csv,
                "reference_effect_row": reference["_source_row"],
            })
    if len(result) != 14:
        raise ValueError("Expected 14 complete block contrasts")
    return result


def summarize(runs: list[dict], effects: list[dict], checks: list[dict]) -> dict:
    output = {
        "main_runs": len(runs), "effect_rows": len(effects),
        "updates": sum(row["updates"] for row in runs),
        "supervised_tokens": sum(row["supervised_tokens"] for row in runs),
        "periodic_commits": sum(row["periodic_commits"] for row in runs),
        "initial_commits": sum(row["initial_commits"] for row in runs),
        "run_counts_by_stage": dict(Counter(row["stage"] for row in runs)),
        "verification": {"comparisons": len(checks),
                         "max_absolute_error": max(item["absolute_error"] for item in checks),
                         "source_hashes_checked": 9,
                         "scope": "Summary-CSV arithmetic, not a revalidation of full experimental traces."},
        "block_contrasts": {},
    }
    for stage in STAGES:
        for work in sorted({row["workload"] for row in effects if row["stage"] == stage}):
            rows = [row for row in effects if row["stage"] == stage and row["workload"] == work]
            values = [row["interaction_original_value"] for row in rows]
            output["block_contrasts"][f"{stage}/{work}"] = {
                "metric": rows[0]["interaction_original_name"],
                "unit": rows[0]["interaction_original_unit"], "n_blocks": len(rows),
                "values": values, "mean": statistics.fmean(values),
                "median": statistics.median(values), "min": min(values), "max": max(values),
                "sample_sd": statistics.stdev(values),
                "sd_scope": "Descriptive SD across the listed complete-block contrasts; not a standard error or confidence interval.",
                "positive": sum(value > 0 for value in values),
                "negative": sum(value < 0 for value in values),
                "leave_one_out_means": [statistics.fmean(values[:i] + values[i + 1:]) for i in range(len(values))],
                "cache_savings_target_s": [row["cache_savings_target_s"] for row in rows],
                "cache_savings_target_pct": [row["cache_savings_target_pct"] for row in rows],
                "mean_cache_savings_target_s": statistics.fmean(row["cache_savings_target_s"] for row in rows),
                "mean_cache_savings_reference_s": statistics.fmean(row["cache_savings_reference_s"] for row in rows),
                "mean_reported_B_target_pct": statistics.fmean(row["reported_B_target_pct"] for row in rows),
            }
    expected_totals = {"updates": 17304, "supervised_tokens": 190265456,
                       "periodic_commits": 240, "initial_commits": 44}
    for key, expected in expected_totals.items():
        if output[key] != expected:
            raise ValueError(f"Workload total mismatch: {key}")
    return output


def make_figure(effects: list[dict]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "case_study_paper_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.55), sharey=True)
    specs = [
        ("frequency", "A. Saving frequency: K47 versus K94", "#2563a6", "I"),
        ("overlap", "B. Save scheduling: ASYNC versus WAIT", "#bc5b16", "H"),
    ]
    for ax, (stage, title, color, symbol) in zip(axes, specs):
        rows = sorted((row for row in effects if row["stage"] == stage), key=lambda row: row["block"])
        x = [row["block"] for row in rows]
        y = [row["interaction_s_per_mtoken"] for row in rows]
        mean = statistics.fmean(y)
        ax.axhline(0, color="#555555", linewidth=0.9, zorder=1)
        ax.axhline(mean, color=color, linestyle="--", linewidth=1.2,
                   label=f"Mean {symbol} = {mean:+.3f}", zorder=2)
        ax.plot(x, y, color=color, linestyle="none", marker="o", markersize=5, zorder=3)
        for xx, yy in zip(x, y):
            ax.annotate(f"{yy:+.2f}", (xx, yy), xytext=(0, 7 if yy >= 0 else -13),
                        textcoords="offset points", ha="center", va="center", fontsize=9,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 0.3})
        ax.set_title(title, loc="left", pad=10)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlim(0.7, 4.3)
        ax.set_ylim(-16.5, 10.0)
        ax.set_xlabel("Temporal block")
        ax.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower left", frameon=False, fontsize=9, handlelength=2.2)
    axes[0].set_ylabel("Interaction (s / million supervised tokens)")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.25, top=0.86, wspace=0.15)
    fig.text(0.5, 0.055,
             "Same U₁ workload, four temporal blocks per panel. Different interventions; no pooled estimate.",
             ha="center", va="center", fontsize=8.5)
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "block_contrasts.png", dpi=300, facecolor="white")
    fig.savefig(figure_dir / "block_contrasts.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-figure", action="store_true", help="Recompute CSV/JSON without matplotlib.")
    args = parser.parse_args()
    checks = []
    paths = original_paths()
    runs = make_runs(paths, checks)
    effects = make_effects(runs, checks)
    summary = summarize(runs, effects, checks)
    write_csv(DATA / "prior_main_runs.csv", MAIN_COLUMNS, runs)
    write_csv(DATA / "prior_effects.csv", EFFECT_COLUMNS, effects)
    (DATA / "prior_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    if not args.skip_figure:
        make_figure(effects)
    print(json.dumps({"main_runs": len(runs), "effect_rows": len(effects),
                      "updates": summary["updates"], "supervised_tokens": summary["supervised_tokens"],
                      "arithmetic_checks": len(checks),
                      "max_absolute_error": summary["verification"]["max_absolute_error"],
                      "figure_generated": not args.skip_figure}, indent=2))


if __name__ == "__main__":
    main()
