#!/usr/bin/env python3
"""Recompute the case study placement tables from archived source fields, without a GPU.

This script only writes placement_* files and the placement_contrasts figure.
It does not pool this intervention with earlier case study designs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ORIGINAL = DATA / "original" / "placement"
SCOPE = "six temporal blocks of fixed U1; block 3 crosses user pause and WSL reboot"
METRICS = ["B0", "BS", "BI", "ES", "EI", "J", "RO", "RP"]
CELLS = ["O0", "P0", "OS", "PS", "OI", "PI"]
TOKENS = 3975934
# Two-sided 95% Student-t critical values for the sample sizes used below.
# Reporting these intervals does not establish independence or normality.
T975 = {3: 4.302652729696142, 6: 2.570581835636314}

MAIN_COLUMNS = "stage design workload asset_block repeat_block position condition input_mode save_mode save_interval_updates slot_id elapsed_s supervised_tokens seconds_per_million_tokens updates periodic_commits initial_commits fetch_s fetch_fraction replicate_scope source_csv source_row cost_source_csv cost_source_row original_source_path original_cost_source_path".split()
MAIN_EXTRA = "attempt_id placement training_pool measurement_segment boot_id crosses_user_pause training_cpus writer_cpus source_result original_result_path original_result_sha256 pre_first_save_s final_drain_s".split()
EFFECT_COLUMNS = "stage design workload block replicate_scope tokens_per_run reference_setting target_setting cache_savings_reference_s cache_savings_target_s cache_savings_reference_s_per_mtoken cache_savings_target_s_per_mtoken cache_savings_reference_pct cache_savings_target_pct interaction_s interaction_s_per_mtoken interaction_original_name interaction_original_value interaction_original_unit interaction_normalized_pct interaction_normalization reported_B_target_pct source_rows reference_effect_csv reference_effect_row".split()
EFFECT_EXTRA = CELLS + METRICS + ["training_pool", "measurement_segments", "crosses_user_pause", "J_pre47_s", "J_post47_s", "J_before_final_drain_s", "J_final_drain_s"]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows, 2):
        row["_source_row"] = i
    return rows


def verify_close(actual, expected, label, checks, tolerance=1e-7):
    actual, expected = float(actual), float(expected)
    if not (math.isfinite(actual) and math.isfinite(expected)):
        raise ValueError(f"Non-finite value: {label}")
    error = abs(actual - expected)
    if error > tolerance:
        raise ValueError(f"Mismatch {label}: {actual} versus {expected}")
    checks.append({"label": label, "absolute_error": error})


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def descriptive(values):
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else None
    half = T975[n] * sd / math.sqrt(n) if n in T975 else None
    return {
        "n": n, "mean": mean, "median": statistics.median(values),
        "sample_sd": sd, "min": min(values), "max": max(values),
        "positive": sum(x > 0 for x in values),
        "negative": sum(x < 0 for x in values),
        "zero": sum(x == 0 for x in values),
        "ci95_t_approx": [mean-half, mean+half] if half is not None else None,
        "loo_means": [statistics.fmean(values[:i]+values[i+1:]) for i in range(n)] if n > 1 else [],
    }


def verify_sources():
    manifest = json.loads((DATA / "placement_source_manifest.json").read_text())
    lookup = {}
    for item in manifest:
        path = ROOT / item["package_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Frozen input changed: {item['package_path']}")
        lookup[item["package_path"]] = item
    return lookup


def reconstruct_runs(sources, checks):
    rows = read_csv(ORIGINAL / "runs.csv")
    if len(rows) != 36:
        raise ValueError("Expected 36 selected runs")
    groups, output, starts = {}, [], set()
    for row in rows:
        attempt = row["attempt_id"]
        relative = f"data/original/placement/results/{attempt}.core.json"
        result = json.loads((ROOT / relative).read_text())
        for flag in ["valid", "training_valid", "timing_valid", "placement_valid"]:
            if result[flag] is not True:
                raise ValueError(f"Selected run is not valid: {attempt}: {flag}")
        if result["primary_clock"] != "CLOCK_MONOTONIC_RAW":
            raise ValueError(f"Unexpected time domain: {attempt}")
        records = result["update_records"]
        if result["completed_updates"] != 282 or len(records) != 282:
            raise ValueError(f"Work amount mismatch: {attempt}")
        if sum(r["tokens"] for r in records) != TOKENS or result["effective_supervised_tokens"] != TOKENS:
            raise ValueError(f"Token amount mismatch: {attempt}")
        starts.add(result["measurement_initial_state"]["complete_digest"])
        elapsed = (result["measurement_end_raw_ns"] - result["measurement_start_raw_ns"]) / 1e9
        fetch = sum(r["fetch_s"] for r in records)
        prefix = records[46]["end"] - result["measurement_start_raw_s"]
        verify_close(elapsed, row["T_total_raw_s"], attempt+": RAW versus CSV", checks)
        verify_close(elapsed, result["elapsed_s"], attempt+": RAW versus result", checks)
        verify_close(fetch, row["fetch_sum_s"], attempt+": fetch", checks)
        verify_close(prefix, row["pre_first_save_s"], attempt+": prefix", checks)
        verify_close(result["final_drain_s"], row["final_drain_s"], attempt+": final drain", checks)
        block, cell = int(row["repeat_block"]), row["cell_id"]
        if cell != result["cell_id"] or block != result["repeat_block"]:
            raise ValueError(f"Identity mismatch: {attempt}")
        groups.setdefault(block, {})[cell] = {"result": result, "row": row, "elapsed_s": elapsed, "fetch_s": fetch, "prefix_s": prefix}
        saving = cell not in ["O0", "P0"]
        expected_commits = 6 if saving else 0
        expected_initial = 1 if saving else 0
        if result["checkpoint_commits"] != expected_commits or result["initial_checkpoint_commits"] != expected_initial:
            raise ValueError(f"Save count mismatch: {attempt}")
        source_csv = "data/original/placement/runs.csv"
        output.append({
            "stage": "placement", "design": "input_path_x_writer_cpu_pool", "workload": "UltraChat",
            "asset_block": 1, "repeat_block": block, "position": int(row["position"]), "condition": cell,
            "input_mode": "online" if cell.startswith("O") else "pretokenized",
            "save_mode": "async" if saving else "none", "save_interval_updates": 47 if saving else "",
            "slot_id": result["slot_id"], "elapsed_s": elapsed, "supervised_tokens": TOKENS,
            "seconds_per_million_tokens": elapsed / (TOKENS / 1e6),
            "updates": 282, "periodic_commits": expected_commits, "initial_commits": expected_initial,
            "fetch_s": fetch, "fetch_fraction": fetch / elapsed, "replicate_scope": SCOPE,
            "source_csv": source_csv, "source_row": row["_source_row"],
            "cost_source_csv": source_csv, "cost_source_row": row["_source_row"],
            "original_source_path": sources[source_csv]["original_source_path"],
            "original_cost_source_path": sources[source_csv]["original_source_path"],
            "attempt_id": attempt, "placement": "none" if not saving else "shared" if cell.endswith("S") else "separated",
            "training_pool": result["training_pool"], "measurement_segment": result["measurement_segment"],
            "boot_id": result["boot_id"], "crosses_user_pause": block == 3,
            "training_cpus": json.dumps(result["training_cpus"]), "writer_cpus": json.dumps(result["writer_cpus"]),
            "source_result": relative, "original_result_path": result["_provenance"]["original_source_path"],
            "original_result_sha256": result["_provenance"]["original_source_sha256"],
            "pre_first_save_s": prefix, "final_drain_s": result["final_drain_s"],
        })
    if len(starts) != 1 or len(groups) != 6 or len({r["slot_id"] for r in output}) != 36:
        raise ValueError("Starting-state or run identity mismatch")
    return output, groups


def reconstruct_effects(groups, checks):
    refs = {int(r["repeat_block"]): r for r in read_csv(ORIGINAL / "blocks.csv")}
    effects = []
    for block, cells in sorted(groups.items()):
        if set(cells) != set(CELLS):
            raise ValueError(f"Incomplete block {block}")
        value = {c: cells[c]["elapsed_s"] for c in CELLS}
        value["B0"] = value["O0"]-value["P0"]
        value["BS"] = value["OS"]-value["PS"]
        value["BI"] = value["OI"]-value["PI"]
        value["ES"] = value["BS"]-value["B0"]
        value["EI"] = value["BI"]-value["B0"]
        value["J"] = value["BS"]-value["BI"]
        value["RO"] = value["OS"]-value["OI"]
        value["RP"] = value["PS"]-value["PI"]
        verify_close(value["J"], value["RO"]-value["RP"], f"block {block}: interaction identity", checks)
        for label, extract in [
            ("J_pre47_s", lambda r: r["prefix_s"]),
            ("J_final_drain_s", lambda r: r["result"]["final_drain_s"]),
            ("J_before_final_drain_s", lambda r: r["result"]["elapsed_without_final_drain_s"]),
        ]:
            vals = {c: extract(cells[c]) for c in CELLS}
            value[label] = vals["OS"]-vals["PS"]-vals["OI"]+vals["PI"]
        value["J_post47_s"] = value["J"]-value["J_pre47_s"]
        for key, val in value.items():
            verify_close(val, refs[block][key], f"block {block}: {key}", checks)
        segments = sorted({int(cells[c]["result"]["measurement_segment"]) for c in CELLS})
        boots = {cells[c]["result"]["boot_id"] for c in CELLS}
        cross = len(boots) > 1
        if cross != (block == 3):
            raise ValueError("Unexpected boot-crossing block")
        effects.append({
            "stage": "placement", "design": "input_path_x_writer_cpu_pool", "workload": "UltraChat",
            "block": block, "replicate_scope": SCOPE, "tokens_per_run": TOKENS,
            "reference_setting": "separated_writer", "target_setting": "shared_writer",
            "cache_savings_reference_s": value["BI"], "cache_savings_target_s": value["BS"],
            "cache_savings_reference_s_per_mtoken": value["BI"]/(TOKENS/1e6),
            "cache_savings_target_s_per_mtoken": value["BS"]/(TOKENS/1e6),
            "cache_savings_reference_pct": 100*value["BI"]/value["OI"],
            "cache_savings_target_pct": 100*value["BS"]/value["OS"],
            "interaction_s": value["J"], "interaction_s_per_mtoken": value["J"]/(TOKENS/1e6),
            "interaction_original_name": "J", "interaction_original_value": value["J"],
            "interaction_original_unit": "seconds_per_training_window",
            "interaction_normalized_pct": "", "interaction_normalization": "not_reported",
            "reported_B_target_pct": "", # Prior-round B-percent convention is not used in this round.
            "source_rows": ";".join(f"data/original/placement/runs.csv:{cells[c]['row']['_source_row']}" for c in CELLS),
            "reference_effect_csv": "data/original/placement/blocks.csv", "reference_effect_row": refs[block]["_source_row"],
            **value, "training_pool": cells["O0"]["result"]["training_pool"],
            "measurement_segments": ",".join(map(str, segments)), "crosses_user_pause": cross,
        })
    return effects


def summarize(runs, effects, groups, checks):
    summary = {
        "stage": "placement", "run_count": len(runs), "complete_blocks": len(effects),
        "workload": "fixed U1", "supervised_tokens_per_run": TOKENS,
        "updates": sum(r["updates"] for r in runs),
        "microbatches": sum(r["updates"]*8 for r in runs),
        "supervised_tokens": sum(r["supervised_tokens"] for r in runs),
        "periodic_commits": sum(r["periodic_commits"] for r in runs),
        "initial_commits": sum(r["initial_commits"] for r in runs),
        "statistics_seconds": {m: descriptive([e[m] for e in effects]) for m in METRICS},
        "pool_J_seconds": {p: descriptive([e["J"] for e in effects if e["training_pool"] == p]) for p in ["P", "Q"]},
        "decomposition_means_seconds": {m: statistics.fmean(e[m] for e in effects) for m in ["J_pre47_s", "J_post47_s", "J_before_final_drain_s", "J_final_drain_s"]},
        "cell_means": {c: {field: statistics.fmean(r[field] for r in runs if r["condition"] == c) for field in ["elapsed_s", "fetch_s", "final_drain_s", "pre_first_save_s"]} for c in CELLS},
        "block3_cross_pause_boot": True,
        "mean_J_s_per_mtoken": statistics.fmean(e["J"] for e in effects)/(TOKENS/1e6),
        "ci_qualification": "Two-sided t approximations based on complete time blocks; independence, exchangeability and normality are not established. These intervals do not remove time drift, CPU-pool differences, or block-3 pause/reboot confounding. They are not equivalence tests.",
        "loo_qualification": "All blocks remain in the primary descriptive estimates. Leave-one-block-out values are sensitivity descriptions, not selected replacements.",
        "scope_qualification": "This module reconstructs reported arithmetic from retained result fields. It does not independently reproduce training, certify all device traces, or identify a unique CPU-contention cause.",
    }
    refs = json.loads((ORIGINAL / "summary.json").read_text())
    for m, s in summary["statistics_seconds"].items():
        for key in ["mean", "median", "min", "max"]:
            verify_close(s[key], refs["statistics"][m][key], m+": summary "+key, checks)
        for a, b in zip(s["ci95_t_approx"], refs["statistics"][m]["ci95"]):
            verify_close(a, b, m+": t approximation", checks, tolerance=1e-6)
        for i, (a, b) in enumerate(zip(s["loo_means"], refs["statistics"][m]["loo_means"]), 1):
            verify_close(a, b, f"{m}: LOO {i}", checks)
    expected = {"run_count":36, "complete_blocks":6, "updates":10152, "microbatches":81216, "supervised_tokens":143133624, "periodic_commits":144, "initial_commits":24}
    for key, expected_value in expected.items():
        if summary[key] != expected_value:
            raise ValueError(f"Unexpected aggregate: {key}")
    summary["verification"] = {"arithmetic_comparisons":len(checks), "max_absolute_error":max(c["absolute_error"] for c in checks), "all_source_file_hashes_matched":True}
    return summary


def atomic_bytes(path, content):
    temp = path.with_suffix(path.suffix+".tmp")
    with temp.open("wb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def make_figure(effects):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"case_study_placement_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.titlesize":10,
                         "pdf.fonttype":42, "ps.fonttype":42})
    fig, axes = plt.subplots(1,2,figsize=(8.1,3.6))
    panels = [
        ("J", "A. Writer-placement interaction (J)", (-20,80)),
        ("ES", "B. Cache × shared-saving interaction (ES)", (-33,48)),
    ]
    for ax, (metric,title,limits) in zip(axes,panels):
        values=[e[metric] for e in effects]
        ax.axvspan(2.7,3.3,color="#e4e7eb",alpha=.8,zorder=0)
        ax.axhline(0,color="#555555",linewidth=.8,zorder=1)
        mean=statistics.fmean(values)
        ax.axhline(mean,color="#24669b",linestyle="--",linewidth=1,
                   label=f"Mean = {mean:+.2f} s",zorder=1)
        for e, value in zip(effects,values):
            block=e["block"]
            ax.scatter(block,value,s=25,facecolors="white" if block==3 else "#24669b",
                       edgecolors="#b96416" if block==3 else "#24669b",linewidths=1.2,zorder=3)
            ax.annotate(f"{value:+.2f}",(block,value),xytext=(0,8 if value>=0 else -12),
                        textcoords="offset points",ha="center",va="center",fontsize=8,
                        bbox={"facecolor":"white","edgecolor":"none","alpha":.92,"pad":.2})
        ax.set_title(title,loc="left",pad=11)
        ax.set_xlim(.6,6.4);ax.set_ylim(*limits)
        ax.set_xticks(range(1,7),["1","2","3*","4","5","6"])
        ax.set_xlabel("Temporal block");ax.set_ylabel("Contrast (seconds per window)")
        ax.grid(axis="y",linewidth=.45,color="#dddddd",zorder=0)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(loc="upper right",frameon=False,fontsize=8)
    fig.subplots_adjust(left=.085,right=.985,bottom=.24,top=.85,wspace=.29)
    fig.text(.5,.055,"* Block 3 spans the user pause and WSL reboot. All six blocks are retained.",
             ha="center",va="center",fontsize=8)
    output=ROOT/"figures";output.mkdir(exist_ok=True)
    for extension in ["png","pdf"]:
        buffer=io.BytesIO()
        fig.savefig(buffer,format=extension,dpi=300,facecolor="white")
        atomic_bytes(output/f"placement_contrasts.{extension}",buffer.getvalue())
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-figure",action="store_true")
    args=parser.parse_args()
    checks=[]
    sources=verify_sources()
    runs,groups=reconstruct_runs(sources,checks)
    effects=reconstruct_effects(groups,checks)
    summary=summarize(runs,effects,groups,checks)
    write_csv(DATA/"placement_runs.csv",MAIN_COLUMNS+MAIN_EXTRA,runs)
    write_csv(DATA/"placement_effects.csv",EFFECT_COLUMNS+EFFECT_EXTRA,effects)
    (DATA/"placement_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    if not args.skip_figure:
        make_figure(effects)
        if (ROOT/"figures/placement_contrasts.pdf").stat().st_size < 1000:
            raise ValueError("PDF figure is empty or incomplete")
    print(json.dumps({"runs":len(runs),"effects":len(effects),"updates":summary["updates"],
                      "tokens":summary["supervised_tokens"],**summary["verification"],
                      "figure_generated":not args.skip_figure},indent=2))


if __name__ == "__main__":
    main()
