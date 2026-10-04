# Phase 4: join Phase 1-3 results into one ranked table plus a short plain-text summary.
# Run this after Phase 1, 2, and 3 have each produced their CSVs.
#
# Example:
#   python run_scripts/analysis/synthesize_report.py

import os

import fire
import pandas as pd


def main(
    phase1_csv="run_scripts/analysis/results/phase1_small_circuit_sweep.csv",
    phase2_csv="run_scripts/analysis/results/phase2_shot_count_sweep.csv",
    phase3_csv="run_scripts/analysis/results/phase3_circuit_resource_report.csv",
    out_path="run_scripts/analysis/results/phase4_summary.txt",
):
    lines = ["=== Phase 4: Synthesis ===\n"]

    phase1 = pd.read_csv(phase1_csv) if os.path.exists(phase1_csv) else None
    # keep_default_na=False: the "shots" column uses the literal string "exact" as a
    # sentinel, which must NOT be parsed as a missing value.
    phase2 = pd.read_csv(phase2_csv, keep_default_na=False) if os.path.exists(phase2_csv) else None
    phase3 = pd.read_csv(phase3_csv) if os.path.exists(phase3_csv) else None

    if phase1 is not None and phase3 is not None:
        merged = phase1.merge(
            phase3[["generator_name", "n_qubits", "circuit_depth", "two_qubit_gate_count",
                    "total_gates", "n_trainable_params"]],
            on=["generator_name", "n_qubits", "circuit_depth"],
            how="left",
        )
        merged = merged.dropna(subset=["kl_div", "two_qubit_gate_count"])
        # Lower KL divergence (better quality) achieved with fewer two-qubit gates (cheaper
        # on real hardware) is the trade-off this is meant to surface -- not a single "best"
        # metric, just a ranking heuristic to inspect by hand.
        merged["kl_per_two_qubit_gate"] = merged["kl_div"] * merged["two_qubit_gate_count"]
        merged = merged.sort_values("kl_per_two_qubit_gate")

        merged_csv = out_path.replace(".txt", "_merged.csv")
        merged.to_csv(merged_csv, index=False)

        lines.append("Best quality-per-two-qubit-gate configs (lower kl_div * gate_count is better):")
        lines.append(merged[["generator_name", "n_qubits", "circuit_depth", "kl_div",
                              "two_qubit_gate_count", "kl_per_two_qubit_gate"]].head(10).to_string(index=False))
        lines.append(f"\nFull merged table: {merged_csv}")
    else:
        lines.append("Phase 1 and/or Phase 3 results not found -- run those first.")

    if phase2 is not None:
        lines.append("\nShot-count sensitivity:")
        for mode, group in phase2.groupby("mode"):
            best = group.sort_values("kl_div").iloc[0]
            lines.append(f"  [{mode}] best KL divergence {best['kl_div']:.4f} at shots={best['shots']}")
    else:
        lines.append("\nPhase 2 results not found -- run that first.")

    report = "\n".join(str(l) for l in lines)
    print(report)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
