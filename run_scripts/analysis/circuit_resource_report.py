# Phase 3: report gate count, circuit depth (layers), and trainable-parameter count for
# generator circuit variants -- no training involved, just circuit construction, so this
# is fast and cheap (safe to run even in a constrained environment). Captures the
# "Generator circuit specs" text each circuit factory already prints via qml.specs(), and
# parses it, rather than depending on PennyLane's internal Resources object API (which
# can differ across versions).
#
# Example:
#   python run_scripts/analysis/circuit_resource_report.py --n_qubits_list='(5,7,9)'

import contextlib
import io
import os
import re

import fire
import numpy as np
import pandas as pd

from qugen.main.generator.quantum_circuits.factory import generator_lookup

# These raise NotImplementedError unconditionally in this codebase today -- skip them.
BROKEN_VARIANTS = ("SU4_blocks", "sparse_blocks")

TWO_QUBIT_GATES = {"CNOT", "CY", "CZ", "CRX", "CRY", "CRZ", "SWAP", "CSWAP", "ISWAP"}

WIRES_RE = re.compile(r"Total wire allocations:\s*(\d+)")
GATES_RE = re.compile(r"Total gates:\s*(\d+)")
DEPTH_RE = re.compile(r"Circuit depth:\s*(\d+)")
GATE_LINE_RE = re.compile(r"^\s{4}(\w+):\s*(\d+)\s*$", re.MULTILINE)


def resource_specs_for(generator_name, n_qubits, circuit_depth):
    gen_fn = generator_lookup[generator_name]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            _, _, _, params_shape = gen_fn(circuit_depth=circuit_depth, n_qubits=n_qubits)
    except NotImplementedError:
        return None
    except Exception as e:
        return {"error": str(e)}

    text = buf.getvalue()
    wires = WIRES_RE.search(text)
    gates = GATES_RE.search(text)
    depth = DEPTH_RE.search(text)
    gate_counts = {m.group(1): int(m.group(2)) for m in GATE_LINE_RE.finditer(text)}
    two_qubit_gate_count = sum(c for g, c in gate_counts.items() if g in TWO_QUBIT_GATES)

    return dict(
        n_wire_allocations=int(wires.group(1)) if wires else None,
        total_gates=int(gates.group(1)) if gates else None,
        circuit_depth_layers=int(depth.group(1)) if depth else None,
        two_qubit_gate_count=two_qubit_gate_count,
        gate_breakdown=str(gate_counts),
        n_trainable_params=int(np.prod(params_shape)),
    )


def main(
    n_qubits_list=(5, 7, 9),
    circuit_depth=2,
    include_color_variants=False,  # color_rot_* variants need circuit_depth as a (depth, entangling_depth) tuple
    out_csv="run_scripts/analysis/results/phase3_circuit_resource_report.csv",
):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    names = [n for n in generator_lookup if not any(b in n for b in BROKEN_VARIANTS)]
    if not include_color_variants:
        names = [n for n in names if not n.startswith("color")]

    rows = []
    for name in names:
        for n_qubits in n_qubits_list:
            specs = resource_specs_for(name, n_qubits, circuit_depth)
            if specs is None:
                continue
            row = dict(generator_name=name, n_qubits=n_qubits, circuit_depth=circuit_depth)
            row.update(specs)
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"Saved {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    fire.Fire(main)
