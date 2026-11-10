# Small-circuit plot generation

Scripts behind the figures in `plots_small_circuits/` (4x4 `_area` MNIST, FRQI). Data scripts write `plotdata/`; `make_plots.py`
renders from `plotdata/` only, so figures can be re-rendered without re-running anything expensive. `plotdata/` is not committed: rebuild it
with steps 1-4, which need the training runs in `apps/logistics/experiments/` on your machine.

Run with the `qugen` environment from any directory. Paths are resolved relative to the repo.

| Step | Script | Writes | Figures |
|---|---|---|---|
| 1 | `collect_diagnostics.py <0\|1>` | `diag_<k>`, `refs.pkl` (real-data classifier and references) | 06-08, 10-12, 14, 15, 17, baseline lines in 18-22 |
| 2 | `collect_tests.py <B\|C\|D\|E>` | `B` export/hardware, `C` shot noise, `D` short training checks, `E` uncompiled vs compiled | 02 (B), 03 (C), 04 (D), 05 (E), 16 (B, C) |
| 3 | `draw_circuits.py [ids]` | `circuits/`, and the diagrams in `plots_small_circuits/circuits/` | `circuits/c00_*` |
| 4 | `run_compare_all.py` (runs `collect_compare.py` rows 2-7, then `collect_nocache.py`) | `cmp_<row>`, `D2` | 04, 18-22 |
| 5 | `make_plots.py [names]` | `plots_small_circuits/*.png`, `decoder_comparison.csv` | all except 01 |

- Figures 09 and 13 read `evaluation_summary.csv` of the two baseline runs in `apps/logistics/experiments/` instead of `plotdata/`.
- Figure 01 (export round trip) came from an earlier script that is not kept here.
- Step 1 must run before step 4 (`collect_compare.py` reads `refs.pkl`).
- Run sections C, D and E of `collect_tests.py` in a fresh process, separate from B: they set `jax_enable_x64` before any JAX work.
- Step 4 needs the trained decoder-comparison runs (rows 2-7 of `experiments_configs/runs_small_circuits.csv`) in
  `apps/logistics/experiments/`.
- `check_pairing.py` checks that both decoders of a seed start from the same weights; `peek_compare.py <iteration>` is a quick
  coverage check of running trainings.
