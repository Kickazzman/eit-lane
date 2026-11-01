# eit-lane

Electrical-impedance-tomography difference-imaging solver and spectral test
on the EIT mesh. Deterministic, CPU-only. Calibration case C-0: the Cheeger
bound holds on the EIT mesh (|Γ₂| = 8.122 ≥ bound 2.683, ratio 3.03).

## Spectral verdicts (2026-10-31)

- Crossover: no sharp mobility edge. Cheeger 2-way confirmed. K* = 2 vs
  N_loc = 73 gap. Modal frequencies confirmed. ker D untested.
- Resolution: mean 0.137, centre 0.042, edge 0.259; PSF 0.209; effective
  rank 52; two-inclusion limit 0.35; d = 0.2 unmeasured.

## Contents

- `eit_solver.py`, `run_tests.py` — solver and tests.
- `results_seed7.json`, `results_seed19.json`,
  `results_seed7_phantoms.npz`, `results_seed19_phantoms.npz` — data as run.
- `eit-lane-report.md`, `spectral-test-report.md` — reports.

## Credit

Research by Joshua McClain (Kickazzman).
Research partner: cobrajet (GitHub: cobrajet302gt).
License: MIT — see LICENSE. Free use for anyone, with the copyright and
permission notice kept. Provenance: see RECORD.md and REDACTION-LOG.md.
