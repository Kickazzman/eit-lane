# §8 Spectral Test on the EIT Mesh — Report

Date: 2026-10-31 (UTC). CPU-only, no model calls, no fleet/seat/service
changes. Protocol pre-registered BEFORE any computation:
`spectral-test/protocol.md`, sha256
`e5318007f4e803a279264e3f8fd210a2bf1867778a95ff6751d7824aac1e4903`.
No rule was adjusted after seeing results. EIT files used read-only
via a verbatim copy (`eit_solver_copy.py`, md5
`be034dcd58ad289523e2b206c905582b`, identical to the lane file).

## Discretization (held constant throughout)

- Mesh: EIT reconstruction mesh only — unit disk, P1 FEM, 48 angles,
  radii [0.22, 0.42, 0.60, 0.76, 0.89, 1.0] → V = 289 nodes, 528 elements.
- Laplacian: FEM stiffness/mass generalized eigenproblem
  K φ = λ M φ, K = unit-conductivity P1 stiffness, M = consistent P1
  mass matrix. Natural Neumann BCs; the constant mode (λ ≈ 0) retained.
  Solved by Cholesky transform + numpy `eigh`; M-orthonormality
  residual 3.1e-15.
- Validation of the discretization (independent check, not a verdict
  input): λ₂ = 3.4155 vs the analytic unit-disk Neumann value
  (j′₁,₁)² = 3.3899 — 0.75% high, the expected direction and size for
  a coarse P1 mesh. The spectrum shows exact degenerate pairs
  (3.415/3.415, 9.464/9.464, …), as disk symmetry requires.

## Claim 1 — Spectrum + IPR: verdict CROSSOVER (sharp edge refuted)

IPR_k = Σφ⁴/(Σφ²)², nodal, unweighted. Full 289-mode table in
`spectrum_ipr.json`.

| Quantity | Value | Sharp-edge rule |
|---|---|---|
| IPR min (mode 1, constant) | 0.003460 = 1/V exactly | — |
| IPR max | 0.031161 | — |
| IPR median | 0.011285 | — |
| IPR span (max/min) | 9.01 (0.95 decades) | required > 100 — FAILS |
| Median adjacent jump | 1.014 | — |
| Max adjacent jump | 3.142 (modes 220→221) | required > 5× median = 5.07 — FAILS |
| N_loc (IPR > 5/V = 0.017301) | 73 of 289 | — |

Both pre-registered conditions fail, the span by an order of
magnitude and the jump by a factor of 1.6. Verdict: CROSSOVER.
The IPR profile is flat across most of the spectrum (decile medians
0.0089–0.0119 for the first eight deciles) and rises only in the top
two deciles (0.0253, 0.0281). The 73 above-threshold modes are
concentrated at HIGH λ (0 in the bottom third, 11 in the middle
third, 61 in the top third).

That concentration inverts the framework's low=localized story on
this mesh, and the reason is stated plainly: on a clean, uniform
mesh, nodal IPR measures grid-scale oscillation, not Anderson
localization. The highest modes oscillate at the mesh scale and
score the highest IPR. There is no disorder and no inclusion
geometry here, so no mobility edge is expected — and none is found.
The crossover verdict agrees with the evaluation's correction.

## Claim 2 — Fiedler partition + interface

Recursive Fiedler bisection (subdomain eigenproblem recomputed per
split), K = 2..8. Interface |Γ_K| = total length of cut mesh edges;
|Ω| = π.

| K | \|Γ_K\| (length) | C(K) = K + \|Γ_K\| |
|---|---|---|
| 2 | 8.122 | 10.122 |
| 3 | 12.184 | 15.184 |
| 4 | 13.605 | 17.605 |
| 5 | 16.122 | 21.122 |
| 6 | 17.539 | 23.539 |
| 7 | 18.956 | 25.956 |
| 8 | 20.374 | 28.374 |

- 2-way Cheeger bound: |Γ| ≥ λ₂|Ω|/4 = 2.683. Measured 8.122,
  ratio 3.03. Verdict: CONFIRMED (inequality holds with margin).
- K-way √λ_{K+1} scaling, as hypothesis: log-log fit of |Γ_K|
  vs √λ_{K+1} over K = 2..8 gives slope 0.905, intercept fit,
  R² = 0.971, residuals within ±0.083. Pre-registered rule
  (|slope−1| < 0.2 AND R² > 0.9) is met. Verdict: CONFIRMED
  AS A FIT ON THIS MESH ONLY. This does not revive the general
  theorem: the 1D counterexample from the evaluation (K-piece
  interface of measure zero vs √λ ∼ K) still refutes the claim
  as a general statement. What is confirmed is a 7-point fit on
  one clean disk mesh — scope it that way and no further.

## Claim 3 — Communication + K*: verdict GAP (K* ≈ N_loc refuted)

- N_loc = 73. K* = argmin C(K) = 2 (cost strictly increasing:
  the interface length grows with K, so the stated normalized
  cost has its minimum at the boundary, not an interior optimum).
- |K* − N_loc| = 71 > 1. Verdict: GAP, not approximately equal.
- Plateau check: (|Γ_8| − |Γ_5|)/|Γ_5| = 0.264 > 0.10.
  Verdict: NO PLATEAU — the interface keeps growing (8.12 → 20.37)
  across the whole K range. The framework's "H¹ stops shrinking
  past K*" shape is not observed, because the measured interface
  never shrinks in the first place: total interface length is
  monotone in K here. The communication quantity that decreases
  with K in Schwarz theory is the per-subdomain interface / coarse
  correction burden, not the total cut length — the stated cost
  formula does not capture that distinction, and on this mesh it
  produces a boundary minimum. Reported as measured, not repaired:
  the formula was pre-registered.

## Claim 4 — Time-as-phase check: frequency statement CONFIRMED (within integration error)

One initial condition (Gaussian bump at (0.3, 0.2), σ = 0.2,
u_t(0) = 0). Modal synthesis at ω_k = √λ_k (all 289 modes) vs
leapfrog integration (dt = 0.1/ω_max = 8.06e-4, 2481 steps to T = 2):

| t | max abs err | mean abs err | rel max err |
|---|---|---|---|
| 0.5 | 3.68e-04 | 1.08e-04 | 7.84e-04 |
| 1.0 | 6.63e-04 | 2.58e-04 | 1.85e-03 |
| 2.0 | 1.27e-05 | 2.21e-06 | 3.36e-05 |

Relative errors ≤ 1.8e-3, consistent with leapfrog truncation at
this dt (halving dt was not run; the error scale matches second-order
integration error, not a frequency mismatch). Verdict: the frequency
statement ω_k = √λ_k is CONFIRMED to integration accuracy. LIMIT
(stated in the protocol and repeated here): this does NOT test the
phase-space group claim or any generator identity — those are
unpinned and untested here.

## Claim 5 — Floor / kernel: NOT TESTED

The ker D identity is unpinned. No computation is claimed for it
in this run. Stated explicitly, as the protocol requires.

## What transfers to the multiplexer partition/decider design

One paragraph, kept separate from the verdicts. (a) The
crossover-not-sharp-edge result is the same moral as the SAT wall:
thresholds on clean systems are S-curves, and any claim of a sharp
capacity edge needs disorder or a demonstrated jump, not an
assertion. (b) The communication correction survives contact with
a real mesh: cost tracks interface size, which grows with the
number of pieces — so "more seats = less communication" is false
as a blanket claim, and the measured escalation rates (lean rule
12.5–37.5% on constructed sets, 100% on the re-derivation runs)
are the interface cost made visible. The decider-as-coarse-space
analogy holds only in the weak sense: the calculator is the
correction that glues local answers, and its cost is per-escalation,
not per-rank. (c) The IPR inversion is a warning against importing
the particle/wave vocabulary directly: on a clean mesh, the
"localized" set is the grid-scale tail, so a multiplexer
localization measure needs either real disorder/inclusion geometry
or a different statistic than nodal IPR. (d) The √λ fit-confirmation
on this mesh licenses nothing general; it is one geometry's fit.
Nothing here changes a seat, a rule, or a bar: the transfer is
diagnostic discipline, not a design change.

## Limits

1. One mesh (289 nodes, clean disk, no inclusions, no disorder).
   A nested/inclusion geometry could show different IPR structure;
   the bacterium mesh was not in the record and was not used.
2. Nodal, unweighted IPR: node density is non-uniform (centre +
   rings), so IPR values carry a density weight the formula does
   not remove. The span/jump verdicts are robust to this (both
   conditions fail by wide margins), but absolute IPR levels are
   mesh-dependent.
3. Recursive bisection recomputes the subdomain eigenproblem per
   split; partitions are nested by construction and need not be
   the global K-way optimum.
4. The wave check uses one initial condition and one dt; no
   convergence study was run.
5. The cost formula's normalized units (c_solve = b = I = 1) are
   the task's stipulation, not a measured machine cost.

## Files

- `~/workspace/spectral-test-report.md` — this report.
- `~/workspace/spectral-test/protocol.md` — pre-registered protocol,
  sha256 `e5318007f4e803a279264e3f8fd210a2bf1867778a95ff6751d7824aac1e4903`.
- `~/workspace/spectral-test/spectral_test.py` — computation script.
- `~/workspace/spectral-test/eit_solver_copy.py` — verbatim EIT copy.
- `~/workspace/spectral-test/spectrum_ipr.json` — all 289
  eigenvalues + IPR values, jump/span/N_loc inputs.
- `~/workspace/spectral-test/partitions.json` — per-K partitions,
  interface lengths, Cheeger numbers, scaling fit, cost curve.
- `~/workspace/spectral-test/wave_check.json` — modal vs leapfrog
  errors, dt, step count.
- `~/workspace/spectral-test/summary.json` — headline numbers.
- `~/workspace/spectral-test/check_results.py` — inspection helper
  (re-prints the saved numbers; computes nothing new).
