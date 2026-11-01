# EIT solver lane — report

## What was built

A standalone, deterministic electrical-impedance-tomography
difference-imaging solver: the shape seat's code lane (map U-M02 /
U-X01). It reconstructs a conductivity CHANGE inside a 2D disk from
boundary voltage differences. No language model is involved; no
fleet, dispatch, router, or live service is touched; numpy only
(stdlib + numpy, verified by import grep — no scipy, since mrfloppy
has numpy 2.5.3 and no scipy, and the constructed mesh needs none).

Files (identical local ↔ mrfloppy, md5-verified):

- `~/workspace/eit-lane/eit_solver.py` (md5 `be034dcd58ad289523e2b206c905582b`) — solver module + CLI
- `~/workspace/eit-lane/run_tests.py` (md5 `6a3a2dda5b8d0d6c5ef32f7519b3ece5`) — harness, 5 tests + Jacobian check
- `~/workspace/eit-lane/results_seed7.json`, `results_seed19.json` — full harness output (tables, headers, spectrum)
- `~/workspace/eit-lane/results_seed7_phantoms.npz`, `results_seed19_phantoms.npz` — phantom data (reconstructions, ground truth on the reconstruction mesh, raw ΔV, resolution + uncertainty maps)
- Mirror: mrfloppy `~/eit-lane/` (same files, same md5s). Authoritative runs executed on mrfloppy.

## Discretization + electrode model (stated, no hidden defaults)

- Domain: unit disk, radius 1. P1 finite elements, conductivity
  piecewise constant per element.
- Mesh: constructed polar grid (deterministic, no mesher). Nodes:
  centre + rings × 48 equally spaced angles.
  - Reconstruction mesh: radii [0.22, 0.42, 0.60, 0.76, 0.89, 1.0] →
    289 nodes, 528 elements.
  - Data-generation mesh (finer, to avoid the inverse crime of
    sharing the reconstruction mesh): radii [0.14, 0.27, 0.39, 0.50,
    0.60, 0.69, 0.78, 0.86, 0.93, 1.0] → 481 nodes, 912 elements.
  - Residual caveat: the two meshes share the 48-angle structure and
    differ only radially, so the crime is reduced, not eliminated.
- Electrodes: 16 POINT electrodes on boundary nodes at angles
  2πk/16 (gap model: current injected at one node, voltage read at
  one node). Contact impedance neglected — stated limitation;
  difference imaging cancels first-order contact/boundary errors.
- Protocol: adjacent drive / adjacent measure. 16 drive pairs, 13
  measurement pairs per drive (pairs touching a drive electrode
  skipped) → 208 measurements, fixed enumeration.
- Forward model: K(σ)u = I, centre node grounded (measurement
  differences are ground-free). Jacobian by the adjoint/reciprocity
  form J[(d,m),e] = −A_e ∇u_d·∇w_m, verified against finite
  differences (test T0).

## Prior (words + mathematics + provenance)

Words: first-difference smoothness — neighbouring elements should
have similar conductivity change; no edge-preserving or learned term.
Mathematics: the estimate solves
min_x ‖Jx − ΔV‖² + λ²‖Dx‖², where D is the element-adjacency
difference operator (one row per interior edge shared by two
elements, +1/−1).
Provenance: standard first-difference Tikhonov smoothness prior,
this implementation. No citation is claimed for this specific D;
the Tikhonov family and the λ(noise) trade are the standard
formulation (pass-2 report, transfer A2).

## λ rule (stated + amendment history)

Rule: Morozov discrepancy principle. λ² = ρ·mean(diag(JᵀJ)), ρ
swept over 29 log-spaced values 1e-6..1e1; the chosen ρ minimizes
| ‖Jx−ΔV‖ − s·√m |, s = stated noise std, m = 208. λ is therefore
a function of the noise level (λ→0 as s→0), as the pass-2 report
requires. The L-curve corner over the same sweep is computed and
published as a diagnostic, but is NOT the choice rule.

Amendment (recorded, not hidden): the first implementation used the
L-curve corner. The seed-7 exploration run measured it
over-smoothing to +183% size error (ρ = 5.62; metrics at the corner:
loc 0.113, size +1.83, contrast recovery 0.22). Diagnosis: with 528
unknowns and 208 measurements the residual vanishes at small ρ, the
L-curve has no clean corner, and the curvature maximum lands in the
over-smoothed region. The rule was amended to the discrepancy
principle BEFORE the authoritative mrfloppy runs; the bars were not
changed. The L-curve table is still published with every result.

## Noise assumption (printed with every result)

Measurement noise is white Gaussian, std s = noise_frac × RMS of the
noiseless ΔV. Uncertainty map = √(diag Cov), Cov = s²M⁻¹JᵀJM⁻¹,
M = JᵀJ + λ²DᵀD. Resolution matrix R = M⁻¹JᵀJ; the resolution map
is diag(R). Every reconstruction result carries: prior, λ rule, ρ,
λ, noise assumption, discretization, difference-vs-absolute choice
(reference: uniform σ = 1.0), resolution mean, uncertainty mean/max,
condition number, noise amplification, iteration count. A bare
image is never emitted; the CLI prints the full header.

## Phantom test table (synthetic only, known ground truth)

Phantoms are analytic disks on the fine mesh; data = nonlinear
forward solves (not the Jacobian). Noise 0.2% of signal RMS unless
stated. Bars pre-registered in `run_tests.py`. Both seeds pass all
tests; the second seed reproduces the same qualitative verdicts.

| Test | Bar | Seed 7 | Seed 19 |
|---|---|---|---|
| T0 Jacobian FD check | max rel err < 1e-3 | 5.5e-07 ✓ | 1.3e-06 ✓ |
| Ta single inclusion, r=0.18, σ=1.5, centre seed-placed | loc < 0.20; \|size\| < 1.00; contrast rec. 0.05–1.5 | loc 0.023; size −0.01; contrast 0.87 ✓ | loc 0.016; size −0.11; contrast 0.88 ✓ |
| Ta, 3-iteration Gauss–Newton | (same, reported) | loc 0.042; size +0.10; contrast 1.19 | loc 0.036; size −0.06; contrast 1.20 |
| Tb two inclusions, r=0.15, dip ratio < 0.85 = resolved | resolved at d=0.9, 0.7 | d=0.9: −0.34; 0.7: −0.19; 0.5: 0.25; 0.35: 0.61; 0.25: 0.89; 0.2: n/a → limit 0.35 ✓ | d=0.9: −0.35; 0.7: −0.25; 0.5: 0.21; 0.35: 0.58; 0.25: 0.89; 0.2: n/a → limit 0.35 ✓ |
| Tc noise only, absolute noise = Ta level | max\|x\| < 10% of Ta peak; ≥90% within 2σ | 7.2e-05 of peak; coverage 0.996 ✓ | 9.6e-05 of peak; coverage 0.985 ✓ |
| Td λ sweep (29 ρ values) | table complete; λ/10, λ×10 change image > 5% | ρ* 5.6e-05 (interior); corner ρ 3.2e-06; λ/10 rel 0.16; λ×10 rel 0.21; flat-prior rel 0.37 ✓ | ρ* 5.6e-05 (interior); corner ρ 3.2e-06; λ/10 rel 0.18; λ×10 rel 0.22; flat-prior rel 0.36 ✓ |
| Te noise sweep, loc error | < 0.35 at 2% noise; < 0.5 anywhere | 0%: 0.018; 0.05%: 0.012; 0.2%: 0.023; 1%: 0.048; 2%: 0.072 ✓ | 0%: 0.007; 0.05%: 0.016; 0.2%: 0.015; 1%: 0.043; 2%: 0.053 ✓ |

Ta inclusion centres: seed 7 (0.150, 0.477); seed 19 (−0.096, 0.511).
Te size errors (same order as loc above): seed 7 −0.26, −0.11, −0.01,
+0.16, +0.44; seed 19 −0.28, −0.23, −0.11, +0.15, +0.46 — graceful
degradation; the half-max area is a λ-sensitive size proxy (see Limits).
Negative dip ratios at large separations mean the valley between the
two inclusions dips below zero (reconstruction undershoot lobes) —
resolved, not an error. d = 0.2 unmeasured: too few element centroids
in the profile band (reported n/a, not interpolated into existence).

Supplementary check (not a gate): negative-contrast inclusion
(σ = 0.5, centre (0.3, −0.25), 0.2% noise) reconstructs with the
correct sign and location (loc 0.042, size −0.02) but contrast
recovery 1.49 — the linear model overshoots resistive inclusions.
Sign handling is correct; magnitude is biased high for decreases.

## Resolution / spectrum characterization (geometry, at ρ*)

- Jacobian 208 × 528. Singular values: σ₁ = 0.0878, σ₁₀ = 0.0401,
  σ₅₀ = 0.00122, σ₁₀₀ = 2.99e-07, σ_last ≈ 1e-19 — roughly
  exponential decay with spatial frequency, as U-M03 states.
  Effective rank: 52 modes above 1% of σ₁, 72 above 0.1%.
- Condition number ≈ 2.3e19 is a double-precision-floor artifact
  (singular values decay to ~1e-19); the effective rank is the
  meaningful number, and it is reported alongside.
- Resolution diag(R): mean 0.137, centre element 0.042, edge mean
  0.259 — the centre is the least resolved region, the edge the most.
  PSF radius at the centre element: 0.209 (domain radii).
- Noise amplification (Ta header): 1245× — voltage noise std
  1.9e-06 maps to mean conductivity uncertainty 0.0022 in σ units,
  against a 0.5 inclusion contrast (SNR ≈ 200 at 0.2% noise).

## Gates

- Harness: all 5 tests + Jacobian check PASS on both seeds, numbers
  recorded in the JSON files above.
- Second seed (19): same pass pattern, same resolution limit (0.35),
  same λ behaviour (discrepancy ρ* 5.6e-05 both seeds; L-curve corner
  at the grid floor both seeds), same graceful noise degradation.
- Standalone: `import eit_solver` + CLI (`spectrum`, `reconstruct`)
  run on mrfloppy with no fleet/dispatch dependency; the CLI prints
  the full publication header with every result. Verified 2026-10-30.

## Limits (what this lane does NOT do)

1. Synthetic phantoms only. No rig exists (pinned, equipment not
   acquired). Nothing here has seen real plant data; no claim about
   real data is made anywhere in the outputs.
2. 2D unit disk, 16 point electrodes, no contact impedance, no
   electrode shape, no 3D effects, no anisotropy, no frequency
   dependence. A real rig (stem cross-section, bark, soil contact)
   breaks the head-model assumptions in exactly the way the pass-2
   report warns; difference imaging from a recent reference is the
   defensible mode, absolute imaging is not offered.
3. The data and reconstruction meshes share the 48-angle structure
   (radial density differs). The inverse crime is reduced, not
   eliminated; a fully independent mesh pair is a follow-up.
4. The smoothness prior blurs edges by design; the half-max size
   proxy is λ-sensitive (size error moves −0.26 → +0.44 across the
   noise sweep at fixed rule). Sharp-boundary claims need a
   different prior (TV/blocky), which is not built.
5. Three-iteration Gauss–Newton overshoots contrast (recovery
   1.19–1.20 vs 0.87–0.88 one-step) while slightly worsening
   localization. One-step is the conservative default; iteration is
   opt-in and reported as such.
6. Resolution limit 0.35 is bracketed only from above by the tested
   grid (0.35 resolved, 0.25 not); d = 0.2 could not be measured
   with this mesh's centroid sampling.
7. The L-curve corner is degenerate on this system (sits at the grid
   floor in both seeds). It is published as a diagnostic only; any
   future λ-rule change must be re-derived, not swapped silently.
8. No LM seat renders a shape verdict from this output; per U-X01 an
   LM may summarize the published solver output (prior, λ, resolution,
   uncertainty attached) but never originates the shape.

## Nothing else launched

Wire v2, repo push, Phase 2 downloads, media test, and rig all stay
pending. The rig stops at spec/budget approval — no purchase, no
acquisition step taken here. No fleet, dispatch, router, or service
state was touched; no resident seat was run.
