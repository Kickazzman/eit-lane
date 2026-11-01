"""Harness for the EIT solver lane. Synthetic phantoms ONLY, known
ground truth, two phantom seeds. Bars are pre-registered below; a
bar that fails is reported as a failed gate, never silently relaxed.

Tests:
  T0  Jacobian finite-difference check (adjoint form vs perturbed solves)
  Ta  single inclusion: localization error + size error (+ 3-iteration GN)
  Tb  two inclusions: separation / resolution limit (valley-dip criterion)
  Tc  no inclusion, noise only at the Ta absolute noise level: false-structure check
  Td  lambda sweep table + discrepancy choice + L-curve corner comparison +
      lambda/10, x10 + flat prior (misuse-signature diagnostic)
  Te  noise-level sweep: graceful degradation

Bars:
  T0: max FD relative error < 1e-3
  Ta: localization < 0.20 radius; |size error| < 1.00; contrast recovery in [0.05, 1.5]
  Tb: valley-dip ratio < 0.85 at separations 0.9 and 0.7 (limit <= 0.7 reported as measured)
  Tc: max|x| < 10% of the same-seed phantom-(a) max|x|; >=90% of elements within 2*uncertainty
  Td: table complete (29 rho values); lambda/10 and lambda x10 images differ
      from the discrepancy-choice image by > 5% relative (lambda matters)
  Te: localization < 0.35 at the highest noise (2%); no localization above 0.5 anywhere

Lambda rule (amended 2026-10-30, recorded in eit_solver.py): Morozov
discrepancy principle, after the L-curve corner measured over-smoothing
(+183% size error) on the seed-7 exploration run. Bars unchanged.

Usage: python3 run_tests.py --seed 7 --out results_seed7.json
"""

import argparse
import json

import numpy as np

from eit_solver import (build_geometry, count_peaks, dip_ratio,
                        discrepancy_choice, jacobian_check, lambda_sweep,
                        make_data, psf_area, reconstruct, region_metrics,
                        result_header, solve_tikhonov)

BARS = {
    "T0_fd_max": 1e-3,
    "Ta_loc": 0.20,
    "Ta_size": 1.00,
    "Ta_contrast": (0.05, 1.5),
    "Tb_dip": 0.85,
    "Tb_sep_required": (0.9, 0.7),
    "Tc_max_frac": 0.10,
    "Tc_coverage": 0.90,
    "Td_rel_change": 0.05,
    "Te_loc_max": 0.35,
    "Te_loc_any": 0.50,
}


def phantom_center(rng, r_min=0.25, r_max=0.55):
    """Random inclusion centre inside the disk, seeded."""
    while True:
        cx, cy = rng.uniform(-0.6, 0.6, size=2)
        if cx ** 2 + cy ** 2 <= r_max ** 2 and cx ** 2 + cy ** 2 >= r_min ** 2:
            return float(cx), float(cy)


def run(seed, out_path):
    rng = np.random.default_rng(seed)
    geo = build_geometry()
    recon = geo["recon"]
    J = geo["J_ref"]
    D = recon["D"]
    area, cent = recon["area"], recon["centroids"]
    pairs = recon["pairs"]
    results = {"seed": seed, "bars": {k: (list(v) if isinstance(v, tuple) else v)
                                      for k, v in BARS.items()},
               "tests": {}}

    # ---- T0: Jacobian check
    fd_max, fd_errs = jacobian_check(
        recon["pts"], recon["tris"], recon["ke"], recon["elec"],
        geo["adj_pairs"], geo["rows"], check_elements=5, seed=seed)
    results["tests"]["T0_jacobian"] = {
        "max_rel_err": fd_max, "per_element": fd_errs,
        "pass": bool(fd_max < BARS["T0_fd_max"])}

    # ---- Ta: single inclusion
    cx, cy = phantom_center(rng)
    contrast = 1.5 if seed % 2 == 1 else 0.5
    sign = 1 if contrast > 1.0 else -1
    incl = [(cx, cy, 0.18, contrast)]
    data = make_data(geo["fine"]["pts"], geo["fine"]["tris"], geo["fine"]["ke"],
                     geo["fine"]["elec"], geo["rows"], geo["adj_pairs"],
                     incl, noise_frac=0.002, noise_seed=seed)
    res1 = reconstruct(J, D, data["dv"], data["noise_std"])
    met1 = region_metrics(res1["x"], area, cent, incl, sign)
    res3 = reconstruct(J, D, data["dv"], data["noise_std"], iterations=3,
                       mesh=geo["mesh_iter"])
    met3 = region_metrics(res3["x"], area, cent, incl, sign)
    lo, hi = BARS["Ta_contrast"]
    results["tests"]["Ta_single"] = {
        "inclusion": incl, "noise_frac": 0.002,
        "one_step": met1, "three_iter": met3,
        "header": result_header(res1),
        "pass": bool(met1["localization_error"] is not None
                     and met1["localization_error"] < BARS["Ta_loc"]
                     and abs(met1["size_error"]) < BARS["Ta_size"]
                     and lo <= met1["contrast_recovery"] <= hi)}
    ref_peak = met1["peak"]

    # ---- Tb: two inclusions, separation sweep (valley-dip criterion)
    tb = {}
    for d in (0.9, 0.7, 0.5, 0.35, 0.25, 0.2):
        c1, c2 = (-d / 2, 0.0), (d / 2, 0.0)
        incl2 = [(c1[0], c1[1], 0.15, 1.5), (c2[0], c2[1], 0.15, 1.5)]
        data2 = make_data(geo["fine"]["pts"], geo["fine"]["tris"],
                          geo["fine"]["ke"], geo["fine"]["elec"], geo["rows"],
                          geo["adj_pairs"], incl2, noise_frac=0.002,
                          noise_seed=seed + 5)
        res2 = reconstruct(J, D, data2["dv"], data2["noise_std"])
        n, peaks = count_peaks(res2["x"], area, cent, pairs, sign=1)
        ratio, n_band = dip_ratio(res2["x"], cent, c1, c2, sign=1)
        tb[str(d)] = {"peaks": n, "peak_positions": peaks,
                      "dip_ratio": ratio, "n_band": n_band}
    tb["resolution_limit"] = min([float(d) for d, v in tb.items()
                                  if isinstance(v, dict) and v["dip_ratio"] is not None
                                  and v["dip_ratio"] < BARS["Tb_dip"]],
                                 default=None)
    tb["pass"] = bool(all(tb[str(d)]["dip_ratio"] is not None
                          and tb[str(d)]["dip_ratio"] < BARS["Tb_dip"]
                          for d in BARS["Tb_sep_required"]))
    results["tests"]["Tb_two"] = tb

    # ---- Tc: noise only, at the SAME ABSOLUTE noise level as Ta
    # (relative scaling would be degenerate: RMS of a zero signal is 0)
    data0 = make_data(geo["fine"]["pts"], geo["fine"]["tris"],
                      geo["fine"]["ke"], geo["fine"]["elec"], geo["rows"],
                      geo["adj_pairs"], [],
                      noise_std_abs=data["noise_std"],
                      noise_seed=seed + 9)
    res0 = reconstruct(J, D, data0["dv"], data0["noise_std"])
    x0 = res0["x"]
    frac_max = float(np.max(np.abs(x0)) / ref_peak) if ref_peak else None
    coverage = float(np.mean(np.abs(x0) <= 2.0 * res0["uncertainty_map"]))
    results["tests"]["Tc_noise_only"] = {
        "max_abs_x": float(np.max(np.abs(x0))),
        "ref_peak_Ta": ref_peak,
        "max_as_frac_of_Ta_peak": frac_max,
        "coverage_within_2sigma": coverage,
        "uncertainty_mean": res0["uncertainty_mean"],
        "pass": bool(frac_max < BARS["Tc_max_frac"]
                     and coverage >= BARS["Tc_coverage"])}

    # ---- Td: lambda sweep on phantom (a): discrepancy choice,
    # L-curve corner comparison, lambda sensitivity, flat prior.
    table, corner_idx, corner_fallback = lambda_sweep(
        J, D, data["dv"], 10.0 ** (-6.0 + 0.25 * np.arange(29)))
    disc_idx, disc_edge, target = discrepancy_choice(
        table, data["noise_std"], len(data["dv"]))
    rho_star = float(10.0 ** (-6.0 + 0.25 * disc_idx))
    rho_corner = float(10.0 ** (-6.0 + 0.25 * corner_idx))
    x_star, _, _ = solve_tikhonov(J, D, data["dv"], rho_star)
    met_star = region_metrics(x_star, area, cent, incl, sign)
    x_corner, _, _ = solve_tikhonov(J, D, data["dv"], rho_corner)
    met_corner = region_metrics(x_corner, area, cent, incl, sign)
    rel_changes = {}
    for tag, rho in (("lambda_div_10", rho_star / 10.0),
                     ("lambda_x_10", rho_star * 10.0)):
        x_alt, _, _ = solve_tikhonov(J, D, data["dv"], rho)
        rel_changes[tag] = float(np.linalg.norm(x_alt - x_star)
                                 / np.linalg.norm(x_star))
    # Flat-prior diagnostic at the same rho (misuse signature check).
    x_flat, _, _ = solve_tikhonov(J, np.eye(J.shape[1]), data["dv"], rho_star)
    flat_rel = float(np.linalg.norm(x_flat - x_star) / np.linalg.norm(x_star))
    results["tests"]["Td_lambda"] = {
        "table": table, "discrepancy_rho": rho_star,
        "discrepancy_edge": bool(disc_edge),
        "discrepancy_target_residual": target,
        "lcurve_corner_rho": rho_corner,
        "lcurve_corner_fallback": bool(corner_fallback),
        "metrics_at_discrepancy": met_star,
        "metrics_at_corner": met_corner,
        "rel_changes": rel_changes, "flat_prior_rel_change": flat_rel,
        "pass": bool(len(table) == 29
                     and all(v > BARS["Td_rel_change"]
                             for v in rel_changes.values()))}

    # ---- Te: noise sweep on phantom (a)
    te = {}
    for frac in (0.0, 5e-4, 2e-3, 1e-2, 2e-2):
        dat = make_data(geo["fine"]["pts"], geo["fine"]["tris"],
                        geo["fine"]["ke"], geo["fine"]["elec"], geo["rows"],
                        geo["adj_pairs"], incl, noise_frac=frac,
                        noise_seed=seed + 11)
        rr = reconstruct(J, D, dat["dv"], dat["noise_std"])
        mm = region_metrics(rr["x"], area, cent, incl, sign)
        te[str(frac)] = {"localization_error": mm["localization_error"],
                         "size_error": mm["size_error"],
                         "peak": mm["peak"]}
    locs = [v["localization_error"] for v in te.values()
            if v["localization_error"] is not None]
    te["pass"] = bool(locs and max(locs) < BARS["Te_loc_any"]
                      and te["0.02"]["localization_error"] < BARS["Te_loc_max"])
    results["tests"]["Te_noise_sweep"] = te

    # ---- Spectrum / resolution characterization (geometry, once)
    sv = np.linalg.svd(J, compute_uv=False)
    M = J.T @ J + rho_star * float(np.mean(np.diag(J.T @ J))) * (D.T @ D)
    R = np.linalg.inv(M) @ (J.T @ J)
    psf, j0 = psf_area(R, area, cent)
    edge = np.mean(np.diag(R)[np.linalg.norm(cent, axis=1) > 0.85])
    results["spectrum"] = {
        "singular_values": sv.tolist(),
        "condition_number": float(sv[0] / sv[-1]),
        "condition_note": ("singular values decay to the double-precision "
                           "floor; the condition number is a numerical "
                           "artifact and the effective rank is the "
                           "meaningful number"),
        "sv_at_10_50_100_200": [float(sv[i]) for i in (9, 49, 99, 199)],
        "effective_rank_1pct": int(np.sum(sv > 0.01 * sv[0])),
        "effective_rank_0p1pct": int(np.sum(sv > 0.001 * sv[0])),
        "resolution_diag_mean": float(np.mean(np.diag(R))),
        "resolution_center": float(np.diag(R)[j0]),
        "resolution_edge_mean": float(edge),
        "psf_area_center": psf,
        "psf_radius_center": float(np.sqrt(psf / np.pi)),
        "jacobian_shape": list(J.shape),
        "n_measurements": len(geo["rows"]),
    }
    results["all_pass"] = bool(all(t.get("pass") for t in results["tests"].values()
                                   if isinstance(t, dict)))
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    # Phantom data export: Ta reconstruction + ground truth on the
    # reconstruction mesh, Tc noise-only reconstruction, raw dV.
    from eit_solver import phantom_sigma
    np.savez(out_path.replace(".json", "_phantoms.npz"),
             x_Ta=res1["x"], sigma_true_recon=phantom_sigma(cent, incl),
             resolution_map=res1["resolution_map"],
             uncertainty_map=res1["uncertainty_map"],
             x_Tc_noise_only=res0["x"], dv_Ta=data["dv"],
             dv_Ta_clean=data["dv_clean"], centroids=cent, area=area,
             inclusion=np.array(incl))
    return results


def print_summary(res):
    t = res["tests"]
    print(f"seed {res['seed']}  all_pass={res['all_pass']}")
    print(f"  T0 jacobian FD max rel err {t['T0_jacobian']['max_rel_err']:.3e} "
          f"pass={t['T0_jacobian']['pass']}")
    a = t["Ta_single"]
    print(f"  Ta single {a['inclusion']} one-step loc="
          f"{a['one_step']['localization_error']:.3f} size_err="
          f"{a['one_step']['size_error']:+.2f} contrast_rec="
          f"{a['one_step']['contrast_recovery']:.3f} | 3-iter loc="
          f"{a['three_iter']['localization_error']:.3f} size_err="
          f"{a['three_iter']['size_error']:+.2f} pass={a['pass']}")
    b = t["Tb_two"]
    def _dip(d):
        v = b[d]["dip_ratio"]
        return f"{v:.3f}" if v is not None else "n/a"
    print(f"  Tb two-inclusion dip ratios " +
          ", ".join(f"d={d}: {_dip(d)} (peaks {b[d]['peaks']})"
                    for d in ("0.9", "0.7", "0.5", "0.35", "0.25", "0.2")) +
          f" limit={b['resolution_limit']} pass={b['pass']}")
    c = t["Tc_noise_only"]
    print(f"  Tc noise-only max|x|={c['max_abs_x']:.4f} "
          f"({c['max_as_frac_of_Ta_peak']:.3f} of Ta peak) "
          f"2-sigma coverage={c['coverage_within_2sigma']:.3f} pass={c['pass']}")
    d = t["Td_lambda"]
    print(f"  Td lambda discrepancy rho={d['discrepancy_rho']:.3g} "
          f"corner rho={d['lcurve_corner_rho']:.3g} "
          f"metrics@disc loc={d['metrics_at_discrepancy']['localization_error']:.3f} "
          f"size={d['metrics_at_discrepancy']['size_error']:+.2f} "
          f"metrics@corner loc={d['metrics_at_corner']['localization_error']:.3f} "
          f"size={d['metrics_at_corner']['size_error']:+.2f} "
          f"rel-changes={ {k: round(v, 3) for k, v in d['rel_changes'].items()} } "
          f"flat-prior rel={d['flat_prior_rel_change']:.3f} pass={d['pass']}")
    e = t["Te_noise_sweep"]
    print("  Te noise sweep loc: " +
          ", ".join(f"{k}: {v['localization_error']:.3f}"
                    for k, v in e.items() if isinstance(v, dict)) +
          f" pass={e['pass']}")
    s = res["spectrum"]
    print(f"  spectrum cond={s['condition_number']:.3e} "
          f"resol mean={s['resolution_diag_mean']:.3f} "
          f"center={s['resolution_center']:.3f} edge={s['resolution_edge_mean']:.3f} "
          f"PSF radius={s['psf_radius_center']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    res = run(args.seed, args.out)
    print_summary(res)
