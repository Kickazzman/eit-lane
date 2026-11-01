"""EIT difference-imaging solver lane — deterministic, numpy-only, standalone.

Reconstructs a conductivity CHANGE (delta-sigma) inside a 2D circular
domain from boundary voltage DIFFERENCES (time-difference EIT), per
map entry U-M02 / U-X01: difference imaging first, a deliberately
named prior, a published Jacobian spectrum / resolution map, and an
uncertainty map with every reconstruction. No language model is
involved; this module originates shapes from data + stated prior only.

DISCRETIZATION (stated, no hidden defaults)
  * Domain: unit disk, radius 1.
  * Mesh: constructed polar grid (no mesher, fully deterministic).
    Nodes: centre + R rings x N_ANG equally spaced nodes.
    Reconstruction mesh: N_ANG=48, radii=[0.22,0.42,0.60,0.76,0.89,1.0]
    -> 289 nodes, 528 triangular P1 elements.
    Data-generation mesh (finer, avoids the inverse crime of sharing
    the reconstruction mesh): N_ANG=48,
    radii=[0.14,0.27,0.39,0.50,0.60,0.69,0.78,0.86,0.93,1.0]
    -> 481 nodes, 912 elements.
  * Conductivity: piecewise constant per element.
  * Electrodes: 16 POINT electrodes on boundary nodes at angles
    2*pi*k/16 (gap model). Current is injected at a single node,
    voltage is read at a single node. Contact impedance is neglected
    (stated limitation); difference imaging cancels first-order
    contact/boundary errors, which is why it is used here.
  * Protocol: adjacent drive / adjacent measure. 16 drive pairs,
    13 measurement pairs per drive (pairs touching a drive electrode
    are skipped) -> 208 measurements, fixed enumeration.

FORWARD MODEL: standard P1 finite elements. K(sigma) u = I, with the
centre node grounded (measurement differences are ground-free).
Solved densely with numpy (meshes are small by design).

JACOBIAN (sensitivity): adjoint / reciprocity form,
    J[row(d,m), e] = -A_e * grad(u_d)|_e . grad(w_m)|_e,
where u_d is the drive-d potential field and w_m the field of a unit
current on measurement pair m, both at the reference conductivity.
Verified against finite differences (see jacobian_check()).

PRIOR (stated in words and in mathematics)
  Words: first-difference smoothness — neighbouring elements should
  have similar conductivity change; no edge-preserving or learned
  term.
  Mathematics: penalty lambda^2 * ||D x||^2, where D is the
  element-adjacency difference operator (one row per interior edge
  shared by two elements, +1/-1). The estimate solves
      min_x  ||J x - dV||^2 + lambda^2 ||D x||^2.
  Provenance: standard first-difference Tikhonov smoothness prior,
  this implementation. No citation is claimed for this specific D;
  the Tikhonov family and the lambda(noise) trade are the standard
  formulation (see the pass-2 report, transfer A2).

LAMBDA RULE (stated, amended 2026-10-30 after measurement):
Morozov discrepancy principle. lambda^2 = rho *
mean(diag(J^T J)), rho swept over 29 log-spaced values 1e-6..1e1;
the chosen rho minimizes | ||Jx-dV|| - s*sqrt(m) |, where s is the
stated noise std and m = 208 measurements. This makes lambda a
function of the noise level (lambda -> 0 as s -> 0), as the
regularization theory in the pass-2 report requires. The L-curve
corner over the same sweep is computed and published as a
diagnostic, but is NOT the choice rule: on this underdetermined
system (528 unknowns, 208 measurements) the residual vanishes at
small rho, the L-curve has no clean corner, and its curvature
maximum landed at rho = 5.62, over-smoothing to +183% size error
(local seed-7 exploration, retained in the record). The L-curve
table is still published with every result.

NOISE ASSUMPTION (stated with every result): measurement noise is
white Gaussian, std s = noise_frac * RMS(noiseless dV). The
uncertainty map is sqrt(diag(Cov)), Cov = s^2 M^-1 J^T J M^-1,
M = J^T J + lambda^2 D^T D. Resolution matrix R = M^-1 J^T J;
the resolution map is diag(R).

Difference vs absolute: DIFFERENCE imaging throughout. The reference
is the uniform background sigma_ref = 1.0 on the data mesh; dV =
V(sigma_true) - V(sigma_ref) (+ noise). No absolute image is produced.

SYNTHETIC DATA ONLY. No rig exists (rig pinned, equipment not
acquired). Nothing in this module has seen real plant data, and no
claim about real data is made anywhere in its outputs.
"""

import json
import math

import numpy as np

# ---------------------------------------------------------------- mesh

N_ELEC = 16


def build_mesh(n_ang, radii):
    """Constructed polar mesh. Returns (points, triangles).

    points: (N, 2) array, index 0 is the centre. triangles: (E, 3)
    int array, counter-clockwise (positive signed area).
    """
    pts = [(0.0, 0.0)]
    for r in radii:
        for i in range(n_ang):
            th = 2.0 * math.pi * i / n_ang
            pts.append((r * math.cos(th), r * math.sin(th)))
    pts = np.asarray(pts, dtype=float)
    tris = []
    base = 1
    for i in range(n_ang):
        tris.append((0, base + i, base + (i + 1) % n_ang))
    for j in range(len(radii) - 1):
        r0 = 1 + j * n_ang
        r1 = 1 + (j + 1) * n_ang
        for i in range(n_ang):
            i2 = (i + 1) % n_ang
            tris.append((r0 + i, r0 + i2, r1 + i2))
            tris.append((r0 + i, r1 + i2, r1 + i))
    tris = np.asarray(tris, dtype=int)
    # Enforce counter-clockwise orientation (positive signed area).
    p = pts[tris]
    area2 = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
             - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    flip = area2 < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return pts, tris


def mesh_geometry(pts, tris):
    """Per-element areas, centroids, and P1 basis gradients."""
    p = pts[tris]  # (E, 3, 2)
    x1, y1 = p[:, 0, 0], p[:, 0, 1]
    x2, y2 = p[:, 1, 0], p[:, 1, 1]
    x3, y3 = p[:, 2, 0], p[:, 2, 1]
    area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * np.abs(area2)
    centroids = p.mean(axis=1)
    grads = np.empty((len(tris), 3, 2))
    grads[:, 0, 0] = (y2 - y3) / area2
    grads[:, 0, 1] = (x3 - x2) / area2
    grads[:, 1, 0] = (y3 - y1) / area2
    grads[:, 1, 1] = (x1 - x3) / area2
    grads[:, 2, 0] = (y1 - y2) / area2
    grads[:, 2, 1] = (x2 - x1) / area2
    # Unit-conductivity element stiffness matrices: Ke[i, j] = A g_i.g_j
    ke = area[:, None, None] * np.einsum("eik,ejk->eij", grads, grads)
    return area, centroids, grads, ke


def electrode_nodes(n_ang, radii):
    """Boundary node index of each of the 16 point electrodes."""
    ring0 = 1 + (len(radii) - 1) * n_ang
    stride = n_ang // N_ELEC
    return [ring0 + k * stride for k in range(N_ELEC)]


def element_adjacency(tris):
    """Pairs of elements sharing an edge -> rows of D."""
    edge_to_elems = {}
    for e, tri in enumerate(tris):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_to_elems.setdefault((min(a, b), max(a, b)), []).append(e)
    pairs = [v for v in edge_to_elems.values() if len(v) == 2]
    return np.asarray(pairs, dtype=int)


def difference_operator(pairs, n_elem):
    D = np.zeros((len(pairs), n_elem))
    D[np.arange(len(pairs)), pairs[:, 0]] = 1.0
    D[np.arange(len(pairs)), pairs[:, 1]] = -1.0
    return D

# ---------------------------------------------------------------- forward


def assemble_stiffness(ke, tris, sigma_elem, n_nodes):
    K = np.zeros((n_nodes, n_nodes))
    rows = np.repeat(tris, 3, axis=1).ravel()
    cols = np.tile(tris, (1, 3)).ravel()
    vals = (sigma_elem[:, None, None] * ke).ravel()
    np.add.at(K, (rows, cols), vals)
    return K


def forward_solve(pts, tris, ke, sigma_elem, elec, pairs):
    """Solve K u = I for each adjacent drive pair.

    pairs: list of (a, b) electrode indices (current in at a, out at b).
    Returns potentials array (len(pairs), n_nodes); node 0 grounded.
    """
    n_nodes = len(pts)
    K = assemble_stiffness(ke, tris, sigma_elem, n_nodes)
    Kr = K[1:, 1:]
    U = np.zeros((len(pairs), n_nodes))
    for p, (a, b) in enumerate(pairs):
        rhs = np.zeros(n_nodes - 1)
        for node, val in ((elec[a], 1.0), (elec[b], -1.0)):
            if node != 0:
                rhs[node - 1] += val
        U[p, 1:] = np.linalg.solve(Kr, rhs)
    return U


def protocol():
    """Fixed measurement enumeration: list of (drive_pair, meas_pair).

    Drive pairs and measurement pairs are adjacent electrode pairs.
    Measurements touching a drive electrode are skipped: 208 rows.
    """
    rows = []
    for d in range(N_ELEC):
        drive = {d, (d + 1) % N_ELEC}
        for m in range(N_ELEC):
            if m in drive or (m + 1) % N_ELEC in drive:
                continue
            rows.append((d, m))
    return rows


def measure(U, elec, rows):
    """Voltage differences for the fixed protocol."""
    out = np.empty(len(rows))
    for i, (d, m) in enumerate(rows):
        out[i] = U[d, elec[m]] - U[d, elec[(m + 1) % N_ELEC]]
    return out


def element_gradients(U, tris, grads):
    """Gradient of each potential field on each element: (P, E, 2)."""
    vals = U[:, tris]  # (P, E, 3)
    return np.einsum("pei,eik->pek", vals, grads)


def jacobian(grad_fields, area, rows, adj_pairs):
    """Adjoint-form Jacobian. grad_fields: (16, E, 2) at sigma_ref.

    Row (d, m): -A_e * grad(u_d) . grad(w_m), where w_m is the field
    of a unit current on adjacent pair m (same 16 solves as drives).
    """
    n_elem = grad_fields.shape[1]
    J = np.empty((len(rows), n_elem))
    for i, (d, m) in enumerate(rows):
        gd = grad_fields[d]
        gm = grad_fields[m]  # measurement pair m == adjacent pair m
        J[i] = -area * (gd[:, 0] * gm[:, 0] + gd[:, 1] * gm[:, 1])
    return J


def jacobian_check(pts, tris, ke, elec, adj_pairs, rows, sigma_ref=1.0,
                   delta=1e-6, check_elements=5, seed=0):
    """Finite-difference validation of the adjoint Jacobian.

    Returns (max_rel_err, per-element rel errs). J is the exact
    linearization, so FD with a small step must agree closely.
    """
    n_elem = len(tris)
    rng = np.random.default_rng(seed)
    elems = sorted(rng.choice(n_elem, size=min(check_elements, n_elem),
                              replace=False).tolist())
    sigma0 = np.full(n_elem, sigma_ref)
    U0 = forward_solve(pts, tris, ke, sigma0, elec, adj_pairs)
    V0 = measure(U0, elec, rows)
    G = element_gradients(U0, tris, mesh_geometry(pts, tris)[2])
    area = mesh_geometry(pts, tris)[0]
    J = jacobian(G, area, rows, adj_pairs)
    errs = []
    for e in elems:
        sigma1 = sigma0.copy()
        sigma1[e] += delta
        V1 = measure(forward_solve(pts, tris, ke, sigma1, elec, adj_pairs),
                     elec, rows)
        fd = (V1 - V0) / delta
        errs.append(float(np.linalg.norm(fd - J[:, e])
                          / np.linalg.norm(J[:, e])))
    return max(errs), errs

# ---------------------------------------------------------------- phantoms


def phantom_sigma(centroids, inclusions, sigma_bg=1.0):
    """Piecewise-constant conductivity from analytic disk inclusions.

    inclusions: list of (cx, cy, radius, sigma_inside).
    """
    sigma = np.full(len(centroids), sigma_bg)
    for cx, cy, r, s_in in inclusions:
        inside = (centroids[:, 0] - cx) ** 2 + (centroids[:, 1] - cy) ** 2 <= r ** 2
        sigma[inside] = s_in
    return sigma


def make_data(pts_f, tris_f, ke_f, elec_f, rows, adj_pairs, inclusions,
              sigma_bg=1.0, noise_frac=0.0, noise_seed=0, noise_std_abs=None):
    """Synthetic difference data on the FINE mesh (no inverse crime).

    Noise std defaults to noise_frac * RMS(noiseless dV). Pass
    noise_std_abs to set an absolute noise level instead (used by the
    noise-only test, where the noiseless signal is zero and a
    relative level would be degenerate). Returns dict with
    clean/noisy dV, noise std, reference RMS.
    """
    geo_f = mesh_geometry(pts_f, tris_f)
    cent_f = geo_f[1]
    sig_ref = np.full(len(tris_f), sigma_bg)
    sig_true = phantom_sigma(cent_f, inclusions, sigma_bg)
    V_ref = measure(forward_solve(pts_f, tris_f, ke_f, sig_ref, elec_f, adj_pairs),
                    elec_f, rows)
    V_true = measure(forward_solve(pts_f, tris_f, ke_f, sig_true, elec_f, adj_pairs),
                     elec_f, rows)
    dv_clean = V_true - V_ref
    rms = float(np.sqrt(np.mean(dv_clean ** 2)))
    s = float(noise_std_abs) if noise_std_abs is not None else noise_frac * rms
    if s > 0:
        rng = np.random.default_rng(noise_seed)
        dv = dv_clean + rng.normal(0.0, s, size=dv_clean.shape)
    else:
        dv = dv_clean.copy()
    return {"dv": dv, "dv_clean": dv_clean, "noise_std": s,
            "noise_frac": noise_frac, "rms_clean": rms,
            "sigma_true_fine": sig_true}

# ---------------------------------------------------------------- solve


def lcurve_corner(residuals, roughnesses):
    """Max-curvature interior point of the log-log L-curve.

    Returns (index, fallback_flag). Fallback if the max sits on an edge.
    Diagnostic only; the choice rule is the discrepancy principle.
    """
    x = np.log(np.maximum(residuals, 1e-300))
    y = np.log(np.maximum(roughnesses, 1e-300))
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx ** 2 + dy ** 2) ** 1.5
    numer = np.abs(dx * ddy - dy * ddx)
    curv = np.where(denom > 0, numer / np.maximum(denom, 1e-300), 0.0)
    i = int(np.argmax(curv[1:-1])) + 1
    fallback = bool(i <= 1 or i >= len(curv) - 2)
    return i, fallback, curv


def solve_tikhonov(J, D, dv, rho):
    """Solve min ||Jx-dv||^2 + lam^2||Dx||^2 for one rho.

    lam^2 = rho * mean(diag(J^T J)). Returns (x, residual, roughness).
    """
    scale = float(np.mean(np.diag(J.T @ J)))
    lam2 = rho * scale
    M = J.T @ J + lam2 * (D.T @ D)
    x = np.linalg.solve(M, J.T @ dv)
    return x, float(np.linalg.norm(J @ x - dv)), float(np.linalg.norm(D @ x))


def lambda_sweep(J, D, dv, rhos):
    """Full sweep table + L-curve corner choice."""
    table = []
    for rho in rhos:
        x, res, rough = solve_tikhonov(J, D, dv, rho)
        table.append({"rho": float(rho), "residual": res, "roughness": rough})
    res = np.array([t["residual"] for t in table])
    rough = np.array([t["roughness"] for t in table])
    idx, fallback, _ = lcurve_corner(res, rough)
    return table, idx, fallback


def discrepancy_choice(table, noise_std, n_meas):
    """Morozov choice: grid rho minimizing |residual - s*sqrt(m)|.

    Returns (index, edge_flag). Edge flag is set when the argmin sits
    on a grid edge (reported, not hidden).
    """
    target = noise_std * math.sqrt(n_meas)
    res = np.array([t["residual"] for t in table])
    idx = int(np.argmin(np.abs(res - target)))
    return idx, bool(idx == 0 or idx == len(table) - 1), float(target)


def reconstruct(J, D, dv, noise_std, rhos=None, iterations=1,
                sigma_bg=1.0, mesh=None, prior="smoothness"):
    """Full reconstruction with all published diagnostics.

    Lambda is chosen by the discrepancy principle over the rho sweep;
    the L-curve corner over the same sweep is published as a
    diagnostic. mesh (optional, for iterations): dict with pts/tris/
    ke/elec/adj_pairs/rows so J can be recomputed at the updated
    conductivity. prior: 'smoothness' (D) or 'flat' (identity
    penalty, diagnostic). Returns a result dict carrying prior,
    lambda rule, noise assumption, resolution map, uncertainty map,
    and spectrum with every image.
    """
    if rhos is None:
        rhos = 10.0 ** (-6.0 + 0.25 * np.arange(29))
    Duse = D if prior == "smoothness" else np.eye(J.shape[1])
    table, corner_idx, corner_fallback = lambda_sweep(J, Duse, dv, rhos)
    idx, edge_flag, target = discrepancy_choice(table, noise_std, len(dv))
    rho_star = float(rhos[idx])
    x, _, _ = solve_tikhonov(J, Duse, dv, rho_star)
    scale = float(np.mean(np.diag(J.T @ J)))
    lam2 = rho_star * scale
    J_cur = J
    if iterations > 1 and mesh is not None:
        sigma = np.full(J.shape[1], sigma_bg) + x
        for _ in range(iterations - 1):
            sigma = np.clip(sigma, 0.05, 5.0)
            U = forward_solve(mesh["pts"], mesh["tris"], mesh["ke"],
                              sigma, mesh["elec"], mesh["adj_pairs"])
            G = element_gradients(U, mesh["tris"], mesh["grads"])
            J_cur = jacobian(G, mesh["area"], mesh["rows"], mesh["adj_pairs"])
            x_new, _, _ = solve_tikhonov(J_cur, Duse, dv, rho_star)
            sigma = np.full(J.shape[1], sigma_bg) + x_new
            x = x_new
    M = J_cur.T @ J_cur + lam2 * (Duse.T @ Duse)
    Minv = np.linalg.inv(M)
    R = Minv @ (J_cur.T @ J_cur)
    # Covariance under white noise std s: s^2 M^-1 J^T J M^-1.
    B = np.linalg.solve(M, J_cur.T)  # M^-1 J^T, (E, m)
    F = J_cur @ Minv  # (m, E); Cov = s^2 B F, diag via contraction
    cov_diag = (noise_std ** 2) * np.einsum("ik,ki->i", B, F)
    uncertainty = np.sqrt(np.maximum(cov_diag, 0.0))
    sv = np.linalg.svd(J_cur, compute_uv=False)
    return {
        "x": x,
        "prior": ("first-difference smoothness: penalty "
                  "lambda^2*||D x||^2, D = element-adjacency difference "
                  "operator (standard Tikhonov, this implementation)"
                  if prior == "smoothness" else
                  "flat: penalty lambda^2*||x||^2 (diagnostic only)"),
        "lambda_rule": ("Morozov discrepancy principle over rho in "
                        "1e-6..1e1, lambda^2 = rho*mean(diag(J^T J)): "
                        "rho minimizing | ||Jx-dV|| - s*sqrt(m) |"),
        "rho": rho_star,
        "lambda": float(math.sqrt(lam2)),
        "lambda_edge_flag": bool(edge_flag),
        "lambda_target_residual": target,
        "lcurve_corner_rho": float(rhos[corner_idx]),
        "lcurve_corner_fallback": bool(corner_fallback),
        "lambda_table": table,
        "noise_assumption": ("white Gaussian, std s = %.6g "
                             "(noise_frac x RMS of noiseless dV)" % noise_std),
        "noise_std": float(noise_std),
        "discretization": ("unit disk, P1 FEM, reconstruction mesh 289 "
                           "nodes / 528 elements, 16 point electrodes, "
                           "adjacent drive/measure, 208 measurements"),
        "imaging": "difference imaging; reference uniform sigma=1.0",
        "resolution_map": np.diag(R),
        "resolution_mean": float(np.mean(np.diag(R))),
        "uncertainty_map": uncertainty,
        "uncertainty_mean": float(np.mean(uncertainty)),
        "uncertainty_max": float(np.max(uncertainty)),
        "singular_values": sv,
        "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        "noise_amplification": (float(np.sqrt(np.mean(cov_diag)) / noise_std)
                                if noise_std > 0 else None),
        "iterations": iterations,
    }


def psf_area(R, area, centroids):
    """Resolution-cell area at the centre element: region where the
    point-spread row exceeds half its diagonal value."""
    j0 = int(np.argmin(centroids[:, 0] ** 2 + centroids[:, 1] ** 2))
    row = R[j0]
    mask = row > 0.5 * row[j0]
    return float(np.sum(area[mask])), j0

# ---------------------------------------------------------------- metrics


def region_metrics(x, area, centroids, inclusions, sign):
    """Localization + size error of the half-max region (given sign).

    sign: +1 for positive-contrast inclusions, -1 for negative.
    Returns dict; empty region -> None values (reported, not hidden).
    """
    xs = sign * x
    peak = float(np.max(xs))
    if peak <= 0:
        return {"peak": peak, "localization_error": None,
                "size_error": None, "area_rec": 0.0, "contrast_recovery": None}
    mask = xs >= 0.5 * peak
    w = xs[mask]
    cen = np.average(centroids[mask], axis=0, weights=w)
    true_c = np.mean([[c[0], c[1]] for c in inclusions], axis=0)
    loc = float(np.linalg.norm(cen - true_c))
    area_rec = float(np.sum(area[mask]))
    area_true = float(sum(math.pi * c[2] ** 2 for c in inclusions))
    true_contrast = float(np.mean([abs(c[3] - 1.0) for c in inclusions]))
    return {"peak": peak, "localization_error": loc,
            "size_error": (area_rec - area_true) / area_true,
            "area_rec": area_rec, "area_true": area_true,
            "contrast_recovery": peak / true_contrast if true_contrast else None,
            "centroid": cen.tolist(), "true_center": true_c.tolist()}


def count_peaks(x, area, centroids, pairs, sign, frac=0.3, min_sep=0.15):
    """Count distinct peaks of sign*x above frac*max, separated by min_sep.

    A peak is an element above all its neighbours (adjacency pairs).
    """
    xs = sign * x
    peak = float(np.max(xs))
    if peak <= 0:
        return 0, []
    neigh = [set() for _ in range(len(x))]
    for a, b in pairs:
        neigh[a].add(b)
        neigh[b].add(a)
    cands = []
    for e in range(len(x)):
        if xs[e] >= frac * peak and all(xs[e] >= xs[n] for n in neigh[e]):
            cands.append(e)
    cands.sort(key=lambda e: -xs[e])
    kept = []
    for e in cands:
        if all(np.linalg.norm(centroids[e] - centroids[k]) >= min_sep
               for k in kept):
            kept.append(e)
    return len(kept), [centroids[k].tolist() for k in kept]


def dip_ratio(x, centroids, c1, c2, sign, band=0.12):
    """Valley-dip resolution criterion along the segment c1 -> c2.

    Takes elements whose centroids lie within `band` of the segment,
    projects them to t in [0, 1], linearly interpolates the sampled
    profile onto a fine t grid (bridging mesh-sampling gaps, e.g. the
    centroid-free centre of a short segment), and compares the
    minimum in the middle third (0.35 < t < 0.65) against the maxima
    of the two outer thirds. Ratio < 0.85 counts as resolved (two
    distinct structures with a valley between them). Returns (ratio,
    n_band). A plateau with spurious local maxima does not pass.
    """
    xs = sign * x
    a = np.asarray(c1, dtype=float)
    b = np.asarray(c2, dtype=float)
    ab = b - a
    L2 = float(ab @ ab)
    rel = centroids - a
    t = (rel @ ab) / L2
    perp = np.abs(rel[:, 0] * ab[1] - rel[:, 1] * ab[0]) / math.sqrt(L2)
    sel = (t >= 0.0) & (t <= 1.0) & (perp <= band)
    tt, vv = t[sel], xs[sel]
    if len(tt) < 6:
        return None, int(len(tt))
    order = np.argsort(tt)
    t_grid = np.linspace(0.0, 1.0, 51)
    v_grid = np.interp(t_grid, tt[order], vv[order])
    left = float(np.max(v_grid[t_grid < 0.4]))
    right = float(np.max(v_grid[t_grid > 0.6]))
    valley = float(np.min(v_grid[(t_grid >= 0.35) & (t_grid <= 0.65)]))
    return valley / min(left, right), int(len(tt))


def result_header(result):
    """The publication block printed with every result (U-M02 rule)."""
    return {
        "prior": result["prior"],
        "lambda_rule": result["lambda_rule"],
        "rho": result["rho"],
        "lambda": result["lambda"],
        "lambda_edge_flag": result["lambda_edge_flag"],
        "lcurve_corner_rho": result["lcurve_corner_rho"],
        "noise_assumption": result["noise_assumption"],
        "discretization": result["discretization"],
        "imaging": result["imaging"],
        "resolution_mean": result["resolution_mean"],
        "uncertainty_mean": result["uncertainty_mean"],
        "uncertainty_max": result["uncertainty_max"],
        "condition_number": result["condition_number"],
        "noise_amplification": result["noise_amplification"],
        "iterations": result["iterations"],
    }


def build_geometry(n_ang=48, radii_recon=(0.22, 0.42, 0.60, 0.76, 0.89, 1.0),
                   radii_fine=(0.14, 0.27, 0.39, 0.50, 0.60, 0.69, 0.78, 0.86, 0.93, 1.0)):
    """Both meshes + protocol + reference Jacobian pieces."""
    adj_pairs = [(k, (k + 1) % N_ELEC) for k in range(N_ELEC)]
    rows = protocol()
    out = {"adj_pairs": adj_pairs, "rows": rows}
    for tag, radii in (("recon", radii_recon), ("fine", radii_fine)):
        pts, tris = build_mesh(n_ang, list(radii))
        area, cent, grads, ke = mesh_geometry(pts, tris)
        out[tag] = {"pts": pts, "tris": tris, "area": area, "centroids": cent,
                    "grads": grads, "ke": ke,
                    "elec": electrode_nodes(n_ang, list(radii)),
                    "pairs": element_adjacency(tris)}
        out[tag]["D"] = difference_operator(out[tag]["pairs"], len(tris))
    # Reference Jacobian on the reconstruction mesh at sigma = 1.
    r = out["recon"]
    U = forward_solve(r["pts"], r["tris"], r["ke"],
                      np.ones(len(r["tris"])), r["elec"], adj_pairs)
    G = element_gradients(U, r["tris"], r["grads"])
    out["J_ref"] = jacobian(G, r["area"], rows, adj_pairs)
    out["mesh_iter"] = {"pts": r["pts"], "tris": r["tris"], "ke": r["ke"],
                        "elec": r["elec"], "adj_pairs": adj_pairs,
                        "rows": rows, "grads": r["grads"], "area": r["area"]}
    return out


def cli_summary(result, extra=None):
    lines = [json.dumps(result_header(result), indent=1)]
    if extra:
        lines.append(json.dumps(extra, indent=1))
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="EIT difference-imaging lane")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("spectrum", help="Jacobian spectrum + resolution stats")
    sp.add_argument("--out", default=None)
    rc = sub.add_parser("reconstruct", help="single synthetic reconstruction")
    rc.add_argument("--cx", type=float, default=0.3)
    rc.add_argument("--cy", type=float, default=0.2)
    rc.add_argument("--radius", type=float, default=0.18)
    rc.add_argument("--sigma", type=float, default=1.5)
    rc.add_argument("--noise-frac", type=float, default=0.002)
    rc.add_argument("--seed", type=int, default=0)
    rc.add_argument("--iterations", type=int, default=1)
    rc.add_argument("--out", default=None)
    args = ap.parse_args()
    geo = build_geometry()
    if args.cmd == "spectrum":
        J = geo["J_ref"]
        sv = np.linalg.svd(J, compute_uv=False)
        r = geo["recon"]
        res = reconstruct(J, r["D"], np.zeros(len(geo["rows"])), 1e-6)
        R = None
        M = J.T @ J + res["rho"] * float(np.mean(np.diag(J.T @ J))) * (r["D"].T @ r["D"])
        R = np.linalg.inv(M) @ (J.T @ J)
        psf, j0 = psf_area(R, r["area"], r["centroids"])
        summ = {"singular_values_top": sv[:10].tolist(),
                "singular_value_50": float(sv[49]),
                "singular_value_100": float(sv[99]),
                "singular_value_last": float(sv[-1]),
                "condition_number": float(sv[0] / sv[-1]),
                "resolution_mean": float(np.mean(np.diag(R))),
                "resolution_center": float(np.diag(R)[j0]),
                "resolution_edge_min": float(np.min(np.diag(R))),
                "psf_area_center": psf,
                "psf_radius_center": float(math.sqrt(psf / math.pi))}
        print(json.dumps(summ, indent=1))
        if args.out:
            np.savez(args.out, singular_values=sv, resolution_diag=np.diag(R))
    else:
        data = make_data(geo["fine"]["pts"], geo["fine"]["tris"],
                         geo["fine"]["ke"], geo["fine"]["elec"], geo["rows"],
                         geo["adj_pairs"],
                         [(args.cx, args.cy, args.radius, args.sigma)],
                         noise_frac=args.noise_frac, noise_seed=args.seed)
        res = reconstruct(geo["J_ref"], geo["recon"]["D"], data["dv"],
                          data["noise_std"], iterations=args.iterations,
                          mesh=geo["mesh_iter"] if args.iterations > 1 else None)
        print(cli_summary(res))
        if args.out:
            np.savez(args.out, x=res["x"],
                     resolution_map=res["resolution_map"],
                     uncertainty_map=res["uncertainty_map"],
                     dv=data["dv"], centroids=geo["recon"]["centroids"],
                     area=geo["recon"]["area"])
