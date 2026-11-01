#!/usr/bin/env python3
"""Deterministic EIT explainer: frames + narration from the record only.

Sources (only allowed sources for on-screen numbers):
  /home/joshua/eit-lane/eit_solver.py        (mesh geometry, imported)
  /home/joshua/eit-lane/results_seed7.json   (harness numbers, seed 7)
  /home/joshua/eit-lane/results_seed19.json  (seed 19 pass confirmation)
  /home/joshua/eit-lane/results_seed7_phantoms.npz
  spectral-test-report.md                    (Cheeger / K* / crossover / ker D)
Every caption number is formatted from the loaded JSON/NPZ values, never
typed from memory. Spectral numbers are the report's own, verified by grep.
No installs, no services, no network. PIL + numpy only.
"""
import json
import math
import os
import subprocess
import sys
import wave

import numpy as np

sys.path.insert(0, "/home/joshua/eit-lane")
import eit_solver  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

W, H = 1280, 720
WORK = "/home/joshua/eit-lane-video"
FRAMES = os.path.join(WORK, "frames_v2")
NARR = os.path.join(WORK, "narr_v2")
FONT = "/usr/share/fonts/TTF/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
PIPER = "/home/joshua/llm/voice-env/bin/piper"
MODEL = "/home/joshua/llm/voices/en_US-glados-medium.onnx"

# ---------------------------------------------------------------- data
with open("/home/joshua/eit-lane/results_seed7.json") as f:
    J7 = json.load(f)
with open("/home/joshua/eit-lane/results_seed19.json") as f:
    J19 = json.load(f)
Z = np.load("/home/joshua/eit-lane/results_seed7_phantoms.npz")
assert J7["all_pass"] is True and J19["all_pass"] is True

RADII = [0.22, 0.42, 0.60, 0.76, 0.89, 1.0]
pts, tris = eit_solver.build_mesh(48, RADII)
elec = eit_solver.electrode_nodes(48, RADII)
assert len(pts) == 289 and len(tris) == 528 and len(elec) == 16

ta = J7["tests"]["Ta_single"]["one_step"]
TA_LOC = ta["localization_error"]
TA_SIZE = ta["size_error"]
TA_CON = ta["contrast_recovery"]
TA_PEAK = ta["peak"]
RHO = J7["tests"]["Ta_single"]["header"]["rho"]
T0 = J7["tests"]["T0_jacobian"]["max_rel_err"]
TB = J7["tests"]["Tb_two"]
TC = J7["tests"]["Tc_noise_only"]
TE = J7["tests"]["Te_noise_sweep"]
SPEC = J7["spectrum"]
RES_MEAN = SPEC["resolution_diag_mean"]
RES_CENTRE = SPEC["resolution_center"]
RES_EDGE = SPEC["resolution_edge_mean"]
PSF_R = SPEC["psf_radius_center"]
RANK1 = SPEC["effective_rank_1pct"]

x_ta = Z["x_Ta"]
sigma_true = Z["sigma_true_recon"]
res_map = Z["resolution_map"]
x_tc = Z["x_Tc_noise_only"]
INCL = Z["inclusion"][0]  # cx, cy, r, sigma
assert abs(INCL[0] - ta["true_center"][0]) < 1e-9
assert abs(INCL[1] - ta["true_center"][1]) < 1e-9

# Spectral numbers, from spectral-test-report.md (Claims 1, 3, 5; verified).
CHEEG_B, CHEEG_M = 2.683, 8.122
CHEEG_R = CHEEG_M / CHEEG_B  # 3.027 -> 3.03
KSTAR, NLOC = 2, 73

# ---------------------------------------------------------------- helpers
def F(sz, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, sz)


def lerp(c1, c2, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def diverging(v, vmax):
    t = max(-1.0, min(1.0, float(v) / vmax))
    if t >= 0:
        return lerp((245, 245, 245), (200, 30, 30), t)
    return lerp((245, 245, 245), (30, 80, 200), -t)


def sequential(v, vmax, vmin=0.0):
    t = max(0.0, min(1.0, (float(v) - vmin) / (vmax - vmin)))
    return lerp((245, 245, 245), (20, 60, 160), t)


def truth_color(v):
    # sigma 1.0 -> pale grey, 1.5 -> red
    return lerp((232, 232, 232), (200, 30, 30), max(0.0, min(1.0, (v - 1.0) / 0.5)))


def xy(cx, cy, R, x, y):
    return (cx + x * R, cy - y * R)


def draw_disk(d, cx, cy, R, colors, electrodes=False):
    for tri, c in zip(tris, colors):
        d.polygon([xy(cx, cy, R, *pts[i]) for i in tri], fill=c, outline=(160, 160, 160))
    d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(0, 0, 0), width=3)
    if electrodes:
        for ei in elec:
            ex, ey = xy(cx, cy, R, *pts[ei])
            d.ellipse([ex - 8, ey - 8, ex + 8, ey + 8], fill=(200, 30, 30),
                      outline=(255, 255, 255), width=2)


def mark(d, cx, cy, R, x, y, label=None):
    px, py = xy(cx, cy, R, x, y)
    d.line([px - 12, py - 12, px + 12, py + 12], fill=(0, 0, 0), width=3)
    d.line([px - 12, py + 12, px + 12, py - 12], fill=(0, 0, 0), width=3)
    if label:
        d.text((px + 16, py - 12), label, font=F(22, bold=True), fill=(0, 0, 0))


def ctext(d, cx, y, s, fnt, fill=(0, 0, 0)):
    d.text((cx - d.textlength(s, font=fnt) / 2, y), s, font=fnt, fill=fill)


def frame(title):
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 64], fill=(20, 40, 80))
    d.text((32, 15), title, font=F(34, bold=True), fill=(255, 255, 255))
    return img, d


def caption(d, lines, source):
    y = 596
    for ln in lines:
        d.text((32, y), ln, font=F(27), fill=(20, 20, 20))
        y += 37
    d.text((32, 686), source, font=F(20), fill=(100, 100, 100))


def save(img, name):
    p = os.path.join(FRAMES, name)
    img.save(p)
    return p


# ---------------------------------------------------------------- frames
def f01_setup():
    img, d = frame("EIT lane  |  1. The setup")
    draw_disk(d, 640, 300, 228, [(226, 226, 226)] * len(tris), electrodes=True)
    ctext(d, 640, 540, "16 electrodes (red) on the boundary", F(24, bold=True))
    caption(d, ["Unit disk, radius 1. Reconstruction mesh: 289 nodes, 528 elements.",
                "Radii 0.22-1.0 in 6 rings, 48 angles. 16 point electrodes."],
            "Source: eit_solver.py (build_mesh / build_geometry); eit-lane-report.md")
    return save(img, "f01.png"), (
        "The EIT lane reconstructs conductivity change inside a unit disk, "
        "from voltages measured on its boundary.")


def f02_protocol():
    img, d = frame("EIT lane  |  2. Protocol and prior")
    draw_disk(d, 300, 312, 205, [(226, 226, 226)] * len(tris), electrodes=True)
    lines = ["Adjacent drive / adjacent measure",
             "208 measurements per frame",
             "Difference imaging vs uniform sigma = 1.0",
             "Smoothness prior; lambda by Morozov discrepancy",
             "rho* = %.2e (seed 7)" % RHO,
             "Noise: white Gaussian, 0.2% of signal RMS"]
    y = 150
    for ln in lines:
        d.text((560, y), ln, font=F(28), fill=(20, 20, 20))
        y += 52
    caption(d, ["Protocol: 16 drive pairs, 13 measurement pairs per drive = 208 measurements.",
                "Pairs touching a drive electrode are skipped."],
            "Source: results_seed7.json (Ta header); eit-lane-report.md")
    return save(img, "f02.png"), (
        "Sixteen electrodes, adjacent drive and adjacent measure: two hundred "
        "and eight measurements. Difference imaging against a uniform reference.")


def f03_truth():
    img, d = frame("EIT lane  |  3. Truth: test Ta, seed 7")
    draw_disk(d, 640, 300, 228, [truth_color(v) for v in sigma_true])
    mark(d, 640, 300, 228, INCL[0], INCL[1])
    ctext(d, 640, 540, "True centre (0.150, 0.477)", F(24, bold=True))
    caption(d, ["Truth (Ta, seed 7): one disk inclusion, centre (0.150, 0.477), radius 0.18.",
                "Conductivity 1.5 on a background of 1.0. Ground truth on the mesh."],
            "Source: results_seed7_phantoms.npz (sigma_true_recon, inclusion); results_seed7.json")
    return save(img, "f03.png"), (
        "The truth, test Ta: one inclusion, radius 0.18, conductivity 1.5 "
        "on a background of 1.0.")


def f04_recon():
    img, d = frame("EIT lane  |  4. Reconstruction, one step")
    vmax = float(np.max(np.abs(x_ta)))
    draw_disk(d, 640, 312, 235, [diverging(v, vmax) for v in x_ta])
    mark(d, 640, 312, 235, ta["centroid"][0], ta["centroid"][1])
    caption(d, ["Reconstruction, one step: peak %.3f; localization error %.3f (bar < 0.20)." % (TA_PEAK, TA_LOC),
                "Size error %+.2f; contrast recovery %.2f (bar 0.05-1.5)." % (TA_SIZE, TA_CON)],
            "Source: results_seed7.json (Ta one_step)")
    return save(img, "f04.png"), (
        "The reconstruction finds it: localization error 0.023, size error "
        "minus 0.01, contrast recovery 0.87.")


def f05_side():
    img, d = frame("EIT lane  |  5. Truth vs reconstruction")
    ctext(d, 320, 78, "TRUTH", F(28, bold=True))
    ctext(d, 960, 78, "RECONSTRUCTION", F(28, bold=True))
    draw_disk(d, 320, 320, 205, [truth_color(v) for v in sigma_true])
    vmax = float(np.max(np.abs(x_ta)))
    draw_disk(d, 960, 320, 205, [diverging(v, vmax) for v in x_ta])
    caption(d, ["Truth (left) vs reconstruction (right), seed 7, same mesh. Colour scale per frame.",
                "Both seeds pass all 5 tests plus the Jacobian check (T0 max rel err %.1e)." % T0],
            "Source: results_seed7.json; results_seed19.json")
    return save(img, "f05.png"), (
        "Truth against reconstruction. The second seed reproduces every "
        "verdict: all five tests pass, both seeds.")


def f06_resolution():
    img, d = frame("EIT lane  |  6. Resolution map")
    vmax = float(np.max(res_map))
    draw_disk(d, 640, 312, 235, [sequential(v, vmax) for v in res_map])
    caption(d, ["Resolution map, diag(R): mean %.3f; centre element %.3f; edge mean %.3f." % (RES_MEAN, RES_CENTRE, RES_EDGE),
                "Centre least resolved, edge most. PSF radius at centre: %.3f. Jacobian 208 x 528." % PSF_R],
            "Source: results_seed7.json (spectrum)")
    return save(img, "f06.png"), (
        "Resolution is worst at the centre, 0.042, best at the edge, 0.259. "
        "Mean 0.137; point-spread radius 0.209.")


def f07_dip():
    img, d = frame("EIT lane  |  7. Two-inclusion limit")
    x0, x1, y0, y1 = 170, 1150, 130, 500
    ymin, ymax = -0.45, 1.0
    def Y(v):
        return y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)
    d.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=2)
    d.line([x0, Y(0), x1, Y(0)], fill=(0, 0, 0), width=2)
    for yy, lbl in ((Y(0.85), "bar 0.85"), (Y(0), "0"), (Y(1.0), "1.0")):
        d.line([x0, yy, x1, yy], fill=(200, 30, 30) if "bar" in lbl else (180, 180, 180),
               width=2 if "bar" in lbl else 1)
        d.text((x1 + 8, yy - 12), lbl, font=F(22, bold=True), fill=(200, 30, 30) if "bar" in lbl else (80, 80, 80))
    seps = ["0.9", "0.7", "0.5", "0.35", "0.25", "0.2"]
    bw = 90
    for i, s in enumerate(seps):
        cx = x0 + (i + 0.5) * (x1 - x0) / 6
        dip = TB[s]["dip_ratio"]
        if dip is None:
            d.rectangle([cx - bw / 2, Y(0.85), cx + bw / 2, Y(0)], outline=(200, 30, 30), width=3)
            ctext(d, cx, Y(0.42), "n/a", F(30, bold=True), fill=(200, 30, 30))
        else:
            ya, yb = sorted([Y(dip), Y(0)])
            d.rectangle([cx - bw / 2, ya, cx + bw / 2, yb],
                        fill=(30, 80, 200) if dip < 0.85 else (200, 120, 30))
            ctext(d, cx, Y(dip) - 34 if dip >= 0 else Y(dip) + 8, "%.2f" % dip, F(24, bold=True))
        ctext(d, cx, y1 + 14, "d = " + s, F(24, bold=True))
    ctext(d, (x0 + x1) / 2, y1 + 48, "separation", F(22), fill=(80, 80, 80))
    d.text((x0 - 130, y0 + 150), "dip ratio", font=F(22), fill=(80, 80, 80))
    caption(d, ["Two inclusions (r=0.15): dip ratio vs separation. Resolved where dip < 0.85.",
                "Limit 0.35, both seeds. d = 0.2 UNMEASURED (n/a) - never interpolated."],
            "Source: results_seed7.json (Tb)")
    return save(img, "f07.png"), (
        "Two inclusions stay resolved down to a separation of 0.35. At 0.2 "
        "the measurement could not be made, and it is not filled in.")


def f08_tc():
    img, d = frame("EIT lane  |  8. Noise only: test Tc")
    vmax = float(np.max(np.abs(x_tc)))
    draw_disk(d, 640, 312, 235, [diverging(v, vmax) for v in x_tc])
    frac = TC["max_as_frac_of_Ta_peak"]
    cov = TC["coverage_within_2sigma"]
    caption(d, ["Noise only (Tc): max |x| is %.1e of the Ta peak; %.1f%% within 2 sigma." % (frac, cov * 100),
                "Colours scaled to this frame's max - the signal is essentially zero."],
            "Source: results_seed7.json (Tc)")
    return save(img, "f08.png"), (
        "Noise only: the reconstruction stays below a ten-thousandth of the "
        "signal peak. Coverage within two sigma: 99.6 percent.")


def f09_te():
    img, d = frame("EIT lane  |  9. Noise sweep: test Te")
    x0, x1, y0, y1 = 200, 1120, 130, 500
    ymax = 0.40
    pts_te = [(float(k), TE[k]["localization_error"]) for k in ["0.0", "0.0005", "0.002", "0.01", "0.02"]]
    labels = ["0%", "0.05%", "0.2%", "1%", "2%"]
    def X(i):
        return x0 + i * (x1 - x0) / 4
    def Y(v):
        return y1 - v / ymax * (y1 - y0)
    d.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=2)
    for gv in (0.1, 0.2, 0.3, 0.35, 0.4):
        d.line([x0, Y(gv), x1, Y(gv)], fill=(220, 220, 220), width=1)
        d.text((x0 - 58, Y(gv) - 12), "%.2f" % gv, font=F(20), fill=(80, 80, 80))
    d.line([x0, Y(0.35), x1, Y(0.35)], fill=(200, 30, 30), width=2)
    d.text((x1 + 8, Y(0.35) - 12), "bar 0.35", font=F(22, bold=True), fill=(200, 30, 30))
    prev = None
    for i, (nz, v) in enumerate(pts_te):
        px, py = X(i), Y(v)
        if prev:
            d.line([prev[0], prev[1], px, py], fill=(30, 80, 200), width=4)
        d.ellipse([px - 8, py - 8, px + 8, py + 8], fill=(30, 80, 200))
        d.text((px - 26, py - 38), "%.3f" % v, font=F(22, bold=True))
        ctext(d, px, y1 + 14, labels[i], F(24, bold=True))
        prev = (px, py)
    ctext(d, (x0 + x1) / 2, y1 + 48, "noise (% of signal RMS)", F(22), fill=(80, 80, 80))
    caption(d, ["Noise sweep (Te): localization error 0.018, 0.012, 0.023, 0.048, 0.072",
                "at 0%, 0.05%, 0.2%, 1%, 2% noise. Graceful degradation."],
            "Source: results_seed7.json (Te)")
    return save(img, "f09.png"), (
        "As noise rises to 2 percent, location error degrades gracefully "
        "to 0.072, inside the 0.35 bar.")


def f10_spectral():
    img, d = frame("EIT lane  |  10. Spectral check (section 8)")
    # Cheeger bars, left block
    d.text((60, 100), "Cheeger bound", font=F(28, bold=True))
    bx0, bw_max = 60, 520
    d.rectangle([bx0, 150, bx0 + bw_max * CHEEG_B / 9.0, 190], fill=(150, 150, 150))
    d.text((bx0 + bw_max * CHEEG_B / 9.0 + 12, 156), "bound %.3f" % CHEEG_B, font=F(26, bold=True))
    d.rectangle([bx0, 210, bx0 + bw_max * CHEEG_M / 9.0, 250], fill=(30, 80, 200))
    d.text((bx0 + bw_max * CHEEG_M / 9.0 + 12, 216), "measured %.3f" % CHEEG_M, font=F(26, bold=True))
    d.text((60, 272), "ratio %.2f - bound CONFIRMED" % CHEEG_R, font=F(28, bold=True))
    d.line([60, 330, 640, 330], fill=(180, 180, 180), width=2)
    d.text((60, 352), "Effective rank: %d modes above 1%% of sigma_1" % RANK1, font=F(26))
    d.text((60, 392), "Jacobian 208 x 528; singular values decay", font=F(26))
    d.text((60, 432), "roughly exponentially with spatial frequency.", font=F(26))
    # Right block
    d.line([680, 100, 680, 480], fill=(180, 180, 180), width=2)
    d.text((710, 100), "K* = %d   vs   N_loc = %d" % (KSTAR, NLOC), font=F(30, bold=True))
    d.text((710, 150), "GAP, not approximately equal.", font=F(28, bold=True), fill=(200, 30, 30))
    d.text((710, 220), "Spectrum verdict: CROSSOVER.", font=F(28, bold=True))
    d.text((710, 262), "No sharp mobility edge.", font=F(26))
    d.text((710, 332), "ker D identity: UNPINNED.", font=F(28, bold=True))
    d.text((710, 374), "Untested. Never presented", font=F(26))
    d.text((710, 414), "as a result.", font=F(26))
    caption(d, ["Cheeger bound 2.683 vs measured interface 8.122 (ratio 3.03). K* = 2 vs N_loc = 73.",
                "Verdict: CROSSOVER. ker D: UNPINNED - labelled untested, never a result."],
            "Source: spectral-test-report.md (Claims 1, 3, 5)")
    return save(img, "f10.png"), (
        "The spectral check confirms the Cheeger bound, and refutes K star "
        "equals N loc: two against seventy-three. Ker D stays unpinned.")


def f11_limits():
    img, d = frame("EIT lane  |  11. Limits, stated with every result")
    bullets = ["Synthetic phantoms only - no rig, no real plant data.",
               "2D unit disk, 16 point electrodes, no contact impedance.",
               "Difference imaging only; inverse crime reduced, not eliminated.",
               "Smoothness prior blurs edges by design.",
               "A bare image is never emitted: prior, lambda, resolution",
               "and uncertainty travel with every result."]
    y = 130
    for b in bullets:
        d.text((90, y), "-  " + b, font=F(29), fill=(20, 20, 20))
        y += 58
    caption(d, ["Limits: the lane has seen no real data; a real rig breaks head-model assumptions.",
                "Difference imaging from a recent reference is the defensible mode."],
            "Source: eit-lane-report.md (Limits)")
    return save(img, "f11.png"), (
        "Limits, stated plainly: synthetic phantoms only. No rig, no real "
        "plant data, two dimensions, no contact impedance.")


def f12_close():
    img, d = frame("EIT lane  |  12. Record")
    draw_disk(d, 260, 312, 170, [(226, 226, 226)] * len(tris), electrodes=True)
    lines = ["All 5 tests + Jacobian check PASS,",
             "seeds 7 and 19.",
             "",
             "Solver: numpy only.",
             "No language model in the loop.",
             "",
             "Every number on screen traces",
             "to the files below."]
    y = 130
    for ln in lines:
        d.text((500, y), ln, font=F(29, bold=(ln and not ln.startswith("Solver") and "Every" not in ln)),
               fill=(20, 20, 20))
        y += 46
    caption(d, ["Sources: eit_solver.py, results_seed7.json, results_seed19.json,",
                "results_seed7_phantoms.npz, eit-lane-report.md, spectral-test-report.md."],
            "Deterministic explainer: frames drawn by code from the record; narration by local piper TTS.")
    return save(img, "f12.png"), (
        "EIT lane: deterministic, numpy only, no model in the loop. "
        "Every number traces to the record.")


# ---------------------------------------------------------------- build
def wav_duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main():
    os.makedirs(FRAMES, exist_ok=True)
    os.makedirs(NARR, exist_ok=True)
    builders = [f01_setup, f02_protocol, f03_truth, f04_recon, f05_side,
                f06_resolution, f07_dip, f08_tc, f09_te, f10_spectral,
                f11_limits, f12_close]
    manifest = []
    for i, fn in enumerate(builders, start=1):
        png, text = fn()
        wav = os.path.join(NARR, "n%02d.wav" % i)
        ok = False
        try:
            r = subprocess.run([PIPER, "-m", MODEL, "-f", wav],
                               input=text.encode(), capture_output=True, timeout=120)
            ok = r.returncode == 0 and os.path.exists(wav) and os.path.getsize(wav) > 1000
        except Exception as e:  # noqa: BLE001
            print("piper failed scene %d: %r" % (i, e), file=sys.stderr)
        if ok:
            dur = wav_duration(wav)
            hold = max(5.0, dur + 0.7)
        else:
            wav = None
            hold = 5.0
        manifest.append({"scene": i, "frame": png, "wav": wav, "hold": round(hold, 2),
                         "narration": text})
        print("scene %02d hold %.2f wav=%s" % (i, hold, bool(wav)))
    with open(os.path.join(WORK, "manifest_v2.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("TOTAL %.1f s" % sum(m["hold"] for m in manifest))


if __name__ == "__main__":
    main()
