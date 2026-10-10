"""Draw the meeg-utils architecture diagram.

Usage::

    python resources/make_architecture.py resources/architecture.svg
    cp resources/architecture.svg docs/source/_static/architecture.svg

Four modules (read, acquisition QC, per-run preprocessing, epoching), the
BIDS data passed between them, and the class behind every step. The style
follows the HAD-MEEG pipeline figure (preprocessing_pipeline.png).
"""

import sys
from html import escape
from itertools import pairwise
from pathlib import Path

W, H = 1400, 640
SANS = "'Helvetica Neue', Helvetica, Arial, Inter, sans-serif"
MONO = "'SFMono-Regular', Menlo, Consolas, 'DejaVu Sans Mono', monospace"

INK, MUTED, LINE = "#1c2740", "#707b8e", "#3b4a60"
BLUE = "#2f5fb3"  # steps and module names
CARD_FILL, CARD_STROKE = "#f4f7fc", "#cfdaee"
DATA_FILL, DATA_STROKE, DATA_INK = "#edf7e8", "#9ccb82", "#28481c"
INPUT_FILL, INPUT_STROKE = "#eceef2", "#c3c9d2"

out: list[str] = []


def add(s: str) -> None:
    out.append(s)


def text(x, y, s, size=13, weight=400, fill=INK, anchor="middle", mono=False):
    add(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{MONO if mono else SANS}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">'
        f"{escape(s)}</text>"
    )


def card(x, y, w, h, title, module, libs):
    add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{CARD_FILL}" '
        f'stroke="{CARD_STROKE}" stroke-width="1.5"/>'
    )
    text(x + 20, y + 30, title, size=17, weight=600, anchor="start")
    text(x + 20 + len(title) * 17 * 0.53 + 12, y + 29, module, size=12.5, fill=BLUE, anchor="start",
         mono=True)  # fmt: skip
    text(x + w - 16, y + h - 12, libs, size=11, fill=MUTED, anchor="end")


def step(x, y, w, h, title, code):
    add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="white" stroke="{BLUE}" '
        'stroke-width="1.4"/>'
    )
    text(x + w / 2, y + 23, title, size=13, weight=600)
    text(x + w / 2, y + 41, code, size=10.5, fill=BLUE, mono=True)


def para(x, y, w, h, title, sub, fill=DATA_FILL, stroke=DATA_STROKE, ink=DATA_INK, skew=12):
    pts = f"{x + skew},{y} {x + w},{y} {x + w - skew},{y + h} {x},{y + h}"
    add(
        f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.5" '
        'stroke-linejoin="round"/>'
    )
    text(x + w / 2, y + h / 2 - 3, title, size=14.5, weight=600, fill=ink)
    text(x + w / 2, y + h / 2 + 14, sub, size=11, fill=ink)


def arrow(points, head=True):
    d = "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    end = ' marker-end="url(#head)"' if head else ""
    add(
        f'<path d="{d}" fill="none" stroke="{LINE}" stroke-width="1.5" stroke-linejoin="round"'
        f"{end}/>"
    )


# ------------------------------------------------------------------------------------------------
add(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
    'role="img" aria-labelledby="title desc">'
)
add('<title id="title">meeg-utils architecture</title>')
add(
    '<desc id="desc">Recordings are read with their BIDS sidecars, checked by the acquisition QC, '
    "preprocessed run by run (an EEG and an MEG lane), epoched and combined per session. Every "
    "box names the class behind it; every step can be changed.</desc>"
)
add(
    '<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" '
    f'markerHeight="6.5" orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 z" fill="{LINE}"/>'
    "</marker></defs>"
)
add(f'<rect width="{W}" height="{H}" fill="white"/>')

TOP, TOP_H = 30, 186

# --- read ------------------------------------------------------------------------------------------
para(30, TOP, 260, 52, "Raw recordings", "any format MNE reads", INPUT_FILL, INPUT_STROKE, INK)
ry = TOP + 84
card(30, ry, 260, TOP + TOP_H - ry, "Read", "meu.io", "mne-bids")
text(160, ry + 54, "BIDS sidecars · system detection", size=12, fill=INK)
text(160, ry + 72, "Neuromag · CTF · KIT · OPM · EEG", size=11.5, fill=MUTED)
arrow([(250, TOP + 52), (250, ry - 2)])

# --- acquisition QC ----------------------------------------------------------------------------------
qx, qw = 322, 480
card(qx, TOP, qw, TOP_H, "Acquisition QC", "meu.qc", "mne · mne-denoise")
rows = (
    "flat · clipped · bridged · impedance · outlying channels",
    "line & narrowband noise · muscle · blinks · heartbeat",
    "head movement · cHPI · SQUID jumps · empty room",
    "digitization · events · photodiode · BIDS metadata",
)
for i, r in enumerate(rows):
    text(qx + qw / 2, TOP + 62 + i * 21, r, size=12, fill=INK)
text(qx + qw / 2, TOP + 62 + 4 * 21 + 8, "qc.inspect(raw) → fail · warn · ok · info",
     size=11.5, fill=BLUE, mono=True)  # fmt: skip

# --- epoching ------------------------------------------------------------------------------------------
ex, ew = 834, 536
card(ex, TOP, ew, TOP_H, "Epoching", "meu.epochs", "mne · autoreject")
bw, bh, by = 116, 52, TOP + 86
gap = (ew - 32 - 4 * bw) / 3
xs = [round(ex + 16 + i * (bw + gap)) for i in range(4)]
step(xs[0], by, bw, bh, "Epoch", "Epoch(event_id)")
step(xs[1], by, bw, bh, "Align heads", "HeadAlign()")
step(xs[2], by, bw, bh, "Repair / drop", "AutoReject()")
step(xs[3], by, bw, bh, "Combine runs", "epochs.combine")
for a, b in pairwise(range(4)):
    arrow([(xs[a] + bw + 2, by + bh / 2), (xs[b] - 3, by + bh / 2)])
top = by - 16
arrow(
    [
        (xs[0] + bw / 2, by - 2),
        (xs[0] + bw / 2, top),
        (xs[2] + bw / 2, top),
        (xs[2] + bw / 2, by - 3),
    ]
)
text(xs[1] + bw / 2, top - 5, "EEG", size=10.5, fill=MUTED)
text(xs[1] + bw / 2, by + bh + 16, "MEG: common head position", size=10.5, fill=MUTED)

# --- data ------------------------------------------------------------------------------------------------
band_y, band_h = TOP + TOP_H + 26, 86
add(
    f'<rect x="16" y="{band_y}" width="{W - 32}" height="{band_h}" rx="18" fill="none" '
    'stroke="#c9ced6" stroke-width="1.3" stroke-dasharray="8 6"/>'
)
py0, ph0 = band_y + 17, 52
para(30, py0, 260, ph0, "BIDS raw data", "per run")
para(qx + 90, py0, 300, ph0, "QC report", "HTML · outlying recordings")
pre_x = 834
para(pre_x, py0, 250, ph0, "Preprocessed runs", "desc-preproc · provenance")
para(1120, py0, 250, ph0, "Epochs", "per session · desc-epochs")

arrow([(160, TOP + TOP_H), (160, py0 - 2)])
arrow([(278, py0 + 14), (304, py0 + 14), (304, TOP + 130), (qx - 3, TOP + 130)])
arrow([(qx + qw / 2, TOP + TOP_H), (qx + qw / 2, py0 - 2)])
arrow([(xs[0] + bw / 2, py0), (xs[0] + bw / 2, by + bh + 22)])
arrow([(xs[3] + bw / 2, by + bh), (xs[3] + bw / 2, py0 - 2)])

# --- preprocessing ------------------------------------------------------------------------------------------
px, pyy = 86, band_y + band_h + 26
pw = W - 16 - px
sw, sh = 150, 52
eeg_y = pyy + 56
meg_y = eeg_y + sh + 42
phh = meg_y + sh + 48 - pyy
card(px, pyy, pw, phh, "Preprocessing · one pipeline per run", "meu.steps",
     "mne · pyprep · mne-denoise · mne-icalabel")  # fmt: skip
cols = 7
x0 = px + 34
sgap = (px + pw - 16 - x0 - cols * sw) / (cols - 1)
cx = [round(x0 + i * (sw + sgap)) for i in range(cols)]
lanes = {
    "EEG": (eeg_y, [
        ("Bridged electrodes", "BridgedElectrodes()"),
        ("High-pass", "Filter(0.1, None)"),
        ("Line noise", "LineNoise()"),
        ("Bad channels", 'BadChannels("prep")'),
        ("Interpolate", "Interpolate()"),
        ("Re-reference", "Reference()"),
        ("ICA · ICLabel", "ICA()"),
    ]),
    "MEG": (meg_y, [
        ("Bad channels", 'BadChannels("maxwell")'),
        ("SSS", "Maxwell()"),
        ("High-pass", "Filter(0.1, None)"),
        ("Resample", "Resample(250)"),
        ("Line noise", "LineNoise()"),
        None,
        ("ICA · MEGnet", 'ICA(labeler="megnet")'),
    ]),
}  # fmt: skip
for lane, (ly, steps) in lanes.items():
    text(px + 34, ly - 8, lane, size=11.5, weight=600, fill=MUTED, anchor="start")
    placed = [i for i, s in enumerate(steps) if s]
    for i in placed:
        step(cx[i], ly, sw, sh, *steps[i])
    for a, b in pairwise(placed):
        arrow([(cx[a] + sw + 2, ly + sh / 2), (cx[b] - 3, ly + sh / 2)])
text(cx[1] + sw / 2, meg_y + sh + 14, "CTF: Reference() · KIT: Regression()", size=10.5, fill=MUTED)
text(px + pw / 2, pyy + phh - 12,
     "Presets eeg-erp · eeg-rest · meg-erp · meg-rest · had-meeg are ordinary pipelines: "
     "set_params, replace, insert or remove any step",
     size=11.5, fill=MUTED)  # fmt: skip

bus_x = 52
arrow([(bus_x, py0 + ph0), (bus_x, meg_y + sh / 2)], head=False)
arrow([(bus_x, eeg_y + sh / 2), (cx[0] - 3, eeg_y + sh / 2)])
arrow([(bus_x, meg_y + sh / 2), (cx[0] - 3, meg_y + sh / 2)])
add(f'<circle cx="{bus_x}" cy="{eeg_y + sh / 2}" r="2.6" fill="{LINE}"/>')
rx = cx[6] + sw + 9
bus_top = band_y + band_h + 13
arrow([(cx[6] + sw, eeg_y + sh / 2), (rx, eeg_y + sh / 2)], head=False)
arrow([(cx[6] + sw, meg_y + sh / 2), (rx, meg_y + sh / 2), (rx, bus_top), (pre_x + 125, bus_top),
       (pre_x + 125, py0 + ph0 + 3)])  # fmt: skip

add("</svg>")
Path(sys.argv[1]).write_text("\n".join(out), encoding="utf-8")
