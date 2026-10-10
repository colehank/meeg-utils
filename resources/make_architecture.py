"""Draw the meeg-utils architecture diagram.

Usage::

    python resources/make_architecture.py resources/architecture.svg
    cp resources/architecture.svg docs/source/_static/architecture.svg

Layout (left to right, top to bottom): how to run it; reading, acquisition
QC and epoching; the BIDS data passed between the stages; preprocessing in
two lanes (EEG and MEG, each in the order of the recommended presets); what
every step records. The style follows the HAD-MEEG pipeline figure
(preprocessing_pipeline.png): module groups, steps as boxes, data as
parallelograms.
"""

import sys
from html import escape
from itertools import pairwise
from pathlib import Path

W, H = 1600, 930
SANS = "'Helvetica Neue', Helvetica, Arial, Inter, sans-serif"
MONO = "'SFMono-Regular', Menlo, Consolas, 'DejaVu Sans Mono', monospace"

INK, MUTED, LINE = "#16233a", "#6b778c", "#3b4a60"
STEP_STROKE = "#30486e"
DATA_FILL, DATA_STROKE, DATA_INK = "#ecf7e6", "#97c97a", "#24461a"
INPUT_FILL, INPUT_STROKE = "#eceef2", "#c2c8d1"
#: module name -> (accent, card fill, card stroke)
THEME = {
    "read": ("#56657c", "#f6f7f9", "#d6dbe2"),
    "qc": ("#0e7f86", "#f0f9f9", "#bfe1e1"),
    "pre": ("#2d63c8", "#f3f7fe", "#c9d9f3"),
    "epo": ("#6b4fd8", "#f6f4fe", "#d8d0f6"),
}

out: list[str] = []


def add(s: str) -> None:
    out.append(s)


def width(s: str, size: float, mono: bool = False) -> float:
    """Approximate rendered width of a label."""
    return len(s) * size * (0.61 if mono else 0.55)


def text(x, y, s, size=14, weight=400, fill=INK, anchor="middle", mono=False, extra=""):
    family = MONO if mono else SANS
    add(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{extra}>{escape(s)}</text>'
    )


def pill(x, y, s, color, size=12.5, mono=True, fill="white", h=22, pad=9):
    x, w = round(x), round(width(s, size, mono) + 2 * pad)
    add(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h}" rx="{h / 2}" fill="{fill}" '
        f'stroke="{color}" stroke-width="1.2"/>'
    )
    text(x + w / 2, y + h / 2 + size * 0.36, s, size=size, fill=color, mono=mono)
    return w


def card(x, y, w, h, key, number, title, module, libs=None, libs_x=None):
    accent, fill, stroke = THEME[key]
    add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.5"/>'
    )
    add(f'<circle cx="{x + 28}" cy="{y + 27}" r="12" fill="{accent}"/>')
    text(x + 28, y + 32, str(number), size=13, weight=700, fill="white")
    text(x + 50, y + 33, title, size=18, weight=600, anchor="start")
    pill(x + 50 + len(title) * 18 * 0.52 + 16, y + 16, module, accent)
    if libs:
        if libs_x is None:
            text(x + w - 16, y + h - 13, libs, size=11.5, fill=MUTED, anchor="end")
        else:
            text(libs_x, y + h - 13, libs, size=11.5, fill=MUTED)


def step(x, y, w, h, title, sub=None, accent=STEP_STROKE, size=13.5):
    add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="white" '
        f'stroke="{accent}" stroke-width="1.5"/>'
    )
    if sub:
        text(x + w / 2, y + h / 2 - 2, title, size=size, weight=600)
        text(x + w / 2, y + h / 2 + 14, sub, size=11.5, fill=MUTED)
    else:
        text(x + w / 2, y + h / 2 + 5, title, size=size, weight=600)


def chip(x, y, w, h, s, accent):
    add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" fill="white" '
        f'stroke="{accent}" stroke-width="1.2"/>'
    )
    text(x + w / 2, y + h / 2 + 4.3, s, size=12.5)


def para(x, y, w, h, title, sub=None, fill=DATA_FILL, stroke=DATA_STROKE, ink=DATA_INK, skew=14):
    pts = f"{x + skew},{y} {x + w},{y} {x + w - skew},{y + h} {x},{y + h}"
    add(
        f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.6" '
        'stroke-linejoin="round"/>'
    )
    if sub:
        text(x + w / 2, y + h / 2 - 3, title, size=15.5, weight=600, fill=ink)
        text(x + w / 2, y + h / 2 + 15, sub, size=11.5, fill=ink, extra=' fill-opacity="0.8"')
    else:
        text(x + w / 2, y + h / 2 + 5, title, size=15.5, weight=600, fill=ink)


def arrow(points, head=True, color=LINE, dashed=False):
    d = "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    extra = ' stroke-dasharray="5 4"' if dashed else ""
    end = ' marker-end="url(#head)"' if head else ""
    add(
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.6" stroke-linejoin="round" '
        f'stroke-linecap="round"{extra}{end}/>'
    )


def label(x, y, s, size=11.5, fill=MUTED, anchor="middle"):
    text(x, y, s, size=size, fill=fill, anchor=anchor)


# ----------------------------------------------------------------------------------------------
add(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
    'role="img" aria-labelledby="title desc">'
)
add('<title id="title">meeg-utils architecture</title>')
add(
    '<desc id="desc">Recordings are read with their BIDS sidecars, checked by the acquisition QC, '
    "preprocessed run by run (an EEG and an MEG lane), epoched and combined per session; every "
    "step records QC metrics, figures and provenance for the reports.</desc>"
)
add(
    "<defs>"
    '<marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" '
    f'orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 z" fill="{LINE}"/></marker>'
    "</defs>"
)
add(f'<rect width="{W}" height="{H}" fill="white"/>')

# --- how to run it --------------------------------------------------------------------------------
add(f'<rect x="20" y="18" width="{W - 40}" height="46" rx="12" fill="#f7f8fa" stroke="#e1e5ea"/>')
text(
    42, 47, "RUN IT", size=12, weight=700, fill=MUTED, anchor="start", extra=' letter-spacing="1.5"'
)
x = 120
for name, code in (
    ("Python", "meu.Pipeline · steps · presets"),
    ("Whole dataset", "meu.Dataset · meu.process_dataset"),
    ("Command line", "meu qc | run | preset | report"),
):
    text(x, 46, name, size=14, weight=600, anchor="start")
    x += width(name, 14) + 12
    x += pill(x, 30, code, THEME["pre"][0], size=12.5) + 54

# --- 1 read -----------------------------------------------------------------------------------------
para(40, 100, 300, 58, "Raw recordings", sub="any format MNE reads", fill=INPUT_FILL,
     stroke=INPUT_STROKE, ink=INK)  # fmt: skip
card(40, 190, 300, 132, "read", 1, "Read", "meu.io")
for i, (s, c) in enumerate((("BIDS sidecars: bads, line frequency", INK),
                            ("detects the acquisition system", INK),
                            ("Neuromag · CTF · KIT · OPM · EEG", MUTED))):  # fmt: skip
    text(190, 250 + i * 20, s, size=12.5, fill=c)
arrow([(300, 158), (300, 188)])

# --- 2 acquisition QC --------------------------------------------------------------------------------
qx, qy, qw, qh = 376, 100, 548, 222
card(qx, qy, qw, qh, "qc", 2, "Acquisition QC", "meu.qc", libs="MNE · mne-denoise")
QC_CHIP = "#8cc9cb"  # the QC accent, lightened (solid: no transparency)
chips = [
    "Flat · clipped · NaN", "Line & narrowband", "Outlying channels",
    "Bridged electrodes", "Impedance", "Blinks · heart · muscle",
    "Head movement · cHPI", "SQUID jumps", "Empty room",
    "Digitization", "Events · photodiode", "BIDS metadata",
]  # fmt: skip
cw, chh, gx, gy = 164, 26, 10, 8
for i, c in enumerate(chips):
    r, k = divmod(i, 3)
    chip(qx + 21 + k * (cw + gx), qy + 54 + r * (chh + gy), cw, chh, c, QC_CHIP)
ly = qy + qh - 13
x = qx + 21
for lvl, col in (("fail", "#c62828"), ("warn", "#e07b00"), ("ok", "#2e7d32"), ("info", "#78859b")):
    add(f'<circle cx="{x + 4}" cy="{ly - 4}" r="4" fill="{col}"/>')
    text(x + 12, ly, lvl, size=11.5, fill=MUTED, anchor="start")
    x += width(lvl, 11.5) + 28
text(x - 6, ly, "(info: measured, not judged)", size=11.5, fill=MUTED, anchor="start")

# --- 4 epoching ----------------------------------------------------------------------------------------
ex, ey, ew, eh = 960, 100, 600, 222
epo = THEME["epo"][0]
card(
    ex, ey, ew, eh, "epo", 4, "Epoching", "meu.epochs", libs="MNE · autoreject", libs_x=ex + ew / 2
)
bw, bh, by = 126, 56, ey + 100
gap = (ew - 40 - 4 * bw) / 3
xs = [round(ex + 20 + i * (bw + gap)) for i in range(4)]
step(xs[0], by, bw, bh, "Epoch each run", "events · fixed length", epo, size=13)
step(xs[1], by, bw, bh, "Align heads", "MEG · HeadAlign", epo, size=13)
step(xs[2], by, bw, bh, "autoreject", "local: repair or drop", epo, size=13)
step(xs[3], by, bw, bh, "Combine runs", "per session", epo, size=13)
mid = by + bh / 2
for a, b in ((0, 1), (1, 2), (2, 3)):
    arrow([(xs[a] + bw + 3, mid), (xs[b] - 4, mid)])
top = by - 20
arrow(
    [
        (xs[0] + bw / 2, by - 2),
        (xs[0] + bw / 2, top),
        (xs[2] + bw / 2, top),
        (xs[2] + bw / 2, by - 4),
    ]
)
label(xs[1] + bw / 2, top - 6, "EEG")

# --- data on disk ------------------------------------------------------------------------------------------
band_y, band_h = 352, 104
add(f'<rect x="20" y="{band_y}" width="{W - 40}" height="{band_h}" rx="20" fill="none" '
    'stroke="#c7ccd4" stroke-width="1.4" stroke-dasharray="9 7"/>')  # fmt: skip
add(f'<rect x="40" y="{band_y - 9}" width="196" height="18" fill="white"/>')
text(46, band_y + 4, "DATA ON DISK · BIDS", size=11.5, weight=700, fill=MUTED, anchor="start",
     extra=' letter-spacing="1.2"')  # fmt: skip
py0, ph0 = band_y + 22, 60
para(40, py0, 300, ph0, "BIDS raw data", sub="one file per run")
para(qx + 74, py0, 400, ph0, "QC report", sub="HTML · summary table · outlying recordings")
pre_x = 960
para(pre_x, py0, 290, ph0, "Preprocessed runs", sub="desc-preproc · JSON provenance")
para(1270, py0, 290, ph0, "Epochs", sub="per session · desc-epochs")

arrow([(190, 322), (190, py0 - 2)])  # read -> raw data
arrow([(326, py0 + 18), (358, py0 + 18), (358, 270), (qx - 3, 270)])  # raw data -> QC
arrow([(qx + qw / 2, qy + qh), (qx + qw / 2, py0 - 2)])  # QC -> report
arrow([(xs[0] + bw / 2, py0), (xs[0] + bw / 2, by + bh + 4)])  # preprocessed -> epoching
arrow([(xs[3] + bw / 2, by + bh), (xs[3] + bw / 2, py0 - 2)])  # epoching -> epochs

# --- 3 preprocessing -------------------------------------------------------------------------------------------
px, pyy, pw, phh = 110, 488, W - 130, 300
pre = THEME["pre"][0]
card(px, pyy, pw, phh, "pre", 3, "Preprocessing · one pipeline per run", "meu.steps",
     libs="MNE · PyPREP · mne-denoise · mne-icalabel")  # fmt: skip
cols = 7
sw, sh = 166, 54
x0 = px + 86
sgap = (px + pw - 20 - x0 - (cols * sw)) / (cols - 1)
cx = [round(x0 + i * (sw + sgap)) for i in range(cols)]
lanes = {
    "EEG": (560, [
        ("Bridged electrodes", "interpolate small groups"),
        ("High-pass", "0.1 Hz"),
        ("Line noise", "ZapLine-plus"),
        ("Bad channels", "PREP, robust reference"),
        ("Interpolate", "spherical splines"),
        ("Re-reference", "average"),
        ("ICA", "ICLabel ≥ 0.8"),
    ]),
    "MEG": (672, [
        ("Bad channels", "Maxwell"),
        ("Noise reduction", "SSS · CTF · KIT"),
        ("High-pass", "0.1 Hz"),
        ("Resample", "250 Hz"),
        ("Line noise", "ZapLine-plus"),
        None,
        ("ICA", "MEGnet ≥ 0.8"),
    ]),
}  # fmt: skip
for ly, steps in lanes.values():
    placed = [i for i, s in enumerate(steps) if s]
    for i in placed:
        step(cx[i], ly, sw, sh, *steps[i], accent=pre)
    for a, b in pairwise(placed):
        arrow([(cx[a] + sw + 3, ly + sh / 2), (cx[b] - 4, ly + sh / 2)])
label(cx[1] + sw / 2, 672 - 10, "by system (OPM: S.HFC, no preset yet)", anchor="middle")
text(px + 22, pyy + phh - 13, "Presets", size=11.5, weight=700, fill=MUTED, anchor="start")
text(px + 72, pyy + phh - 13, "eeg-erp · eeg-rest · meg-erp · meg-rest · had-meeg", size=11.5,
     fill=MUTED, anchor="start")  # fmt: skip
text(px + 400, pyy + phh - 13, "More steps", size=11.5, weight=700, fill=MUTED, anchor="start")
text(px + 470, pyy + phh - 13, "BadSegments · ASR · SNS · ByChannelType", size=11.5, fill=MUTED,
     anchor="start")  # fmt: skip

# raw data -> both lanes (a bus left of the card)
bus_x = 64
arrow([(bus_x, py0 + ph0), (bus_x, 672 + sh / 2)], head=False)
arrow([(bus_x, 560 + sh / 2), (cx[0] - 4, 560 + sh / 2)])
arrow([(bus_x, 672 + sh / 2), (cx[0] - 4, 672 + sh / 2)])
add(f'<circle cx="{bus_x}" cy="{560 + sh / 2}" r="3" fill="{LINE}"/>')
for lane, (ly, _) in lanes.items():
    pill(px + 20, ly + sh / 2 - 11, lane, pre, size=12, mono=False, fill="white")
# both ICA outputs -> preprocessed runs (a bus along the right of the card)
rx = cx[6] + sw + 10
arrow([(cx[6] + sw, 560 + sh / 2), (rx, 560 + sh / 2)], head=False)
arrow([(cx[6] + sw, 672 + sh / 2), (rx, 672 + sh / 2), (rx, 472), (pre_x + 145, 472),
       (pre_x + 145, py0 + ph0 + 3)])  # fmt: skip

# --- what every step records ------------------------------------------------------------------------------------
fy = 812
add(f'<rect x="20" y="{fy}" width="{W - 40}" height="46" rx="12" fill="#f7f8fa" stroke="#e1e5ea"/>')
text(42, fy + 28, "EVERY STEP AND CHECK", size=12, weight=700, fill=MUTED, anchor="start",
     extra=' letter-spacing="1.5"')  # fmt: skip
x = 238
for s in ("qc_ metrics", "plot() figures", "provenance in the JSON sidecar"):
    x += pill(x, fy + 12, s, LINE, size=12.5, mono=False) + 10
arrow([(x + 6, fy + 23), (x + 40, fy + 23)])
x += 52
x += pill(x, fy + 12, "meu.report", THEME["pre"][0]) + 12
text(x, fy + 28, "HTML report per run · dataset summary with outlying runs", size=13,
     anchor="start")  # fmt: skip

# --- legend ---------------------------------------------------------------------------------------------------------
ly = 882
x = 1010
add(f'<polygon points="{x + 9},{ly} {x + 46},{ly} {x + 37},{ly + 20} {x},{ly + 20}" '
    f'fill="{INPUT_FILL}" stroke="{INPUT_STROKE}" stroke-width="1.4"/>')  # fmt: skip
text(x + 54, ly + 15, "input", size=12.5, fill=MUTED, anchor="start")
x += 120
add(f'<polygon points="{x + 9},{ly} {x + 46},{ly} {x + 37},{ly + 20} {x},{ly + 20}" '
    f'fill="{DATA_FILL}" stroke="{DATA_STROKE}" stroke-width="1.4"/>')  # fmt: skip
text(x + 54, ly + 15, "data (BIDS)", size=12.5, fill=MUTED, anchor="start")
x += 150
add(f'<rect x="{x}" y="{ly}" width="46" height="20" rx="6" fill="{THEME["pre"][1]}" '
    f'stroke="{THEME["pre"][2]}" stroke-width="1.4"/>')  # fmt: skip
text(x + 54, ly + 15, "module", size=12.5, fill=MUTED, anchor="start")
x += 120
add(f'<rect x="{x}" y="{ly}" width="46" height="20" rx="5" fill="white" stroke="{STEP_STROKE}" '
    'stroke-width="1.4"/>')  # fmt: skip
text(x + 54, ly + 15, "step", size=12.5, fill=MUTED, anchor="start")

add("</svg>")
Path(sys.argv[1]).write_text("\n".join(out), encoding="utf-8")
