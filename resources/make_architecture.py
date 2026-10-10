"""Draw the meeg-utils architecture diagram.

Usage::

    python resources/make_architecture.py resources/architecture.svg
    cp resources/architecture.svg docs/source/_static/architecture.svg

The style follows the HAD-MEEG pipeline figure (preprocessing_pipeline.png):
module groups in light blue, steps as boxes, data as green parallelograms.
"""

import sys
from html import escape
from pathlib import Path

W, H = 1600, 900
FONT = "'Helvetica Neue', Helvetica, Arial, Inter, sans-serif"
MONO = "'SFMono-Regular', Menlo, Consolas, 'DejaVu Sans Mono', monospace"

GROUP_FILL, GROUP_STROKE = "#eef4fb", "#c6d9f0"
STEP_STROKE = "#1f3d6b"
DATA_FILL, DATA_STROKE = "#e4f2d7", "#a9d18e"
INPUT_FILL, INPUT_STROKE = "#d6d6d6", "#bdbdbd"
TEXT, MUTED, ACCENT = "#1a1a1a", "#7a7a7a", "#1f3d6b"

out: list[str] = []


def add(s: str) -> None:
    out.append(s)


def text(x, y, s, size=15, weight="normal", fill=TEXT, anchor="middle", family=FONT, style=""):
    add(
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}"{style}>{escape(s)}</text>'
    )


def lines(x, y, rows, size=15, gap=None, **kw):
    """Centered multi-line text; rows of (string, size, weight, fill)."""
    gap = gap or size * 1.3
    total = sum((r[1] if len(r) > 1 else size) * 1.3 for r in rows)
    cy = y - total / 2 + (rows[0][1] if len(rows[0]) > 1 else size) * 0.95
    for r in rows:
        s, sz, wt, fl = (list(r) + [size, "normal", TEXT][len(r) - 1 :])[:4]
        text(x, cy, s, size=sz, weight=wt, fill=fl, **kw)
        cy += sz * 1.3


def group(x, y, w, h, title, module, libs=None, tab_x=18, libs_x=None):
    add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" fill="{GROUP_FILL}" '
        f'stroke="{GROUP_STROKE}" stroke-width="4"/>'
    )
    title_w = len(title) * 12.6
    module_w = len(module) * 8.6
    tw = title_w + module_w + 46
    add(
        f'<rect x="{x + tab_x}" y="{y - 18}" width="{tw}" height="38" rx="10" fill="{GROUP_FILL}" '
        f'stroke="{GROUP_STROKE}" stroke-width="4"/>'
    )
    text(x + tab_x + 16, y + 9, title, size=23, anchor="start")
    text(
        x + tab_x + 16 + title_w + 14,
        y + 8,
        module,
        size=14,
        fill=ACCENT,
        anchor="start",
        family=MONO,
    )
    if libs:
        lx = x + w - 18 if libs_x is None else libs_x
        text(
            lx,
            y + h - 12,
            libs,
            size=12.5,
            fill=MUTED,
            anchor="end" if libs_x is None else "middle",
        )


def step(x, y, w, h, title, sub=None, size=15):
    add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white" stroke="{STEP_STROKE}" '
        f'stroke-width="2.2"/>'
    )
    rows = [(title, size, "normal", TEXT)]
    if sub:
        rows += [(s, 12.5, "normal", MUTED) for s in ([sub] if isinstance(sub, str) else sub)]
    lines(x + w / 2, y + h / 2, rows)


def chip(x, y, w, h, title):
    add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="white" stroke="{STEP_STROKE}" '
        f'stroke-width="1.4"/>'
    )
    text(x + w / 2, y + h / 2 + 5, title, size=13.5)


def para(x, y, w, h, title, sub=None, fill=DATA_FILL, stroke=DATA_STROKE, skew=18, size=19):
    pts = f"{x + skew},{y} {x + w},{y} {x + w - skew},{y + h} {x},{y + h}"
    add(f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
    rows = [(title, size, "normal", TEXT)]
    if sub:
        rows.append((sub, 13, "normal", TEXT))
    lines(x + w / 2, y + h / 2, rows)


def arrow(points, label=None, label_at=None, width=2.4, dashed=False):
    d = "M " + " L ".join(f"{px},{py}" for px, py in points)
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    add(
        f'<path d="{d}" fill="none" stroke="#111" stroke-width="{width}"{dash} '
        f'marker-end="url(#head)"/>'
    )
    if label:
        lx, ly, rot = label_at
        add(
            f'<text x="{lx}" y="{ly}" font-family="{FONT}" font-size="14" fill="{TEXT}" '
            f'text-anchor="middle" transform="rotate({rot} {lx} {ly})">{label}</text>'
        )


add(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
    f'role="img" aria-labelledby="title desc">'
)
add('<title id="title">meeg-utils architecture</title>')
add(
    '<desc id="desc">Raw recordings are read with their BIDS sidecars, checked by the '
    "acquisition QC, preprocessed run by run, epoched and combined per session; every step "
    "records QC metrics, figures and provenance for the reports.</desc>"
)
add(
    '<defs><marker id="head" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" '
    'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#111"/>'
    "</marker></defs>"
)
add(f'<rect width="{W}" height="{H}" fill="white"/>')

# --- top bar: how to run it -------------------------------------------------------------
add(
    f'<rect x="20" y="16" width="{W - 40}" height="50" rx="12" fill="#f4f4f4" stroke="#d0d0d0" stroke-width="2"/>'
)
text(40, 47, "Run it", size=17, weight="bold", anchor="start")
items = [
    ("Python", "meu.Pipeline · steps · presets"),
    ("Dataset", "meu.Dataset · meu.process_dataset"),
    ("Command line", "meu qc | run | preset | report"),
]
x = 120
for label, code in items:
    text(x, 47, label, size=15, anchor="start")
    text(x + len(label) * 8.6 + 10, 47, code, size=14, fill=ACCENT, anchor="start", family=MONO)
    x += len(label) * 8.6 + 10 + len(code) * 8.5 + 60

# --- column 1: input and reading ---------------------------------------------------------
para(
    40,
    108,
    300,
    62,
    "Raw recordings",
    sub="any format MNE reads",
    fill=INPUT_FILL,
    stroke=INPUT_STROKE,
)
group(40, 222, 300, 102, "Read", "meu.io")
arrow([(290, 170), (290, 220)])
lines(
    190,
    280,
    [
        ("BIDS sidecars · system detection", 13, "normal", TEXT),
        ("Neuromag · CTF · KIT · OPM · EEG", 13, "normal", MUTED),
    ],
)

# --- acquisition QC -----------------------------------------------------------------------
gx, gy, gw, gh = 380, 116, 530, 208
group(gx, gy, gw, gh, "Acquisition QC", "meu.qc")
chips = [
    "Flat · clipped · NaN",
    "Line & narrowband",
    "Outlying channels",
    "Bridged electrodes",
    "Impedance",
    "Blinks · heart · muscle",
    "Head movement · cHPI",
    "SQUID jumps",
    "Empty room",
    "Digitization",
    "Events · photodiode",
    "BIDS metadata",
]
cw, ch_, gap = 158, 30, 9
for i, c in enumerate(chips):
    r, k = divmod(i, 3)
    chip(gx + 20 + k * (cw + gap), gy + 34 + r * (ch_ + 8), cw, ch_, c)
text(
    gx + 20,
    gy + gh - 12,
    "fail · warn · ok · info (measured only)",
    size=12.5,
    fill=MUTED,
    anchor="start",
)
text(gx + gw - 18, gy + gh - 12, "MNE, mne-denoise", size=12.5, fill=MUTED, anchor="end")

# --- epoching -------------------------------------------------------------------------------
ex, ey, ew, eh = 950, 116, 610, 208
group(ex, ey, ew, eh, "Epoching", "meu.epochs", libs="MNE, autoreject", libs_x=ex + 300)
bw, bh, by = 122, 64, ey + 84
xs = [ex + 22 + i * (bw + 33) for i in range(4)]
step(xs[0], by, bw, bh, "Epoch each run", sub=["events or", "fixed length"], size=14)
step(xs[1], by, bw, bh, "Align head", sub=["MEG only", "(HeadAlign)"], size=14)
step(xs[2], by, bw, bh, "autoreject", sub=["local: repair", "or drop"], size=14)
step(xs[3], by, bw, bh, "Combine runs", sub=["per session"], size=14)
arrow([(xs[0] + bw, by + bh / 2), (xs[1] - 2, by + bh / 2)])
arrow([(xs[1] + bw, by + bh / 2), (xs[2] - 2, by + bh / 2)])
arrow([(xs[2] + bw, by + bh / 2), (xs[3] - 2, by + bh / 2)])
arrow(
    [
        (xs[0] + bw / 2, by),
        (xs[0] + bw / 2, by - 22),
        (xs[2] + bw / 2, by - 22),
        (xs[2] + bw / 2, by - 2),
    ],
    "EEG",
    (xs[1] + bw / 2, by - 28, 0),
)

# --- shared data band -------------------------------------------------------------------------
add(
    f'<rect x="20" y="360" width="{W - 40}" height="108" rx="26" fill="none" stroke="#b5b5b5" '
    'stroke-width="3" stroke-dasharray="22 14"/>'
)
para(40, 380, 300, 68, "BIDS raw data", sub="per run")
para(470, 380, 330, 68, "QC report", sub="HTML · table · outlying recordings")
para(950, 380, 300, 68, "Preprocessed data", sub="per run · desc-preproc")
para(1270, 380, 290, 68, "Epoched data", sub="per session · desc-epochs")

# arrows between the band and the groups
arrow([(190, 324), (190, 378)])  # read -> BIDS raw data
arrow([(330, 395), (360, 395), (360, 300), (378, 300)])  # raw data -> QC
arrow([(635, 324), (635, 378)])  # QC -> report
arrow([(xs[0] + bw / 2, 380), (xs[0] + bw / 2, by + bh + 2)])  # preprocessed -> epoching
arrow([(xs[3] + bw / 2, by + bh), (xs[3] + bw / 2, 378)])  # epoching -> epochs

# --- preprocessing ------------------------------------------------------------------------------
px, py, pw, ph = 20, 506, W - 40, 222
group(
    px,
    py,
    pw,
    ph,
    "Preprocessing, one pipeline per run",
    "meu.steps",
    libs="MNE, PyPREP, mne-denoise, mne-icalabel",
    tab_x=360,
)
row = py + 104
b1 = (50, row - 32, 280, 64)
step(*b1, "Bad channels", sub="PREP (EEG) · Maxwell (MEG)")
arrow([(190, 448), (190, b1[1] - 2)])  # raw data -> bad channels
meg = (420, row - 82, 330, 64)
eeg = (420, row + 18, 330, 64)
step(*meg, "System noise reduction", sub="SSS · CTF gradients · ref. regression · HFC")
step(*eeg, "Bridges · interpolation", sub="robust average reference")
arrow([(b1[0] + b1[2], row), (meg[0] - 2, meg[1] + 32)], "MEG", (375, row - 32, -29))
arrow([(b1[0] + b1[2], row), (eeg[0] - 2, eeg[1] + 32)], "EEG", (375, row + 40, 29))
f = (850, row - 32, 210, 64)
step(*f, "Filter · resample", sub="0.1 Hz high-pass")
arrow([(meg[0] + meg[2], meg[1] + 32), (f[0] - 2, row)])
arrow([(eeg[0] + eeg[2], eeg[1] + 32), (f[0] - 2, row)])
ln = (1110, row - 32, 190, 64)
step(*ln, "Line noise", sub="ZapLine-plus")
arrow([(f[0] + f[2], row), (ln[0] - 2, row)])
ica = (1350, row - 32, 200, 64)
step(*ica, "ICA", sub="ICLabel (EEG) · MEGnet (MEG)")
arrow([(ln[0] + ln[2], row), (ica[0] - 2, row)])
arrow([(xs[0] + bw / 2, 506), (xs[0] + bw / 2, 450)])  # preprocessing -> preprocessed data
text(
    50,
    py + ph - 12,
    "also: BadSegments · ASR · SNS · ByChannelType (separate steps per channel type)",
    size=12.5,
    fill=MUTED,
    anchor="start",
)

# --- bottom bar: what every step records ------------------------------------------------------------
add(
    f'<rect x="20" y="752" width="{W - 40}" height="50" rx="12" fill="#f4f4f4" stroke="#d0d0d0" stroke-width="2"/>'
)
text(40, 783, "Every step and check", size=17, weight="bold", anchor="start")
text(
    232,
    783,
    "qc_ metrics · plot() figures · provenance in the JSON sidecar  →",
    size=15,
    anchor="start",
)
text(712, 783, "meu.report", size=14, fill=ACCENT, anchor="start", family=MONO)
text(812, 783, "HTML report per run · dataset summary with outlying runs", size=15, anchor="start")

# --- legend ----------------------------------------------------------------------------------------
ly = 840
lx = 860
add(
    f'<polygon points="{lx + 12},{ly} {lx + 70},{ly} {lx + 58},{ly + 26} {lx},{ly + 26}" '
    f'fill="{INPUT_FILL}" stroke="{INPUT_STROKE}" stroke-width="2"/>'
)
text(lx + 82, ly + 19, "Input data", size=15, anchor="start")
lx += 200
add(
    f'<rect x="{lx}" y="{ly}" width="62" height="26" rx="8" fill="{GROUP_FILL}" stroke="{GROUP_STROKE}" stroke-width="3"/>'
)
text(lx + 74, ly + 19, "Module", size=15, anchor="start")
lx += 170
add(
    f'<rect x="{lx}" y="{ly + 2}" width="56" height="22" fill="white" stroke="{STEP_STROKE}" stroke-width="2"/>'
)
text(lx + 68, ly + 19, "Step", size=15, anchor="start")
lx += 140
add(
    f'<rect x="{lx}" y="{ly - 4}" width="74" height="34" rx="8" fill="none" stroke="#b5b5b5" stroke-width="2" stroke-dasharray="8 6"/>'
)
add(
    f'<polygon points="{lx + 18},{ly + 3} {lx + 62},{ly + 3} {lx + 56},{ly + 23} {lx + 12},{ly + 23}" '
    f'fill="{DATA_FILL}" stroke="{DATA_STROKE}" stroke-width="2"/>'
)
text(lx + 86, ly + 19, "Data (BIDS)", size=15, anchor="start")

add("</svg>")
Path(sys.argv[1]).write_text("\n".join(out), encoding="utf-8")
