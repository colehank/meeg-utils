"""Draw the meeg-utils architecture diagram.

Usage::

    python resources/make_architecture.py resources/architecture.svg
    cp resources/architecture.svg docs/source/_static/architecture.svg

One line, left to right: read, check, preprocess (an EEG and an MEG lane),
epoch. Under each stage its module; in each lane, the steps of the
recommended presets.
"""

import sys
from html import escape
from pathlib import Path

W, H = 1400, 384
SANS = "'Helvetica Neue', Helvetica, Arial, Inter, sans-serif"
MONO = "'SFMono-Regular', Menlo, Consolas, 'DejaVu Sans Mono', monospace"
INK, MUTED, FAINT, LINE = "#1d1d1f", "#6e6e73", "#aeaeb2", "#c7c7cc"
ACCENT = "#0a66d6"

out: list[str] = []


def add(s: str) -> None:
    out.append(s)


def text(x, y, s, size=13, weight=400, fill=INK, anchor="middle", mono=False):
    add(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{MONO if mono else SANS}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">'
        f"{escape(s)}</text>"
    )


def node(x, y, title, module):
    """A stage on the line: a dot, the stage name above, the module below."""
    add(f'<circle cx="{x}" cy="{y}" r="7" fill="white" stroke="{INK}" stroke-width="2"/>')
    text(x, y - 22, title, size=20, weight=600)
    text(x, y + 30, module, size=12.5, fill=ACCENT, mono=True)


def lane(x0, x1, y, label, preset, steps):
    """A lane under the preprocessing stage: a thin line and its steps as plain words."""
    add(f'<path d="M{x0},{y} H{x1}" stroke="{LINE}" stroke-width="1.5"/>')
    text(x0 - 14, y + 1, label, size=12, weight=600, fill=MUTED, anchor="end")
    text(x0 - 14, y + 16, preset, size=10, fill=ACCENT, anchor="end", mono=True)
    n = len(steps)
    for i, (name, cls) in enumerate(steps):
        x = x0 + (x1 - x0) * (i + 0.5) / n
        add(f'<circle cx="{x:.1f}" cy="{y}" r="3.5" fill="{INK}"/>')
        text(x, y - 13, name, size=12, fill=INK)
        text(x, y + 20, cls, size=10, fill=MUTED, mono=True)


add(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
    'role="img" aria-labelledby="title desc">'
)
add('<title id="title">meeg-utils: read, check, preprocess, epoch</title>')
add(
    '<desc id="desc">Recordings are read with their BIDS sidecars (meu.io), checked by the '
    "acquisition QC (meu.qc), preprocessed run by run in an EEG or an MEG lane (meu.steps), "
    "and epoched and combined per session (meu.epochs).</desc>"
)
add(f'<rect width="{W}" height="{H}" fill="white"/>')

# the line
Y = 92
X = [120, 400, 760, 1280]
add(f'<path d="M{X[0]},{Y} H{X[-1]}" stroke="{INK}" stroke-width="2"/>')
node(X[0], Y, "Read", "meu.io")
node(X[1], Y, "Check", "meu.qc")
node(X[2], Y, "Preprocess", "meu.steps")
node(X[3], Y, "Epoch", "meu.epochs")

# what each stage gives
for x, s in zip(
    X,
    (
        "BIDS sidecars, any system",
        "fail · warn · ok · info",
        "one pipeline per run",
        "runs combined per session",
    ),
    strict=True,
):
    text(x, Y + 50, s, size=12, fill=MUTED)

# the two lanes
x0, x1 = 236, 1280
lane(x0, x1, 200, "EEG", "eeg-erp", [
    ("Bridged electrodes", "BridgedElectrodes"),
    ("High-pass", "Filter"),
    ("Line noise", "LineNoise"),
    ("Bad channels", "BadChannels"),
    ("Interpolate", "Interpolate"),
    ("Re-reference", "Reference"),
    ("ICA · ICLabel", "ICA"),
])  # fmt: skip
lane(x0, x1, 290, "MEG", "meg-erp", [
    ("Bad channels", "BadChannels"),
    ("SSS", "Maxwell"),
    ("High-pass", "Filter"),
    ("Resample", "Resample"),
    ("Line noise", "LineNoise"),
    ("ICA · MEGnet", "ICA"),
])  # fmt: skip
# the split from the preprocessing stage into the lanes
add(
    f'<path d="M{X[2]},{Y + 64} V{Y + 78} H{x0 - 60} V290 H{x0 - 44}" fill="none" '
    f'stroke="{LINE}" stroke-width="1.5"/>'
)
add(f'<path d="M{x0 - 60},200 H{x0 - 44}" stroke="{LINE}" stroke-width="1.5"/>')

text((x0 + x1) / 2, 336,
     "Neuromag shown. CTF: Interpolate + Reference(ctf_grade=3) instead of SSS · "
     "KIT: Regression instead of bad channels and SSS",
     size=11, fill=MUTED)  # fmt: skip
text(W / 2, H - 12, "Presets are ordinary pipelines: change, replace or remove any step.",
     size=12, fill=FAINT)  # fmt: skip

add("</svg>")
Path(sys.argv[1]).write_text("\n".join(out), encoding="utf-8")
