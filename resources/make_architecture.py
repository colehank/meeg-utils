"""Draw the meeg-utils module map.

Usage::

    python resources/make_architecture.py resources/architecture.svg
    cp resources/architecture.svg docs/source/_static/architecture.svg

The public modules as tiles (name, what it is, its main members), on the
libraries it builds on. No order or flow is implied.
"""

import sys
from html import escape
from pathlib import Path

W, H = 1200, 520
SANS = "'Helvetica Neue', Helvetica, Arial, Inter, sans-serif"
MONO = "'SFMono-Regular', Menlo, Consolas, 'DejaVu Sans Mono', monospace"
INK, MUTED, TILE, ACCENT = "#1d1d1f", "#6e6e73", "#f5f5f7", "#0a66d6"

TILES = [
    ("meu.io", "Read and write", "read · detect_system · save_derivative"),
    ("meu.qc", "Acquisition QC", "inspect · inspect_dataset · 16 checks"),
    ("meu.steps", "Steps", "Filter · LineNoise · BadChannels · Maxwell · ICA · …"),
    ("meu.Pipeline", "Pipelines", "fit · transform · set_params · to_yaml"),
    ("meu.presets", "Presets", "eeg-erp · eeg-rest · meg-erp · meg-rest · had-meeg"),
    ("meu.epochs", "Combining runs", "combine: channels · events · head position"),
    ("meu.Dataset", "BIDS datasets", "Dataset · process_dataset · process"),
    ("meu.report", "Reports", "build · from_derivative · summarize"),
    ("meu", "Command line", "meu qc · run · preset · report"),
]
BASE = "MNE-Python · MNE-BIDS · mne-denoise · PyPREP · mne-icalabel · autoreject · scikit-learn"

out: list[str] = []


def add(s: str) -> None:
    out.append(s)


def text(x, y, s, size, weight=400, fill=INK, anchor="start", mono=False):
    add(
        f'<text x="{x}" y="{y}" font-family="{MONO if mono else SANS}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>'
    )


add(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
    'role="img" aria-labelledby="title desc">'
)
add('<title id="title">meeg-utils modules</title>')
add(
    '<desc id="desc">The modules of meeg-utils: io, qc, steps, Pipeline, presets, epochs, '
    "Dataset, report and the meu command, built on MNE-Python and its ecosystem.</desc>"
)
add(f'<rect width="{W}" height="{H}" fill="white"/>')

M, G, COLS = 24, 16, 3
TW = (W - 2 * M - (COLS - 1) * G) // COLS
TH = 128
for i, (module, title, members) in enumerate(TILES):
    r, c = divmod(i, COLS)
    x, y = M + c * (TW + G), M + r * (TH + G)
    add(f'<rect x="{x}" y="{y}" width="{TW}" height="{TH}" rx="16" fill="{TILE}"/>')
    text(x + 24, y + 36, module, 14, fill=ACCENT, mono=True)
    text(x + 24, y + 70, title, 21, weight=600)
    text(x + 24, y + 100, members, 13, fill=MUTED)

y = M + 3 * TH + 2 * G + 24
add(f'<path d="M{M},{y} H{W - M}" stroke="#d2d2d7" stroke-width="1"/>')
text(W / 2, y + 30, "Built on " + BASE, 13, fill=MUTED, anchor="middle")

add("</svg>")
Path(sys.argv[1]).write_text("\n".join(out), encoding="utf-8")
