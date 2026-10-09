"""The ``meu`` command line.

Examples
--------
::

    meu qc /data/bids --out qc/                      # check every recording
    meu qc sub-01_task-rest_eeg.vhdr --out qc/       # one recording, with an HTML report
    meu run --preset had-meeg --datatype eeg --sources /data/bids --out /data/bids/derivatives/meu -j 8
    meu run pipeline.yaml --sources rec1_raw.fif rec2_raw.fif --out derivatives/
    meu preset                                        # list the presets
    meu preset meg-erp --option system=ctf -o pipeline.yaml
    meu report derivatives/sub-01/eeg/sub-01_task-rest_desc-preproc_eeg.fif
    meu report derivatives/ --desc preproc            # summary of every run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

LEVEL_ORDER = ("ok", "warn", "fail")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``meu`` command line.

    Parameters
    ----------
    argv : sequence of str | None
        Arguments (default: ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit status: 0 on success, 1 if a recording failed, 2 if QC findings
        reached ``--fail-on``.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    from .logger import setup_logging

    setup_logging("DEBUG" if args.verbose else "WARNING" if args.quiet else "INFO")
    return int(args.func(args))


def _parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="meu", description="MEG/EEG quality control and preprocessing (meeg-utils)."
    )
    parser.add_argument("--version", action="version", version=f"meeg-utils {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    common.add_argument("-q", "--quiet", action="store_true", help="only warnings and errors")
    common.add_argument("-j", "--jobs", type=int, default=1, help="recordings in parallel")
    sub = parser.add_subparsers(dest="command", metavar="command")

    qc = sub.add_parser(
        "qc",
        parents=[common],
        help="check the acquisition quality of recordings",
        description="Run the acquisition-quality checks (meeg_utils.qc) on recordings or on "
        "every MEG/EEG recording of a BIDS dataset.",
    )
    qc.add_argument("sources", nargs="+", help="recordings, or the root of a BIDS dataset")
    qc.add_argument("--out", type=Path, required=True, help="output directory")
    qc.add_argument(
        "--checks",
        help="comma-separated check names (default: all applicable), e.g. amplitude,bridging",
    )
    qc.add_argument("--no-report", action="store_true", help="skip the HTML reports")
    qc.add_argument(
        "--fail-on",
        choices=("warn", "fail"),
        help="exit with status 2 if any finding reaches this level",
    )
    qc.set_defaults(func=_qc)

    run = sub.add_parser(
        "run",
        parents=[common],
        help="preprocess recordings with a pipeline configuration or a preset",
        description="Fit a pipeline on each recording separately and save BIDS derivatives "
        "(meeg_utils.process).",
    )
    run.add_argument("config", nargs="?", type=Path, help="pipeline YAML (Pipeline.to_yaml)")
    run.add_argument("--preset", help="use a preset instead of a configuration file")
    run.add_argument("--datatype", choices=("meg", "eeg"), help="datatype for the preset / BIDS")
    run.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="preset option, repeatable; values are YAML, e.g. stage=epochs, "
        "event_id=[target,standard], epoch_duration=2 (MEG system: detected if not given)",
    )
    run.add_argument("--sources", nargs="+", required=True, help="recordings or a BIDS root")
    run.add_argument("--out", type=Path, required=True, help="derivatives root")
    run.add_argument("--desc", default="preproc", help="BIDS desc entity of the outputs")
    run.add_argument(
        "--set",
        action="append",
        default=[],
        dest="set_params",
        metavar="STEP__PARAM=VALUE",
        help="change a step parameter (pipe.set_params), repeatable; values are YAML, e.g. "
        "bridged__large_groups=bad, ica__threshold=0.9",
    )
    run.add_argument(
        "--epochs",
        nargs="?",
        const="preset",
        metavar="CONFIG",
        help="also epoch each run and combine the runs of each session (sources: one BIDS "
        "root); without CONFIG, the epochs stage of --preset",
    )
    existing = run.add_mutually_exclusive_group()
    existing.add_argument("--skip-existing", action="store_true", help="skip finished recordings")
    existing.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    run.set_defaults(func=_run)

    preset = sub.add_parser(
        "preset",
        parents=[common],
        help="list the presets, or write one as a pipeline configuration",
        description="Without a name, list the presets and their options. With a name, write "
        "the preset as YAML (to edit, then use with 'meu run CONFIG').",
    )
    preset.add_argument("name", nargs="?", help="preset name")
    preset.add_argument(
        "--option", action="append", default=[], metavar="KEY=VALUE",
        help="preset option, repeatable; values are YAML (e.g. system=neuromag, stage=epochs)",
    )  # fmt: skip
    preset.add_argument("--datatype", choices=("meg", "eeg"), help="datatype, for had-meeg")
    preset.add_argument("-o", "--out", type=Path, help="output file (default: print)")
    preset.set_defaults(func=_preset)

    report = sub.add_parser(
        "report",
        parents=[common],
        help="HTML report of a saved derivative, or summary of a derivatives folder",
        description="For a derivative file (.fif written by meu run): its pipeline, QC metrics, "
        "provenance and data. For a folder: the QC metrics of every derivative in a table "
        "(CSV) and a report with outlying files and the distribution of each metric.",
    )
    report.add_argument("path", type=Path, help="derivative .fif file, or derivatives folder")
    report.add_argument("--out", type=Path, help="output file (file) or directory (folder)")
    report.add_argument("--desc", help="folder: only derivatives with this desc (e.g. preproc)")
    report.set_defaults(func=_report)
    return parser


# ----------------------------------------------------------------------


def _qc(args: argparse.Namespace) -> int:
    from . import qc, report

    checks = _select_checks(args.checks)
    sources = _expand(args.sources)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    result = qc.inspect_dataset(sources, checks, n_jobs=args.jobs)

    for source, rep in result.reports.items():
        stem = _stem(source)
        _write_json(out / f"{stem}_qc.json", rep.to_dict())
        if not args.no_report:
            html = report.build(qc=rep, title=f"QC: {Path(source).name}")
            html.save(out / f"{stem}_qc.html", open_browser=False, overwrite=True, verbose=False)
        print(rep)
    summary = result.to_csv(out / "qc_summary.csv")
    if len(result.reports) > 1:
        import matplotlib.pyplot as plt

        result.plot()["levels"].savefig(out / "qc_overview.png", dpi=120, bbox_inches="tight")
        plt.close("all")
    for outlier in result.outliers:
        print(
            f"outlier: {Path(outlier['source']).name}: {outlier['metric']} = "
            f"{outlier['value']:g} (z = {outlier['z']:g})"
        )
    for source, error in result.errors.items():
        print(f"error: {source}: {error}", file=sys.stderr)
    print(f"\n{result!r}\nSummary: {summary}")

    if result.errors:
        return 1
    if args.fail_on is not None:
        worst = max((r.level for r in result.reports.values()), key=LEVEL_ORDER.index, default="ok")
        if LEVEL_ORDER.index(worst) >= LEVEL_ORDER.index(args.fail_on):
            return 2
    return 0


def _run(args: argparse.Namespace) -> int:
    from . import process
    from .core import Pipeline

    if (args.config is None) == (args.preset is None):
        raise SystemExit("meu run: give either a configuration file or --preset, not both.")
    sources = _expand(args.sources, datatype=args.datatype)
    options: dict[str, Any] = {}
    if args.preset is not None:
        options = _preset_options(args, sources)
        if args.epochs is not None:
            options.pop("stage", None)
        pipeline = Pipeline.preset(args.preset, **options)
    else:
        if args.option:
            raise SystemExit("meu run: --option only applies to --preset.")
        pipeline = Pipeline.from_yaml(args.config)
    if args.epochs is not None:
        return _run_dataset(args, pipeline, options)
    _apply_set_params(args.set_params, [pipeline])
    records = process(
        pipeline,
        sources,
        args.out,
        desc=args.desc,
        n_jobs=args.jobs,
        on_error="warn",
        skip_existing=args.skip_existing,
        overwrite=args.overwrite,
    )
    for record in records:
        line = f"{record['status']:<7} {record['source']}"
        line += f" -> {record['output']}" if record["output"] else f": {record['error']}"
        print(line)
    n_failed = sum(r["status"] == "failed" for r in records)
    print(f"\n{len(records) - n_failed}/{len(records)} recordings done, {n_failed} failed.")
    return 1 if n_failed else 0


def _preset(args: argparse.Namespace) -> int:
    import yaml

    from . import presets
    from .core import Pipeline

    if args.name is None:
        for name, summary in presets.available().items():
            print(f"{name:<10} {summary}\n{'':<10} options: {', '.join(presets.options(name))}")
        return 0
    if args.name not in presets.available():
        raise SystemExit(f"Unknown preset {args.name!r}; available: {sorted(presets.available())}")
    args.preset = args.name
    options = _preset_options(args, [])
    if "system" in presets.options(args.name) and "system" not in options:
        stage = options.get("stage", "preprocessing")
        if stage == "preprocessing":
            raise SystemExit(
                f"meu preset {args.name}: give --option system=... (neuromag, ctf, kit)."
            )
    text = yaml.safe_dump(
        Pipeline.preset(args.name, **options).to_dict(), sort_keys=False, allow_unicode=True
    )
    if args.out is None:
        print(text, end="")
    else:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


def _report(args: argparse.Namespace) -> int:
    from . import report

    path: Path = args.path
    if path.is_dir():
        out = args.out or path
        out.mkdir(parents=True, exist_ok=True)
        summary = report.summarize(path, desc=args.desc)
        csv_path = summary.to_csv(out / "derivatives_summary.csv")
        html_path = out / "derivatives_summary.html"
        summary.report().save(html_path, open_browser=False, overwrite=True, verbose=False)
        for o in summary.outliers:
            print(f"outlier: {o['source']}: {o['metric']} = {o['value']:g} (z = {o['z']:g})")
        print(f"{len(summary.table)} derivatives; {len(summary.outliers)} outlying values.")
        print(f"Wrote {csv_path} and {html_path}")
        return 0
    if not path.exists():
        raise SystemExit(f"{path} does not exist.")
    out = args.out or path.with_suffix(".html")
    report.from_derivative(path).save(out, open_browser=False, overwrite=True, verbose=False)
    print(f"Wrote {out}")
    return 0


def _run_dataset(args: argparse.Namespace, pipeline: Any, options: dict[str, Any]) -> int:
    """meu run --epochs: preprocess, epoch and combine a BIDS dataset (meu.process_dataset)."""
    from .core import Pipeline
    from .dataset import Dataset, process_dataset

    if len(args.sources) != 1 or not Path(args.sources[0]).is_dir():
        raise SystemExit("meu run --epochs: --sources must be the root of one BIDS dataset.")
    if args.epochs == "preset":
        if args.preset is None:
            raise SystemExit("meu run --epochs: give a configuration file, or use --preset.")
        epochs = Pipeline.preset(args.preset, stage="epochs", **options)
    else:
        epochs = Pipeline.from_yaml(args.epochs)
    _apply_set_params(args.set_params, [pipeline, epochs])
    dataset = Dataset(args.sources[0], datatype=args.datatype)
    result = process_dataset(
        dataset,
        args.out,
        preprocessing=pipeline,
        epochs=epochs,
        desc=args.desc,
        n_jobs=args.jobs,
        skip_existing=not args.overwrite,
        on_error="warn",
    )
    log = result.to_csv(Path(args.out) / "meu_batch.csv")
    for record in result.failed:
        print(f"failed  [{record['stage']}] {record['source']}: {record['error']}")
    print(f"\n{dataset!r}\n{result!r}\nLog: {log}")
    return 1 if result.failed else 0


# ----------------------------------------------------------------------


def _apply_set_params(items: list[str], pipelines: list[Any]) -> None:
    """Apply --set STEP__PARAM=VALUE to the pipeline that has the parameter."""
    import yaml

    for item in items:
        key, sep, value = item.partition("=")
        if not sep or "__" not in key:
            raise SystemExit(f"meu run: --set expects STEP__PARAM=VALUE, got {item!r}.")
        target = next((p for p in pipelines if key in p.get_params(deep=True)), None)
        if target is None:
            raise SystemExit(f"meu run: no parameter {key!r} in the pipeline(s).")
        target.set_params(**{key: yaml.safe_load(value)})


def _preset_options(args: argparse.Namespace, sources: list[Any]) -> dict[str, Any]:
    """Preset options from --option and --datatype; the MEG system from the first recording."""
    import yaml

    from . import presets

    accepted = presets.options(args.preset)
    options: dict[str, Any] = {}
    for item in args.option:
        key, sep, value = item.partition("=")
        if not sep or key not in accepted:
            raise SystemExit(
                f"meu run: invalid --option {item!r}; {args.preset} accepts {accepted} as KEY=VALUE."
            )
        options[key] = yaml.safe_load(value)
    if args.datatype and "datatype" in accepted:
        options.setdefault("datatype", args.datatype)
    if "system" in accepted and "system" not in options and sources:
        from .io import detect_system, read

        options["system"] = detect_system(read(sources[0], preload=False))
    return options


def _select_checks(names: str | None) -> list[Any] | None:
    from . import qc

    if names is None:
        return None
    available = {cls.name: cls for cls in qc.DEFAULT_CHECKS}
    selected = [n.strip() for n in names.split(",") if n.strip()]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise SystemExit(f"Unknown checks {unknown}; available: {sorted(available)}")
    return [available[n]() for n in selected]


def _expand(sources: Sequence[str], datatype: str | None = None) -> list[Any]:
    """Recordings from the command line; a directory is a BIDS dataset root."""
    from .qc import find_recordings

    items: list[Any] = []
    for source in sources:
        path = Path(source)
        if path.is_dir() and path.suffix != ".ds":  # CTF recordings are directories
            found = find_recordings(path)
            if datatype is not None:
                found = [p for p in found if p.datatype == datatype]
            if not found:
                raise SystemExit(f"No MEG/EEG recordings found in {path}.")
            items.extend(found)
        else:
            items.append(path)
    return items


def _stem(source: str) -> str:
    name = Path(source).name
    for suffix in (".fif", ".ds", ".vhdr", ".edf", ".bdf", ".set", ".con", ".sqd", ".cnt"):
        name = name.removesuffix(suffix)
    return name


def _write_json(fname: Path, data: dict[str, Any]) -> None:
    def default(value: Any) -> Any:
        if hasattr(value, "tolist"):  # numpy arrays and scalars
            return value.tolist()
        return str(value)

    fname.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=default), encoding="utf-8"
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
