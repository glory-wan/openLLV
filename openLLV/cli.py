"""Command-line interface for openLLV."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import openLLV as llv


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Run the openLLV command-line interface and return an exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    _print_result(result)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the root CLI parser and all supported subcommands."""
    parser = argparse.ArgumentParser(
        prog="openllv",
        description="Command-line tools for openLLV.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_predict_parser(subparsers)
    _add_train_parser(subparsers)
    _add_evaluate_parser(subparsers, name="evaluate")
    _add_evaluate_parser(subparsers, name="eval")
    _add_imwrite_parser(subparsers)
    _add_list_parser(subparsers)

    return parser


def _add_predict_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the prediction subcommand."""
    parser = subparsers.add_parser(
        "predict",
        help="Process an image or folder with a model or algorithm.",
    )
    parser.add_argument(
        "target",
        help="Model name, checkpoint path, or traditional algorithm name.",
    )
    parser.add_argument("source", help="Input image path or image directory.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path or directory.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        help="Predictor backend: auto, deep, or traditional.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device used for deep-learning prediction.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default predictor output directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch-size metadata used by the deep predictor.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Data-loader worker metadata used by the deep predictor.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar for directory prediction.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save a single-image prediction.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output filename when saving to a directory.",
    )
    parser.add_argument(
        "--output-ext",
        default=None,
        help="Output extension override.",
    )
    parser.add_argument(
        "--kwargs",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Additional keyword arguments forwarded to openLLV.predict().",
    )
    parser.set_defaults(func=_cmd_predict)


def _add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the training subcommand."""
    parser = subparsers.add_parser("train", help="Train a model.")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Built-in configuration name or YAML configuration path.",
    )
    parser.add_argument(
        "--kwargs",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Additional keyword arguments forwarded to openLLV.train().",
    )
    parser.set_defaults(func=_cmd_train)


def _add_evaluate_parser(
    subparsers: argparse._SubParsersAction,
    name: str = "evaluate",
) -> None:
    """Add an evaluation subcommand under ``name``."""
    parser = subparsers.add_parser(
        name,
        help="Evaluate a directory of processed images.",
    )
    parser.add_argument(
        "--en",
        "--en-img-dir",
        dest="en_img_dir",
        required=True,
        help="Directory containing processed images.",
    )
    parser.add_argument(
        "--ref",
        "--ref-img-dir",
        dest="ref_img_dir",
        default=None,
        help="Optional reference-image directory.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Metric names.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Path used to save evaluation results.",
    )
    parser.add_argument(
        "--return-evaluator",
        action="store_true",
        help="Return the evaluator object instead of only its results.",
    )
    parser.add_argument(
        "--kwargs",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Additional keyword arguments forwarded to openLLV.evaluate().",
    )
    parser.set_defaults(func=_cmd_evaluate)


def _add_imwrite_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the image-writing subcommand."""
    parser = subparsers.add_parser(
        "imwrite",
        help="Write or convert an image.",
    )
    parser.add_argument(
        "image",
        help="Input image path or another supported string source.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path or directory.",
    )
    parser.add_argument(
        "--save-format",
        default=None,
        help="Output image format override.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output filename when saving to a directory.",
    )
    parser.add_argument(
        "--kwargs",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Additional keyword arguments forwarded to openLLV.imwrite().",
    )
    parser.set_defaults(func=_cmd_imwrite)


def _add_list_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add the component-listing subcommand."""
    parser = subparsers.add_parser(
        "list",
        help="List available openLLV components.",
    )
    parser.set_defaults(func=_cmd_list_available)


def _cmd_predict(args: argparse.Namespace) -> Any:
    """Run prediction from parsed CLI arguments."""
    kwargs = _parse_key_value_args(args.kwargs)
    kwargs.update(
        _drop_none(
            {
                "backend": args.backend,
                "device": args.device,
                "output_dir": args.output_dir,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "progress_bar": not args.no_progress,
                "save": not args.no_save,
                "output_name": args.output_name,
                "output_ext": args.output_ext,
            }
        )
    )
    return llv.predict(
        args.target,
        args.source,
        output=args.output,
        **kwargs,
    )


def _cmd_train(args: argparse.Namespace) -> Any:
    """Run training from parsed CLI arguments."""
    kwargs = _parse_key_value_args(args.kwargs)
    return llv.train(args.config, **kwargs)


def _cmd_evaluate(args: argparse.Namespace) -> Any:
    """Run evaluation from parsed CLI arguments."""
    kwargs = _parse_key_value_args(args.kwargs)
    return llv.evaluate(
        en=args.en_img_dir,
        ref=args.ref_img_dir,
        metrics=args.metrics,
        save_path=args.save_path,
        return_evaluator=args.return_evaluator,
        **kwargs,
    )


def _cmd_imwrite(args: argparse.Namespace) -> Any:
    """Run image writing from parsed CLI arguments."""
    kwargs = _parse_key_value_args(args.kwargs)
    return llv.imwrite(
        args.image,
        output=args.output,
        save_format=args.save_format,
        output_name=args.output_name,
        **kwargs,
    )


def _cmd_list_available(args: argparse.Namespace) -> Any:
    """Return the detailed component listing."""
    return llv.list_available()


def _format_component_table(
    category: str,
    rows: Iterable[Dict[str, Any]],
) -> str:
    """Format one component category as an ASCII table."""
    table_rows = []
    for row in rows:
        aliases = row.get("aliases", [])
        aliases_text = (
            ", ".join(str(alias) for alias in aliases)
            if aliases
            else "-"
        )
        table_rows.append((str(row.get("name", "")), aliases_text))

    name_width = max([len("name"), *(len(name) for name, _ in table_rows)])
    aliases_width = max(
        [len("aliases"), *(len(aliases) for _, aliases in table_rows)]
    )
    separator = f"+-{'-' * name_width}-+-{'-' * aliases_width}-+"

    lines = [
        category,
        separator,
        f"| {'name'.ljust(name_width)} | {'aliases'.ljust(aliases_width)} |",
        separator,
    ]
    lines.extend(
        f"| {name.ljust(name_width)} | {aliases.ljust(aliases_width)} |"
        for name, aliases in table_rows
    )
    lines.append(separator)
    return "\n".join(lines)


def _is_component_listing(value: Any) -> bool:
    """Return whether ``value`` is a detailed component listing."""
    if not isinstance(value, dict) or not value:
        return False

    for rows in value.values():
        if not isinstance(rows, list):
            return False
        if any(
            not isinstance(row, dict)
            or not {"name", "aliases"}.issubset(row)
            for row in rows
        ):
            return False
    return True


def _parse_key_value_args(items: Iterable[str]) -> Dict[str, Any]:
    """Parse repeated ``KEY=VALUE`` arguments into a dictionary."""
    parsed: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE format, got {item!r}.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Keyword name must not be empty: {item!r}.")
        parsed[key] = _parse_value(value)
    return parsed


def _parse_value(value: str) -> Any:
    """Parse a CLI string as a common Python literal when possible."""
    value = value.strip()
    lowered = value.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _drop_none(values: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``values`` without keys whose value is ``None``."""
    return {key: value for key, value in values.items() if value is not None}


def _print_result(result: Any) -> None:
    """Print a command result in a compact, deterministic representation."""
    if result is None:
        return

    if _is_component_listing(result):
        print("Available components:")
        for category, rows in result.items():
            print()
            print(_format_component_table(category, rows))
        return

    if _is_path_like(result):
        print(result)
        return

    if _is_single_prediction_result(result):
        enhanced, saved_path = result
        if saved_path is not None:
            print(saved_path)
            return
        result = {
            "image": _json_safe(enhanced),
            "saved_path": None,
        }

    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))


def _is_single_prediction_result(value: Any) -> bool:
    """Check for the ``(processed_image, saved_path)`` result contract."""
    if not isinstance(value, tuple) or len(value) != 2:
        return False

    enhanced, saved_path = value
    return _is_image_like(enhanced) and (
        saved_path is None or _is_path_like(saved_path)
    )


def _is_image_like(value: Any) -> bool:
    """Return whether ``value`` resembles an in-memory image."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return True
    return (
        hasattr(value, "__class__")
        and value.__class__.__module__.startswith("PIL")
    )


def _is_path_like(value: Any) -> bool:
    """Return whether ``value`` is a path string or ``Path``."""
    return isinstance(value, (str, Path))


def _json_safe(value: Any) -> Any:
    """Convert common command results to compact JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    if _is_image_like(value):
        summary: Dict[str, Any] = {
            "type": value.__class__.__name__,
        }
        if hasattr(value, "shape"):
            summary["shape"] = list(value.shape)
        if hasattr(value, "dtype"):
            summary["dtype"] = str(value.dtype)
        if value.__class__.__module__.startswith("PIL"):
            summary["mode"] = getattr(value, "mode", None)
            summary["size"] = list(getattr(value, "size", ()))
        return summary
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (RuntimeError, TypeError, ValueError):
            pass
    if hasattr(value, "results"):
        return _json_safe(value.results)
    return repr(value)


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
