"""训练后整数关系图的标准库便携入口。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


MINIMUM_PYTHON = (3, 11)
sys.dont_write_bytecode = True


def _root() -> Path:
    return Path(__file__).resolve().parent


def _add_application(root: Path) -> None:
    application = root / "app"
    package = application / "pure_integer_ai" / "__init__.py"
    if not package.is_file():
        raise SystemExit("便携包缺少 app/pure_integer_ai")
    sys.path.insert(0, str(application))


def _models(root: Path) -> tuple[Path, ...]:
    parent = root / "model"
    if not parent.is_dir():
        return ()
    direct = parent / "trained_graph_release.json"
    if direct.is_file():
        return (parent.resolve(),)
    return tuple(sorted(
        (path.resolve() for path in parent.iterdir()
         if path.is_dir() and (path / "trained_graph_release.json").is_file()),
        key=lambda path: path.name,
    ))


def _model(root: Path, value: str | None) -> Path:
    if value is not None:
        path = Path(value).expanduser().resolve()
        if not (path / "trained_graph_release.json").is_file():
            raise SystemExit("--model 必须指向含 trained_graph_release.json 的目录")
        return path
    choices = _models(root)
    if len(choices) != 1:
        raise SystemExit("无法唯一发现模型，请使用 --model 指定发布根")
    return choices[0]


def _run(root: Path, model: Path, args: argparse.Namespace) -> int:
    _add_application(root)
    from pure_integer_ai.experiments.run_trained_relation_graph_terminal import main

    forwarded = ["--release-root", str(model), "--protocol", args.protocol]
    if args.session is not None:
        forwarded.extend(("--memory-database", str(Path(args.session).expanduser().resolve())))
    if args.metrics is not None:
        forwarded.extend(("--metrics-output", str(Path(args.metrics).expanduser().resolve())))
    return main(forwarded)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行训练后整数关系图便携包")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("terminal", "jsonl"):
        child = sub.add_parser(name, help="启动人类终端或 JSONL 交流入口")
        child.add_argument("--model", default=None)
        child.add_argument("--session", default=None,
                           help="可选会话 SQLite 路径，模型目录之外")
        child.add_argument("--metrics", default=None)
        child.set_defaults(protocol=name)
    verify = sub.add_parser("verify", help="核验训练后发布根")
    verify.add_argument("--model", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    if sys.version_info < MINIMUM_PYTHON:
        raise SystemExit("需要 CPython 3.11 或更高版本")
    root = _root()
    args = _parser().parse_args(argv)
    model = _model(root, args.model)
    _add_application(root)
    from pure_integer_ai.experiments.trained_graph_release import (
        load_trained_graph_release,
    )
    if args.command == "verify":
        release = load_trained_graph_release(model, verify_payload_hashes=True)
        print(json.dumps({
            "release_id": release.release_id,
            "model_root": str(model),
            "status": "PASS",
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0
    return _run(root, model, args)


if __name__ == "__main__":
    raise SystemExit(main())
