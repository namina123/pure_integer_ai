"""训练后整数关系图的标准库便携入口。"""
from __future__ import annotations

import argparse
import io
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


def _validate_boundary(root: Path, model: Path, args: argparse.Namespace) -> int:
    """启动前/后快照模型根与便携包根，验证包外 session 与文件闭合。

    使用进程内 JSONL 回放（便携包不安装 -m 包入口），退出码非零只代表
    strict fail-closed，不代表边界漂移；文件集合变化才是 FAIL 判定依据。
    """
    _add_application(root)
    from pure_integer_ai.experiments.runtime_boundary_validator import (
        _jsonl_turn,
        compare_snapshots,
        snapshot_tree,
    )
    from pure_integer_ai.experiments.run_trained_relation_graph_terminal import (
        run_trained_relation_graph_terminal,
    )
    from pure_integer_ai.experiments.trained_graph_release import (
        load_trained_graph_release,
    )
    session = Path(args.session).expanduser().resolve()
    try:
        session.relative_to(model)
    except ValueError:
        session_external = True
    else:
        session_external = False
    before_model = snapshot_tree(model)
    before_root = snapshot_tree(root)
    release = load_trained_graph_release(model, verify_payload_hashes=True)
    turns = tuple(
        value.strip() for value in args.turns.split(",") if value.strip())
    if not turns:
        raise SystemExit("--turns 不能为空")
    payload = b"".join(
        _jsonl_turn(text, request_id=ordinal + 1)
        for ordinal, text in enumerate(turns))
    payload += b"{\"op\":\"quit\"}\n"
    stream_in = io.BytesIO(payload)
    stream_out = io.BytesIO()
    exit_code = 0
    stderr_tail = ""
    try:
        run_trained_relation_graph_terminal(
            training_database=release.training_database,
            fallback_surfaces=(),
            memory_database=str(session),
            input_stream=stream_in,
            output_stream=stream_out,
            protocol_stream=True,
            strict_graph=True,
        )
    except BaseException as error:  # strict fail-closed 是合法终止
        exit_code = 1
        stderr_tail = str(error)[-2000:]
    after_model = snapshot_tree(model)
    after_root = snapshot_tree(root)
    model_compare = compare_snapshots(before_model, after_model)
    root_compare = compare_snapshots(before_root, after_root)
    model_closed = bool(model_compare["closed"])
    package_closed = bool(root_compare["closed"])
    drift = not model_closed or not package_closed or not session_external
    status = "FAIL" if drift else "PASS"
    print(json.dumps({
        "format": "PURE_INTEGER_RUNTIME_BOUNDARY_VALIDATION_V1",
        "schema_version": 1,
        "status": status,
        "model_tree_closed": model_closed,
        "package_tree_closed": package_closed,
        "session_external": session_external,
        "exit_code": exit_code,
        "stderr_tail": stderr_tail,
        "model_tree": model_compare,
        "package_tree": root_compare,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if status == "PASS" else 1


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
    boundary = sub.add_parser(
        "validate-boundary",
        help="运行前后快照校验模型根与便携包文件闭合、session 外置性")
    boundary.add_argument("--model", default=None)
    boundary.add_argument("--session", required=True,
                          help="包外会话 SQLite 路径（不得位于模型根内）")
    boundary.add_argument("--turns", default="你好,一加一等于几？")
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
    if args.command == "validate-boundary":
        return _validate_boundary(root, model, args)
    return _run(root, model, args)


if __name__ == "__main__":
    raise SystemExit(main())
