"""训练后整数关系图的标准库便携入口。"""
from __future__ import annotations

import argparse
import hashlib
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


def _verify_bundle(root: Path, model: Path) -> int:
    manifest_path = root / "portable_bundle_manifest.json"
    digest_path = root / "portable_bundle_manifest.json.sha256"
    if not manifest_path.is_file() or not digest_path.is_file():
        raise SystemExit("便携包缺少完整性清单")

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    if digest_path.read_text(encoding="ascii").strip() != sha256(manifest_path):
        raise SystemExit("便携包清单 SHA-256 不匹配")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("便携包 inventory 非法")
    declared = set()
    for row in rows:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit("便携包 inventory 含越界路径")
        source_path = root / relative
        if source_path.is_symlink():
            raise SystemExit("便携包 inventory 不允许符号链接")
        path = source_path.resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise SystemExit("便携包文件越界") from error
        if (not path.is_file() or path.is_symlink()
                or path.stat().st_size != row["size_bytes"]
                or sha256(path) != row["sha256"]):
            raise SystemExit(f"便携包文件 SHA-256 不匹配: {relative}")
        declared.add(relative.as_posix())
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
        and path.name not in {
            "portable_bundle_manifest.json",
            "portable_bundle_manifest.json.sha256",
        }
    }
    if declared != actual:
        raise SystemExit("便携包文件集合与 inventory 不一致")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise SystemExit("便携包目录不允许符号链接")
    if (model / "model" / "training.sqlite3").stat().st_nlink != 1:
        raise SystemExit("模型数据库不是独立物理文件")
    _add_application(root)
    from pure_integer_ai.experiments.trained_graph_release import (
        load_trained_graph_release,
    )
    release = load_trained_graph_release(model, verify_payload_hashes=True)
    print(json.dumps({
        "package_id": manifest.get("package_id"),
        "release_id": release.release_id,
        "model_sha256": sha256(release.training_database),
        "file_count": len(declared),
        "runtime_dependencies": [],
        "physical_model_copy": True,
        "free_dialogue_complete": 0,
        "status": "PASS",
    }, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


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
    terminal = sub.add_parser("terminal", help="启动终端文本对话")
    terminal.add_argument("--model", default=None)
    terminal.add_argument("--session", default=None,
                          help="可选会话 SQLite 路径，模型目录之外")
    terminal.add_argument("--metrics", default=None)
    terminal.set_defaults(protocol="terminal")
    verify = sub.add_parser("verify", help="核验整个便携包及训练后发布根")
    verify.add_argument("--model", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    if sys.implementation.name != "cpython" or sys.version_info < MINIMUM_PYTHON:
        raise SystemExit("需要 CPython 3.11 或更高版本")
    root = _root()
    args = _parser().parse_args(argv)
    model = _model(root, args.model)
    if args.command == "verify":
        return _verify_bundle(root, model)
    _add_application(root)
    return _run(root, model, args)


if __name__ == "__main__":
    raise SystemExit(main())
