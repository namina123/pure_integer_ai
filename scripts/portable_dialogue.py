"""Portable, standard-library-only entry point for a closed dialogue release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


MINIMUM_PYTHON = (3, 11)
MANIFEST_NAME = "portable_bundle_manifest.json"
sys.dont_write_bytecode = True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _bundle_root() -> Path:
    return Path(__file__).resolve().parent


def _add_application_path(root: Path) -> None:
    application = root / "app"
    package = application / "pure_integer_ai" / "__init__.py"
    if not package.is_file():
        raise SystemExit("运行代码不完整：缺少 app/pure_integer_ai/__init__.py")
    sys.path.insert(0, str(application))


def _model_candidates(root: Path) -> tuple[Path, ...]:
    model_parent = root / "model"
    if not model_parent.is_dir():
        return ()
    direct = model_parent / "public_model_release.json"
    if direct.is_file():
        return (model_parent.resolve(),)
    return tuple(sorted(
        (path.resolve() for path in model_parent.iterdir()
         if path.is_dir() and (path / "public_model_release.json").is_file()),
        key=lambda path: path.name,
    ))


def _resolve_model(root: Path, value: str | None) -> Path:
    if value is not None:
        model = Path(value).expanduser().resolve()
        if not (model / "public_model_release.json").is_file():
            raise SystemExit("--model 必须指向含 public_model_release.json 的目录")
        return model
    candidates = _model_candidates(root)
    if len(candidates) != 1:
        raise SystemExit(
            "无法唯一发现模型；请用 --model 指定 release root")
    return candidates[0]


def _resolve_session(model: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    session = Path(value).expanduser().resolve()
    if session == model or model in session.parents:
        raise SystemExit("--session 必须位于只读模型目录之外")
    session.mkdir(parents=True, exist_ok=True)
    if not session.is_dir():
        raise SystemExit("--session 无法创建或不是目录")
    return session


def _load_bundle_manifest(root: Path) -> dict[str, object]:
    path = root / MANIFEST_NAME
    digest_path = root / (MANIFEST_NAME + ".sha256")
    if not path.is_file() or not digest_path.is_file():
        raise SystemExit("便携包缺少完整性 manifest")
    expected = digest_path.read_text(encoding="ascii").strip()
    if expected != _sha256(path):
        raise SystemExit("便携包 manifest SHA-256 不匹配")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit("便携包 manifest 不可回读") from error
    if (not isinstance(value, dict)
            or value.get("format") != "PURE_INTEGER_AI_PORTABLE_BUNDLE"
            or value.get("schema_version") != 1):
        raise SystemExit("便携包 manifest 格式不兼容")
    return value


def _verify_code(root: Path, manifest: dict[str, object]) -> int:
    files = manifest.get("code_files")
    if not isinstance(files, list) or not files:
        raise SystemExit("便携包 manifest 缺少 code_files")
    declared: set[str] = set()
    for ordinal, item in enumerate(files):
        if not isinstance(item, dict):
            raise SystemExit(f"code_files[{ordinal}] 非法")
        relative = item.get("path")
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if (not isinstance(relative, str) or not relative
                or type(size) is not int or size < 0
                or not isinstance(digest, str) or len(digest) != 64):
            raise SystemExit(f"code_files[{ordinal}] 描述非法")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise SystemExit("code_files 路径越出便携包") from error
        if relative in declared or not path.is_file():
            raise SystemExit(f"代码文件缺失或重复：{relative}")
        declared.add(relative)
        if path.stat().st_size != size or _sha256(path) != digest:
            raise SystemExit(f"代码文件发生漂移：{relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in (root / "app").rglob("*") if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    declared_app = {item for item in declared if item.startswith("app/")}
    if actual != declared_app:
        raise SystemExit("app 代码文件集合不闭合")
    return len(declared)


def _verify(root: Path, model: Path) -> int:
    manifest = _load_bundle_manifest(root)
    code_count = _verify_code(root, manifest)
    _add_application_path(root)
    from pure_integer_ai.experiments.public_model_release import (
        load_public_model_release,
    )

    release = load_public_model_release(
        model, require_k_drive=False, verify_payload_hashes=True)
    result = {
        "code_file_count": code_count,
        "model_root": str(model),
        "python": ".".join(str(item) for item in sys.version_info[:3]),
        "release_id": release.release_id,
        "status": "PASS",
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0


def _run(root: Path, model: Path, *, protocol: str,
         session: Path | None, performance_tier: str) -> int:
    _add_application_path(root)
    from pure_integer_ai.experiments.run_trained_dialogue_terminal import main

    arguments = [
        "--project-root", str(model),
        "--release-root", str(model),
        "--protocol", protocol,
        "--performance-tier", performance_tier,
    ]
    if session is not None:
        arguments.extend(("--session-root", str(session)))
    return main(arguments)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行或验证 Pure Integer AI 便携对话包")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
            ("terminal", "启动人类可读的交互终端"),
            ("jsonl", "启动标准输入/输出 JSONL 协议")):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument(
            "--model", default=None,
            help="模型 release root；省略时使用包内唯一 model")
        child.add_argument(
            "--session", default=None,
            help="可选会话 checkpoint 目录，必须位于模型之外")
        child.add_argument(
            "--performance-tier",
            choices=("strict", "deferred-narrow", "deferred-narrow-fast"),
            default="deferred-narrow-fast",
            help="启动校验/延迟策略；搬运后首次运行前应先执行 verify")
    verify = subparsers.add_parser("verify", help="严格核验代码和模型 SHA-256")
    verify.add_argument(
        "--model", default=None,
        help="模型 release root；省略时使用包内唯一 model")
    return parser


def main(argv: list[str] | None = None) -> int:
    if sys.version_info < MINIMUM_PYTHON:
        raise SystemExit("需要 CPython 3.11 或更高版本")
    root = _bundle_root()
    args = _parser().parse_args(argv)
    model = _resolve_model(root, args.model)
    if args.command == "verify":
        return _verify(root, model)
    session = _resolve_session(model, args.session)
    return _run(
        root, model, protocol=args.command, session=session,
        performance_tier=args.performance_tier)


if __name__ == "__main__":
    raise SystemExit(main())
