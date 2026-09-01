"""Assemble code and one closed model release into a portable test bundle."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


ENTRY_MODULES = (
    "pure_integer_ai.experiments.public_model_release",
    "pure_integer_ai.experiments.run_trained_dialogue_terminal",
)
RUNTIME_DEPENDENCIES: tuple[str, ...] = ()
RUNTIME_RESOURCES = (
    "data/ph2/broad_qa_question_slots_v2.json",
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _module_path(source_root: Path, name: str) -> Path | None:
    relative = Path(*name.split("."))
    module = source_root / relative.with_suffix(".py")
    if module.is_file():
        return module
    package = source_root / relative / "__init__.py"
    return package if package.is_file() else None


def _parent_modules(name: str) -> tuple[str, ...]:
    parts = name.split(".")
    return tuple(".".join(parts[:end]) for end in range(1, len(parts)))


def _local_imports(path: Path, module_name: str) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    package = module_name if path.name == "__init__.py" else module_name.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names
                         if alias.name == "pure_integer_ai"
                         or alias.name.startswith("pure_integer_ai."))
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                package_parts = package.split(".") if package else []
                keep = len(package_parts) - node.level + 1
                if keep < 0:
                    continue
                prefix = ".".join(package_parts[:keep])
                base = ".".join(item for item in (prefix, base) if item)
            if base == "pure_integer_ai" or base.startswith("pure_integer_ai."):
                names.add(base)
                for alias in node.names:
                    names.add(base + "." + alias.name)
    return tuple(sorted(names))


def _runtime_source_files(source_root: Path) -> tuple[Path, ...]:
    pending = list(ENTRY_MODULES)
    seen: set[str] = set()
    files: set[Path] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        path = _module_path(source_root, name)
        if path is None:
            continue
        files.add(path)
        pending.extend(_parent_modules(name))
        pending.extend(_local_imports(path, name))
    if not files:
        raise RuntimeError("没有解析出便携运行代码")
    return tuple(sorted(files))


def _copy_runtime_code(project: Path, target: Path) -> int:
    source_root = project / "src"
    files = _runtime_source_files(source_root)
    for source in files:
        relative = source.relative_to(source_root)
        destination = target / "app" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return len(files)


def _vendor_runtime_dependencies(target: Path) -> list[dict[str, object]]:
    """Copy explicitly approved runtime dependencies, if any."""
    inventory: list[dict[str, object]] = []
    for distribution_name in RUNTIME_DEPENDENCIES:
        import importlib.metadata
        import importlib.util

        distribution = importlib.metadata.distribution(distribution_name)
        import_name = distribution_name
        spec = importlib.util.find_spec(import_name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"运行依赖不可定位: {distribution_name}")
        source = Path(spec.origin).resolve().parent
        destination = target / "app" / import_name
        for path in sorted(source.rglob("*")):
            if (not path.is_file() or "__pycache__" in path.parts
                    or path.suffix in {".pyc", ".pyo"}):
                continue
            output = destination / path.relative_to(source)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output)
        license_files = tuple(
            item for item in (distribution.files or ())
            if Path(str(item)).name.upper().startswith("LICENSE"))
        if not license_files:
            raise RuntimeError(f"运行依赖缺少许可证: {distribution_name}")
        notice_root = target / "THIRD_PARTY_NOTICES"
        notice_root.mkdir(exist_ok=True)
        notice = notice_root / (
            f"{distribution_name}-{distribution.version}-LICENSE.txt")
        shutil.copy2(distribution.locate_file(license_files[0]), notice)
        inventory.append({
            "distribution": distribution_name,
            "license": str(distribution.metadata.get("License") or ""),
            "version": distribution.version,
        })
    return inventory


def _copy_runtime_resources(project: Path, target: Path) -> int:
    for relative in RUNTIME_RESOURCES:
        source = project / relative
        if not source.is_file():
            raise RuntimeError(f"运行资源缺失: {relative}")
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)
    return len(RUNTIME_RESOURCES)


def _write_launchers(target: Path) -> None:
    windows = (
        "@echo off\r\n"
        "setlocal\r\n"
        "chcp 65001 >nul\r\n"
        "python \"%~dp0run.py\" terminal --session "
        "\"%~dp0runtime\\session\" %*\r\n"
    )
    (target / "启动终端.cmd").write_text(
        windows, encoding="utf-8-sig", newline="")
    jsonl = (
        "@echo off\r\n"
        "setlocal\r\n"
        "chcp 65001 >nul\r\n"
        "python \"%~dp0run.py\" jsonl --session "
        "\"%~dp0runtime\\session\" %*\r\n"
    )
    (target / "启动JSONL.cmd").write_text(
        jsonl, encoding="utf-8-sig", newline="")
    shell = (
        "#!/bin/sh\n"
        "set -eu\n"
        "ROOT=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\n"
        "exec python3 \"$ROOT/run.py\" terminal "
        "--session \"$ROOT/runtime/session\" \"$@\"\n"
    )
    shell_path = target / "run_terminal.sh"
    shell_path.write_text(shell, encoding="utf-8", newline="\n")
    shell_path.chmod(0o755)


def _git_head(project: Path) -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=project,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="ascii")
    return result.stdout.strip()


def _code_inventory(target: Path) -> list[dict[str, object]]:
    roots = (
        target / "app", target / "run.py", target / "LICENSE",
        target / "README_中文.md", target / "启动终端.cmd",
        target / "启动JSONL.cmd", target / "run_terminal.sh",
        target / "THIRD_PARTY_NOTICES",
        target / "data",
    )
    paths: list[Path] = []
    for root in roots:
        if root.is_dir():
            paths.extend(
                path for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"})
        elif root.is_file():
            paths.append(root)
    return [{
        "path": path.relative_to(target).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    } for path in sorted(paths)]


def build(project_root: str | Path, release_root: str | Path,
          output_root: str | Path, *, package_id: str) -> dict[str, object]:
    project = Path(project_root).resolve()
    release = Path(release_root).resolve()
    output = Path(output_root).resolve()
    if not (project / "src/pure_integer_ai/__init__.py").is_file():
        raise ValueError("project_root 非项目根")
    if not (release / "public_model_release.json").is_file():
        raise ValueError("release_root 非闭合模型 release")
    if output.exists():
        raise FileExistsError("output_root 已存在，禁止覆盖")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        code_count = _copy_runtime_code(project, temporary)
        dependencies = _vendor_runtime_dependencies(temporary)
        resource_count = _copy_runtime_resources(project, temporary)
        shutil.copy2(project / "scripts/portable_dialogue.py", temporary / "run.py")
        shutil.copy2(project / "LICENSE", temporary / "LICENSE")
        shutil.copy2(
            project / "docs/portable_dialogue_bundle_v1.md",
            temporary / "README_中文.md")
        _write_launchers(temporary)
        (temporary / "runtime/session").mkdir(parents=True)
        model_parent = temporary / "model"
        model_parent.mkdir()
        model_target = model_parent / release.name
        shutil.copytree(release, model_target, copy_function=shutil.copy2)
        release_manifest = model_target / "public_model_release.json"
        release_digest = model_target / "public_model_release.sha256"
        expected_release_digest = release_digest.read_text(
            encoding="ascii").strip()
        if expected_release_digest != _sha256(release_manifest):
            raise ValueError("复制后的模型 manifest SHA-256 漂移")
        release_value = json.loads(release_manifest.read_text(encoding="utf-8"))
        manifest = {
            "code_files": _code_inventory(temporary),
            "format": "PURE_INTEGER_AI_PORTABLE_BUNDLE",
            "model": {
                "manifest_sha256": expected_release_digest,
                "path": "model/" + release.name,
                "release_id": release_value.get("release_id"),
            },
            "package_id": package_id,
            "python_requires": ">=3.11",
            "runtime_dependencies": dependencies,
            "runtime_resource_count": resource_count,
            "schema_version": 1,
            "source_git_head": _git_head(project),
        }
        manifest_path = temporary / "portable_bundle_manifest.json"
        manifest_path.write_bytes(_canonical(manifest))
        (temporary / "portable_bundle_manifest.json.sha256").write_text(
            _sha256(manifest_path) + "\n", encoding="ascii", newline="\n")
        os.replace(temporary, output)
    except Exception:
        print(f"partial_root={temporary}", file=sys.stderr)
        raise
    size = sum(path.stat().st_size for path in output.rglob("*")
               if path.is_file())
    return {
        "bytes": size,
        "code_file_count": code_count,
        "model_release": release.name,
        "output_root": str(output),
        "package_id": package_id,
        "status": "BUILT",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build portable dialogue bundle")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--package-id", required=True)
    args = parser.parse_args(argv)
    result = build(
        args.project_root, args.release_root, args.output_root,
        package_id=args.package_id)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
