"""构建训练后整数关系图的可搬运测试包。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


ENTRY_MODULES = (
    "pure_integer_ai.experiments.run_trained_relation_graph_terminal",
    "pure_integer_ai.experiments.trained_graph_release",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _module_path(source_root: Path, name: str) -> Path | None:
    relative = Path(*name.split("."))
    module = source_root / relative.with_suffix(".py")
    if module.is_file():
        return module
    package = source_root / relative / "__init__.py"
    return package if package.is_file() else None


def _parents(name: str) -> tuple[str, ...]:
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
                parts = package.split(".") if package else []
                keep = len(parts) - node.level + 1
                if keep < 0:
                    continue
                prefix = ".".join(parts[:keep])
                base = ".".join(item for item in (prefix, base) if item)
            if base == "pure_integer_ai" or base.startswith("pure_integer_ai."):
                names.add(base)
                # ``from pure_integer_ai.storage import discipline`` 这类
                # 导入的子模块不一定出现在 AST 的 module 字段中；把
                # 具名导入一并加入闭包，避免便携包在新进程中循环导入。
                for alias in node.names:
                    if alias.name != "*":
                        names.add(base + "." + alias.name)
    return tuple(sorted(names))


def _runtime_files(project: Path) -> tuple[Path, ...]:
    pending = list(ENTRY_MODULES)
    seen: set[str] = set()
    files: set[Path] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        path = _module_path(project / "src", name)
        if path is None:
            continue
        files.add(path)
        pending.extend(_parents(name))
        pending.extend(_local_imports(path, name))
    if not files:
        raise RuntimeError("未解析出训练图运行代码")
    return tuple(sorted(files))


def _copy_code(project: Path, target: Path) -> int:
    files = _runtime_files(project)
    source_root = project / "src"
    for source in files:
        destination = target / "app" / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return len(files)


def _write_launchers(target: Path) -> None:
    (target / "启动终端.cmd").write_text(
        "@echo off\r\nsetlocal\r\nchcp 65001 >nul\r\n"
        "python \"%~dp0run.py\" terminal %*\r\n",
        encoding="utf-8-sig", newline="")
    (target / "启动JSONL.cmd").write_text(
        "@echo off\r\nsetlocal\r\nchcp 65001 >nul\r\n"
        "python \"%~dp0run.py\" jsonl %*\r\n",
        encoding="utf-8-sig", newline="")
    shell = "#!/bin/sh\nset -eu\nROOT=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\nexec python3 \"$ROOT/run.py\" terminal \"$@\"\n"
    path = target / "run_terminal.sh"
    path.write_text(shell, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _head(project: Path) -> str:
    return subprocess.run(("git", "rev-parse", "HEAD"), cwd=project,
                          check=True, capture_output=True, text=True,
                          encoding="ascii").stdout.strip()


def _inventory(root: Path) -> list[dict[str, object]]:
    paths = []
    for base in (root / "app", root / "run.py", root / "LICENSE",
                 root / "README_中文.md", root / "启动终端.cmd",
                 root / "启动JSONL.cmd", root / "run_terminal.sh",
                 root / "model"):
        if base.is_file():
            paths.append(base)
        elif base.is_dir():
            paths.extend(path for path in base.rglob("*") if path.is_file()
                         and "__pycache__" not in path.parts
                         and path.suffix not in {".pyc", ".pyo"})
    return [{"path": path.relative_to(root).as_posix(),
             "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(paths)]


def build(project_root: str | Path, release_root: str | Path,
          output_root: str | Path, *, package_id: str) -> dict[str, object]:
    project = Path(project_root).resolve()
    release = Path(release_root).resolve()
    output = Path(output_root).resolve()
    if not (project / "src/pure_integer_ai/__init__.py").is_file():
        raise ValueError("project_root 不是项目根")
    if not (release / "trained_graph_release.json").is_file():
        raise ValueError("release_root 不是训练图发布根")
    if output.exists():
        raise FileExistsError("output_root 已存在，拒绝覆盖")
    temporary = output.with_name(output.name + ".partial-" + uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    try:
        code_count = _copy_code(project, temporary)
        shutil.copy2(project / "scripts/portable_trained_graph.py", temporary / "run.py")
        shutil.copy2(project / "LICENSE", temporary / "LICENSE")
        (temporary / "README_中文.md").write_text(
            "# 训练后整数关系图便携测试包\n\n"
            "本包包含训练后的图模型和独立交流入口，可复制到另一台安装 CPython 3.11+ 的电脑离线测试。\n\n"
            "## 启动\n\n"
            "Windows 双击 `启动终端.cmd`，或运行 `python run.py terminal`。\n"
            "JSONL 接口运行 `python run.py jsonl`，每行输入 `\u007b\"op\":\"turn\",\"text\":\"...\"\u007d`。\n"
            "输入 `\u007b\"op\":\"quit\"\u007d` 结束。\n\n"
            "模型目录只读；需要持久会话时使用 `--session` 指向模型目录之外的 SQLite 文件。\n"
            "本包不携带课程、外部 QA、论文、密钥或 OpenCC。它是当前训练结果的独立测试包。\n",
            encoding="utf-8", newline="\n")
        _write_launchers(temporary)
        shutil.copytree(release, temporary / "model" / release.name,
                        copy_function=shutil.copy2)
        files = _inventory(temporary)
        manifest = {
            "format": "PURE_INTEGER_TRAINED_GRAPH_PORTABLE_BUNDLE_V1",
            "schema_version": 1,
            "package_id": package_id,
            "source_git_head": _head(project),
            "model": {"path": "model/" + release.name,
                      "release_manifest": "trained_graph_release.json"},
            "python_requires": ">=3.11",
            "runtime_dependencies": [],
            "files": files,
        }
        manifest_path = temporary / "portable_bundle_manifest.json"
        manifest_path.write_bytes(_canonical(manifest))
        (temporary / "portable_bundle_manifest.json.sha256").write_text(
            _sha256(manifest_path) + "\n", encoding="ascii", newline="\n")
        temporary.rename(output)
    except Exception:
        print(f"partial_root={temporary}", file=sys.stderr)
        raise
    return {"status": "BUILT", "output_root": str(output),
            "package_id": package_id, "code_file_count": code_count,
            "bytes": sum(path.stat().st_size for path in output.rglob("*")
                          if path.is_file())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建训练图便携包")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--package-id", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.project_root, args.release_root, args.output_root,
                           package_id=args.package_id), ensure_ascii=False,
                     sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
