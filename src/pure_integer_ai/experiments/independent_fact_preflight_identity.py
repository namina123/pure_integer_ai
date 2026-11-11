"""Freeze the current source tree and all physical preflight prerequisites."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

from pure_integer_ai.experiments.natural_relation_preflight import file_sha256


RUNNER = "scripts/run_independent_fact_preflight.py"
LAUNCHER = "scripts/wait_relation_language_preflight.ps1"


def code_identity(project: Path) -> dict:
    files = sorted((project / "src").rglob("*.py"))
    files.extend(project / relative for relative in (RUNNER, LAUNCHER))
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(project).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(path)))
    return {"source_tree_and_runner_sha256": digest.hexdigest(),
            "launcher_sha256": file_sha256(project / LAUNCHER),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()}


def tree_hashes(root: Path, relatives: tuple[str, ...]) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): file_sha256(path)
            for relative in relatives
            for path in sorted((root / relative).rglob("*")) if path.is_file()}


def verify_frozen_files(root: Path, hashes: dict[str, str]) -> None:
    if any(file_sha256(root / relative) != digest for relative, digest in hashes.items()):
        raise ValueError("frozen preflight source, split or physical graph changed")


__all__ = ["RUNNER", "LAUNCHER", "code_identity", "tree_hashes", "verify_frozen_files"]
