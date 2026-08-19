"""直接运行 DLG-05 生产候选并写出不可覆盖的公开 observation。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pure_integer_ai.experiments.conversation_heldout_candidate_runtime import (
    run_dlg05_public_candidate,
    verify_dlg05_candidate_observation,
    write_dlg05_candidate_observation,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = (
    REPOSITORY_ROOT
    / "data"
    / "ph2"
    / "manifests"
    / "dlg05_public_candidate_observation_v3.json"
)


def _target(value: str | Path) -> Path:
    """把 observation 输出限制在公开 manifest 目录。"""
    path = Path(value)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    path = path.resolve()
    expected_parent = (
        REPOSITORY_ROOT / "data" / "ph2" / "manifests").resolve()
    if path.parent != expected_parent:
        raise ValueError("target 必须位于 data/ph2/manifests")
    return path


def run(target: str | Path = DEFAULT_TARGET) -> dict[str, object]:
    """在临时 SQLite 上运行生产候选并返回 observation 验证摘要。"""
    output = _target(target)
    with TemporaryDirectory(prefix="dlg05-public-candidate-") as root:
        result = run_dlg05_public_candidate(
            Path(root) / "candidate.sqlite3")
        write_dlg05_candidate_observation(
            output, REPOSITORY_ROOT, result)
    return verify_dlg05_candidate_observation(output)


def main() -> int:
    """解析 CLI，执行候选或只读验证既有 observation。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = _target(args.target)
    result = (
        verify_dlg05_candidate_observation(output)
        if args.verify_only else run(output)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
