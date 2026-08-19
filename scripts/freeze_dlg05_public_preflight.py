"""运行唯一公开 DLG-05 qualification harness 并写出不可覆盖冻结清单。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pure_integer_ai.experiments.conversation_heldout_candidate_runtime import (
    qualify_dlg05_public_candidate,
    run_dlg05_public_candidate,
    verify_dlg05_candidate_observation,
    write_dlg05_candidate_observation,
)
from pure_integer_ai.experiments.conversation_heldout_freeze import (
    verify_dlg05_public_freeze_document,
    write_dlg05_public_freeze_document,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = (
    REPOSITORY_ROOT
    / "data"
    / "ph2"
    / "manifests"
    / "dlg05_public_preflight_freeze_v3.json"
)
CANDIDATE_OBSERVATION = (
    REPOSITORY_ROOT
    / "data"
    / "ph2"
    / "manifests"
    / "dlg05_public_candidate_observation_v3.json"
)


def _target(value: str | Path) -> Path:
    """把输出限制在公开 manifest 目录。"""
    path = Path(value)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    path = path.resolve()
    expected_parent = (
        REPOSITORY_ROOT / "data" / "ph2" / "manifests").resolve()
    if path.parent != expected_parent:
        raise ValueError("target 必须位于 data/ph2/manifests")
    return path


def freeze(target: str | Path = DEFAULT_TARGET) -> dict[str, object]:
    """运行真实六 case preflight 并返回写出的公开冻结摘要。"""
    output = _target(target)
    with TemporaryDirectory(prefix="dlg05-public-freeze-") as root:
        selection = run_dlg05_public_candidate(
            Path(root) / "selection.sqlite3")
        write_dlg05_candidate_observation(
            CANDIDATE_OBSERVATION, REPOSITORY_ROOT, selection)
        verify_dlg05_candidate_observation(CANDIDATE_OBSERVATION)
        result = qualify_dlg05_public_candidate(
            Path(root) / "qualification.sqlite3")
        write_dlg05_public_freeze_document(
            output,
            REPOSITORY_ROOT,
            result.catalog,
            result.manifest,
            result.qualification,
        )
    payload = output.read_bytes()
    value = json.loads(payload)
    if value.get("formal_run") != 0 or value.get("labels_included") != 0:
        raise RuntimeError("DLG-05 freeze 越过 public/label-free 边界")
    result = verify_dlg05_public_freeze_document(output, REPOSITORY_ROOT)
    if result["sha256"] != hashlib.sha256(payload).hexdigest():
        raise RuntimeError("DLG-05 freeze verifier/output SHA 漂移")
    return result


def main() -> int:
    """解析 CLI 并输出紧凑冻结摘要。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = (
        verify_dlg05_public_freeze_document(
            _target(args.target), REPOSITORY_ROOT)
        if args.verify_only else freeze(args.target)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
