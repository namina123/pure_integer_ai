"""为公开 CI 恢复 W-03 冻结测试所需的 W-02 安全连续性文件。"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from pure_integer_ai.experiments.ph2_w03_continuity import (
    verify_formal_w02_continuity,
)


ARCHIVE_SHA256 = "6834567e1df2609ad6314967e5b3a3e1231916a894a1b48fb67a0fee2027652a"
ARCHIVE_RELATIVE_PATH = "tests/fixtures/w02_public_continuity_v1.tar.gz"
EXPECTED_FILES = (
    "w02_artifacts/formal_candidate_v2/candidate.barrier.sqlite3",
    "w02_artifacts/formal_candidate_v2/candidate.segments.sqlite3",
    "w02_artifacts/formal_candidate_v2/candidate.sqlite3",
    "w02_artifacts/formal_candidate_v2/candidate.worker.sqlite3",
    "w02_artifacts/formal_candidate_v2/candidate_host_freeze_v2.json",
    "w02_artifacts/formal_candidate_v2/runs/3/cursor.json",
    "w02_artifacts/formal_candidate_v2/runs/3/global_identity.dump",
    "w02_artifacts/formal_candidate_v2/runs/3/run.manifest.json",
    "w02_artifacts/formal_candidate_v2/runs/3/run.manifest.sha256",
    "w02_artifacts/formal_candidate_v2/runs/3/space_1.dump",
    "w02_artifacts/formal_candidate_v2/runs/3/space_2.dump",
    "w02_artifacts/formal_candidate_v2/runs/3/space_3.dump",
    "w02_artifacts/formal_candidate_v2_publication_20260730_a/"
    "candidate_publication_attestation.json",
    "w02_artifacts/formal_private_evaluator_v3_20260730_a/publication/"
    "private_evaluation_aggregate.json",
    "w02_artifacts/formal_private_evaluator_v3_20260730_a/publication/"
    "w02_runtime_evidence_receipt.json",
)


class W03CIFixtureError(RuntimeError):
    """CI fixture 归档身份、成员或解包边界错误。"""


def _safe_member_name(value: str) -> str:
    """只接受位于 w02_artifacts 下的规范 POSIX 成员路径。"""
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts
            or path.as_posix() != value
            or not path.parts or path.parts[0] != "w02_artifacts"):
        raise W03CIFixtureError("W-02 CI fixture 成员路径非法")
    return value


def restore_w02_public_continuity_fixture(
        repository_root: str | Path,
        destination_parent: str | Path,
        ) -> Path:
    """严格验证归档成员，以 xb 解包并执行正式 continuity verifier。"""
    repository = Path(repository_root).resolve()
    destination = Path(destination_parent).resolve()
    archive = repository / Path(*PurePosixPath(
        ARCHIVE_RELATIVE_PATH).parts)
    if (not archive.is_file() or archive.is_symlink()
            or hashlib.sha256(archive.read_bytes()).hexdigest()
            != ARCHIVE_SHA256):
        raise W03CIFixtureError("W-02 CI fixture archive identity 漂移")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:gz") as bundle:
        members = bundle.getmembers()
        files = tuple(sorted(
            _safe_member_name(item.name)
            for item in members if item.isfile()))
        if files != tuple(sorted(EXPECTED_FILES)):
            raise W03CIFixtureError("W-02 CI fixture 文件集合漂移")
        for member in members:
            name = _safe_member_name(member.name.rstrip("/"))
            if member.issym() or member.islnk() or member.isdev():
                raise W03CIFixtureError("W-02 CI fixture 含链接或设备成员")
            target = (destination / Path(*PurePosixPath(name).parts)).resolve()
            if not target.is_relative_to(destination):
                raise W03CIFixtureError("W-02 CI fixture 解包路径逃逸")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise W03CIFixtureError("W-02 CI fixture 含未知成员类型")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise W03CIFixtureError("W-02 CI fixture 文件不可读取")
            try:
                with target.open("xb") as handle:
                    shutil.copyfileobj(source, handle)
            except FileExistsError as exc:
                raise W03CIFixtureError("W-02 CI fixture 不可覆盖") from exc
    root = destination / "w02_artifacts"
    continuity = verify_formal_w02_continuity(repository, root)
    if (continuity.dimension_statuses != ("PASS",) * 5
            or continuity.fail_count != 0 or continuity.ne_count != 0):
        raise W03CIFixtureError("W-02 CI fixture continuity 未通过")
    return root


def main() -> None:
    """在 checkout 上一级恢复，与冻结 W-03 测试的路径合同一致。"""
    repository = Path(__file__).resolve().parents[1]
    root = restore_w02_public_continuity_fixture(repository, repository.parent)
    print(f"W-02 continuity CI fixture verified: {len(EXPECTED_FILES)} files at {root.name}")


if __name__ == "__main__":
    main()
