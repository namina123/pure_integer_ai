"""W-03 词义与概念边界阶段的零 payload 可见性合同。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from pathlib import PurePosixPath

import pytest

from pure_integer_ai.experiments.ph2_d03_release_contract import D03ContractError
from pure_integer_ai.experiments.ph2_d03_release_catalog import (
    FORMAL_GLOBAL_MANIFEST_PATH,
)
from pure_integer_ai.experiments.ph2_d03_release_reader import D03ReleaseReader


REPOSITORY = Path(__file__).resolve().parents[1]
VIEW_KINDS = ("candidate", "teacher", "evaluator")
W03_BAD_PATHS = (
    "ph2_dataset_artifacts/d02_source_pack_v1/packs/"
    "WIKIDATA_REVISION_V1--CC0-1.0--source-pack-v1/"
    "owners/teacher/dev.evidence.jsonl.gz",
    "ph2_dataset_artifacts/d02_source_pack_v1/packs/"
    "WIKIDATA_REVISION_V1--CC0-1.0--source-pack-v1/"
    "owners/teacher/held_out.evidence.jsonl.gz",
    "ph2_dataset_artifacts/d02_source_pack_v1/packs/"
    "ZHWIKTIONARY_20260701--CC-BY-SA-4.0--source-pack-v1/"
    "owners/teacher/held_out.evidence.jsonl.gz",
    "ph2_dataset_artifacts/d02_source_pack_v1/packs/"
    "WIKIDATA_REVISION_V1--CC0-1.0--source-pack-v1/"
    "owners/evaluator/train.labels.jsonl.gz",
    "ph2_dataset_artifacts/d02_source_pack_v1/packs/"
    "ZHWIKTIONARY_20260701--CC-BY-SA-4.0--source-pack-v1/"
    "owners/evaluator/train.labels.jsonl.gz",
)
W01_PATH_DIGEST = "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b"
W02_PATH_DIGESTS = (
    "134148a9d603c236d7e8a218c4dca26b2c296414ed555fefcea719f750bef561",
    "d49d3baf53cd3346b728a24211b5bfdf903cf39785cebb03ea91d7cf8e1f035f",
    "79f722d423bbdb67fb24e5976fe4bf79cedd693b55587a6c063500923a016b8c",
)


def _guard_payload_reads(monkeypatch) -> tuple[list[str], list[str]]:
    """禁止 gzip payload 与普通路径写入，并返回两类尝试账。"""
    original_read_bytes = Path.read_bytes
    gzip_read_attempts: list[str] = []
    write_attempts: list[str] = []

    def guarded_read_bytes(path: Path) -> bytes:
        """拒绝 reader 在完成路径授权前打开 gzip payload。"""
        if path.name.endswith(".jsonl.gz"):
            gzip_read_attempts.append(path.as_posix())
            raise AssertionError("W-03 visibility 不得读取 gzip payload")
        return original_read_bytes(path)

    def guarded_write(path: Path, *_args, **_kwargs):
        """拒绝只读 visibility 路径产生任何文件写入。"""
        write_attempts.append(path.as_posix())
        raise AssertionError("W-03 visibility 不得产生文件写入")

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "write_bytes", guarded_write)
    monkeypatch.setattr(Path, "write_text", guarded_write)
    return gzip_read_attempts, write_attempts


def _open_guarded_reader(monkeypatch) -> tuple[D03ReleaseReader, list[str], list[str]]:
    """在零 payload、零写守卫下打开正式 D-03 reader。"""
    gzip_reads, writes = _guard_payload_reads(monkeypatch)
    reader = D03ReleaseReader.open(
        REPOSITORY,
        FORMAL_GLOBAL_MANIFEST_PATH,
        require_publication=True,
    )
    return reader, gzip_reads, writes


def _path_digest(paths: tuple[str, ...]) -> str:
    """对有序路径集合形成带末尾换行的稳定摘要。"""
    return hashlib.sha256(("\n".join(paths) + "\n").encode("utf-8")).hexdigest()


def test_w03_visibility_is_exact_without_reading_payload(monkeypatch) -> None:
    """仅回读规范 manifest，并冻结三类 owner 视图的精确路径数。"""
    reader, gzip_read_attempts, write_attempts = _open_guarded_reader(monkeypatch)
    views = tuple(
        reader.visibility("W-03", view_kind)
        for view_kind in VIEW_KINDS
    )

    assert tuple(len(view.allowed_paths) for view in views) == (12, 18, 29)
    assert tuple(view.payload_reads for view in views) == (0, 0, 0)
    assert tuple(view.payload_bytes for view in views) == (0, 0, 0)
    assert gzip_read_attempts == []
    assert write_attempts == []


def test_w03_allowed_paths_trace_to_unique_owner_and_split(monkeypatch) -> None:
    """每条授权路径唯一追溯到 pack manifest 文件身份、owner 和 split。"""
    reader, gzip_reads, writes = _open_guarded_reader(monkeypatch)
    allowed_pairs = {
        "candidate": {("source", None), ("observation", "train")},
        "teacher": {
            ("source", None),
            ("observation", "train"),
            ("teacher", "train"),
        },
        "evaluator": {
            ("source", None),
            ("observation", "dev"),
            ("observation", "held_out"),
            ("evaluator", "dev"),
            ("evaluator", "held_out"),
        },
    }

    for view_kind in VIEW_KINDS:
        traces = reader.visible_file_identities("W-03", view_kind)
        paths = tuple(item.relative_path for item in traces)
        assert paths == reader.visibility("W-03", view_kind).allowed_paths
        assert len(paths) == len(set(paths))
        for item in traces:
            prefix = PurePosixPath(item.manifest_identity.relative_path).parent
            assert item.relative_path == PurePosixPath(
                prefix,
                item.file_identity.relative_path,
            ).as_posix()
            assert (
                item.file_identity.owner_kind,
                item.file_identity.split,
            ) in allowed_pairs[view_kind]
    assert gzip_reads == []
    assert writes == []


def test_w03_private_views_are_disjoint_and_bad_paths_fail_before_payload(
        monkeypatch,
        ) -> None:
    """五条跨 split 坏路径在 payload 前拒绝，三类 owner 不交叉污染。"""
    reader, gzip_reads, writes = _open_guarded_reader(monkeypatch)
    candidate = set(reader.visibility("W-03", "candidate").allowed_paths)
    teacher = set(reader.visibility("W-03", "teacher").allowed_paths)
    evaluator = set(reader.visibility("W-03", "evaluator").allowed_paths)
    evaluator_private = {path for path in evaluator if "/owners/evaluator/" in path}

    assert candidate.isdisjoint(evaluator_private)
    assert all(path not in teacher for path in W03_BAD_PATHS[:3])
    assert all(path not in evaluator for path in W03_BAD_PATHS[3:])
    assert all("/owners/teacher/" not in path for path in evaluator)
    assert all("/observations/train.jsonl.gz" not in path for path in evaluator)
    for view_kind, path in (
            *(("teacher", path) for path in W03_BAD_PATHS[:3]),
            *(("evaluator", path) for path in W03_BAD_PATHS[3:]),
            ):
        with pytest.raises(D03ContractError, match="不可见"):
            reader.require_visible_path("W-03", view_kind, path)
    assert gzip_reads == []
    assert writes == []


def test_w01_w02_visibility_path_sets_are_unchanged(monkeypatch) -> None:
    """共享 reader 收窄 W-03 时保持 W-01/W-02 三视图逐路径不变。"""
    reader, gzip_reads, writes = _open_guarded_reader(monkeypatch)
    w01 = tuple(reader.visibility("W-01", kind) for kind in VIEW_KINDS)
    w02 = tuple(reader.visibility("W-02", kind) for kind in VIEW_KINDS)

    assert tuple(len(view.allowed_paths) for view in w01) == (0, 0, 0)
    assert tuple(_path_digest(view.allowed_paths) for view in w01) == (
        W01_PATH_DIGEST,
        W01_PATH_DIGEST,
        W01_PATH_DIGEST,
    )
    assert tuple(len(view.allowed_paths) for view in w02) == (4, 6, 9)
    assert tuple(_path_digest(view.allowed_paths) for view in w02) == W02_PATH_DIGESTS
    assert gzip_reads == []
    assert writes == []


@pytest.mark.parametrize(
    ("stage_key", "view_kind", "path"),
    (
        ("W-10", "candidate", W03_BAD_PATHS[0]),
        ("W-03", "trainer", W03_BAD_PATHS[0]),
        ("W-03", "candidate", "../escape.jsonl.gz"),
        ("W-03", "candidate", "/absolute/path.jsonl.gz"),
        ("W-03", "candidate", "not//canonical.jsonl.gz"),
        ("W-03", "candidate", "not\\posix.jsonl.gz"),
    ),
)
def test_w03_unknown_or_unsafe_requests_fail_closed(
        monkeypatch,
        stage_key: str,
        view_kind: str,
        path: str,
        ) -> None:
    """未知 stage/view、路径逃逸和非规范路径均在 payload 前关闭。"""
    reader, gzip_reads, writes = _open_guarded_reader(monkeypatch)
    with pytest.raises(D03ContractError):
        reader.require_visible_path(stage_key, view_kind, path)
    assert gzip_reads == []
    assert writes == []


def test_w03_unknown_pack_fails_before_payload(monkeypatch) -> None:
    """未知 pack 在尝试 manifest 或 payload 读取前拒绝。"""
    reader, gzip_reads, writes = _open_guarded_reader(monkeypatch)
    with pytest.raises(D03ContractError, match="未知 pack"):
        reader.verify_pack_files("UNKNOWN-W03-PACK")
    assert gzip_reads == []
    assert writes == []


def test_w03_pack_manifest_drift_fails_closed_before_payload(
        tmp_path: Path,
        monkeypatch,
        ) -> None:
    """pack manifest 任一字节漂移时，授权集合在 gzip payload 前失效。"""
    baseline = D03ReleaseReader.open(
        REPOSITORY,
        FORMAL_GLOBAL_MANIFEST_PATH,
        require_publication=True,
    )
    manifest_identity = baseline.visible_file_identities(
        "W-03",
        "candidate",
    )[0].manifest_identity
    source = REPOSITORY / Path(*PurePosixPath(
        manifest_identity.relative_path).parts)
    target = tmp_path / Path(*PurePosixPath(
        manifest_identity.relative_path).parts)
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes() + b" ")

    gzip_reads, writes = _guard_payload_reads(monkeypatch)
    reader = D03ReleaseReader.open(
        tmp_path,
        FORMAL_GLOBAL_MANIFEST_PATH,
        dependency_root=REPOSITORY,
        require_publication=True,
    )
    with pytest.raises(D03ContractError, match="身份漂移"):
        reader.visibility("W-03", "candidate")
    assert gzip_reads == []
    assert writes == []
