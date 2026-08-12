"""一周广域问答 10k 纵向片的合成合同与能力测试。"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_contract import (
    BroadQaSelectedPage,
    BroadQaSelectionManifest,
    parse_selection_manifest,
)
from pure_integer_ai.experiments.ph2_broad_qa_index import (
    broad_qa_terms,
    build_broad_qa_index,
)
from pure_integer_ai.experiments.ph2_broad_qa_query import query_broad_qa
from pure_integer_ai.experiments.ph2_broad_qa_query import BroadQaQueryError
from pure_integer_ai.experiments.ph2_broad_qa_question_slots import (
    BroadQaQuestionSlotError,
    load_broad_qa_question_slots,
)
from pure_integer_ai.experiments.ph2_broad_qa_selection import (
    BroadQaSelectionError,
    derive_broad_qa_selection_prefix,
    profile_broad_qa_selection,
)
from pure_integer_ai.experiments.ph2_broad_qa_sharded import (
    BroadQaShardedError,
    build_broad_qa_projection_shards,
    build_broad_qa_sharded_index,
)
from pure_integer_ai.experiments.ph2_broad_qa_source import (
    project_broad_qa_passages,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.run_ph2_broad_qa import (
    _parser,
    _work_path,
)


def _page(page_id: int, title: str, text: str) -> bytes:
    """生成包含完整 revision 身份的单页 multistream XML 片段。"""
    escaped = (text.replace("&", "&amp;").replace("<", "&lt;")
               .replace(">", "&gt;"))
    return (
        f"<page><title>{title}</title><ns>0</ns><id>{page_id}</id>"
        f"<revision><id>{1000 + page_id}</id><parentid>{page_id}</parentid>"
        "<timestamp>2026-07-01T00:00:00Z</timestamp>"
        "<contributor><username>测试贡献者</username><id>7</id></contributor>"
        "<model>wikitext</model><format>text/x-wiki</format>"
        f"<text xml:space=\"preserve\">{escaped}</text>"
        "<sha1>syntheticsha1</sha1></revision></page>"
    ).encode("utf-8")


def _fixture(tmp_path: Path) -> tuple[BroadQaSelectionManifest, Path]:
    """构造三个独立 bzip2 block 和对应选择 manifest。"""
    pages = (
        (1, "都江堰", "都江堰是位于[[四川省]]成都市的水利工程。"
         "它由秦国蜀郡太守李冰主持修建，用于防洪和灌溉。"),
        (2, "祖冲之", "祖冲之是中国南北朝时期的数学家、天文学家。\n\n"
         "他把圆周率计算到小数点后第七位。"),
        (3, "长江", "长江是中国第一长河，发源于青藏高原。\n\n"
         "长江干流流经多个省级行政区，最后注入东海。"),
    )
    compressed = []
    cursor = 0
    selected = []
    for ordinal, (page_id, title, text) in enumerate(pages, start=1):
        payload = bz2.compress(_page(page_id, title, text))
        start = cursor
        cursor += len(payload)
        compressed.append(payload)
        rank = f"{ordinal:064x}"
        selected.append(BroadQaSelectedPage(
            ordinal, rank, title,
            hashlib.sha256(title.encode("utf-8")).hexdigest(),
            page_id, ordinal, start, cursor,
        ))
    xml = tmp_path / "wiki.xml.bz2"
    xml.write_bytes(b"".join(compressed))
    manifest = BroadQaSelectionManifest(
        "ZHWIKIPEDIA_20260701", "synthetic-snapshot", "a" * 64,
        "b" * 64, "c" * 40, "d" * 64, xml.stat().st_size, 3, 3,
        tuple(selected),
    )
    return manifest, xml


def test_selection_manifest_round_trip_is_canonical_and_tamper_evident(
        tmp_path: Path) -> None:
    """选择只绑定 index 坐标，规范字节可回读并拒绝字段漂移。"""
    manifest, _ = _fixture(tmp_path)
    payload = manifest.canonical_bytes()
    assert parse_selection_manifest(payload) == manifest
    assert parse_selection_manifest(payload).sha256() == manifest.sha256()


def test_selection_prefix_preserves_source_identity_and_is_not_rescored(
        tmp_path: Path) -> None:
    """稳定 selection 前缀只截取既有排名，不按正文或答案重选。"""
    manifest, _ = _fixture(tmp_path)
    prefix = derive_broad_qa_selection_prefix(
        manifest, requested_page_count=2)
    assert prefix.requested_page_count == 2
    assert prefix.selected_pages == manifest.selected_pages[:2]
    assert prefix.snapshot_manifest_sha256 == manifest.snapshot_manifest_sha256
    assert prefix.index_local_sha256 == manifest.index_local_sha256
    assert prefix.xml_local_sha256 == manifest.xml_local_sha256
    assert prefix.sha256() != manifest.sha256()
    with pytest.raises(BroadQaSelectionError):
        derive_broad_qa_selection_prefix(manifest, requested_page_count=4)


def test_wikitext_projection_keeps_exact_raw_span_and_link_display() -> None:
    """成熟 parser 处理链接，证据 hash 始终对应原始 Wikitext span。"""
    raw = "导言。\n\n==地理==\n都江堰位于[[四川省|四川]]成都，用于防洪和灌溉。"
    passages = project_broad_qa_passages(raw)
    assert len(passages) == 1
    evidence = passages[0]
    source = raw[evidence.raw_start:evidence.raw_end]
    assert "[[四川省|四川]]" in source
    assert "四川成都" in evidence.text
    assert evidence.raw_sha256 == hashlib.sha256(
        source.encode("utf-8")).hexdigest()


def test_build_query_answer_unknown_and_bit_identical_rebuild(
        tmp_path: Path) -> None:
    """陌生问句从 postings 找证据，未知拒答且两次构建逐字节一致。"""
    manifest, xml = _fixture(tmp_path)
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    report_a = build_broad_qa_index(
        manifest, xml_path=xml, database_path=first, accepted_page_limit=3)
    report_b = build_broad_qa_index(
        manifest, xml_path=xml, database_path=second, accepted_page_limit=3)
    assert report_a["database_sha256"] == report_b["database_sha256"]
    assert first.read_bytes() == second.read_bytes()
    connection = sqlite3.connect(str(first))
    try:
        result = query_broad_qa(connection, "谁主持修建都江堰用于防洪？")
        unknown = query_broad_qa(connection, "火星上的都江堰由谁修建？")
    finally:
        connection.close()
    assert result.status == "ANSWER"
    assert result.title == "都江堰"
    assert "李冰" in result.answer
    assert result.evidence_raw_sha256 is not None
    assert result.revision_timestamp == "2026-07-01T00:00:00Z"
    assert result.to_dict()["citation"]["contributor"] == {
        "kind": "registered", "user_id": 7, "username": "测试贡献者"}
    assert result.to_dict()["citation"]["attribution"] == (
        "Wikipedia contributors")
    assert result.source_url == (
        "https://zh.wikipedia.org/w/index.php?curid=1&oldid=1001")
    assert unknown.status in {"UNKNOWN", "CLARIFY"}
    assert unknown.answer is None


def test_terms_are_integer_index_features_not_full_question_keys() -> None:
    """改写问题共享局部特征，但不保存完整问题字符串答案表。"""
    first = set(broad_qa_terms("谁主持修建都江堰？"))
    second = set(broad_qa_terms("都江堰的修建者是谁？"))
    assert {"c:都江", "c:江堰"}.issubset(first & second)
    assert "谁主持修建都江堰？" not in first


def test_query_rejects_unbounded_question_and_posting_work(
        tmp_path: Path) -> None:
    """查询长度、原始特征和 posting visit 都有运行时硬预算。"""
    manifest, xml = _fixture(tmp_path)
    database = tmp_path / "budget.sqlite"
    build_broad_qa_index(
        manifest, xml_path=xml, database_path=database,
        accepted_page_limit=3,
    )
    connection = sqlite3.connect(str(database))
    try:
        with pytest.raises(BroadQaQueryError, match="question"):
            query_broad_qa(connection, "长" * 2049)
        with pytest.raises(BroadQaQueryError, match="posting visit"):
            query_broad_qa(
                connection, "谁主持修建都江堰？", max_posting_visits=1)
    finally:
        connection.close()


def test_large_work_path_is_absolute_and_windows_stays_on_k(
        tmp_path: Path) -> None:
    """公开 CLI 可跨平台复现，本机 Windows 仍严格执行 K 盘纪律。"""
    with pytest.raises(argparse.ArgumentTypeError, match="absolute"):
        _work_path("relative-run-root")
    if sys.platform == "win32":
        with pytest.raises(argparse.ArgumentTypeError, match="K:"):
            _work_path(str(tmp_path))
    else:
        assert _work_path(str(tmp_path)) == tmp_path.resolve()


def test_relation_without_literal_overlap_keeps_minimum_adjacent_context(
        tmp_path: Path) -> None:
    """关系词尚无同义映射时保留相邻解释句，不把定义句冒充答案。"""
    manifest, xml = _fixture(tmp_path)
    database = tmp_path / "relation.sqlite"
    build_broad_qa_index(
        manifest, xml_path=xml, database_path=database,
        accepted_page_limit=3,
    )
    connection = sqlite3.connect(str(database))
    try:
        result = query_broad_qa(connection, "都江堰有什么作用？")
    finally:
        connection.close()
    assert result.status == "ANSWER"
    assert result.answer is not None
    assert "用于防洪和灌溉" in result.answer


def test_unanchored_entity_relation_query_fails_closed(
        tmp_path: Path) -> None:
    """未收录实体不得被关系词共现的弱相关页面冒答。"""
    manifest, xml = _fixture(tmp_path)
    database = tmp_path / "unanchored.sqlite3"
    build_broad_qa_index(
        manifest, xml_path=xml, database_path=database,
        accepted_page_limit=3,
    )
    connection = sqlite3.connect(str(database))
    try:
        result = query_broad_qa(
            connection, "谁主持修建未收录工程用于防洪？")
    finally:
        connection.close()
    assert result.status == "UNKNOWN"
    assert result.answer is None


def test_shared_block_is_read_once_without_changing_rank_selection(
        tmp_path: Path) -> None:
    """物理 offset 顺序可提速，但最终有效页面仍由冻结 hash ordinal 决定。"""
    payload = bz2.compress(
        _page(1, "低排名页", "低排名页包含足够长的第一段证据内容。")
        + _page(2, "高排名页", "高排名页同样包含足够长的第一段证据内容。")
    )
    xml = tmp_path / "shared.xml.bz2"
    xml.write_bytes(payload)
    pages = (
        BroadQaSelectedPage(
            1, "1" * 64, "高排名页",
            hashlib.sha256("高排名页".encode("utf-8")).hexdigest(),
            2, 2, 0, len(payload),
        ),
        BroadQaSelectedPage(
            2, "2" * 64, "低排名页",
            hashlib.sha256("低排名页".encode("utf-8")).hexdigest(),
            1, 1, 0, len(payload),
        ),
    )
    manifest = BroadQaSelectionManifest(
        "ZHWIKIPEDIA_20260701", "synthetic-shared", "a" * 64,
        "b" * 64, "c" * 40, "d" * 64, len(payload), 2, 2, pages,
    )
    profile = profile_broad_qa_selection(manifest)
    assert profile["compressed_block_count"] == 1
    assert profile["reused_candidate_block_count"] == 1
    assert profile["compressed_bytes_read"] == len(payload)
    database = tmp_path / "shared.sqlite"
    report = build_broad_qa_index(
        manifest, xml_path=xml, database_path=database,
        accepted_page_limit=1,
    )
    connection = sqlite3.connect(str(database))
    try:
        titles = tuple(connection.execute(
            "SELECT title FROM document ORDER BY doc_id"))
    finally:
        connection.close()
    assert report["compressed_block_count"] == 1
    assert report["compressed_bytes_read"] == len(payload)
    assert titles == (("高排名页",),)


def test_one_and_four_workers_build_bit_identical_database(
        tmp_path: Path) -> None:
    """worker 数只影响执行调度，不得改变紧凑 artifact 任一字节。"""
    manifest, xml = _fixture(tmp_path)
    single = tmp_path / "single.sqlite"
    parallel = tmp_path / "parallel.sqlite"
    report_single = build_broad_qa_index(
        manifest, xml_path=xml, database_path=single,
        accepted_page_limit=3, worker_count=1,
    )
    report_parallel = build_broad_qa_index(
        manifest, xml_path=xml, database_path=parallel,
        accepted_page_limit=3, worker_count=4,
    )
    assert report_single["worker_count"] == 1
    assert report_parallel["worker_count"] == 4
    assert report_single["database_sha256"] == report_parallel[
        "database_sha256"]
    assert single.read_bytes() == parallel.read_bytes()


def test_sharded_fresh_resume_and_worker_outputs_are_bit_identical(
        tmp_path: Path) -> None:
    """projection 中断可恢复，fresh/resume 与 1/4 worker 最终逐字节相同。"""
    manifest, xml = _fixture(tmp_path)
    resumed_root = tmp_path / "resumed-shards"
    partial = build_broad_qa_projection_shards(
        manifest,
        xml_path=xml,
        shard_root=resumed_root,
        max_blocks_per_shard=1,
        worker_count=4,
        max_new_shards=1,
    )
    assert partial["status"] == "INCOMPLETE"
    assert partial["completed_shard_count"] == 1
    resumed = tmp_path / "resumed.sqlite3"
    resumed_report = build_broad_qa_sharded_index(
        manifest,
        xml_path=xml,
        shard_root=resumed_root,
        database_path=resumed,
        accepted_page_count=3,
        max_blocks_per_shard=1,
        worker_count=4,
    )
    fresh = tmp_path / "fresh.sqlite3"
    fresh_report = build_broad_qa_sharded_index(
        manifest,
        xml_path=xml,
        shard_root=tmp_path / "fresh-shards",
        database_path=fresh,
        accepted_page_count=3,
        max_blocks_per_shard=1,
        worker_count=1,
    )
    assert resumed_report["database_sha256"] == fresh_report["database_sha256"]
    assert resumed.read_bytes() == fresh.read_bytes()
    connection = sqlite3.connect(str(resumed))
    try:
        result = query_broad_qa(connection, "谁主持修建都江堰用于防洪？")
    finally:
        connection.close()
    assert result.status == "ANSWER"
    assert "李冰" in result.answer


def test_sharded_stage_budgets_resume_and_share_projection_across_targets(
        tmp_path: Path) -> None:
    """projection/posting 可分别限额恢复，2/3 页目标不得互相占用 segment。"""
    manifest, xml = _fixture(tmp_path)
    root = tmp_path / "shards"
    database = tmp_path / "three.sqlite3"
    projection = build_broad_qa_sharded_index(
        manifest,
        xml_path=xml,
        shard_root=root,
        database_path=database,
        accepted_page_count=3,
        max_blocks_per_shard=1,
        max_new_projection_shards=1,
    )
    assert projection["status"] == "PROJECTION_INCOMPLETE"
    posting = build_broad_qa_sharded_index(
        manifest,
        xml_path=xml,
        shard_root=root,
        database_path=database,
        accepted_page_count=3,
        max_blocks_per_shard=1,
        max_new_posting_shards=1,
        publish=False,
    )
    assert posting["status"] == "POSTING_INCOMPLETE"
    ready = build_broad_qa_sharded_index(
        manifest,
        xml_path=xml,
        shard_root=root,
        database_path=database,
        accepted_page_count=3,
        max_blocks_per_shard=1,
        publish=False,
    )
    assert ready["status"] == "READY_TO_PUBLISH"
    complete = build_broad_qa_sharded_index(
        manifest,
        xml_path=xml,
        shard_root=root,
        database_path=database,
        accepted_page_count=3,
        max_blocks_per_shard=1,
    )
    assert complete["status"] == "COMPLETE"
    assert complete == build_broad_qa_sharded_index(
        manifest,
        xml_path=xml,
        shard_root=root,
        database_path=database,
        accepted_page_count=3,
        max_blocks_per_shard=1,
    )
    smaller = tmp_path / "two.sqlite3"
    smaller_report = build_broad_qa_sharded_index(
        manifest,
        xml_path=xml,
        shard_root=root,
        database_path=smaller,
        accepted_page_count=2,
        max_blocks_per_shard=1,
    )
    assert smaller_report["accepted_page_count"] == 2
    assert smaller_report["database_sha256"] != complete["database_sha256"]
    assert (root / "targets" / "pages-000000002").is_dir()
    assert (root / "targets" / "pages-000000003").is_dir()


def test_sharded_target_projection_stops_only_after_cutoff_is_closed(
        tmp_path: Path) -> None:
    """目标第 N 页早于下一未处理 shard 时可发布，较大目标仍需继续读取。"""
    manifest, xml = _fixture(tmp_path)
    root = tmp_path / "target-shards"
    one = tmp_path / "one.sqlite3"
    report = build_broad_qa_sharded_index(
        manifest,
        xml_path=xml,
        shard_root=root,
        database_path=one,
        accepted_page_count=1,
        max_blocks_per_shard=1,
    )
    assert report["status"] == "COMPLETE"
    assert report["active_projection_shard_count"] == 1
    assert report["completed_projection_shard_count"] == 1
    assert not (root / "projection-000002.sqlite3").exists()
    connection = sqlite3.connect(str(one))
    try:
        assert connection.execute("SELECT count(*) FROM document").fetchone() == (1,)
    finally:
        connection.close()
    three = tmp_path / "three-after-one.sqlite3"
    report = build_broad_qa_sharded_index(
        manifest,
        xml_path=xml,
        shard_root=root,
        database_path=three,
        accepted_page_count=3,
        max_blocks_per_shard=1,
    )
    assert report["active_projection_shard_count"] == 3
    assert report["completed_projection_shard_count"] == 3


def test_sharded_partial_requires_explicit_discard_and_then_recovers(
        tmp_path: Path) -> None:
    """未知 partial 默认 fail closed，显式恢复仅丢弃确定的单文件后重建。"""
    manifest, xml = _fixture(tmp_path)
    root = tmp_path / "shards"
    root.mkdir()
    partial = root / "projection-000001.sqlite3.partial"
    partial.write_bytes(b"interrupted")
    with pytest.raises(BroadQaShardedError):
        build_broad_qa_projection_shards(
            manifest,
            xml_path=xml,
            shard_root=root,
            max_blocks_per_shard=1,
            max_new_shards=1,
        )
    report = build_broad_qa_projection_shards(
        manifest,
        xml_path=xml,
        shard_root=root,
        max_blocks_per_shard=1,
        max_new_shards=1,
        discard_unsealed=True,
    )
    assert report["completed_shard_count"] == 1
    assert not partial.exists()


@pytest.mark.parametrize("artifact", ("database", "receipt", "posting"))
def test_sharded_resume_rejects_projection_and_posting_tampering(
        tmp_path: Path, artifact: str) -> None:
    """sealed projection 数据库/receipt 与 posting segment 任一漂移均拒绝恢复。"""
    manifest, xml = _fixture(tmp_path)
    root = tmp_path / artifact
    database = tmp_path / f"{artifact}.sqlite3"
    build_broad_qa_sharded_index(
        manifest,
        xml_path=xml,
        shard_root=root,
        database_path=database,
        accepted_page_count=3,
        max_blocks_per_shard=1,
        publish=False,
    )
    if artifact == "database":
        target = root / "projection-000001.sqlite3"
    elif artifact == "receipt":
        target = root / "projection-000001.receipt.json"
    else:
        target = (root / "targets" / "pages-000000003"
                  / "posting-000001.sqlite3")
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(BroadQaShardedError):
        build_broad_qa_sharded_index(
            manifest,
            xml_path=xml,
            shard_root=root,
            database_path=database,
            accepted_page_count=3,
            max_blocks_per_shard=1,
        )


def test_sharded_publication_partial_is_targeted_and_explicitly_recoverable(
        tmp_path: Path) -> None:
    """publication 中断不被静默覆盖，授权后可由已封存 shards 重建。"""
    manifest, xml = _fixture(tmp_path)
    root = tmp_path / "shards"
    database = tmp_path / "published.sqlite3"
    ready = build_broad_qa_sharded_index(
        manifest,
        xml_path=xml,
        shard_root=root,
        database_path=database,
        accepted_page_count=3,
        max_blocks_per_shard=1,
        publish=False,
    )
    assert ready["status"] == "READY_TO_PUBLISH"
    partial = database.with_suffix(".sqlite3.partial")
    partial.write_bytes(b"interrupted")
    with pytest.raises(BroadQaShardedError):
        build_broad_qa_sharded_index(
            manifest,
            xml_path=xml,
            shard_root=root,
            database_path=database,
            accepted_page_count=3,
            max_blocks_per_shard=1,
        )
    report = build_broad_qa_sharded_index(
        manifest,
        xml_path=xml,
        shard_root=root,
        database_path=database,
        accepted_page_count=3,
        max_blocks_per_shard=1,
        discard_unsealed=True,
    )
    assert report["status"] == "COMPLETE"
    assert not partial.exists()


def test_build_sharded_cli_exposes_bounded_resume_controls() -> None:
    """长构建的 projection/posting/publish/recovery 控制必须可由 CLI 驱动。"""
    root = Path("K:/broad-qa-test")
    args = _parser().parse_args([
        "build-sharded",
        "--run-root", str(root),
        "--snapshot-manifest", "snapshot.json",
        "--index", str(root / "index.txt.bz2"),
        "--selection", str(root / "selection.json"),
        "--candidate-count", "100000",
        "--xml", str(root / "wiki.xml.bz2"),
        "--database", str(root / "index.sqlite3"),
        "--page-count", "100000",
        "--shard-root", str(root / "shards"),
        "--max-new-projection-shards", "2",
        "--max-new-posting-shards", "3",
        "--no-publish",
        "--discard-unsealed",
    ])
    assert args.max_new_projection_shards == 2
    assert args.max_new_posting_shards == 3
    assert args.no_publish is True
    assert args.discard_unsealed is True


def test_question_slot_artifact_is_external_canonical_and_tamper_evident(
        tmp_path: Path) -> None:
    """问式开放类 surface 来自独立 artifact，SHA 漂移必须 fail closed。"""
    slots = load_broad_qa_question_slots()
    assert {"多少", "何时", "哪些", "是谁"}.issubset(
        set(slots.surfaces))
    assert "矮寨大桥" in slots.strip_slots("矮寨大桥何时通车？")
    assert "何时" not in slots.strip_slots("矮寨大桥何时通车？")
    assert "地区" in slots.strip_slots("黄山松分布在哪些地区？")
    tampered = tmp_path / "slots.json"
    source = Path(
        "data/ph2/broad_qa_question_slots_v1.json").resolve()
    tampered.write_bytes(source.read_bytes().replace(
        "多少".encode("utf-8"), "几多".encode("utf-8"), 1))
    with pytest.raises(BroadQaQuestionSlotError):
        load_broad_qa_question_slots(tampered)


def test_public_dev_question_artifact_is_canonical_and_not_held_out() -> None:
    """公开问题集明确是开发探针，不能被后续报告冒充 held-out。"""
    source = Path("data/ph2/broad_qa_dev_questions_v1.json").resolve()
    payload = source.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    assert canonical_json_bytes(value) + b"\n" == payload
    assert value["scope"] == "DEVELOPMENT_VERTICAL_PROBE_NOT_HELD_OUT"
    assert len(value["questions"]) == len(set(value["questions"])) == 24


def test_public_10k_receipt_does_not_claim_week_minimum_pass() -> None:
    """10k 机器证据可公开复核，但不得越级声明 100k 周最低门通过。"""
    source = Path(
        "data/ph2/broad_qa_10k_dev_preview_receipt_v1.json").resolve()
    payload = source.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    assert canonical_json_bytes(value) + b"\n" == payload
    assert value["scope"] == "DEVELOPMENT_VERTICAL_PROBE_NOT_HELD_OUT"
    assert value["status"] == (
        "DAY_1_VERTICAL_SLICE_RUNTIME_EVIDENCED_NOT_WEEK_MINIMUM_PASS")
    assert value["build"]["accepted_page_count"] == 10_000
    assert value["probe"]["citation_audit_failure_count"] == 0


def test_public_20k_receipt_is_canonical_and_explicitly_bounded() -> None:
    """20k receipt 的字节、身份和边界字段必须可机械复核。"""
    source = Path(
        "data/ph2/broad_qa_20k_preview_receipt_v1.json").resolve()
    payload = source.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    assert canonical_json_bytes(value) + b"\n" == payload
    assert value["artifact_kind"] == "PH2_BROAD_QA_20K_PREVIEW_RECEIPT_V1"
    assert value["status"] == (
        "SOURCE_BOUND_EXTRACTIVE_PREVIEW_NOT_HELD_OUT_NOT_GENERAL_QA")
    assert value["candidate_page_count"] == 100_000
    assert value["accepted_page_count"] == 20_000
    assert value["passage_count"] == 109_006
    assert value["term_count"] == 3_608_002
    assert value["database_bytes"] == 251_494_400
    assert len(value["database_sha256"]) == 64
    assert value["citation_probe"]["scope"] == (
        "DEVELOPMENT_VERTICAL_PROBE_NOT_HELD_OUT")
    assert value["citation_probe"]["citation_audit_failure_count"] == 0
