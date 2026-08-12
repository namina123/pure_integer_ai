"""为 100k/300k 广域问答构建可恢复 projection 与 posting 分片。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
import hashlib
import heapq
import json
from pathlib import Path
import sqlite3
import time

from pure_integer_ai.experiments.ph2_broad_qa_contract import (
    BroadQaSelectedPage,
    BroadQaSelectionManifest,
)
from pure_integer_ai.experiments.ph2_broad_qa_index import (
    INDEX_SCHEMA_VERSION,
    broad_qa_terms,
)
from pure_integer_ai.experiments.ph2_broad_qa_source import (
    iter_broad_qa_selected_pages,
    project_broad_qa_passages,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.v02_run_store import HostProcessMemory
from pure_integer_ai.storage.integer_codec import (
    decode_integer_tuple,
    encode_integer_tuple,
)


PLAN_KIND = "PH2_BROAD_QA_SHARD_PLAN_V2"
PROJECTION_RECEIPT_KIND = "PH2_BROAD_QA_PROJECTION_SHARD_RECEIPT_V1"
POSTING_RECEIPT_KIND = "PH2_BROAD_QA_POSTING_SHARD_RECEIPT_V1"
PUBLICATION_RECEIPT_KIND = "PH2_BROAD_QA_SHARDED_PUBLICATION_RECEIPT_V1"


# object-model: exception
class BroadQaShardedError(RuntimeError):
    """分片计划、checkpoint、segment 或最终合并不满足恢复合同。"""


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaShardSpec:
    """一个物理 block 唯一分配的 projection shard 计划。"""

    shard_id: int
    selected_pages: tuple[BroadQaSelectedPage, ...]
    block_count: int
    compressed_bytes: int
    page_inventory_sha256: str

    def __post_init__(self) -> None:
        """核验 shard id、页面唯一性、block 预算和 inventory 承诺。"""
        if (type(self.shard_id) is not int or self.shard_id <= 0
                or not self.selected_pages
                or type(self.block_count) is not int or self.block_count <= 0
                or type(self.compressed_bytes) is not int
                or self.compressed_bytes <= 0
                or len(self.page_inventory_sha256) != 64
                or len({item.ordinal for item in self.selected_pages})
                != len(self.selected_pages)):
            raise BroadQaShardedError("broad QA shard spec 非规范")

    def to_dict(self) -> dict[str, object]:
        """导出不重复 selection 页面明细的紧凑计划记录。"""
        return {
            "block_count": self.block_count,
            "compressed_bytes": self.compressed_bytes,
            "first_selection_ordinal": min(
                item.ordinal for item in self.selected_pages),
            "last_selection_ordinal": max(
                item.ordinal for item in self.selected_pages),
            "page_count": len(self.selected_pages),
            "page_inventory_sha256": self.page_inventory_sha256,
            "shard_id": self.shard_id,
        }


def _sha256_path(path: Path) -> str:
    """流式计算分片、segment 或最终数据库 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _memory_sample() -> tuple[int, int]:
    """读取当前工作集与进程峰值；宿主不支持时保留零证据。"""
    values = HostProcessMemory()()
    return (
        int(values.get("current_working_set_bytes", 0)),
        int(values.get("process_peak_working_set_bytes", 0)),
    )


def _canonical_line(value: object) -> bytes:
    """返回带单换行的 canonical JSON artifact。"""
    return canonical_json_bytes(value) + b"\n"


def _write_idempotent(path: Path, payload: bytes) -> None:
    """独占发布小 artifact；已存在时只接受逐字节相同内容。"""
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise BroadQaShardedError(f"artifact 已存在且漂移: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _read_canonical(path: Path, kind: str) -> dict[str, object]:
    """严格读取 canonical JSON object 并核验 artifact kind。"""
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaShardedError(f"artifact 不可读: {path.name}") from error
    if (not isinstance(value, dict) or value.get("artifact_kind") != kind
            or _canonical_line(value) != payload):
        raise BroadQaShardedError(f"artifact 非规范: {path.name}")
    return value


def _inventory_sha256(pages: tuple[BroadQaSelectedPage, ...]) -> str:
    """承诺 shard 中按全局 ordinal 排序的完整页面身份与坐标。"""
    value = [item.to_dict() for item in sorted(
        pages, key=lambda item: item.ordinal)]
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def make_broad_qa_shard_specs(
        selection: BroadQaSelectionManifest,
        *,
        max_blocks_per_shard: int,
        ) -> tuple[BroadQaShardSpec, ...]:
    """按物理 block offset 唯一分配页面，并形成有界稳定 shard。"""
    if not isinstance(selection, BroadQaSelectionManifest):
        raise TypeError("broad QA selection 类型错误")
    if (type(max_blocks_per_shard) is not int
            or not 1 <= max_blocks_per_shard <= 4096):
        raise BroadQaShardedError("max blocks per shard 非法")
    ordered = sorted(
        selection.selected_pages,
        key=lambda item: (
            item.compressed_block_offset,
            item.compressed_block_end_offset,
            item.ordinal,
        ),
    )
    physical_block_groups = tuple(
        (key, tuple(group)) for key, group in groupby(
            ordered,
            key=lambda item: (
                item.compressed_block_offset,
                item.compressed_block_end_offset,
            ),
        )
    )
    block_groups = tuple(sorted(
        physical_block_groups,
        key=lambda item: (
            min(page.ordinal for page in item[1]),
            item[0][0],
            item[0][1],
        ),
    ))
    specs = []
    for start in range(0, len(block_groups), max_blocks_per_shard):
        groups = block_groups[start:start + max_blocks_per_shard]
        pages = tuple(
            item for _, values in groups for item in values)
        specs.append(BroadQaShardSpec(
            len(specs) + 1,
            pages,
            len(groups),
            sum(end - offset for (offset, end), _ in groups),
            _inventory_sha256(pages),
        ))
    if (not specs
            or sum(item.block_count for item in specs) != len(block_groups)
            or sum(len(item.selected_pages) for item in specs)
            != len(selection.selected_pages)):
        raise BroadQaShardedError("broad QA shard plan inventory 漂移")
    return tuple(specs)


def _plan_value(
        selection: BroadQaSelectionManifest,
        specs: tuple[BroadQaShardSpec, ...],
        *,
        max_blocks_per_shard: int,
        ) -> dict[str, object]:
    """形成只绑定选择与分片算法的 canonical plan。"""
    return {
        "artifact_kind": PLAN_KIND,
        "format_version": 2,
        "max_blocks_per_shard": max_blocks_per_shard,
        "schedule": "MIN_SELECTION_ORDINAL_BLOCK_V1",
        "selection_sha256": selection.sha256(),
        "shard_count": len(specs),
        "shards": [item.to_dict() for item in specs],
        "source_key": selection.source_key,
    }


def _database_receipt_valid(
        database: Path,
        receipt: Path,
        *,
        kind: str,
        expected: dict[str, object],
        ) -> dict[str, object] | None:
    """只在 receipt 字段和数据库 SHA/size 全匹配时允许 resume。"""
    if not database.exists() and not receipt.exists():
        return None
    if not database.is_file() or not receipt.is_file():
        raise BroadQaShardedError("database/receipt 非成对存在")
    value = _read_canonical(receipt, kind)
    for key, item in expected.items():
        if value.get(key) != item:
            raise BroadQaShardedError(f"resume receipt 字段漂移: {key}")
    if (value.get("database_bytes") != database.stat().st_size
            or value.get("database_sha256") != _sha256_path(database)):
        raise BroadQaShardedError("resume database identity 漂移")
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
    finally:
        connection.close()
    if integrity != ("ok",):
        raise BroadQaShardedError("resume database integrity 失败")
    return value


def _partial_path(database: Path) -> Path:
    """返回仅属于给定数据库发布动作的固定 partial 路径。"""
    return database.with_suffix(database.suffix + ".partial")


def _discard_exact_file(path: Path) -> None:
    """只删除已解析出的单个普通文件，拒绝目录和链接。"""
    if path.is_symlink() or not path.is_file():
        raise BroadQaShardedError(f"未封存残留不是普通文件: {path.name}")
    path.unlink()


def _prepare_artifact_pair(
        database: Path,
        receipt: Path,
        *,
        discard_unsealed: bool,
        ) -> None:
    """检查 receipt-last 三件套；显式授权时仅丢弃确定的未封存文件。"""
    partial = _partial_path(database)
    database_exists = database.exists()
    receipt_exists = receipt.exists()
    partial_exists = partial.exists()
    if database_exists and receipt_exists:
        if not partial_exists:
            return
        if not discard_unsealed:
            raise BroadQaShardedError("sealed artifact 旁存在 partial 残留")
        _discard_exact_file(partial)
        return
    if not database_exists and not receipt_exists and not partial_exists:
        return
    if not discard_unsealed:
        raise BroadQaShardedError("artifact 存在未封存残留")
    for path in (partial, database, receipt):
        if path.exists():
            _discard_exact_file(path)


def _projection_paths(root: Path, shard_id: int) -> tuple[Path, Path]:
    """返回一个 projection shard 的数据库和 receipt 路径。"""
    stem = f"projection-{shard_id:06d}"
    return root / f"{stem}.sqlite3", root / f"{stem}.receipt.json"


def _target_root(root: Path, accepted_page_count: int) -> Path:
    """按发布规模隔离 cutoff 相关 posting，允许 projection 跨规模复用。"""
    return root / "targets" / f"pages-{accepted_page_count:09d}"


def _posting_paths(target_root: Path, shard_id: int) -> tuple[Path, Path]:
    """返回一个目标规模内 posting shard 的数据库和 receipt 路径。"""
    stem = f"posting-{shard_id:06d}"
    return (
        target_root / f"{stem}.sqlite3",
        target_root / f"{stem}.receipt.json",
    )


def _build_projection_shard(
        selection: BroadQaSelectionManifest,
        spec: BroadQaShardSpec,
        *,
        xml_path: Path,
        database: Path,
        receipt: Path,
        plan_sha256: str,
        worker_count: int,
        discard_unsealed: bool,
        ) -> dict[str, object]:
    """不可覆盖地构建单个 source/projection shard，并最后发布 receipt。"""
    expected = {
        "page_inventory_sha256": spec.page_inventory_sha256,
        "plan_sha256": plan_sha256,
        "selection_sha256": selection.sha256(),
        "shard_id": spec.shard_id,
        "worker_count": worker_count,
    }
    _prepare_artifact_pair(
        database, receipt, discard_unsealed=discard_unsealed)
    reused = _database_receipt_valid(
        database, receipt, kind=PROJECTION_RECEIPT_KIND, expected=expected)
    if reused is not None:
        return reused
    partial = _partial_path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    started_ns = time.perf_counter_ns()
    rss_before_bytes, peak_before_bytes = _memory_sample()
    connection = sqlite3.connect(str(partial))
    eligible = 0
    projected = 0
    passage_count = 0
    source_elapsed_ns = 0
    projection_elapsed_ns = 0
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.executescript("""
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE document(
                doc_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                page_id INTEGER NOT NULL UNIQUE,
                revision_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                contributor_json TEXT NOT NULL,
                text_sha256 TEXT NOT NULL
            );
            CREATE TABLE passage(
                passage_id INTEGER PRIMARY KEY,
                doc_id INTEGER NOT NULL,
                ordinal INTEGER NOT NULL,
                raw_start INTEGER NOT NULL,
                raw_end INTEGER NOT NULL,
                raw_sha256 TEXT NOT NULL,
                text TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                section_title TEXT NOT NULL
            );
            CREATE INDEX passage_doc ON passage(doc_id, ordinal);
        """)
        pages = iter(iter_broad_qa_selected_pages(
                spec.selected_pages,
                xml_path=xml_path,
                source_key=selection.source_key,
                xml_compressed_size_bytes=selection.xml_compressed_size_bytes,
                worker_count=worker_count))
        while True:
            source_started_ns = time.perf_counter_ns()
            try:
                page = next(pages)
            except StopIteration:
                source_elapsed_ns += max(
                    1, time.perf_counter_ns() - source_started_ns)
                break
            source_elapsed_ns += max(
                1, time.perf_counter_ns() - source_started_ns)
            projection_started_ns = time.perf_counter_ns()
            eligible += 1
            passages = project_broad_qa_passages(page.wikitext)
            if not passages:
                projection_elapsed_ns += max(
                    1, time.perf_counter_ns() - projection_started_ns)
                continue
            projected += 1
            connection.execute(
                "INSERT INTO document VALUES(?,?,?,?,?,?,?)",
                (page.ordinal, page.title, page.page_id, page.revision_id,
                 page.timestamp, page.contributor_json, page.text_sha256),
            )
            for passage in passages:
                passage_id = page.ordinal * 128 + passage.ordinal
                connection.execute(
                    "INSERT INTO passage VALUES(?,?,?,?,?,?,?,?,?)",
                    (passage_id, page.ordinal, passage.ordinal,
                     passage.raw_start, passage.raw_end, passage.raw_sha256,
                     passage.text, passage.text_sha256,
                     passage.section_title),
                )
                passage_count += 1
            projection_elapsed_ns += max(
                1, time.perf_counter_ns() - projection_started_ns)
        metadata = {
            "page_inventory_sha256": spec.page_inventory_sha256,
            "plan_sha256": plan_sha256,
            "selection_sha256": selection.sha256(),
            "shard_id": str(spec.shard_id),
        }
        connection.executemany(
            "INSERT INTO metadata VALUES(?,?)", sorted(metadata.items()))
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()
    partial.replace(database)
    rss_after_bytes, peak_after_bytes = _memory_sample()
    value = {
        "artifact_kind": PROJECTION_RECEIPT_KIND,
        "database_bytes": database.stat().st_size,
        "database_sha256": _sha256_path(database),
        "eligible_page_count": eligible,
        "elapsed_ns": max(1, time.perf_counter_ns() - started_ns),
        "format_version": 1,
        "page_inventory_sha256": spec.page_inventory_sha256,
        "passage_count": passage_count,
        "plan_sha256": plan_sha256,
        "process_peak_working_set_bytes": max(
            peak_before_bytes, peak_after_bytes),
        "projected_page_count": projected,
        "projection_elapsed_ns": projection_elapsed_ns,
        "rss_after_bytes": rss_after_bytes,
        "rss_before_bytes": rss_before_bytes,
        "selected_page_count": len(spec.selected_pages),
        "selection_sha256": selection.sha256(),
        "shard_id": spec.shard_id,
        "source_elapsed_ns": source_elapsed_ns,
        "worker_count": worker_count,
    }
    _write_idempotent(receipt, _canonical_line(value))
    return value


def build_broad_qa_projection_shards(
        selection: BroadQaSelectionManifest,
        *,
        xml_path: str | Path,
        shard_root: str | Path,
        max_blocks_per_shard: int = 512,
        worker_count: int = 4,
        max_new_shards: int | None = None,
        accepted_page_target: int | None = None,
        discard_unsealed: bool = False,
        ) -> dict[str, object]:
    """发布或复用有序 projection shards，并在 accepted cutoff 封闭时早停。"""
    if worker_count not in {1, 2, 4}:
        raise BroadQaShardedError("worker count 只能为 1/2/4")
    if (max_new_shards is not None
            and (type(max_new_shards) is not int or max_new_shards <= 0)):
        raise BroadQaShardedError("max new shards 非法")
    if (accepted_page_target is not None
            and (type(accepted_page_target) is not int
                 or accepted_page_target <= 0
                 or accepted_page_target > len(selection.selected_pages))):
        raise BroadQaShardedError("accepted page target 非法")
    root = Path(shard_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    specs = make_broad_qa_shard_specs(
        selection, max_blocks_per_shard=max_blocks_per_shard)
    plan = _plan_value(
        selection, specs, max_blocks_per_shard=max_blocks_per_shard)
    plan_payload = _canonical_line(plan)
    plan_sha256 = hashlib.sha256(plan_payload).hexdigest()
    _write_idempotent(root / "shard-plan.json", plan_payload)
    built = 0
    receipts = []
    target_state = None
    accepted_heap: list[int] = []
    for spec in specs:
        database, receipt = _projection_paths(root, spec.shard_id)
        existed = database.exists() and receipt.exists()
        if not existed and max_new_shards is not None and built >= max_new_shards:
            break
        value = _build_projection_shard(
            selection,
            spec,
            xml_path=Path(xml_path).resolve(),
            database=database,
            receipt=receipt,
            plan_sha256=plan_sha256,
            worker_count=worker_count,
            discard_unsealed=discard_unsealed,
        )
        receipts.append(value)
        if not existed:
            built += 1
        if accepted_page_target is not None:
            connection = sqlite3.connect(
                f"file:{database.as_posix()}?mode=ro", uri=True)
            try:
                ordinals = connection.execute(
                    "SELECT doc_id FROM document ORDER BY doc_id")
                for row in ordinals:
                    ordinal = int(row[0])
                    if len(accepted_heap) < accepted_page_target:
                        heapq.heappush(accepted_heap, -ordinal)
                    elif ordinal < -accepted_heap[0]:
                        heapq.heapreplace(accepted_heap, -ordinal)
            finally:
                connection.close()
            if len(accepted_heap) == accepted_page_target:
                cutoff = -accepted_heap[0]
                processed = len(receipts)
                if (processed == len(specs)
                        or cutoff < min(
                            item.ordinal
                            for item in specs[processed].selected_pages)):
                    target_state = (cutoff, sum(
                        int(item["projected_page_count"])
                        for item in receipts))
                    break
    completed = sum(
        int(_projection_paths(root, spec.shard_id)[0].is_file()
            and _projection_paths(root, spec.shard_id)[1].is_file())
        for spec in specs
    )
    status = "COMPLETE" if completed == len(specs) else "INCOMPLETE"
    if target_state is not None:
        status = "TARGET_COMPLETE"
    return {
        "active_shard_count": len(receipts),
        "accepted_cutoff_ordinal": (
            target_state[0] if target_state is not None else None),
        "completed_shard_count": completed,
        "database_bytes": sum(
            int(item["database_bytes"]) for item in receipts),
        "new_shard_count": built,
        "plan_sha256": plan_sha256,
        "process_peak_working_set_bytes": max(
            (int(item.get("process_peak_working_set_bytes", 0))
             for item in receipts), default=0),
        "projected_page_count": sum(
            int(item["projected_page_count"]) for item in receipts),
        "projection_elapsed_ns": sum(
            int(item.get("projection_elapsed_ns", 0))
            for item in receipts),
        "selected_page_count": sum(
            int(item["selected_page_count"]) for item in receipts),
        "shard_count": len(specs),
        "source_elapsed_ns": sum(
            int(item.get("source_elapsed_ns", 0)) for item in receipts),
        "status": status,
    }


def _accepted_cutoff(
        root: Path,
        specs: tuple[BroadQaShardSpec, ...],
        accepted_page_count: int,
        ) -> int:
    """按全局 selection ordinal 选出第 N 个可投影页面的 cutoff。"""
    ordinals = []
    for spec in specs:
        database, _ = _projection_paths(root, spec.shard_id)
        connection = sqlite3.connect(
            f"file:{database.as_posix()}?mode=ro", uri=True)
        try:
            ordinals.extend(
                row[0] for row in connection.execute(
                    "SELECT doc_id FROM document ORDER BY doc_id"))
        finally:
            connection.close()
    ordinals.sort()
    if len(ordinals) < accepted_page_count:
        raise BroadQaShardedError(
            "projection candidate 不足: "
            f"requested={accepted_page_count}, projected={len(ordinals)}")
    return ordinals[accepted_page_count - 1]


def _delta(values: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """把严格递增 passage id 转为正 delta。"""
    result = []
    prior = 0
    for value in values:
        if type(value) is not int or value <= prior:
            raise BroadQaShardedError("posting passage id 非严格递增")
        result.append(value - prior)
        prior = value
    return tuple(result)


def _restore(payload: bytes) -> tuple[int, ...]:
    """恢复局部 posting 的正 delta，并拒绝非递增结果。"""
    current = 0
    values = []
    for item in decode_integer_tuple(payload):
        if item <= 0:
            raise BroadQaShardedError("posting delta 非正")
        current += item
        values.append(current)
    return tuple(values)


def _build_posting_shard(
        spec: BroadQaShardSpec,
        *,
        root: Path,
        target_root: Path,
        cutoff_ordinal: int,
        plan_sha256: str,
        discard_unsealed: bool,
        ) -> dict[str, object]:
    """从 sealed projection 生成只含 cutoff 内 passage 的局部排序 posting。"""
    projection, projection_receipt = _projection_paths(root, spec.shard_id)
    projection_value = _read_canonical(
        projection_receipt, PROJECTION_RECEIPT_KIND)
    database, receipt = _posting_paths(target_root, spec.shard_id)
    expected = {
        "cutoff_ordinal": cutoff_ordinal,
        "plan_sha256": plan_sha256,
        "projection_database_sha256": projection_value["database_sha256"],
        "shard_id": spec.shard_id,
    }
    _prepare_artifact_pair(
        database, receipt, discard_unsealed=discard_unsealed)
    reused = _database_receipt_valid(
        database, receipt, kind=POSTING_RECEIPT_KIND, expected=expected)
    if reused is not None:
        return reused
    partial = _partial_path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    started_ns = time.perf_counter_ns()
    rss_before_bytes, peak_before_bytes = _memory_sample()
    source = sqlite3.connect(
        f"file:{projection.as_posix()}?mode=ro", uri=True)
    target = sqlite3.connect(str(partial))
    postings: dict[str, list[int]] = defaultdict(list)
    passage_count = 0
    try:
        target.execute("PRAGMA journal_mode=OFF")
        target.execute("PRAGMA synchronous=OFF")
        target.execute("""
            CREATE TABLE posting(
                term TEXT PRIMARY KEY,
                document_frequency INTEGER NOT NULL,
                passage_deltas BLOB NOT NULL
            )
        """)
        for passage_id, title, section_title, text in source.execute("""
                SELECT p.passage_id,d.title,p.section_title,p.text
                FROM passage AS p JOIN document AS d ON d.doc_id=p.doc_id
                WHERE d.doc_id<=? ORDER BY p.passage_id
                """, (cutoff_ordinal,)):
            passage_count += 1
            terms = set(broad_qa_terms(title))
            terms.update(broad_qa_terms(section_title))
            terms.update(broad_qa_terms(text))
            for term in terms:
                postings[term].append(passage_id)
        for term in sorted(postings):
            values = postings[term]
            target.execute(
                "INSERT INTO posting VALUES(?,?,?)",
                (term, len(values), encode_integer_tuple(_delta(values))),
            )
        target.commit()
        target.execute("VACUUM")
    finally:
        target.close()
        source.close()
    partial.replace(database)
    rss_after_bytes, peak_after_bytes = _memory_sample()
    value = {
        "artifact_kind": POSTING_RECEIPT_KIND,
        "cutoff_ordinal": cutoff_ordinal,
        "database_bytes": database.stat().st_size,
        "database_sha256": _sha256_path(database),
        "elapsed_ns": max(1, time.perf_counter_ns() - started_ns),
        "format_version": 1,
        "passage_count": passage_count,
        "plan_sha256": plan_sha256,
        "process_peak_working_set_bytes": max(
            peak_before_bytes, peak_after_bytes),
        "projection_database_sha256": projection_value["database_sha256"],
        "rss_after_bytes": rss_after_bytes,
        "rss_before_bytes": rss_before_bytes,
        "shard_id": spec.shard_id,
        "term_count": len(postings),
    }
    _write_idempotent(receipt, _canonical_line(value))
    return value


def _copy_projection(
        target: sqlite3.Connection,
        projection: Path,
        *,
        cutoff_ordinal: int,
        ) -> tuple[int, int]:
    """以稳定行序把 cutoff 内文档和段落复制到最终数据库。"""
    source = sqlite3.connect(
        f"file:{projection.as_posix()}?mode=ro", uri=True)
    document_count = 0
    passage_count = 0
    try:
        for row in source.execute(
                "SELECT * FROM document WHERE doc_id<=? ORDER BY doc_id",
                (cutoff_ordinal,)):
            target.execute("INSERT INTO document VALUES(?,?,?,?,?,?,?)", row)
            document_count += 1
        for row in source.execute("""
                SELECT p.* FROM passage AS p
                JOIN document AS d ON d.doc_id=p.doc_id
                WHERE d.doc_id<=? ORDER BY p.passage_id
                """, (cutoff_ordinal,)):
            target.execute("INSERT INTO passage VALUES(?,?,?,?,?,?,?,?,?)", row)
            passage_count += 1
    finally:
        source.close()
    return document_count, passage_count


def _copy_posting_parts(
        target: sqlite3.Connection,
        segment: Path,
        *,
        shard_id: int,
        ) -> None:
    """把局部 posting payload 复制到最终库的外排合并表。"""
    source = sqlite3.connect(
        f"file:{segment.as_posix()}?mode=ro", uri=True)
    try:
        for term, frequency, payload in source.execute(
                "SELECT term,document_frequency,passage_deltas "
                "FROM posting ORDER BY term"):
            target.execute(
                "INSERT INTO posting_part VALUES(?,?,?,?)",
                (term, shard_id, frequency, payload),
            )
    finally:
        source.close()


def _publish_final_database(
        selection: BroadQaSelectionManifest,
        specs: tuple[BroadQaShardSpec, ...],
        *,
        root: Path,
        target_root: Path,
        database: Path,
        cutoff_ordinal: int,
        accepted_page_count: int,
        plan_sha256: str,
        discard_unsealed: bool,
        ) -> dict[str, object]:
    """合并 sealed shards，按 term 外排 posting，并 receipt-last 发布。"""
    receipt = database.with_suffix(database.suffix + ".receipt.json")
    expected = {
        "accepted_page_count": accepted_page_count,
        "cutoff_ordinal": cutoff_ordinal,
        "plan_sha256": plan_sha256,
        "selection_sha256": selection.sha256(),
    }
    _prepare_artifact_pair(
        database, receipt, discard_unsealed=discard_unsealed)
    reused = _database_receipt_valid(
        database, receipt, kind=PUBLICATION_RECEIPT_KIND, expected=expected)
    if reused is not None:
        return reused
    partial = _partial_path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    started_ns = time.perf_counter_ns()
    rss_before_bytes, peak_before_bytes = _memory_sample()
    connection = sqlite3.connect(str(partial))
    document_count = 0
    passage_count = 0
    document_merge_elapsed_ns = 0
    posting_part_copy_elapsed_ns = 0
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.executescript("""
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE document(
                doc_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                page_id INTEGER NOT NULL UNIQUE,
                revision_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                contributor_json TEXT NOT NULL,
                text_sha256 TEXT NOT NULL
            );
            CREATE TABLE passage(
                passage_id INTEGER PRIMARY KEY,
                doc_id INTEGER NOT NULL,
                ordinal INTEGER NOT NULL,
                raw_start INTEGER NOT NULL,
                raw_end INTEGER NOT NULL,
                raw_sha256 TEXT NOT NULL,
                text TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                section_title TEXT NOT NULL
            );
            CREATE TABLE posting(
                term TEXT PRIMARY KEY,
                document_frequency INTEGER NOT NULL,
                passage_deltas BLOB NOT NULL
            );
            CREATE TABLE posting_part(
                term TEXT NOT NULL,
                shard_id INTEGER NOT NULL,
                document_frequency INTEGER NOT NULL,
                passage_deltas BLOB NOT NULL
            );
            CREATE INDEX passage_doc ON passage(doc_id, ordinal);
            CREATE INDEX document_title ON document(title);
        """)
        for spec in specs:
            projection, _ = _projection_paths(root, spec.shard_id)
            phase_started_ns = time.perf_counter_ns()
            docs, passages = _copy_projection(
                connection, projection, cutoff_ordinal=cutoff_ordinal)
            document_merge_elapsed_ns += max(
                1, time.perf_counter_ns() - phase_started_ns)
            document_count += docs
            passage_count += passages
            segment, _ = _posting_paths(target_root, spec.shard_id)
            phase_started_ns = time.perf_counter_ns()
            _copy_posting_parts(
                connection, segment, shard_id=spec.shard_id)
            posting_part_copy_elapsed_ns += max(
                1, time.perf_counter_ns() - phase_started_ns)
        if document_count != accepted_page_count:
            raise BroadQaShardedError("final document count 漂移")
        connection.execute(
            "CREATE INDEX posting_part_order ON posting_part(term,shard_id)")
        posting_merge_started_ns = time.perf_counter_ns()
        term_count = 0
        rows = connection.execute("""
            SELECT term,shard_id,document_frequency,passage_deltas
            FROM posting_part ORDER BY term,shard_id
        """)
        for term, group in groupby(rows, key=lambda item: item[0]):
            restored = []
            expected_frequency = 0
            for _, _, frequency, payload in group:
                values = _restore(payload)
                if len(values) != frequency:
                    raise BroadQaShardedError(
                        "posting segment frequency 漂移")
                expected_frequency += frequency
                restored.extend(values)
            restored.sort()
            if len(restored) != expected_frequency:
                raise BroadQaShardedError("merged posting frequency 漂移")
            connection.execute(
                "INSERT INTO posting VALUES(?,?,?)",
                (term, len(restored), encode_integer_tuple(_delta(restored))),
            )
            term_count += 1
        posting_merge_elapsed_ns = max(
            1, time.perf_counter_ns() - posting_merge_started_ns)
        connection.execute("DROP INDEX posting_part_order")
        connection.execute("DROP TABLE posting_part")
        metadata = {
            "accepted_page_count": str(accepted_page_count),
            "index_schema_version": str(INDEX_SCHEMA_VERSION),
            "license_id": "CC-BY-SA-4.0",
            "passage_count": str(passage_count),
            "selection_sha256": selection.sha256(),
            "snapshot_id": selection.snapshot_id,
            "source_key": selection.source_key,
            "term_count": str(term_count),
        }
        connection.executemany(
            "INSERT INTO metadata VALUES(?,?)", sorted(metadata.items()))
        finalize_started_ns = time.perf_counter_ns()
        connection.commit()
        connection.execute("VACUUM")
        finalize_elapsed_ns = max(
            1, time.perf_counter_ns() - finalize_started_ns)
    finally:
        connection.close()
    partial.replace(database)
    rss_after_bytes, peak_after_bytes = _memory_sample()
    value = {
        "accepted_page_count": accepted_page_count,
        "artifact_kind": PUBLICATION_RECEIPT_KIND,
        "cutoff_ordinal": cutoff_ordinal,
        "database_bytes": database.stat().st_size,
        "database_sha256": _sha256_path(database),
        "document_merge_elapsed_ns": document_merge_elapsed_ns,
        "elapsed_ns": max(1, time.perf_counter_ns() - started_ns),
        "finalize_elapsed_ns": finalize_elapsed_ns,
        "format_version": 1,
        "passage_count": passage_count,
        "plan_sha256": plan_sha256,
        "posting_merge_elapsed_ns": posting_merge_elapsed_ns,
        "posting_part_copy_elapsed_ns": posting_part_copy_elapsed_ns,
        "process_peak_working_set_bytes": max(
            peak_before_bytes, peak_after_bytes),
        "rss_after_bytes": rss_after_bytes,
        "rss_before_bytes": rss_before_bytes,
        "selection_sha256": selection.sha256(),
        "shard_count": len(specs),
        "term_count": term_count,
    }
    _write_idempotent(receipt, _canonical_line(value))
    return value


def build_broad_qa_sharded_index(
        selection: BroadQaSelectionManifest,
        *,
        xml_path: str | Path,
        shard_root: str | Path,
        database_path: str | Path,
        accepted_page_count: int,
        max_blocks_per_shard: int = 512,
        worker_count: int = 4,
        max_new_projection_shards: int | None = None,
        max_new_posting_shards: int | None = None,
        publish: bool = True,
        discard_unsealed: bool = False,
        ) -> dict[str, object]:
    """按预算推进 projection、posting 和 publication，并返回精确恢复状态。"""
    if (type(accepted_page_count) is not int
            or accepted_page_count <= 0
            or accepted_page_count > len(selection.selected_pages)):
        raise BroadQaShardedError("accepted page count 非法")
    if (max_new_posting_shards is not None
            and (type(max_new_posting_shards) is not int
                 or max_new_posting_shards <= 0)):
        raise BroadQaShardedError("max new posting shards 非法")
    if type(publish) is not bool or type(discard_unsealed) is not bool:
        raise BroadQaShardedError("sharded build 开关类型错误")
    projection = build_broad_qa_projection_shards(
        selection,
        xml_path=xml_path,
        shard_root=shard_root,
        max_blocks_per_shard=max_blocks_per_shard,
        worker_count=worker_count,
        max_new_shards=max_new_projection_shards,
        accepted_page_target=accepted_page_count,
        discard_unsealed=discard_unsealed,
    )
    if projection["status"] not in {"COMPLETE", "TARGET_COMPLETE"}:
        return {
            **projection,
            "accepted_page_count": accepted_page_count,
            "stage": "PROJECTION",
            "status": "PROJECTION_INCOMPLETE",
        }
    root = Path(shard_root).resolve()
    all_specs = make_broad_qa_shard_specs(
        selection, max_blocks_per_shard=max_blocks_per_shard)
    active_shard_count = int(projection["active_shard_count"])
    specs = all_specs[:active_shard_count]
    cutoff = projection["accepted_cutoff_ordinal"]
    if cutoff is None:
        cutoff = _accepted_cutoff(root, specs, accepted_page_count)
    target_root = _target_root(root, accepted_page_count)
    new_posting_count = 0
    posting_receipts = []
    for spec in specs:
        posting_database, posting_receipt = _posting_paths(
            target_root, spec.shard_id)
        existed = posting_database.exists() and posting_receipt.exists()
        if (not existed and max_new_posting_shards is not None
                and new_posting_count >= max_new_posting_shards):
            continue
        posting_receipts.append(_build_posting_shard(
            spec,
            root=root,
            target_root=target_root,
            cutoff_ordinal=cutoff,
            plan_sha256=projection["plan_sha256"],
            discard_unsealed=discard_unsealed,
        ))
        if not existed:
            new_posting_count += 1
    completed_posting_count = sum(
        int(_posting_paths(target_root, spec.shard_id)[0].is_file()
            and _posting_paths(target_root, spec.shard_id)[1].is_file())
        for spec in specs
    )
    stage_report = {
        "completed_projection_shard_count": projection[
            "completed_shard_count"],
        "accepted_page_count": accepted_page_count,
        "completed_posting_shard_count": completed_posting_count,
        "cutoff_ordinal": cutoff,
        "new_posting_shard_count": new_posting_count,
        "plan_sha256": projection["plan_sha256"],
        "posting_database_bytes": sum(
            int(item["database_bytes"]) for item in posting_receipts),
        "posting_elapsed_ns": sum(
            int(item["elapsed_ns"]) for item in posting_receipts),
        "process_peak_working_set_bytes": max(
            int(projection.get("process_peak_working_set_bytes", 0)),
            max((int(item.get("process_peak_working_set_bytes", 0))
                 for item in posting_receipts), default=0)),
        "projected_page_count": projection["projected_page_count"],
        "projection_database_bytes": projection["database_bytes"],
        "projection_elapsed_ns": projection["projection_elapsed_ns"],
        "posting_shard_count": len(specs),
        "projection_shard_count": projection["shard_count"],
        "active_projection_shard_count": active_shard_count,
        "selected_page_count": projection["selected_page_count"],
        "source_elapsed_ns": projection["source_elapsed_ns"],
    }
    if completed_posting_count != len(specs):
        return {
            **stage_report,
            "stage": "POSTING",
            "status": "POSTING_INCOMPLETE",
        }
    if not publish:
        return {
            **stage_report,
            "stage": "PUBLICATION",
            "status": "READY_TO_PUBLISH",
        }
    publication = _publish_final_database(
        selection,
        specs,
        root=root,
        target_root=target_root,
        database=Path(database_path).resolve(),
        cutoff_ordinal=cutoff,
        accepted_page_count=accepted_page_count,
        plan_sha256=projection["plan_sha256"],
        discard_unsealed=discard_unsealed,
    )
    return {
        **publication,
        **stage_report,
        "stage": "COMPLETE",
        "status": "COMPLETE",
    }


__all__ = [
    "BroadQaShardSpec",
    "BroadQaShardedError",
    "build_broad_qa_projection_shards",
    "build_broad_qa_sharded_index",
    "make_broad_qa_shard_specs",
]
