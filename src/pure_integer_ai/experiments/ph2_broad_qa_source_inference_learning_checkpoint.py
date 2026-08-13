"""来源归纳 learner 的 append-only checkpoint 与等价恢复合同。

checkpoint 只保存固定协议、单一 operator family、逻辑 cursor、已处理前缀和
输出计数；不保存 payload，不读取 validation，也不以墙钟或随机数参与身份。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_learning_protocol import (
    SOURCE_INFERENCE_LEARNING_FAMILIES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
    parse_canonical_json_bytes,
)


SOURCE_INFERENCE_LEARNING_CHECKPOINT_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_LEARNING_CHECKPOINT_V2")
SOURCE_INFERENCE_LEARNING_CHECKPOINT_OPEN = "OPEN"
SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE = "COMPLETE"
SOURCE_INFERENCE_LEARNING_CHECKPOINT_STATES = (
    SOURCE_INFERENCE_LEARNING_CHECKPOINT_OPEN,
    SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE,
)
_EMPTY_PREFIX_SHA256 = hashlib.sha256(b"").hexdigest()


def _sha256(value: object, *, label: str, empty: bool = False) -> str:
    """要求 checkpoint 承诺为 SHA-256，可按字段允许空链指针。"""
    if empty and value == "":
        return value
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise BroadQaExternalDataError(f"{label} 必须是 SHA-256")
    return value


def source_inference_learning_prefix_sha256(
        item_ids: tuple[str, ...],
        ) -> str:
    """对有序已处理 item identity 前缀计算无歧义摘要。"""
    if not isinstance(item_ids, tuple):
        raise BroadQaExternalDataError("learning prefix 必须是 tuple")
    digest = hashlib.sha256()
    for item_id in item_ids:
        _sha256(item_id, label="learning prefix item_id")
        digest.update(bytes.fromhex(item_id))
    return digest.hexdigest()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaSourceInferenceLearningCheckpoint:
    """一个可规范回读、只靠逻辑序推进的 learner snapshot。"""

    run_id: str
    protocol_manifest_sha256: str
    operator_family: str
    training_item_count: int
    training_item_order_sha256: str
    revision: int
    previous_checkpoint_sha256: str
    logical_cursor: int
    processed_item_count: int
    processed_item_prefix_sha256: str
    evidence_candidate_count: int
    rule_candidate_count: int
    status: str

    def __post_init__(self) -> None:
        """核验 run/protocol/family、连续 revision 和前缀计数边界。"""
        _sha256(self.run_id, label="learning run_id")
        _sha256(
            self.protocol_manifest_sha256, label="learning protocol manifest")
        if self.operator_family not in SOURCE_INFERENCE_LEARNING_FAMILIES:
            raise BroadQaExternalDataError("learning checkpoint family 未启用")
        if type(self.training_item_count) is not int or self.training_item_count <= 0:
            raise BroadQaExternalDataError(
                "learning checkpoint training item count 非法")
        _sha256(
            self.training_item_order_sha256,
            label="learning training item order",
        )
        if type(self.revision) is not int or self.revision < 0:
            raise BroadQaExternalDataError("learning checkpoint revision 非法")
        _sha256(
            self.previous_checkpoint_sha256,
            label="learning previous checkpoint",
            empty=self.revision == 0,
        )
        if (self.revision == 0) != (self.previous_checkpoint_sha256 == ""):
            raise BroadQaExternalDataError(
                "learning checkpoint previous SHA 链断裂")
        for name in (
                "logical_cursor", "processed_item_count",
                "evidence_candidate_count", "rule_candidate_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise BroadQaExternalDataError(
                    f"learning checkpoint {name} 非法")
        if self.logical_cursor != self.processed_item_count:
            raise BroadQaExternalDataError(
                "learning checkpoint cursor 与 prefix count 漂移")
        if self.processed_item_count > self.training_item_count:
            raise BroadQaExternalDataError(
                "learning checkpoint processed count 超出 TRAIN")
        _sha256(
            self.processed_item_prefix_sha256,
            label="learning processed prefix",
        )
        if (self.processed_item_count == 0
                and self.processed_item_prefix_sha256 != _EMPTY_PREFIX_SHA256):
            raise BroadQaExternalDataError(
                "learning initial prefix SHA 漂移")
        if self.status not in SOURCE_INFERENCE_LEARNING_CHECKPOINT_STATES:
            raise BroadQaExternalDataError("learning checkpoint status 未注册")
        if (self.revision == 0 and any((
                self.logical_cursor,
                self.evidence_candidate_count,
                self.rule_candidate_count,
                self.status != SOURCE_INFERENCE_LEARNING_CHECKPOINT_OPEN,
        ))):
            raise BroadQaExternalDataError(
                "learning initial checkpoint 必须为空 OPEN")
        if (self.status == SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE
                and (self.processed_item_count != self.training_item_count
                     or self.processed_item_prefix_sha256
                     != self.training_item_order_sha256)):
            raise BroadQaExternalDataError(
                "learning complete checkpoint 未覆盖完整 TRAIN")

    def to_dict(self) -> dict[str, object]:
        """导出字段精确的规范 checkpoint。"""
        return {
            "artifact_kind": SOURCE_INFERENCE_LEARNING_CHECKPOINT_KIND,
            "evidence_candidate_count": self.evidence_candidate_count,
            "format_version": 1,
            "logical_cursor": self.logical_cursor,
            "operator_family": self.operator_family,
            "previous_checkpoint_sha256": self.previous_checkpoint_sha256,
            "processed_item_count": self.processed_item_count,
            "processed_item_prefix_sha256": self.processed_item_prefix_sha256,
            "protocol_manifest_sha256": self.protocol_manifest_sha256,
            "revision": self.revision,
            "rule_candidate_count": self.rule_candidate_count,
            "run_id": self.run_id,
            "status": self.status,
            "training_item_count": self.training_item_count,
            "training_item_order_sha256": self.training_item_order_sha256,
        }

    def canonical_bytes(self) -> bytes:
        """返回单换行结尾的规范 checkpoint 字节。"""
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        """返回绑定下一 revision 的 checkpoint 内容摘要。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(
            cls,
            value: object,
            ) -> "BroadQaSourceInferenceLearningCheckpoint":
        """从字段精确的 JSON object 恢复 checkpoint。"""
        expected = {
            "artifact_kind", "evidence_candidate_count", "format_version",
            "logical_cursor", "operator_family", "previous_checkpoint_sha256",
            "processed_item_count", "processed_item_prefix_sha256",
            "protocol_manifest_sha256", "revision", "rule_candidate_count",
            "run_id", "status",
            "training_item_count", "training_item_order_sha256",
        }
        if (not isinstance(value, dict) or set(value) != expected
                or value["artifact_kind"]
                != SOURCE_INFERENCE_LEARNING_CHECKPOINT_KIND
                or type(value["format_version"]) is not int
                or value["format_version"] != 1):
            raise BroadQaExternalDataError(
                "source inference learning checkpoint 字段漂移")
        return cls(
            value["run_id"],
            value["protocol_manifest_sha256"],
            value["operator_family"],
            value["training_item_count"],
            value["training_item_order_sha256"],
            value["revision"],
            value["previous_checkpoint_sha256"],
            value["logical_cursor"],
            value["processed_item_count"],
            value["processed_item_prefix_sha256"],
            value["evidence_candidate_count"],
            value["rule_candidate_count"],
            value["status"],
        )


def initial_source_inference_learning_checkpoint(
        *,
        run_id: str,
        protocol_manifest_sha256: str,
        operator_family: str,
        training_item_ids: tuple[str, ...],
        ) -> BroadQaSourceInferenceLearningCheckpoint:
    """构造 revision 0，不读取任何训练 payload。"""
    training_order_sha = source_inference_learning_prefix_sha256(
        training_item_ids)
    if not training_item_ids or len(set(training_item_ids)) != len(
            training_item_ids):
        raise BroadQaExternalDataError(
            "learning TRAIN item identities 必须非空唯一")
    return BroadQaSourceInferenceLearningCheckpoint(
        run_id,
        protocol_manifest_sha256,
        operator_family,
        len(training_item_ids),
        training_order_sha,
        0,
        "",
        0,
        0,
        _EMPTY_PREFIX_SHA256,
        0,
        0,
        SOURCE_INFERENCE_LEARNING_CHECKPOINT_OPEN,
    )


def advance_source_inference_learning_checkpoint(
        checkpoint: BroadQaSourceInferenceLearningCheckpoint,
        *,
        training_item_ids: tuple[str, ...],
        processed_item_ids: tuple[str, ...],
        evidence_candidate_count: int,
        rule_candidate_count: int,
        complete: bool = False,
        ) -> BroadQaSourceInferenceLearningCheckpoint:
    """以调用方给出的完整前缀推进一次 checkpoint，并拒绝倒退。"""
    if not isinstance(checkpoint, BroadQaSourceInferenceLearningCheckpoint):
        raise TypeError("learning checkpoint 类型非法")
    if checkpoint.status != SOURCE_INFERENCE_LEARNING_CHECKPOINT_OPEN:
        raise BroadQaExternalDataError("已完成 learning checkpoint 不得推进")
    if type(complete) is not bool:
        raise BroadQaExternalDataError("learning checkpoint complete 必须是 bool")
    for name, value in (
            ("evidence_candidate_count", evidence_candidate_count),
            ("rule_candidate_count", rule_candidate_count)):
        if type(value) is not int or value < 0:
            raise BroadQaExternalDataError(
                f"learning checkpoint {name} 非法")
    training_order_sha = source_inference_learning_prefix_sha256(
        training_item_ids)
    if (len(training_item_ids) != checkpoint.training_item_count
            or len(set(training_item_ids)) != len(training_item_ids)
            or training_order_sha != checkpoint.training_item_order_sha256):
        raise BroadQaExternalDataError(
            "learning checkpoint TRAIN identity 漂移")
    prefix_sha256 = source_inference_learning_prefix_sha256(processed_item_ids)
    count = len(processed_item_ids)
    if (count < checkpoint.processed_item_count
            or evidence_candidate_count < checkpoint.evidence_candidate_count
            or rule_candidate_count < checkpoint.rule_candidate_count):
        raise BroadQaExternalDataError("learning checkpoint 不得回退")
    if processed_item_ids != training_item_ids[:count]:
        raise BroadQaExternalDataError(
            "learning checkpoint 只能提交冻结 TRAIN 有序前缀")
    if complete and count != checkpoint.training_item_count:
        raise BroadQaExternalDataError(
            "learning checkpoint 未覆盖完整 TRAIN 不得完成")
    committed_prefix = processed_item_ids[:checkpoint.processed_item_count]
    if (source_inference_learning_prefix_sha256(committed_prefix)
            != checkpoint.processed_item_prefix_sha256):
        raise BroadQaExternalDataError(
            "learning checkpoint 已处理前缀不得替换")
    return BroadQaSourceInferenceLearningCheckpoint(
        checkpoint.run_id,
        checkpoint.protocol_manifest_sha256,
        checkpoint.operator_family,
        checkpoint.training_item_count,
        checkpoint.training_item_order_sha256,
        checkpoint.revision + 1,
        checkpoint.sha256(),
        count,
        count,
        prefix_sha256,
        evidence_candidate_count,
        rule_candidate_count,
        (SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE
         if complete else SOURCE_INFERENCE_LEARNING_CHECKPOINT_OPEN),
    )


def parse_source_inference_learning_checkpoint(
        payload: bytes,
        ) -> BroadQaSourceInferenceLearningCheckpoint:
    """严格回读单个规范 checkpoint，拒绝未知字段和尾随字节。"""
    if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
            or payload.endswith(b"\n\n")):
        raise BroadQaExternalDataError("learning checkpoint 换行非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except ValueError as error:
        raise BroadQaExternalDataError(
            "learning checkpoint 不是规范 JSON") from error
    checkpoint = BroadQaSourceInferenceLearningCheckpoint.from_dict(value)
    if checkpoint.canonical_bytes() != payload:
        raise BroadQaExternalDataError("learning checkpoint 字节漂移")
    return checkpoint


def append_source_inference_learning_checkpoint(
        chain_path: str | Path,
        checkpoint: BroadQaSourceInferenceLearningCheckpoint,
        ) -> None:
    """只允许创建新链或把连续 revision 原子追加到既有链尾。"""
    if not isinstance(checkpoint, BroadQaSourceInferenceLearningCheckpoint):
        raise TypeError("learning checkpoint 类型非法")
    path = Path(chain_path).resolve()
    existing = () if not path.exists() else read_source_inference_learning_chain(
        path)
    if not existing:
        if checkpoint.revision != 0:
            raise BroadQaExternalDataError(
                "learning checkpoint chain 必须从 revision 0 开始")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(checkpoint.canonical_bytes())
        return
    previous = existing[-1]
    if (checkpoint.revision != previous.revision + 1
            or checkpoint.previous_checkpoint_sha256 != previous.sha256()
            or checkpoint.run_id != previous.run_id
            or checkpoint.protocol_manifest_sha256
            != previous.protocol_manifest_sha256
            or checkpoint.operator_family != previous.operator_family
            or checkpoint.training_item_count != previous.training_item_count
            or checkpoint.training_item_order_sha256
            != previous.training_item_order_sha256
            or checkpoint.logical_cursor < previous.logical_cursor
            or checkpoint.processed_item_count < previous.processed_item_count
            or checkpoint.evidence_candidate_count
            < previous.evidence_candidate_count
            or checkpoint.rule_candidate_count < previous.rule_candidate_count
            or previous.status == SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE):
        raise BroadQaExternalDataError(
            "learning checkpoint append 链或运行身份漂移")
    with path.open("ab") as handle:
        handle.write(checkpoint.canonical_bytes())


def read_source_inference_learning_chain(
        path: str | Path,
        ) -> tuple[BroadQaSourceInferenceLearningCheckpoint, ...]:
    """回读完整 append-only 链并重验 revision、SHA 和运行身份。"""
    file = Path(path).resolve()
    try:
        payload = file.read_bytes()
    except OSError as error:
        raise BroadQaExternalDataError(
            "learning checkpoint chain 不可读") from error
    if not payload or not payload.endswith(b"\n"):
        raise BroadQaExternalDataError("learning checkpoint chain 为空或截断")
    checkpoints = tuple(
        parse_source_inference_learning_checkpoint(line + b"\n")
        for line in payload.splitlines()
    )
    first = checkpoints[0]
    for revision, checkpoint in enumerate(checkpoints):
        previous = "" if revision == 0 else checkpoints[revision - 1].sha256()
        if (checkpoint.revision != revision
                or checkpoint.previous_checkpoint_sha256 != previous
                or checkpoint.run_id != first.run_id
                or checkpoint.protocol_manifest_sha256
                != first.protocol_manifest_sha256
                or checkpoint.operator_family != first.operator_family
                or checkpoint.training_item_count != first.training_item_count
                or checkpoint.training_item_order_sha256
                != first.training_item_order_sha256
                or (revision > 0 and (
                    checkpoint.logical_cursor
                    < checkpoints[revision - 1].logical_cursor
                    or checkpoint.processed_item_count
                    < checkpoints[revision - 1].processed_item_count
                    or checkpoint.evidence_candidate_count
                    < checkpoints[revision - 1].evidence_candidate_count
                    or checkpoint.rule_candidate_count
                    < checkpoints[revision - 1].rule_candidate_count))
                or (revision < len(checkpoints) - 1
                    and checkpoint.status
                    == SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE)):
            raise BroadQaExternalDataError(
                "learning checkpoint chain 断裂或身份漂移")
    return checkpoints


def source_inference_learning_result_sha256(
        *,
        protocol_manifest_sha256: str,
        operator_family: str,
        processed_item_ids: tuple[str, ...],
        evidence_record_sha256s: tuple[str, ...],
        rule_record_sha256s: tuple[str, ...],
        ) -> str:
    """形成 fresh/resume 均可独立重算的规范结果摘要。"""
    _sha256(protocol_manifest_sha256, label="learning protocol manifest")
    if operator_family not in SOURCE_INFERENCE_LEARNING_FAMILIES:
        raise BroadQaExternalDataError("learning result family 未启用")
    prefix = source_inference_learning_prefix_sha256(processed_item_ids)
    if (not processed_item_ids
            or len(set(processed_item_ids)) != len(processed_item_ids)):
        raise BroadQaExternalDataError(
            "learning result processed items 必须非空唯一")
    for label, values in (
            ("evidence", evidence_record_sha256s),
            ("rule", rule_record_sha256s)):
        if not isinstance(values, tuple):
            raise BroadQaExternalDataError(f"learning {label} SHAs 必须是 tuple")
        for value in values:
            _sha256(value, label=f"learning {label} record")
        if values != tuple(sorted(set(values))):
            raise BroadQaExternalDataError(
                f"learning {label} SHAs 必须唯一规范排序")
    value = {
        "evidence_record_sha256s": list(evidence_record_sha256s),
        "operator_family": operator_family,
        "processed_item_count": len(processed_item_ids),
        "processed_item_prefix_sha256": prefix,
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "rule_record_sha256s": list(rule_record_sha256s),
    }
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_source_inference_fresh_resume_equivalence(
        fresh_sha256: str,
        resumed_sha256: str,
        ) -> None:
    """把 fresh/resume 逐字节结果等价作为 learner 硬门。"""
    _sha256(fresh_sha256, label="fresh learning result")
    _sha256(resumed_sha256, label="resumed learning result")
    if fresh_sha256 != resumed_sha256:
        raise BroadQaExternalDataError(
            "source inference fresh/resume 结果不等价")


__all__ = [
    "BroadQaSourceInferenceLearningCheckpoint",
    "SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE",
    "SOURCE_INFERENCE_LEARNING_CHECKPOINT_KIND",
    "SOURCE_INFERENCE_LEARNING_CHECKPOINT_OPEN",
    "advance_source_inference_learning_checkpoint",
    "append_source_inference_learning_checkpoint",
    "initial_source_inference_learning_checkpoint",
    "parse_source_inference_learning_checkpoint",
    "read_source_inference_learning_chain",
    "require_source_inference_fresh_resume_equivalence",
    "source_inference_learning_prefix_sha256",
    "source_inference_learning_result_sha256",
]
