"""normalization v3 的 OpenCC 来源承诺与 Evidence 独立重放。

本模块把字典文件、物理行、mapping candidate、phrase trial 和共享
``EvidenceRecord`` 绑定起来。它不选择规则、不读取 evaluation，也不把 OpenCC
来源行为表述为一般语言真值。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    GLOBAL_OWNER_SCOPE,
    ParserVersion,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    ScopeIdentity,
    document_scope,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_contrastive_protocol import (
    NORMALIZATION_CONTRASTIVE_CANDIDATE_KIND,
    NORMALIZATION_CONTRASTIVE_FAMILY,
    NORMALIZATION_CONTRASTIVE_QUALIFICATIONS,
    NORMALIZATION_CONTRASTIVE_TRIAL_KIND,
    read_normalization_contrastive_protocol,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_source_pack import (
    NORMALIZATION_SOURCE_FILES,
    read_normalization_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
    parse_canonical_json_bytes,
)


NORMALIZATION_CONTRASTIVE_PROTOCOL_SOURCE_KIND = 817038
NORMALIZATION_DICTIONARY_FILE_SOURCE_KIND = 817039
NORMALIZATION_EVIDENCE_COMMITMENT_V3_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_EVIDENCE_COMMITMENT_V3")
NORMALIZATION_EVIDENCE_REASON_KEYS = {
    "SOURCE_REPLAY_SUPPORT": (817042, 1),
    "SOURCE_REPLAY_REFUTE": (817042, 2),
}
_CANDIDATE_SOURCE_PATH = "dictionary/TSCharacters.txt"
_TRIAL_SOURCE_PATH = "dictionary/TSPhrases.txt"


def _sha256(value: object, *, label: str) -> str:
    """要求来源承诺为小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise BroadQaExternalDataError(f"{label} 必须是 SHA-256")
    return value


def _strict_codepoint(value: object, *, label: str) -> int:
    """要求值为 Unicode scalar，而不是 bool、代理项或越界整数。"""
    if (type(value) is not int or not 0 <= value <= 0x10FFFF
            or 0xD800 <= value <= 0xDFFF):
        raise BroadQaExternalDataError(f"{label} 不是 Unicode scalar")
    return value


def _strict_key(value: object, *, label: str) -> tuple[int, ...]:
    """要求共享稳定键为非空严格整数数组或 tuple。"""
    if (not isinstance(value, (list, tuple)) or not value
            or any(type(item) is not int for item in value)):
        raise BroadQaExternalDataError(f"{label} 必须是严格整数键")
    return tuple(value)


def _sha_identity(value: str, *, label: str) -> int:
    """将完整 SHA-256 无损投影为正整数身份分量。"""
    return int(_sha256(value, label=label), 16) + 1


def normalization_contrastive_protocol_source(
        protocol_manifest_sha256: str,
        ) -> SourceRef:
    """把 normalization contrastive protocol 映射为独立来源身份。"""
    return SourceRef(
        NORMALIZATION_CONTRASTIVE_PROTOCOL_SOURCE_KIND,
        _sha_identity(
            protocol_manifest_sha256,
            label="normalization contrastive protocol manifest",
        ),
        1,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(curriculum=CurriculumVersion(1)),
    )


def normalization_contrastive_protocol_scope(
        protocol_manifest_sha256: str,
        ) -> ScopeIdentity:
    """返回只指向 contrastive protocol artifact 的 document scope。"""
    return document_scope(
        normalization_contrastive_protocol_source(
            protocol_manifest_sha256))


def normalization_dictionary_file_source(
        *,
        source_pack_manifest_sha256: str,
        relative_path: str,
        ) -> SourceRef:
    """为 OpenCC source pack 中的一份固定文件形成稳定来源引用。"""
    if relative_path not in NORMALIZATION_SOURCE_FILES:
        raise BroadQaExternalDataError(
            "normalization dictionary relative path 未注册")
    document_id = int.from_bytes(
        hashlib.sha256(relative_path.encode("utf-8")).digest(), "big") + 1
    return SourceRef(
        NORMALIZATION_DICTIONARY_FILE_SOURCE_KIND,
        _sha_identity(
            source_pack_manifest_sha256,
            label="normalization source pack manifest",
        ),
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(corpus=CorpusVersion(1), parser=ParserVersion(1)),
    )


def normalization_training_item_ids(
        candidates: tuple[dict[str, object], ...],
        trials: tuple[dict[str, object], ...],
        ) -> tuple[str, ...]:
    """为完整 candidate/trial TRAIN_SOURCE 形成有类型边界的有序身份。"""
    if (not isinstance(candidates, tuple) or not candidates
            or not isinstance(trials, tuple) or not trials):
        raise BroadQaExternalDataError(
            "normalization training source 必须非空 tuple")
    identities = tuple(
        hashlib.sha256(canonical_json_bytes({
            "record_id": record[key],
            "record_kind": kind,
        })).hexdigest()
        for records, kind, key in (
            (candidates, NORMALIZATION_CONTRASTIVE_CANDIDATE_KIND,
             "candidate_id"),
            (trials, NORMALIZATION_CONTRASTIVE_TRIAL_KIND, "trial_id"),
        )
        for record in records
    )
    if len(set(identities)) != len(identities):
        raise BroadQaExternalDataError(
            "normalization training identity 冲突")
    return identities


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaNormalizationSourceLineCommitmentV3:
    """OpenCC source pack 内一条物理 UTF-8 行的完整来源承诺。"""

    source_pack_manifest_sha256: str
    relative_path: str
    file_sha256: str
    line_ordinal: int
    byte_start: int
    byte_end: int
    line_sha256: str
    source_ref: SourceRef

    def __post_init__(self) -> None:
        """核验文件身份、物理坐标和共享 SourceRef 投影。"""
        _sha256(
            self.source_pack_manifest_sha256,
            label="normalization line source pack",
        )
        expected_file = NORMALIZATION_SOURCE_FILES.get(self.relative_path)
        if (expected_file is None
                or self.file_sha256 != expected_file["sha256"]):
            raise BroadQaExternalDataError(
                "normalization line file identity 漂移")
        _sha256(self.file_sha256, label="normalization line file")
        _sha256(self.line_sha256, label="normalization physical line")
        if (type(self.line_ordinal) is not int or self.line_ordinal <= 0
                or type(self.byte_start) is not int or self.byte_start < 0
                or type(self.byte_end) is not int
                or self.byte_end <= self.byte_start):
            raise BroadQaExternalDataError(
                "normalization line physical span 非法")
        expected_source = normalization_dictionary_file_source(
            source_pack_manifest_sha256=self.source_pack_manifest_sha256,
            relative_path=self.relative_path,
        )
        if self.source_ref != expected_source:
            raise BroadQaExternalDataError(
                "normalization line SourceRef 漂移")

    def to_dict(self) -> dict[str, object]:
        """导出字段精确的物理行承诺。"""
        return {
            "byte_end": self.byte_end,
            "byte_start": self.byte_start,
            "file_sha256": self.file_sha256,
            "line_ordinal": self.line_ordinal,
            "line_sha256": self.line_sha256,
            "relative_path": self.relative_path,
            "source_pack_manifest_sha256": self.source_pack_manifest_sha256,
            "source_ref_key": list(self.source_ref.stable_key()),
        }

    @classmethod
    def from_dict(
            cls,
            value: object,
            ) -> "BroadQaNormalizationSourceLineCommitmentV3":
        """从字段精确 JSON object 恢复物理行承诺。"""
        expected = {
            "byte_end", "byte_start", "file_sha256", "line_ordinal",
            "line_sha256", "relative_path", "source_pack_manifest_sha256",
            "source_ref_key",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise BroadQaExternalDataError(
                "normalization line commitment 字段漂移")
        try:
            source_ref = SourceRef.from_stable_key(_strict_key(
                value["source_ref_key"], label="normalization line source"))
        except (TypeError, ValueError) as error:
            raise BroadQaExternalDataError(
                "normalization line SourceRef 非法") from error
        return cls(
            value["source_pack_manifest_sha256"],
            value["relative_path"],
            value["file_sha256"],
            value["line_ordinal"],
            value["byte_start"],
            value["byte_end"],
            value["line_sha256"],
            source_ref,
        )


def normalization_source_line_commitment_from_record(
        *,
        source_pack_manifest_sha256: str,
        source_commitment: object,
        ) -> BroadQaNormalizationSourceLineCommitmentV3:
    """把 contrastive record 的物理行字段提升为一等来源承诺。"""
    expected = {
        "byte_end", "byte_start", "file_sha256", "line_ordinal",
        "line_sha256", "relative_path",
    }
    if not isinstance(source_commitment, dict) or set(
            source_commitment) != expected:
        raise BroadQaExternalDataError(
            "normalization source line record 漂移")
    relative_path = source_commitment["relative_path"]
    return BroadQaNormalizationSourceLineCommitmentV3(
        source_pack_manifest_sha256,
        relative_path,
        source_commitment["file_sha256"],
        source_commitment["line_ordinal"],
        source_commitment["byte_start"],
        source_commitment["byte_end"],
        source_commitment["line_sha256"],
        normalization_dictionary_file_source(
            source_pack_manifest_sha256=source_pack_manifest_sha256,
            relative_path=relative_path,
        ),
    )


def normalization_evidence_payload(
        *,
        candidate_id: str,
        trial_id: str,
        source_codepoint_offset: int,
        input_codepoint: int,
        candidate_output_codepoint: int,
        observed_output_codepoint: int,
        ) -> tuple[int, ...]:
    """形成绑定 candidate、trial、offset 和观察值的共享 Evidence payload。"""
    if type(source_codepoint_offset) is not int or source_codepoint_offset < 0:
        raise BroadQaExternalDataError(
            "normalization Evidence offset 非法")
    return (
        _sha_identity(candidate_id, label="normalization candidate id"),
        _sha_identity(trial_id, label="normalization trial id"),
        source_codepoint_offset,
        _strict_codepoint(input_codepoint, label="normalization input"),
        _strict_codepoint(
            candidate_output_codepoint,
            label="normalization candidate output",
        ),
        _strict_codepoint(
            observed_output_codepoint,
            label="normalization observed output",
        ),
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class BroadQaNormalizationEvidenceCommitmentV3:
    """一个可从 OpenCC mapping 与 phrase line 独立重放的 Evidence。"""

    contrastive_protocol_manifest_sha256: str
    source_pack_manifest_sha256: str
    candidate_id: str
    trial_id: str
    candidate_source: BroadQaNormalizationSourceLineCommitmentV3
    trial_source: BroadQaNormalizationSourceLineCommitmentV3
    source_codepoint_offset: int
    input_codepoint: int
    candidate_output_codepoint: int
    observed_output_codepoint: int
    qualification_kind: str
    evidence_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验资格、来源、Evidence stance、payload 和确定性事件身份。"""
        _sha256(
            self.contrastive_protocol_manifest_sha256,
            label="normalization Evidence protocol",
        )
        _sha256(
            self.source_pack_manifest_sha256,
            label="normalization Evidence source pack",
        )
        _sha256(self.candidate_id, label="normalization Evidence candidate")
        _sha256(self.trial_id, label="normalization Evidence trial")
        if (not isinstance(
                self.candidate_source,
                BroadQaNormalizationSourceLineCommitmentV3)
                or not isinstance(
                    self.trial_source,
                    BroadQaNormalizationSourceLineCommitmentV3)
                or self.candidate_source.source_pack_manifest_sha256
                != self.source_pack_manifest_sha256
                or self.trial_source.source_pack_manifest_sha256
                != self.source_pack_manifest_sha256
                or self.candidate_source.relative_path
                != _CANDIDATE_SOURCE_PATH
                or self.trial_source.relative_path != _TRIAL_SOURCE_PATH):
            raise BroadQaExternalDataError(
                "normalization Evidence source carrier 漂移")
        payload = normalization_evidence_payload(
            candidate_id=self.candidate_id,
            trial_id=self.trial_id,
            source_codepoint_offset=self.source_codepoint_offset,
            input_codepoint=self.input_codepoint,
            candidate_output_codepoint=self.candidate_output_codepoint,
            observed_output_codepoint=self.observed_output_codepoint,
        )
        if self.qualification_kind not in NORMALIZATION_CONTRASTIVE_QUALIFICATIONS:
            raise BroadQaExternalDataError(
                "normalization Evidence qualification 未注册")
        expected_stance = (
            EVIDENCE_SUPPORT
            if self.qualification_kind == "SOURCE_REPLAY_SUPPORT"
            else EVIDENCE_REFUTE)
        if ((expected_stance == EVIDENCE_SUPPORT)
                != (self.candidate_output_codepoint
                    == self.observed_output_codepoint)):
            raise BroadQaExternalDataError(
                "normalization Evidence qualification/observation 漂移")
        try:
            evidence = EvidenceRecord.from_stable_key(self.evidence_key)
        except (TypeError, ValueError) as error:
            raise BroadQaExternalDataError(
                "normalization Evidence key 无法恢复") from error
        expected_evidence_id = _sha_identity(
            self.trial_id, label="normalization Evidence trial id")
        if (evidence.evidence_id != expected_evidence_id
                or evidence.stance != expected_stance
                or evidence.reason_key
                != NORMALIZATION_EVIDENCE_REASON_KEYS[self.qualification_kind]
                or evidence.source != self.trial_source.source_ref
                or evidence.timestamp_seq != expected_evidence_id
                or evidence.payload != payload
                or evidence.supersedes_evidence_id != 0):
            raise BroadQaExternalDataError(
                "normalization Evidence event/source/payload 漂移")

    def to_dict(self) -> dict[str, object]:
        """导出字段精确的来源化 Evidence commitment。"""
        return {
            "artifact_kind": NORMALIZATION_EVIDENCE_COMMITMENT_V3_KIND,
            "candidate_id": self.candidate_id,
            "candidate_output_codepoint": self.candidate_output_codepoint,
            "candidate_source": self.candidate_source.to_dict(),
            "contrastive_protocol_manifest_sha256": (
                self.contrastive_protocol_manifest_sha256),
            "evidence_key": list(self.evidence_key),
            "format_version": 1,
            "input_codepoint": self.input_codepoint,
            "observed_output_codepoint": self.observed_output_codepoint,
            "qualification_kind": self.qualification_kind,
            "source_codepoint_offset": self.source_codepoint_offset,
            "source_pack_manifest_sha256": self.source_pack_manifest_sha256,
            "trial_id": self.trial_id,
            "trial_source": self.trial_source.to_dict(),
        }

    def canonical_bytes(self) -> bytes:
        """返回单换行结尾的规范 commitment 字节。"""
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        """返回包含来源坐标和 Evidence event 的规范摘要。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(
            cls,
            value: object,
            ) -> "BroadQaNormalizationEvidenceCommitmentV3":
        """从字段精确 JSON object 恢复 commitment。"""
        expected = {
            "artifact_kind", "candidate_id", "candidate_output_codepoint",
            "candidate_source", "contrastive_protocol_manifest_sha256",
            "evidence_key", "format_version", "input_codepoint",
            "observed_output_codepoint", "qualification_kind",
            "source_codepoint_offset", "source_pack_manifest_sha256",
            "trial_id", "trial_source",
        }
        if (not isinstance(value, dict) or set(value) != expected
                or value["artifact_kind"]
                != NORMALIZATION_EVIDENCE_COMMITMENT_V3_KIND
                or type(value["format_version"]) is not int
                or value["format_version"] != 1):
            raise BroadQaExternalDataError(
                "normalization Evidence commitment 字段漂移")
        return cls(
            value["contrastive_protocol_manifest_sha256"],
            value["source_pack_manifest_sha256"],
            value["candidate_id"],
            value["trial_id"],
            BroadQaNormalizationSourceLineCommitmentV3.from_dict(
                value["candidate_source"]),
            BroadQaNormalizationSourceLineCommitmentV3.from_dict(
                value["trial_source"]),
            value["source_codepoint_offset"],
            value["input_codepoint"],
            value["candidate_output_codepoint"],
            value["observed_output_codepoint"],
            value["qualification_kind"],
            _strict_key(
                value["evidence_key"],
                label="normalization Evidence key",
            ),
        )


def parse_normalization_evidence_commitment_v3(
        payload: bytes,
        ) -> BroadQaNormalizationEvidenceCommitmentV3:
    """严格回读单个规范 normalization Evidence commitment。"""
    if (not isinstance(payload, bytes) or not payload.endswith(b"\n")
            or payload.endswith(b"\n\n")):
        raise BroadQaExternalDataError(
            "normalization Evidence commitment 换行非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except ValueError as error:
        raise BroadQaExternalDataError(
            "normalization Evidence commitment 不是规范 JSON") from error
    commitment = BroadQaNormalizationEvidenceCommitmentV3.from_dict(value)
    if commitment.canonical_bytes() != payload:
        raise BroadQaExternalDataError(
            "normalization Evidence commitment 字节漂移")
    return commitment


def normalization_evidence_commitment_from_records(
        *,
        contrastive_protocol_manifest_sha256: str,
        source_pack_manifest_sha256: str,
        candidate: dict[str, object],
        trial: dict[str, object],
        hypothesis: HypothesisKey,
        ) -> BroadQaNormalizationEvidenceCommitmentV3:
    """从严格来源 record 构造绑定指定 hypothesis 的确定性 Evidence。"""
    if not isinstance(hypothesis, HypothesisKey):
        raise TypeError("normalization Evidence hypothesis 类型非法")
    candidate_id = candidate.get("candidate_id")
    trial_id = trial.get("trial_id")
    if (candidate.get("record_kind")
            != NORMALIZATION_CONTRASTIVE_CANDIDATE_KIND
            or trial.get("record_kind") != NORMALIZATION_CONTRASTIVE_TRIAL_KIND
            or trial.get("candidate_id") != candidate_id):
        raise BroadQaExternalDataError(
            "normalization candidate/trial identity 漂移")
    qualification = trial["qualification_kind"]
    evidence_id = _sha_identity(
        trial_id, label="normalization source trial id")
    candidate_source = normalization_source_line_commitment_from_record(
        source_pack_manifest_sha256=source_pack_manifest_sha256,
        source_commitment=candidate["source_commitment"],
    )
    trial_source = normalization_source_line_commitment_from_record(
        source_pack_manifest_sha256=source_pack_manifest_sha256,
        source_commitment=trial["source_commitment"],
    )
    payload = normalization_evidence_payload(
        candidate_id=candidate_id,
        trial_id=trial_id,
        source_codepoint_offset=trial["source_codepoint_offset"],
        input_codepoint=candidate["input_codepoint"],
        candidate_output_codepoint=candidate["output_codepoint"],
        observed_output_codepoint=trial["observed_output_codepoint"],
    )
    evidence = EvidenceRecord(
        evidence_id,
        hypothesis,
        (EVIDENCE_SUPPORT
         if qualification == "SOURCE_REPLAY_SUPPORT"
         else EVIDENCE_REFUTE),
        NORMALIZATION_EVIDENCE_REASON_KEYS[qualification],
        trial_source.source_ref,
        evidence_id,
        payload,
    )
    return BroadQaNormalizationEvidenceCommitmentV3(
        contrastive_protocol_manifest_sha256,
        source_pack_manifest_sha256,
        candidate_id,
        trial_id,
        candidate_source,
        trial_source,
        trial["source_codepoint_offset"],
        candidate["input_codepoint"],
        candidate["output_codepoint"],
        trial["observed_output_codepoint"],
        qualification,
        evidence.stable_key(),
    )


def validate_normalization_evidence_commitment(
        commitment: BroadQaNormalizationEvidenceCommitmentV3,
        *,
        protocol_manifest_sha256: str,
        source_pack_manifest_sha256: str,
        candidate_by_id: dict[str, dict[str, object]],
        trial_by_id: dict[str, dict[str, object]],
        expected_hypothesis: HypothesisKey,
        expected_qualification: str,
        ) -> None:
    """从冻结 candidate/trial 独立重造 commitment 并要求逐字段相等。"""
    if not isinstance(commitment, BroadQaNormalizationEvidenceCommitmentV3):
        raise TypeError("normalization Evidence commitment 类型非法")
    candidate = candidate_by_id.get(commitment.candidate_id)
    trial = trial_by_id.get(commitment.trial_id)
    if (candidate is None or trial is None
            or commitment.contrastive_protocol_manifest_sha256
            != protocol_manifest_sha256
            or commitment.source_pack_manifest_sha256
            != source_pack_manifest_sha256
            or commitment.qualification_kind != expected_qualification):
        raise BroadQaExternalDataError(
            "normalization Evidence protocol/source/qualification 漂移")
    expected = normalization_evidence_commitment_from_records(
        contrastive_protocol_manifest_sha256=protocol_manifest_sha256,
        source_pack_manifest_sha256=source_pack_manifest_sha256,
        candidate=candidate,
        trial=trial,
        hypothesis=expected_hypothesis,
    )
    if commitment != expected:
        raise BroadQaExternalDataError(
            "normalization Evidence source replay commitment 漂移")


def read_normalization_training_provenance(
        *,
        source_pack_dir: str | Path,
        contrastive_protocol_dir: str | Path,
        ) -> tuple[
            dict[str, object],
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[str, ...],
        ]:
    """严格回读 OpenCC pack 与 TRAIN_SOURCE protocol，不读取 evaluation。"""
    source_root = Path(source_pack_dir).resolve()
    protocol_root = Path(contrastive_protocol_dir).resolve()
    source_manifest = read_normalization_source_pack(source_root)
    protocol_manifest, candidates, trials = (
        read_normalization_contrastive_protocol(
            protocol_root,
            source_pack_dir=source_root,
        ))
    if (protocol_manifest["operator_family"]
            != NORMALIZATION_CONTRASTIVE_FAMILY
            or protocol_manifest["source_pack_manifest_sha256"]
            != source_manifest["manifest_sha256"]):
        raise BroadQaExternalDataError(
            "normalization training provenance identity 漂移")
    return (
        source_manifest,
        protocol_manifest,
        candidates,
        trials,
        normalization_training_item_ids(candidates, trials),
    )


__all__ = [
    "BroadQaNormalizationEvidenceCommitmentV3",
    "BroadQaNormalizationSourceLineCommitmentV3",
    "NORMALIZATION_CONTRASTIVE_PROTOCOL_SOURCE_KIND",
    "NORMALIZATION_DICTIONARY_FILE_SOURCE_KIND",
    "NORMALIZATION_EVIDENCE_COMMITMENT_V3_KIND",
    "NORMALIZATION_EVIDENCE_REASON_KEYS",
    "normalization_contrastive_protocol_scope",
    "normalization_contrastive_protocol_source",
    "normalization_dictionary_file_source",
    "normalization_evidence_commitment_from_records",
    "normalization_evidence_payload",
    "normalization_source_line_commitment_from_record",
    "normalization_training_item_ids",
    "parse_normalization_evidence_commitment_v3",
    "read_normalization_training_provenance",
    "validate_normalization_evidence_commitment",
]
