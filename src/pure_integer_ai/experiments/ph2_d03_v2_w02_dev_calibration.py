"""PH2-D03-V2 W-02 的只读 dev calibration 与安全公开报告。"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Iterator
import unicodedata

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    exact_dict,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2_STAGE_EVALUATION_POLICIES,
    validate_v2_safe_report,
)
from pure_integer_ai.experiments.ph2_d03_v2_schema import validate_v2_record
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_model import (
    W02_CAPABILITY_CARRIER_RECONSTRUCTION,
    W02_CAPABILITY_OOV_BOUNDARY_LATTICE,
    W02_CAPABILITY_UD_MORPHOLOGY,
    W02_CAPABILITY_UNICODE_ANALYSIS,
    W02CandidatePrediction,
    W02CarrierRule,
    W02MorphologyCandidate,
    W02UnicodeUnit,
    boundary_lattice,
    generate_with_carrier_rules,
    observe_w02_carrier,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_publication import (
    W02_CANDIDATE_RECEIPT_PATH,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    W02CandidatePredictor,
    open_w02_candidate_predictor,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_compiler import _dimension_key
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import (
    W02CompileFreeze,
    W02FileFreeze,
    W02_LAYOUTS,
    read_w02_compile_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_runtime_contract import (
    read_w02_candidate_runtime_freeze,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    parse_canonical_json_bytes,
)


W02_DEV_FREEZE_VERSION = "PH2-D03-V2-W02-DEV-CALIBRATION-FREEZE-V1"
W02_DEV_REPORT_VERSION = "PH2-D03-V2-W02-DEV-CALIBRATION-REPORT-V1"
W02_DEV_FREEZE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_dev_calibration_freeze_v1.json"
)
W02_DEV_REPORT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_dev_calibration_report_v1.json"
)
W02_DEV_CODE_PATHS = (
    "src/pure_integer_ai/experiments/ph2_d03_v2_w02_dev_calibration.py",
    "src/pure_integer_ai/experiments/run_ph2_d03_v2_w02_dev_calibration.py",
    "tests/test_ph2_d03_v2_w02_dev_calibration.py",
)
W02_DEV_LAYOUT_PATHS = {
    "DEV_SOURCE": "source/source_refs.jsonl.gz",
    "DEV_OBSERVATION": "observations/dev.jsonl.gz",
    "DEV_LABEL": "evaluator/dev.labels.jsonl.gz",
}
W02_DEV_DIMENSIONS = (
    "W-02-V2-BOUNDARY-WITHDRAWAL",
    "W-02-V2-MULTI-CANDIDATE",
    "W-02-V2-NEW-CONTENT-MORPHOLOGY",
    "W-02-V2-OOV",
    "W-02-V2-GENERATION-HARD-CONJUNCT",
)


# object-model: exception
class W02DevCalibrationError(RuntimeError):
    """W-02 dev 输入、预测或公开投影不满足冻结合同。"""


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _hash_value(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _safe_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (not relative or "\\" in relative or pure.is_absolute()
            or pure.as_posix() != relative
            or any(part in {"", ".", ".."} for part in pure.parts)):
        raise W02DevCalibrationError("W-02 dev 输入路径非法")
    current = root
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise W02DevCalibrationError("W-02 dev 输入不得经过 symlink")
    target = (root / Path(*pure.parts)).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise W02DevCalibrationError("W-02 dev 输入文件缺失或逃逸")
    return target


def _tree_sha256(root: Path) -> str:
    rows = []
    for target in sorted(root.rglob("*")):
        if target.is_symlink():
            raise W02DevCalibrationError("W-02 dev 审计树不得含 symlink")
        if target.is_file():
            size, digest = _sha256_file(target)
            rows.append({
                "relative": target.relative_to(root).as_posix(),
                "sha256": digest,
                "size_bytes": size,
            })
    return _hash_value(rows)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02DevInputRoot:
    """只允许指向独立 dev-calibration owner 的物理根。"""

    root: Path

    def __post_init__(self) -> None:
        root = Path(self.root).resolve()
        object.__setattr__(self, "root", root)
        if not root.is_dir() or root.name != "dev-calibration" or root.is_symlink():
            raise W02DevCalibrationError("W-02 dev root 身份非法")


def _dev_freeze(freeze: W02CompileFreeze, layout_key: str) -> W02FileFreeze:
    if layout_key not in W02_DEV_LAYOUT_PATHS:
        raise W02DevCalibrationError("W-02 dev layout 未注册")
    matches = tuple(item for item in freeze.files if item.layout_key == layout_key)
    if len(matches) != 1:
        raise W02DevCalibrationError("W-02 dev layout freeze 不唯一")
    item = matches[0]
    expected = W02_LAYOUTS[layout_key]
    if (item.root_key, item.record_kind, item.split, item.storage_relative_path) != expected:
        raise W02DevCalibrationError("W-02 dev layout 与 compile freeze 漂移")
    return item


def iter_w02_dev_records(
        freeze: W02CompileFreeze,
        dev: W02DevInputRoot,
        layout_key: str,
        ) -> Iterator[object]:
    """逐条读取一个 dev gzip，并在 EOF 闭合 transport/content 身份。"""
    identity = _dev_freeze(freeze, layout_key)
    target = _safe_file(dev.root, W02_DEV_LAYOUT_PATHS[layout_key])
    size, digest = _sha256_file(target)
    if size != identity.transport_size_bytes or digest != identity.transport_sha256:
        raise W02DevCalibrationError("W-02 dev transport identity 漂移")
    content_digest = hashlib.sha256()
    content_size = 0
    count = 0
    first_key: tuple[int, ...] | None = None
    last_key: tuple[int, ...] | None = None
    previous_key: tuple[int, ...] | None = None
    try:
        with target.open("rb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.endswith(b"\n") or line.endswith(b"\n\n"):
                        raise W02DevCalibrationError(
                            f"W-02 dev JSONL 第 {line_number} 行换行非法")
                    content_digest.update(line)
                    content_size += len(line)
                    value = parse_canonical_json_bytes(line[:-1], require_object=True)
                    assert isinstance(value, dict)
                    record = validate_v2_record(value)
                    if getattr(record, "RECORD_KIND", None) != identity.record_kind:
                        raise W02DevCalibrationError("W-02 dev record kind 漂移")
                    if (identity.split
                            and getattr(record, "split", identity.split)
                            != identity.split):
                        raise W02DevCalibrationError("W-02 dev split 漂移")
                    key = record.stable_key.components
                    if previous_key is not None and key <= previous_key:
                        raise W02DevCalibrationError("W-02 dev stable key 未严格排序")
                    previous_key = key
                    first_key = key if first_key is None else first_key
                    last_key = key
                    count += 1
                    yield record
    except (OSError, EOFError, ValueError) as error:
        if isinstance(error, W02DevCalibrationError):
            raise
        raise W02DevCalibrationError("W-02 dev gzip/JSONL 读取失败") from error
    if (count != identity.record_count
            or content_size != identity.content_size_bytes
            or content_digest.hexdigest() != identity.content_sha256
            or first_key != identity.first_record_key
            or last_key != identity.last_record_key):
        raise W02DevCalibrationError("W-02 dev content identity 漂移")


def scan_w02_dev_sources(
        freeze: W02CompileFreeze,
        dev: W02DevInputRoot,
        ) -> tuple[int, str]:
    """完整回读 dev SourceRef，但不把来源表层写入报告。"""
    count = 0
    digest = hashlib.sha256()
    for record in iter_w02_dev_records(freeze, dev, "DEV_SOURCE"):
        if not isinstance(record, SourceRefRecord):
            raise W02DevCalibrationError("W-02 dev SourceRef 类型错误")
        digest.update(canonical_json_bytes(record.stable_key.to_list()))
        count += 1
    return count, digest.hexdigest()


def iter_w02_dev_pairs(
        freeze: W02CompileFreeze,
        dev: W02DevInputRoot,
        ) -> Iterator[tuple[ObservationRecord, EvaluatorLabelRecord]]:
    """单遍配对 dev Observation 与 evaluator record。"""
    observations = iter_w02_dev_records(freeze, dev, "DEV_OBSERVATION")
    evaluations = iter_w02_dev_records(freeze, dev, "DEV_LABEL")
    count = 0
    for observation, evaluation in zip(observations, evaluations, strict=True):
        if (not isinstance(observation, ObservationRecord)
                or not isinstance(evaluation, EvaluatorLabelRecord)):
            raise W02DevCalibrationError("W-02 dev pair 类型错误")
        if (evaluation.observation_key != observation.stable_key
                or observation.split != "dev"
                or evaluation.visible_stage != "W-02"
                or evaluation.owner_mode != "read_only"):
            raise W02DevCalibrationError("W-02 dev pair 绑定或 owner 漂移")
        count += 1
        yield observation, evaluation
    if count != freeze.plan.split_total("dev"):
        raise W02DevCalibrationError("W-02 dev pair 数量与 plan 漂移")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02DevCandidateIndex:
    """冻结 Candidate SQLite 的只读批量索引，不持有资源句柄。"""

    carrier_rules: dict[str, tuple[tuple[W02CarrierRule, int], ...]]
    oov_lengths: tuple[int, ...]
    lexemes: dict[str, tuple[tuple[str, str, str, int], ...]]
    capabilities: tuple[str, ...]
    max_lexeme_length: int
    semantic_sha256: str
    row_count: int

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.capabilities))) != self.capabilities:
            raise W02DevCalibrationError("W-02 dev capabilities 未规范排序")
        if type(self.max_lexeme_length) is not int or self.max_lexeme_length < 0:
            raise W02DevCalibrationError("W-02 dev lexeme 上界非法")
        if type(self.row_count) is not int or self.row_count <= 0:
            raise W02DevCalibrationError("W-02 dev Candidate index 为空")
        if len(self.semantic_sha256) != 64:
            raise W02DevCalibrationError("W-02 dev Candidate index SHA 非法")


def load_w02_dev_candidate_index(
        predictor: W02CandidatePredictor,
        ) -> W02DevCandidateIndex:
    """一次读取只读 SQLite 索引，避免逐表层重复 SQL。"""
    if not isinstance(predictor, W02CandidatePredictor):
        raise TypeError("W-02 dev predictor 类型错误")
    rule_rows = tuple(predictor.connection.execute(
        "SELECT carrier_kind,prefix,suffix,root_node_kind,content_node_kind,"
        "support_count FROM carrier_rules ORDER BY carrier_kind,prefix,suffix,"
        "root_node_kind,content_node_kind"))
    rules: dict[str, list[tuple[W02CarrierRule, int]]] = {}
    for row in rule_rows:
        rule = W02CarrierRule(
            str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]))
        rules.setdefault(rule.carrier_kind, []).append((rule, int(row[5])))
    length_rows = tuple(predictor.connection.execute(
        "SELECT DISTINCT unit_length FROM oov_units ORDER BY unit_length"))
    lexeme_rows = tuple(predictor.connection.execute(
        "SELECT form,lemma,upos,feats_json,support_count FROM lexemes "
        "ORDER BY form,lemma,upos,feats_json"))
    lexemes: dict[str, list[tuple[str, str, str, int]]] = {}
    for row in lexeme_rows:
        lexemes.setdefault(str(row[0]), []).append(
            (str(row[1]), str(row[2]), str(row[3]), int(row[4])))
    semantic = {
        "capabilities": list(predictor.capabilities),
        "lexemes": [list(row) for row in lexeme_rows],
        "oov_lengths": [int(row[0]) for row in length_rows],
        "rules": [list(row) for row in rule_rows],
    }
    return W02DevCandidateIndex(
        {key: tuple(value) for key, value in rules.items()},
        tuple(int(row[0]) for row in length_rows),
        {key: tuple(value) for key, value in lexemes.items()},
        predictor.capabilities,
        predictor.max_lexeme_length,
        _hash_value(semantic),
        len(rule_rows) + len(length_rows) + len(lexeme_rows)
        + len(predictor.capabilities),
    )


def predict_w02_dev_observation(
        index: W02DevCandidateIndex,
        observation: ObservationRecord,
        ) -> tuple[W02CandidatePrediction, int]:
    """与冻结 predictor 等价地预测，并返回确定性逻辑操作计数。"""
    if not isinstance(index, W02DevCandidateIndex):
        raise TypeError("W-02 dev Candidate index 类型错误")
    observed = observe_w02_carrier(observation)
    rules = index.carrier_rules.get(observed.carrier_kind, ())
    generation = generate_with_carrier_rules(
        rules, carrier_kind=observed.carrier_kind, surface=observed.surface)
    if W02_CAPABILITY_OOV_BOUNDARY_LATTICE in index.capabilities and index.oov_lengths:
        points = boundary_lattice(
            observed.surface, observed_unit_lengths=index.oov_lengths)
    else:
        points = (0,) if not observed.surface else (0, len(observed.surface))
    unicode_units: tuple[W02UnicodeUnit, ...] = ()
    if W02_CAPABILITY_UNICODE_ANALYSIS in index.capabilities:
        unicode_units = tuple(W02UnicodeUnit(
            ord(char), unicodedata.category(char), unicodedata.combining(char))
            for char in observed.surface)
    morphology: list[W02MorphologyCandidate] = []
    lookup_count = 0
    if (W02_CAPABILITY_UD_MORPHOLOGY in index.capabilities
            and index.max_lexeme_length > 0):
        for start in range(len(observed.surface)):
            limit = min(len(observed.surface), start + index.max_lexeme_length)
            for end in range(start + 1, limit + 1):
                lookup_count += 1
                form = observed.surface[start:end]
                for lemma, upos, feats_json, support in index.lexemes.get(form, ()):
                    morphology.append(W02MorphologyCandidate(
                        start, end, form, lemma, upos, feats_json, support))
    status = {
        "GENERATED": "PREDICTED",
        "AMBIGUOUS": "AMBIGUOUS",
        "UNKNOWN": "UNKNOWN",
    }[generation.status]
    result = W02CandidatePrediction(
        observed.observation_key, status, generation, points,
        unicode_units, tuple(morphology), index.capabilities)
    operations = (
        len(observed.surface) + len(rules) + len(points) + len(unicode_units)
        + lookup_count + len(morphology) + 8
    )
    return result, operations


def _expected_family(expected: dict[str, object]) -> str:
    keys = set(expected)
    if "oov_units" in keys:
        return "AUTHORED_OOV"
    if "morphology" in keys:
        return "UD_ANNOTATION"
    if "code_point_units" in keys:
        return "UNICODE_ANNOTATION"
    raise W02DevCalibrationError("W-02 dev expected family 未注册")


def _expected_boundaries(expected: dict[str, object], family: str) -> set[int]:
    if family == "AUTHORED_OOV":
        raw = expected.get("oov_boundaries")
        if not isinstance(raw, list) or any(type(item) is not int for item in raw):
            raise W02DevCalibrationError("W-02 dev OOV boundaries 非法")
        return set(raw)
    if family == "UD_ANNOTATION":
        raw = expected.get("boundary_spans")
        if not isinstance(raw, list):
            raise W02DevCalibrationError("W-02 dev UD boundaries 非法")
        return {value for row in raw for value in (row["start"], row["end"])}
    raw = expected.get("grapheme_candidate_boundaries")
    if not isinstance(raw, list) or any(type(item) is not int for item in raw):
        raise W02DevCalibrationError("W-02 dev Unicode boundaries 非法")
    return set(raw)


def _content_pass(
        prediction: W02CandidatePrediction,
        expected: dict[str, object],
        family: str,
        ) -> bool:
    if family == "AUTHORED_OOV":
        return _expected_boundaries(expected, family).issubset(
            set(prediction.boundary_lattice))
    if family == "UNICODE_ANNOTATION":
        units = expected.get("code_point_units")
        if not isinstance(units, list):
            return False
        target = [{
            "category": row["category"],
            "code_point": row["code_point"],
            "combining_class": row["combining_class"],
        } for row in units]
        return [item.to_dict() for item in prediction.unicode_units] == target
    rows = expected.get("morphology")
    if not isinstance(rows, list):
        return False
    actual = {
        (item.form, item.lemma, item.upos, item.feats_json)
        for item in prediction.morphology_candidates
    }
    target = {
        (row["form"], str(row["lemma"]), row["upos"],
         canonical_json_bytes(row["feats"]).decode("utf-8"))
        for row in rows
    }
    return bool(target) and target.issubset(actual)


def _evaluate_pair(
        observation: ObservationRecord,
        evaluation: EvaluatorLabelRecord,
        prediction: W02CandidatePrediction,
        dimension_by_key: dict[tuple[int, ...], str],
        ) -> tuple[str, str, bool | None, str]:
    dimension = dimension_by_key.get(evaluation.dimension_key.components)
    if dimension is None or evaluation.expected_state != "TRUE":
        return "UNREGISTERED", "UNREGISTERED", None, _hash_value(evaluation.to_dict())
    expected = evaluation.expected_payload.to_value()
    family = _expected_family(expected)
    language = observation.typed_payload.to_value()["language_payload"]
    generation_ok = (
        prediction.status == "PREDICTED"
        and prediction.generation.carrier_serialization
        == language["carrier_serialization"]
        and prediction.generation.content_span_start
        == language["content_span_start"]
        and prediction.generation.content_span_end
        == language["content_span_end"]
    )
    target_boundaries = _expected_boundaries(expected, family)
    boundary_ok = target_boundaries.issubset(set(prediction.boundary_lattice))
    if dimension == W02_DEV_DIMENSIONS[0]:
        passed = generation_ok and boundary_ok
    elif dimension == W02_DEV_DIMENSIONS[1]:
        passed = boundary_ok and len(prediction.boundary_lattice) >= 2
    elif dimension == W02_DEV_DIMENSIONS[2]:
        passed = _content_pass(prediction, expected, family)
    elif dimension == W02_DEV_DIMENSIONS[3]:
        passed = (
            prediction.status != "UNKNOWN"
            and 0 in prediction.boundary_lattice
            and len(prediction.generation.surface) in prediction.boundary_lattice
            and (family != "AUTHORED_OOV" or boundary_ok)
        )
    else:
        passed = generation_ok
    evidence_sha = _hash_value({
        "dimension": dimension,
        "evaluation_sha256": _hash_value(evaluation.to_dict()),
        "family": family,
        "passed": int(passed),
        "prediction_sha256": _hash_value(prediction.to_dict()),
    })
    return dimension, family, passed, evidence_sha


def _dimension_report(
        dimension: str,
        rows: list[tuple[bool | None, str]],
        ) -> dict[str, object]:
    passed = sum(value is True for value, _ in rows)
    failed = sum(value is False for value, _ in rows)
    ne = sum(value is None for value, _ in rows)
    status = "FAIL" if failed else "NE" if ne or not rows else "PASS"
    return {
        "denominator": len(rows),
        "dimension_key": dimension,
        "evidence_sha256": _hash_value([digest for _, digest in rows]),
        "failed": failed,
        "ne": ne,
        "numerator": passed,
        "status": status,
    }


def _code_rows(repository: Path) -> tuple[list[dict[str, object]], str]:
    rows = []
    for relative in W02_DEV_CODE_PATHS:
        target = (repository / Path(*PurePosixPath(relative).parts)).resolve()
        if (not target.is_relative_to(repository) or not target.is_file()
                or target.is_symlink()):
            raise W02DevCalibrationError("W-02 dev code freeze 文件缺失")
        size, digest = _sha256_file(target)
        rows.append({"repository_file": relative, "sha256": digest,
                     "size_bytes": size})
    return rows, _hash_value(rows)


def build_w02_dev_calibration_freeze(repository_root: str | Path) -> dict[str, object]:
    """冻结 dev runtime、输入、Candidate receipt 和 1/1 policy。"""
    repository = Path(repository_root).resolve()
    parent = read_w02_compile_freeze(repository)
    runtime = read_w02_candidate_runtime_freeze(repository)
    receipt_path = repository / Path(
        *PurePosixPath(W02_CANDIDATE_RECEIPT_PATH).parts)
    receipt = read_canonical_object(receipt_path)
    receipt_size, receipt_sha = _sha256_file(receipt_path)
    if (receipt.get("status") != "W02_CANDIDATE_ARTIFACT_FROZEN"
            or receipt.get("formal_training_runs") != 1
            or receipt.get("formal_private_evaluation_runs") != 0
            or receipt.get("private_payload_reads") != 0
            or receipt.get("teacher_calls") != 0
            or receipt.get("compile_freeze_sha256") != parent.sha256()
            or receipt.get("runtime_freeze_sha256") != runtime.sha256()):
        raise W02DevCalibrationError("W-02 Candidate receipt 状态非法")
    code_rows, code_sha = _code_rows(repository)
    dev_files = [_dev_freeze(parent, key).to_dict() for key in W02_DEV_LAYOUT_PATHS]
    policy = next(item for item in V2_STAGE_EVALUATION_POLICIES
                  if item.stage_key == "W-02")
    return {
        "artifact_kind": "PH2_D03_V2_W02_DEV_CALIBRATION_FREEZE",
        "artifact_version": W02_DEV_FREEZE_VERSION,
        "candidate_artifact_manifest_sha256":
            receipt["candidate_artifact_manifest_sha256"],
        "candidate_receipt_file_sha256": receipt_sha,
        "candidate_receipt_size_bytes": receipt_size,
        "candidate_semantic_sha256": receipt["candidate_semantic_sha256"],
        "code_files": code_rows,
        "code_freeze_sha256": code_sha,
        "compile_freeze_sha256": parent.sha256(),
        "dev_input_commitment": _hash_value(dev_files),
        "dev_input_files": dev_files,
        "evaluator_policy": policy.to_dict(),
        "formal_dev_calibration_runs": 0,
        "formal_private_evaluation_runs": 0,
        "formal_training_runs": 1,
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "next_action": "W02_FORMAL_DEV_CALIBRATION",
        "private_payload_reads": 0,
        "release_key": "PH2-D03-V2",
        "resource_budget": parent.resource_budget.to_dict(),
        "runtime_freeze_sha256": runtime.sha256(),
        "stage_key": "W-02",
        "status": "W02_DEV_CALIBRATION_FREEZE_COMPLETE",
        "teacher_calls": 0,
    }


def publish_w02_dev_calibration_freeze(repository_root: str | Path) -> Path:
    """不可覆盖发布 dev calibration 运行前冻结。"""
    repository = Path(repository_root).resolve()
    value = build_w02_dev_calibration_freeze(repository)
    target = repository / Path(*PurePosixPath(W02_DEV_FREEZE_PATH).parts)
    write_immutable_json(value, target)
    return target


def read_w02_dev_calibration_freeze(repository_root: str | Path) -> dict[str, object]:
    """严格回读 dev freeze，并重算现场代码和父绑定。"""
    repository = Path(repository_root).resolve()
    target = repository / Path(*PurePosixPath(W02_DEV_FREEZE_PATH).parts)
    value = read_canonical_object(target)
    expected = build_w02_dev_calibration_freeze(repository)
    if value != expected:
        raise W02DevCalibrationError("W-02 dev freeze 与现场身份漂移")
    return value


def run_w02_dev_calibration(
        repository_root: str | Path,
        dev_root: str | Path,
        artifact_root: str | Path,
        *,
        run_id: int = 1,
        ) -> dict[str, object]:
    """运行一次只读 dev calibration，真实 FAIL/NE 也原样返回。"""
    if run_id != 1:
        raise W02DevCalibrationError("W-02 dev formal run_id 固定为 1")
    repository = Path(repository_root).resolve()
    artifact = Path(artifact_root).resolve()
    dev = W02DevInputRoot(Path(dev_root))
    freeze = read_w02_dev_calibration_freeze(repository)
    before_dev = _tree_sha256(dev.root)
    before_candidate = _tree_sha256(artifact)
    parent = read_w02_compile_freeze(repository)
    source_count, source_sha = scan_w02_dev_sources(parent, dev)
    dimension_by_key = {
        _dimension_key(name).components: name for name in W02_DEV_DIMENSIONS
    }
    rows: dict[str, list[tuple[bool | None, str]]] = {
        name: [] for name in W02_DEV_DIMENSIONS
    }
    family_counts = {name: 0 for name in (
        "AUTHORED_OOV", "UD_ANNOTATION", "UNICODE_ANNOTATION")}
    logic_operations = 0
    evaluation_count = 0
    with open_w02_candidate_predictor(artifact) as predictor:
        index = load_w02_dev_candidate_index(predictor)
        for observation, evaluation in iter_w02_dev_pairs(parent, dev):
            prediction, operations = predict_w02_dev_observation(index, observation)
            dimension, family, passed, evidence_sha = _evaluate_pair(
                observation, evaluation, prediction, dimension_by_key)
            if dimension not in rows or family not in family_counts:
                raise W02DevCalibrationError("W-02 dev dimension/family 漂移")
            rows[dimension].append((passed, evidence_sha))
            family_counts[family] += 1
            logic_operations += operations + 8
            evaluation_count += 1
            if logic_operations > parent.resource_budget.max_logic_operations:
                raise W02DevCalibrationError("W-02 dev logic resource stop")
    dimensions = [_dimension_report(name, rows[name]) for name in W02_DEV_DIMENSIONS]
    failed = sum(item["status"] == "FAIL" for item in dimensions)
    ne = sum(item["status"] == "NE" for item in dimensions)
    status = "FAIL" if failed else "NE" if ne else "PASS"
    after_candidate = _tree_sha256(artifact)
    after_dev = _tree_sha256(dev.root)
    if before_candidate != after_candidate or before_dev != after_dev:
        raise W02DevCalibrationError("W-02 dev calibration 产生非授权写入")
    input_bytes = sum(
        _dev_freeze(parent, key).transport_size_bytes for key in W02_DEV_LAYOUT_PATHS)
    report = {
        "artifact_kind": "PH2_D03_V2_W02_DEV_CALIBRATION_REPORT",
        "artifact_version": W02_DEV_REPORT_VERSION,
        "candidate_artifact_manifest_sha256":
            freeze["candidate_artifact_manifest_sha256"],
        "candidate_index_row_count": index.row_count,
        "candidate_index_semantic_sha256": index.semantic_sha256,
        "candidate_semantic_sha256": freeze["candidate_semantic_sha256"],
        "code_freeze_sha256": freeze["code_freeze_sha256"],
        "compile_freeze_sha256": freeze["compile_freeze_sha256"],
        "dev_input_commitment": freeze["dev_input_commitment"],
        "dimension_results": dimensions,
        "evaluator_record_reads": evaluation_count,
        "family_counts": family_counts,
        "formal_dev_calibration_runs": 1,
        "formal_private_evaluation_runs": 0,
        "formal_training_runs": 1,
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "logic_operations": logic_operations,
        "next_action": (
            "W02_SHADOW_AUDIT" if status == "PASS"
            else "W02_DEV_FAILED_SUCCESSOR_REQUIRED"),
        "observation_reads": evaluation_count,
        "private_payload_reads": 0,
        "release_key": "PH2-D03-V2",
        "run_id": run_id,
        "runtime_freeze_sha256": freeze["runtime_freeze_sha256"],
        "source_count": source_count,
        "source_identity_sha256": source_sha,
        "stage_key": "W-02",
        "status": status,
        "teacher_calls": 0,
        "transport_bytes_read": input_bytes,
        "validated_layout_count": len(W02_DEV_LAYOUT_PATHS),
        "zero_write_audit": {
            "candidate_writes": 0,
            "companion_writes": 0,
            "core_writes": 0,
            "dev_owner_writes": 0,
            "evidence_writes": 0,
            "memory_writes": 0,
            "use_writes": 0,
        },
    }
    validate_v2_safe_report(report)
    return report


def publish_w02_dev_calibration_report(
        repository_root: str | Path,
        external_report: str | Path,
        ) -> Path:
    """回读 Git 外正式报告并不可覆盖发布 safe public projection。"""
    repository = Path(repository_root).resolve()
    value = read_canonical_object(external_report)
    validate_v2_safe_report(value)
    freeze_path = repository / Path(*PurePosixPath(W02_DEV_FREEZE_PATH).parts)
    freeze_size, freeze_sha = _sha256_file(freeze_path)
    if (value.get("artifact_version") != W02_DEV_REPORT_VERSION
            or value.get("formal_dev_calibration_runs") != 1
            or value.get("formal_private_evaluation_runs") != 0
            or value.get("private_payload_reads") != 0
            or value.get("teacher_calls") != 0
            or value.get("code_freeze_sha256")
            != read_canonical_object(freeze_path).get("code_freeze_sha256")):
        raise W02DevCalibrationError("W-02 dev 正式报告状态非法")
    public = dict(value)
    public["dev_freeze_file_sha256"] = freeze_sha
    public["dev_freeze_size_bytes"] = freeze_size
    validate_v2_safe_report(public)
    target = repository / Path(*PurePosixPath(W02_DEV_REPORT_PATH).parts)
    write_immutable_json(public, target)
    return target


__all__ = [
    "W02_DEV_CODE_PATHS",
    "W02_DEV_DIMENSIONS",
    "W02_DEV_FREEZE_PATH",
    "W02_DEV_REPORT_PATH",
    "W02DevCalibrationError",
    "W02DevCandidateIndex",
    "W02DevInputRoot",
    "build_w02_dev_calibration_freeze",
    "iter_w02_dev_pairs",
    "iter_w02_dev_records",
    "load_w02_dev_candidate_index",
    "predict_w02_dev_observation",
    "publish_w02_dev_calibration_freeze",
    "publish_w02_dev_calibration_report",
    "read_w02_dev_calibration_freeze",
    "run_w02_dev_calibration",
    "scan_w02_dev_sources",
]
