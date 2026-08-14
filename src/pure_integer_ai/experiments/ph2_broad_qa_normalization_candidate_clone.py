"""把 normalization v3 learned records 编译为禁用态 candidate clone。

clone 只执行一对一码点规则及来源化 exact-context defeater。它不改变公开
production gate，不读取 evaluation，也不把执行结果写回 learned pack。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_rule_records_v3 import (
    BroadQaNormalizationAcceptedRuleV3,
    BroadQaNormalizationRejectedTrialV3,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_CANDIDATE_CLONE_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_CANDIDATE_CLONE_V1")
NORMALIZATION_CANDIDATE_CLONE_STATUS = (
    "CANDIDATE_CLONE_ONLY_PUBLIC_PRODUCTION_DISABLED")


def _sha256(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _codepoint(value: object, *, label: str) -> int:
    """核验 Unicode scalar value。"""
    if (type(value) is not int or not 0 <= value <= 0x10FFFF
            or 0xD800 <= value <= 0xDFFF):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _text(value: object, *, label: str) -> str:
    """核验非空且不含 surrogate 的运行文本。"""
    if (not isinstance(value, str) or not value
            or any(0xD800 <= ord(item) <= 0xDFFF for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationContextDefeaterProgram:
    """一个绑定完整短语、offset 和观察输出的可执行 context predicate。"""

    trial_id: str
    candidate_id: str
    defeater_key: tuple[int, ...]
    phrase_source: str
    source_codepoint_offset: int
    input_codepoint: int
    candidate_output_codepoint: int
    observed_output_codepoint: int

    def __post_init__(self) -> None:
        """拒绝 identity-only、错 offset 或未形成反例的 defeater。"""
        _sha256(self.trial_id, label="normalization clone trial id")
        _sha256(self.candidate_id, label="normalization clone candidate id")
        phrase = _text(
            self.phrase_source, label="normalization clone defeater phrase")
        for label, value in (
                ("input codepoint", self.input_codepoint),
                ("candidate output", self.candidate_output_codepoint),
                ("observed output", self.observed_output_codepoint)):
            _codepoint(value, label=f"normalization clone {label}")
        if (not isinstance(self.defeater_key, tuple) or not self.defeater_key
                or any(type(item) is not int or item < 0
                       for item in self.defeater_key)
                or type(self.source_codepoint_offset) is not int
                or not 0 <= self.source_codepoint_offset < len(phrase)
                or ord(phrase[self.source_codepoint_offset])
                != self.input_codepoint
                or self.observed_output_codepoint
                == self.candidate_output_codepoint):
            raise BroadQaExternalDataError(
                "normalization clone defeater 不是可执行 context 反例")

    def matches(self, text: str, position: int) -> bool:
        """判断当前位置是否满足冻结 phrase 与 offset。"""
        start = position - self.source_codepoint_offset
        return (start >= 0
                and start + len(self.phrase_source) <= len(text)
                and text[start:start + len(self.phrase_source)]
                == self.phrase_source)

    def to_dict(self) -> dict[str, object]:
        """导出规范 clone defeater。"""
        return {
            "candidate_id": self.candidate_id,
            "candidate_output_codepoint": self.candidate_output_codepoint,
            "defeater_key": list(self.defeater_key),
            "input_codepoint": self.input_codepoint,
            "observed_output_codepoint": self.observed_output_codepoint,
            "phrase_source": self.phrase_source,
            "source_codepoint_offset": self.source_codepoint_offset,
            "trial_id": self.trial_id,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationCandidateRuleProgram:
    """一个 accepted mapping 及其声明和可执行 defeater。"""

    mapping_candidate_id: str
    input_codepoint: int
    output_codepoint: int
    declared_defeater_keys: tuple[tuple[int, ...], ...]
    defeaters: tuple[NormalizationContextDefeaterProgram, ...]

    def __post_init__(self) -> None:
        """要求规则和 defeater 规范排序且不冲突。"""
        _sha256(
            self.mapping_candidate_id,
            label="normalization clone mapping candidate",
        )
        _codepoint(self.input_codepoint, label="normalization clone rule input")
        _codepoint(self.output_codepoint, label="normalization clone rule output")
        declared = self.declared_defeater_keys
        if (not isinstance(declared, tuple) or not declared
                or any(not isinstance(item, tuple) or not item
                       or any(type(value) is not int or value < 0
                              for value in item)
                       for item in declared)
                or declared != tuple(sorted(set(declared)))
                or not isinstance(self.defeaters, tuple)
                or any(not isinstance(item, NormalizationContextDefeaterProgram)
                       or item.candidate_id != self.mapping_candidate_id
                       or item.input_codepoint != self.input_codepoint
                       or item.candidate_output_codepoint != self.output_codepoint
                       for item in self.defeaters)
                or tuple(item.trial_id for item in self.defeaters)
                != tuple(sorted(set(item.trial_id for item in self.defeaters)))):
            raise BroadQaExternalDataError(
                "normalization clone rule/defeater inventory 漂移")

    def to_dict(self) -> dict[str, object]:
        """导出规范 clone rule。"""
        return {
            "declared_defeater_keys": [
                list(item) for item in self.declared_defeater_keys],
            "defeaters": [item.to_dict() for item in self.defeaters],
            "input_codepoint": self.input_codepoint,
            "mapping_candidate_id": self.mapping_candidate_id,
            "output_codepoint": self.output_codepoint,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationCandidateCloneProgram:
    """禁用公开生产 gate 的确定性 normalization clone 程序。"""

    rule_pack_manifest_sha256: str
    rules: tuple[NormalizationCandidateRuleProgram, ...]
    production_enabled: int = 0
    status: str = NORMALIZATION_CANDIDATE_CLONE_STATUS

    def __post_init__(self) -> None:
        """要求输入码点唯一并固定禁用态。"""
        _sha256(
            self.rule_pack_manifest_sha256,
            label="normalization clone rule pack manifest",
        )
        inputs = tuple(item.input_codepoint for item in self.rules)
        identities = tuple(item.mapping_candidate_id for item in self.rules)
        if (not isinstance(self.rules, tuple) or not self.rules
                or any(not isinstance(item, NormalizationCandidateRuleProgram)
                       for item in self.rules)
                or inputs != tuple(sorted(set(inputs)))
                or len(set(identities)) != len(identities)
                or type(self.production_enabled) is not int
                or self.production_enabled != 0
                or self.status != NORMALIZATION_CANDIDATE_CLONE_STATUS):
            raise BroadQaExternalDataError(
                "normalization clone program 非唯一或错误启用")

    def to_dict(self) -> dict[str, object]:
        """导出不含运行输入的 clone identity。"""
        return {
            "artifact_kind": NORMALIZATION_CANDIDATE_CLONE_KIND,
            "format_version": 1,
            "production_enabled": self.production_enabled,
            "rule_pack_manifest_sha256": self.rule_pack_manifest_sha256,
            "rules": [item.to_dict() for item in self.rules],
            "status": self.status,
        }

    def sha256(self) -> str:
        """返回 clone 程序规范摘要。"""
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationCloneStep:
    """一个输入位置的规则选择和 defeater 命中轨迹。"""

    position: int
    input_codepoint: int
    output_codepoint: int
    mapping_candidate_id: str
    defeater_trial_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """导出单位置规范轨迹。"""
        return {
            "defeater_trial_ids": list(self.defeater_trial_ids),
            "input_codepoint": self.input_codepoint,
            "mapping_candidate_id": self.mapping_candidate_id,
            "output_codepoint": self.output_codepoint,
            "position": self.position,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class NormalizationCloneResult:
    """candidate clone 的长度保持输出和逐位置轨迹。"""

    input_text: str
    output_text: str
    steps: tuple[NormalizationCloneStep, ...]

    def __post_init__(self) -> None:
        """要求轨迹完整覆盖输入且输出逐位置一致。"""
        _text(self.input_text, label="normalization clone input")
        _text(self.output_text, label="normalization clone output")
        if (len(self.input_text) != len(self.output_text)
                or len(self.steps) != len(self.input_text)
                or tuple(item.position for item in self.steps)
                != tuple(range(len(self.input_text)))
                or any(item.input_codepoint != ord(self.input_text[item.position])
                       or item.output_codepoint
                       != ord(self.output_text[item.position])
                       for item in self.steps)):
            raise BroadQaExternalDataError(
                "normalization clone 轨迹与文本不一致")

    def to_dict(self) -> dict[str, object]:
        """导出可重放的规范运行结果。"""
        return {
            "input_text": self.input_text,
            "output_text": self.output_text,
            "steps": [item.to_dict() for item in self.steps],
        }

    def sha256(self) -> str:
        """返回规范运行摘要。"""
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


def compile_normalization_candidate_clone(
        *,
        rule_pack_manifest_sha256: str,
        accepted_rules: tuple[BroadQaNormalizationAcceptedRuleV3, ...],
        rejected_trials: tuple[BroadQaNormalizationRejectedTrialV3, ...],
        contrastive_trials: tuple[dict[str, object], ...],
        ) -> NormalizationCandidateCloneProgram:
    """从已严格回读的 pack 与 TRAIN_SOURCE 编译 exact-context clone。"""
    _sha256(
        rule_pack_manifest_sha256,
        label="normalization clone rule pack manifest",
    )
    if (not isinstance(accepted_rules, tuple) or not accepted_rules
            or not isinstance(rejected_trials, tuple) or not rejected_trials
            or not isinstance(contrastive_trials, tuple)
            or not contrastive_trials):
        raise BroadQaExternalDataError(
            "normalization clone 输入 records 为空或类型漂移")
    trial_by_id = {}
    for trial in contrastive_trials:
        trial_id = trial.get("trial_id") if isinstance(trial, dict) else None
        if (not isinstance(trial_id, str) or trial_id in trial_by_id):
            raise BroadQaExternalDataError(
                "normalization clone contrastive trial identity 漂移")
        trial_by_id[trial_id] = trial
    rejected_by_candidate: dict[
        str, list[BroadQaNormalizationRejectedTrialV3]] = {}
    for rejected in rejected_trials:
        if not isinstance(rejected, BroadQaNormalizationRejectedTrialV3):
            raise BroadQaExternalDataError(
                "normalization clone rejected trial 类型漂移")
        rejected_by_candidate.setdefault(
            rejected.candidate.mapping_candidate_id, []).append(rejected)
    programs = []
    for accepted in accepted_rules:
        if not isinstance(accepted, BroadQaNormalizationAcceptedRuleV3):
            raise BroadQaExternalDataError(
                "normalization clone accepted rule 类型漂移")
        candidate = accepted.candidate
        compiled_defeaters = []
        rejected_records = rejected_by_candidate.get(
            candidate.mapping_candidate_id, [])
        if (tuple(sorted(item.sha256() for item in rejected_records))
                != accepted.rejection_record_sha256s):
            raise BroadQaExternalDataError(
                "normalization clone rejection ledger 漂移")
        for rejected in sorted(rejected_records, key=lambda item: item.trial_id):
            trial = trial_by_id.get(rejected.trial_id)
            if not isinstance(trial, dict):
                raise BroadQaExternalDataError(
                    "normalization clone rejected trial 缺少来源记录")
            expected = {
                "candidate_id": candidate.mapping_candidate_id,
                "candidate_output_codepoint": candidate.output_codepoint,
                "source_codepoint": candidate.input_codepoint,
                "qualification_kind": "SOURCE_REPLAY_REFUTE",
                "trial_id": rejected.trial_id,
            }
            if any(trial.get(key) != value for key, value in expected.items()):
                raise BroadQaExternalDataError(
                    "normalization clone rejected trial/source 漂移")
            commitments = rejected.evidence_commitments
            if (not commitments or any(
                    item.trial_id != rejected.trial_id
                    or item.candidate_id != candidate.mapping_candidate_id
                    or item.input_codepoint != trial.get("source_codepoint")
                    or item.candidate_output_codepoint
                    != trial.get("candidate_output_codepoint")
                    or item.observed_output_codepoint
                    != trial.get("observed_output_codepoint")
                    or item.source_codepoint_offset
                    != trial.get("source_codepoint_offset")
                    or item.qualification_kind != "SOURCE_REPLAY_REFUTE"
                    for item in commitments)):
                raise BroadQaExternalDataError(
                    "normalization clone rejected Evidence/source 漂移")
            compiled_defeaters.append(NormalizationContextDefeaterProgram(
                rejected.trial_id,
                candidate.mapping_candidate_id,
                rejected.context_defeater.stable_key(),
                trial.get("phrase_source"),
                trial.get("source_codepoint_offset"),
                candidate.input_codepoint,
                candidate.output_codepoint,
                trial.get("observed_output_codepoint"),
            ))
        programs.append(NormalizationCandidateRuleProgram(
            candidate.mapping_candidate_id,
            candidate.input_codepoint,
            candidate.output_codepoint,
            tuple(item.stable_key() for item in candidate.defeaters),
            tuple(compiled_defeaters),
        ))
    return NormalizationCandidateCloneProgram(
        rule_pack_manifest_sha256,
        tuple(sorted(programs, key=lambda item: item.input_codepoint)),
    )


def _matching_defeaters(
        rule: NormalizationCandidateRuleProgram,
        text: str,
        position: int,
        ) -> tuple[NormalizationContextDefeaterProgram, ...]:
    """返回当前位置全部 exact-context 命中并拒绝输出冲突。"""
    matches = tuple(item for item in rule.defeaters
                    if item.matches(text, position))
    outputs = {item.observed_output_codepoint for item in matches}
    if len(outputs) > 1:
        raise BroadQaExternalDataError(
            "normalization clone 同一位置 defeater 输出冲突")
    return matches


def execute_normalization_candidate_clone(
        program: NormalizationCandidateCloneProgram,
        text: str,
        ) -> NormalizationCloneResult:
    """通过码点索引执行 candidate clone，公开 production gate 保持关闭。"""
    if not isinstance(program, NormalizationCandidateCloneProgram):
        raise TypeError("normalization clone program 类型非法")
    source = _text(text, label="normalization clone runtime input")
    by_input = {item.input_codepoint: item for item in program.rules}
    output = []
    steps = []
    for position, character in enumerate(source):
        input_codepoint = ord(character)
        rule = by_input.get(input_codepoint)
        if rule is None:
            output_codepoint = input_codepoint
            candidate_id = ""
            trial_ids: tuple[str, ...] = ()
        else:
            matches = _matching_defeaters(rule, source, position)
            output_codepoint = (
                matches[0].observed_output_codepoint
                if matches else rule.output_codepoint)
            candidate_id = rule.mapping_candidate_id
            trial_ids = tuple(item.trial_id for item in matches)
        output.append(chr(output_codepoint))
        steps.append(NormalizationCloneStep(
            position,
            input_codepoint,
            output_codepoint,
            candidate_id,
            trial_ids,
        ))
    return NormalizationCloneResult(source, "".join(output), tuple(steps))


def reference_normalization_candidate_rewrite(
        program: NormalizationCandidateCloneProgram,
        text: str,
        ) -> NormalizationCloneResult:
    """用线性规则扫描独立重放 clone 语义，供 evaluator 对照。"""
    if not isinstance(program, NormalizationCandidateCloneProgram):
        raise TypeError("normalization reference program 类型非法")
    source = _text(text, label="normalization reference input")
    output = []
    steps = []
    for position, character in enumerate(source):
        input_codepoint = ord(character)
        candidates = tuple(
            item for item in program.rules
            if item.input_codepoint == input_codepoint)
        if len(candidates) > 1:
            raise BroadQaExternalDataError(
                "normalization reference 规则输入冲突")
        if not candidates:
            output_codepoint = input_codepoint
            candidate_id = ""
            trial_ids: tuple[str, ...] = ()
        else:
            rule = candidates[0]
            matches = _matching_defeaters(rule, source, position)
            output_codepoint = (
                matches[0].observed_output_codepoint
                if matches else rule.output_codepoint)
            candidate_id = rule.mapping_candidate_id
            trial_ids = tuple(item.trial_id for item in matches)
        output.append(chr(output_codepoint))
        steps.append(NormalizationCloneStep(
            position,
            input_codepoint,
            output_codepoint,
            candidate_id,
            trial_ids,
        ))
    return NormalizationCloneResult(source, "".join(output), tuple(steps))


__all__ = [
    "NORMALIZATION_CANDIDATE_CLONE_KIND",
    "NORMALIZATION_CANDIDATE_CLONE_STATUS",
    "NormalizationCandidateCloneProgram",
    "NormalizationCandidateRuleProgram",
    "NormalizationCloneResult",
    "NormalizationCloneStep",
    "NormalizationContextDefeaterProgram",
    "compile_normalization_candidate_clone",
    "execute_normalization_candidate_clone",
    "reference_normalization_candidate_rewrite",
]
