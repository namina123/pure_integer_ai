"""首个来源绑定纵向问答的不可变合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_contract import (
    W03W04W05VerticalQuery,
    W03W04W05VerticalResult,
)


W03_W04_W05_QUESTION_ANSWER_STATUSES = {"ANSWER", "UNKNOWN", "CLARIFY"}


# object-model: exception
class W03W04W05QuestionAnswerError(ValueError):
    """类型化问题、已学证明路径或回答投影发生漂移。"""


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _text(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise W03W04W05QuestionAnswerError(
            f"{where} is not canonical text")
    return value


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W03W04W05QuestionAnswerError(
            f"{where} is not a strict integer key")
    return value


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise W03W04W05QuestionAnswerError(f"{where} is not SHA-256")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03W04W05QuestionRequest:
    """保存真实问题表层与类型化目标角色，不携带回答标签。

    ``target_role_keys`` 是问题理解的输出边界；理解有歧义时可包含多个
    候选，本探针不会在候选之间猜测。
    """

    question_surface: str
    vertical_query: W03W04W05VerticalQuery
    target_role_keys: tuple[tuple[int, ...], ...]
    source_record_key: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _text(self.question_surface, where="question surface")
        if not isinstance(self.vertical_query, W03W04W05VerticalQuery):
            raise W03W04W05QuestionAnswerError(
                "question vertical query drifted")
        if (not isinstance(self.target_role_keys, tuple)
                or not self.target_role_keys):
            raise W03W04W05QuestionAnswerError(
                "question target roles are empty")
        for item in self.target_role_keys:
            _strict_key(item, where="question target role")
        if (tuple(sorted(self.target_role_keys)) != self.target_role_keys
                or len(set(self.target_role_keys))
                != len(self.target_role_keys)):
            raise W03W04W05QuestionAnswerError(
                "question target roles are not canonical")
        if self.source_record_key is not None:
            _strict_key(
                self.source_record_key,
                where="question source record key",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "question_surface": self.question_surface,
            "source_record_key": (
                None if self.source_record_key is None
                else list(self.source_record_key)
            ),
            "target_role_keys": [
                list(item) for item in self.target_role_keys],
            "vertical_query": self.vertical_query.to_dict(),
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03W04W05AnswerProof:
    """从三阶段 Observation 到唯一回答 occurrence 的精确已学路径。"""

    source_record_key: tuple[int, ...]
    source_ref_key: tuple[int, ...]
    source_commitment: str
    w03_observation_key: tuple[int, ...]
    w04_observation_key: tuple[int, ...]
    w05_observation_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    role_binding_key: tuple[int, ...]
    role_key: tuple[int, ...]
    filler_key: tuple[int, ...]
    answer_occurrence_key: tuple[int, ...]
    answer_start: int
    answer_end: int
    reasoning_status: str
    generation_status: str
    generation_construction_key: tuple[int, ...]
    generation_option_sha256: str
    generated_proposition_surface: str

    def __post_init__(self) -> None:
        for name in (
                "source_record_key", "source_ref_key",
                "w03_observation_key", "w04_observation_key",
                "w05_observation_key", "proposition_key", "predicate_key",
                "role_binding_key", "role_key", "filler_key",
                "answer_occurrence_key", "generation_construction_key"):
            _strict_key(getattr(self, name), where=f"answer proof {name}")
        _sha256(self.source_commitment, where="answer source commitment")
        _sha256(
            self.generation_option_sha256,
            where="answer generation option",
        )
        if (type(self.answer_start) is not int
                or type(self.answer_end) is not int
                or self.answer_start < 0
                or self.answer_end <= self.answer_start):
            raise W03W04W05QuestionAnswerError(
                "answer occurrence span drifted")
        _text(self.reasoning_status, where="answer reasoning status")
        _text(self.generation_status, where="answer generation status")
        _text(
            self.generated_proposition_surface,
            where="generated proposition surface",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "answer_end": self.answer_end,
            "answer_occurrence_key": list(self.answer_occurrence_key),
            "answer_start": self.answer_start,
            "filler_key": list(self.filler_key),
            "generated_proposition_surface": (
                self.generated_proposition_surface),
            "generation_construction_key": list(
                self.generation_construction_key),
            "generation_option_sha256": self.generation_option_sha256,
            "generation_status": self.generation_status,
            "predicate_key": list(self.predicate_key),
            "proposition_key": list(self.proposition_key),
            "reasoning_status": self.reasoning_status,
            "role_binding_key": list(self.role_binding_key),
            "role_key": list(self.role_key),
            "source_commitment": self.source_commitment,
            "source_record_key": list(self.source_record_key),
            "source_ref_key": list(self.source_ref_key),
            "w03_observation_key": list(self.w03_observation_key),
            "w04_observation_key": list(self.w04_observation_key),
            "w05_observation_key": list(self.w05_observation_key),
        }


def _validate_answer_path(
        result: "W03W04W05QuestionAnswerResult",
        ) -> None:
    vertical = result.vertical_result
    proof = result.proof
    if vertical.status != "BRIDGED" or vertical.link is None or proof is None:
        raise W03W04W05QuestionAnswerError(
            "ANSWER lacks a bridged vertical proof")
    link = vertical.link
    if (
            result.request.target_role_keys != (proof.role_key,)
            or (result.request.source_record_key is not None
                and result.request.source_record_key
                != proof.source_record_key)
            or proof.source_record_key != link.source_ref_key
            or proof.source_commitment != link.source_commitment
            or proof.w03_observation_key != link.w03_observation_key
            or proof.w04_observation_key != link.w04_observation_key
            or proof.w05_observation_key != link.w05_observation_key
            or proof.proposition_key != link.proposition_key
            or proof.predicate_key != link.predicate_key):
        raise W03W04W05QuestionAnswerError(
            "answer proof escaped the vertical link")
    bridge = vertical.w04_w05.link
    if (bridge is None
            or proof.role_binding_key not in bridge.role_binding_keys
            or proof.answer_occurrence_key not in bridge.occurrence_order):
        raise W03W04W05QuestionAnswerError(
            "answer proof escaped the W-05 bridge structure")
    w05 = vertical.w04_w05.w05_result
    if (w05.status != "UNIQUE"
            or w05.selected_proposition_key != proof.proposition_key
            or w05.selected_reasoning_status != "AUTHORIZED"
            or w05.generation_status != "READY"):
        raise W03W04W05QuestionAnswerError(
            "answer is not authorized by learned reasoning and generation")
    candidates = tuple(
        item for item in w05.candidates
        if item.proposition_key == proof.proposition_key)
    if len(candidates) != 1:
        raise W03W04W05QuestionAnswerError(
            "answer Proposition is not unique")
    candidate = candidates[0]
    if (candidate.active != 1
            or candidate.lifecycle_status != "ACTIVE"
            or candidate.reasoning_status != proof.reasoning_status
            or proof.reasoning_status != "AUTHORIZED"
            or candidate.source_record_key != proof.source_record_key
            or candidate.source_ref_key != proof.source_ref_key
            or candidate.source_commitment != proof.source_commitment
            or candidate.predicate_key != proof.predicate_key
            or candidate.occurrence_order != bridge.occurrence_order
            or tuple(item.identity_key for item in candidate.role_bindings)
            != bridge.role_binding_keys):
        raise W03W04W05QuestionAnswerError(
            "answer candidate is not the exact learned Proposition")
    bindings = tuple(
        item for item in candidate.role_bindings
        if item.identity_key == proof.role_binding_key
        and item.role_key == proof.role_key
        and item.filler_key == proof.filler_key)
    occurrences = tuple(
        item for item in candidate.occurrences
        if item.identity_key == proof.answer_occurrence_key
        and item.semantic_object_key == proof.filler_key)
    if (len(bindings) != 1 or len(occurrences) != 1
            or occurrences[0].start != proof.answer_start
            or occurrences[0].end != proof.answer_end
            or occurrences[0].surface_fragment != result.answer_surface):
        raise W03W04W05QuestionAnswerError(
            "answer surface is not projected from the learned RoleBinding")
    generation = tuple(
        item for item in w05.generation_options
        if _sha(item.to_dict()) == proof.generation_option_sha256)
    if len(generation) != 1:
        raise W03W04W05QuestionAnswerError(
            "answer generation option is not unique")
    option = generation[0]
    if (proof.generation_status != "READY"
            or option.construction_key != proof.generation_construction_key
            or option.surface != proof.generated_proposition_surface
            or option.target_proposition_key != candidate.proposition_key
            or option.target_predicate_key != candidate.predicate_key
            or option.target_source_ref_key != candidate.source_ref_key
            or option.target_source_commitment != candidate.source_commitment
            or option.context_key != candidate.context_key
            or option.occurrence_order != candidate.occurrence_order
            or option.role_binding_keys
            != tuple(item.identity_key for item in candidate.role_bindings)):
        raise W03W04W05QuestionAnswerError(
            "answer is not bound to the learned Generation option")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03W04W05QuestionAnswerResult:
    """保存一个回答或闭锁停止，以及不可变的纵向审计轨迹。"""

    request: W03W04W05QuestionRequest
    status: str
    answer_surface: str | None
    proof: W03W04W05AnswerProof | None
    vertical_result: W03W04W05VerticalResult
    state_before_sha256: str
    state_after_sha256: str
    experimental: int = 1
    formal_mastery_claim: int = 0
    w03_started: int = 0
    w04_started: int = 0
    w05_started: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.request, W03W04W05QuestionRequest)
                or self.status not in W03_W04_W05_QUESTION_ANSWER_STATUSES
                or not isinstance(
                    self.vertical_result, W03W04W05VerticalResult)):
            raise W03W04W05QuestionAnswerError(
                "question answer result projection drifted")
        if self.vertical_result.query != self.request.vertical_query:
            raise W03W04W05QuestionAnswerError(
                "question answer replaced the requested vertical query")
        _sha256(self.state_before_sha256, where="question state before")
        _sha256(self.state_after_sha256, where="question state after")
        if self.state_before_sha256 != self.state_after_sha256:
            raise W03W04W05QuestionAnswerError(
                "question query changed learned public state")
        if self.status == "ANSWER":
            _text(self.answer_surface, where="answer surface")
            if not isinstance(self.proof, W03W04W05AnswerProof):
                raise W03W04W05QuestionAnswerError(
                    "ANSWER lacks an immutable proof")
            _validate_answer_path(self)
        elif self.answer_surface is not None or self.proof is not None:
            raise W03W04W05QuestionAnswerError(
                "non-answer result published answer content")
        if (self.experimental, self.formal_mastery_claim, self.w03_started,
                self.w04_started, self.w05_started) != (1, 0, 0, 0, 0):
            raise W03W04W05QuestionAnswerError(
                "question answer boundary flags drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "answer_surface": self.answer_surface,
            "experimental": self.experimental,
            "formal_mastery_claim": self.formal_mastery_claim,
            "proof": None if self.proof is None else self.proof.to_dict(),
            "request": self.request.to_dict(),
            "state_after_sha256": self.state_after_sha256,
            "state_before_sha256": self.state_before_sha256,
            "status": self.status,
            "vertical_result": self.vertical_result.to_dict(),
            "w03_started": self.w03_started,
            "w04_started": self.w04_started,
            "w05_started": self.w05_started,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


__all__ = [
    "W03_W04_W05_QUESTION_ANSWER_STATUSES",
    "W03W04W05AnswerProof",
    "W03W04W05QuestionAnswerError",
    "W03W04W05QuestionAnswerResult",
    "W03W04W05QuestionRequest",
]
