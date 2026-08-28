"""Portable integer routing model for lazily loaded dialogue experts."""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path

from pure_integer_ai.experiments.conversation_dialogue_experts import (
    learned_domain_activation_features,
    learned_grounded_domain_activation_features,
)
from pure_integer_ai.experiments.conversation_learned_dialogue_response import (
    LearnedDialogueResponseModel,
    dialogue_prompt_features,
)
from pure_integer_ai.storage.integer_codec import (
    decode_integer_tuple,
    encode_integer_tuple,
)


DIALOGUE_EXPERT_ROUTING_FILE = "model/dialogue_expert_router.int"
DIALOGUE_EXPERT_ROUTING_MAGIC = (21404, 260827, 74)
DIALOGUE_EXPERT_ROUTING_SCHEMA = 2


class DialogueExpertRoutingError(ValueError):
    """The ordered domain activation model is malformed or incompatible."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# object-model: value; representation=struct; interop=dialogue-expert-router-v1
@dataclass(frozen=True, slots=True)
class DialogueExpertRoutingModel:
    """Ordered domain activation feature sets with no host object identity."""

    general_course_sha256: tuple[int, ...]
    domain_course_sha256s: tuple[tuple[int, ...], ...]
    domain_activation_features: tuple[tuple[tuple[int, ...], ...], ...]

    def __post_init__(self) -> None:
        shas = (self.general_course_sha256, *self.domain_course_sha256s)
        if (any(not isinstance(value, tuple) or len(value) != 32
                or any(type(item) is not int or not 0 <= item <= 255
                       for item in value) for value in shas)
                or len(self.domain_course_sha256s)
                != len(self.domain_activation_features)):
            raise DialogueExpertRoutingError("routing course SHA 绑定非法")
        if (not isinstance(self.domain_activation_features, tuple)
                or not self.domain_activation_features):
            raise DialogueExpertRoutingError("domain activation family 为空")
        for features in self.domain_activation_features:
            if (not isinstance(features, tuple) or not features
                    or features != tuple(sorted(set(features)))):
                raise DialogueExpertRoutingError("domain activation feature 非规范")
            for feature in features:
                if (not isinstance(feature, tuple) or not feature
                        or any(type(item) is not int or item < 0
                               or item > 0x10FFFF or 0xD800 <= item <= 0xDFFF
                               for item in feature)):
                    raise DialogueExpertRoutingError(
                        "domain activation scalar 非规范")

    def integer_stream(self) -> tuple[int, ...]:
        result = [
            *DIALOGUE_EXPERT_ROUTING_MAGIC,
            DIALOGUE_EXPERT_ROUTING_SCHEMA,
            *self.general_course_sha256,
            len(self.domain_activation_features),
        ]
        for course_sha, features in zip(
                self.domain_course_sha256s,
                self.domain_activation_features):
            result.extend(course_sha)
            result.append(len(features))
            for feature in features:
                result.extend((len(feature), *feature))
        return tuple(result)

    @classmethod
    def from_integer_stream(
            cls, stream: tuple[int, ...],
            ) -> "DialogueExpertRoutingModel":
        if not isinstance(stream, tuple) or any(type(item) is not int
                                                for item in stream):
            raise DialogueExpertRoutingError("routing stream 非整数 tuple")
        cursor = 0

        def take() -> int:
            nonlocal cursor
            if cursor >= len(stream):
                raise DialogueExpertRoutingError("routing stream 被截断")
            value = stream[cursor]
            cursor += 1
            return value

        if tuple(take() for _ in DIALOGUE_EXPERT_ROUTING_MAGIC) != (
                DIALOGUE_EXPERT_ROUTING_MAGIC):
            raise DialogueExpertRoutingError("routing magic 不兼容")
        if take() != DIALOGUE_EXPERT_ROUTING_SCHEMA:
            raise DialogueExpertRoutingError("routing schema 不兼容")
        general_sha = tuple(take() for _ in range(32))
        domain_count = take()
        domains = []
        domain_shas = []
        for _ in range(domain_count):
            domain_shas.append(tuple(take() for _ in range(32)))
            feature_count = take()
            features = []
            for _ in range(feature_count):
                width = take()
                features.append(tuple(take() for _ in range(width)))
            domains.append(tuple(features))
        if cursor != len(stream):
            raise DialogueExpertRoutingError("routing stream 存在尾随整数")
        model = cls(general_sha, tuple(domain_shas), tuple(domains))
        if model.integer_stream() != stream:
            raise DialogueExpertRoutingError("routing stream 非规范")
        return model


def build_dialogue_expert_routing_model(
        general: LearnedDialogueResponseModel,
        domains: tuple[LearnedDialogueResponseModel, ...],
        ) -> DialogueExpertRoutingModel:
    if not domains:
        raise DialogueExpertRoutingError("routing domains 为空")
    return DialogueExpertRoutingModel(
        general.course_sha256,
        tuple(domain.course_sha256 for domain in domains),
        tuple(tuple(sorted(learned_domain_activation_features(general, domain)))
              for domain in domains))


def _course_grounding_feature_counts(
        course: Path,
        ) -> tuple[dict[tuple[int, ...], int],
                   dict[tuple[int, ...], int], int, int]:
    grounded: dict[tuple[int, ...], int] = {}
    conversational: dict[tuple[int, ...], int] = {}
    grounded_documents = 0
    conversational_documents = 0
    try:
        with course.open("r", encoding="utf-8", newline="") as stream:
            for ordinal, line in enumerate(stream, 1):
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or value.get("split") not in {"train", "heldout"}
                        or value.get("intent_support") not in {0, 1}):
                    raise DialogueExpertRoutingError(
                        f"domain course line {ordinal} 身份非法")
                if value["split"] != "train":
                    continue
                turns = value.get("dialogue_turns")
                if (not isinstance(turns, list) or len(turns) < 2
                        or not isinstance(turns[-2], dict)
                        or turns[-2].get("speaker_role") != 1
                        or not isinstance(turns[-2].get("surface"), str)):
                    raise DialogueExpertRoutingError(
                        f"domain course line {ordinal} prompt 非法")
                features = frozenset(dialogue_prompt_features(
                    turns[-2]["surface"]))
                target = grounded if value["intent_support"] == 0 else conversational
                for feature in features:
                    target[feature] = target.get(feature, 0) + 1
                if value["intent_support"] == 0:
                    grounded_documents += 1
                else:
                    conversational_documents += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DialogueExpertRoutingError("domain course 无法读取") from error
    return (grounded, conversational, grounded_documents,
            conversational_documents)


def build_dialogue_expert_routing_model_from_courses(
        general: LearnedDialogueResponseModel,
        domains: tuple[LearnedDialogueResponseModel, ...],
        courses: tuple[Path, ...],
        ) -> DialogueExpertRoutingModel:
    if not domains or len(domains) != len(courses):
        raise DialogueExpertRoutingError("domain model/course 数量不一致")
    families = []
    for course in courses:
        (grounded, conversational, grounded_documents,
         conversational_documents) = _course_grounding_feature_counts(course)
        families.append(tuple(sorted(
            learned_grounded_domain_activation_features(
                general,
                grounded_feature_counts=grounded,
                conversational_feature_counts=conversational,
                grounded_document_count=grounded_documents,
                conversational_document_count=conversational_documents))))
    return DialogueExpertRoutingModel(
        general.course_sha256,
        tuple(domain.course_sha256 for domain in domains),
        tuple(families))


def encode_dialogue_expert_routing_model(
        model: DialogueExpertRoutingModel,
        ) -> bytes:
    return encode_integer_tuple(model.integer_stream())


def load_dialogue_expert_routing_model(
        path: str | Path,
        ) -> DialogueExpertRoutingModel:
    target = Path(path).resolve()
    if not target.is_file():
        raise DialogueExpertRoutingError("routing artifact 不存在")
    try:
        stream = decode_integer_tuple(target.read_bytes())
        return DialogueExpertRoutingModel.from_integer_stream(stream)
    except (OSError, TypeError, ValueError) as error:
        if isinstance(error, DialogueExpertRoutingError):
            raise
        raise DialogueExpertRoutingError("routing artifact 非法") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="build portable learned dialogue expert routing model")
    parser.add_argument("--general-artifact-root", required=True)
    parser.add_argument("--domain-artifact-root", action="append", required=True)
    parser.add_argument("--domain-course", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from pure_integer_ai.experiments.build_learned_dialogue_response_artifact import (
        load_learned_dialogue_response_artifact,
    )
    general = load_learned_dialogue_response_artifact(
        args.general_artifact_root)
    domains = tuple(load_learned_dialogue_response_artifact(item)
                    for item in args.domain_artifact_root)
    courses = tuple(Path(item).resolve() for item in args.domain_course)
    if len(courses) != len(domains):
        raise DialogueExpertRoutingError("domain course/artifact 数量不一致")
    for course, artifact in zip(courses, domains):
        if (course.drive.upper() != "K:" or not course.is_file()
                or _sha256_file(course) != artifact.course_sha256):
            raise DialogueExpertRoutingError("domain course SHA 或路径漂移")
    model = build_dialogue_expert_routing_model_from_courses(
        general.model, tuple(item.model for item in domains), courses)
    payload = encode_dialogue_expert_routing_model(model)
    target = Path(args.output).resolve()
    if target.drive.upper() != "K:" or not target.parent.is_dir():
        raise DialogueExpertRoutingError("routing output 必须是 K 盘既有目录下文件")
    with target.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        "bytes": len(payload),
        "domain_count": len(model.domain_activation_features),
        "feature_counts": [len(item)
                           for item in model.domain_activation_features],
        "sha256": hashlib.sha256(payload).hexdigest(),
    }, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "DIALOGUE_EXPERT_ROUTING_FILE", "DialogueExpertRoutingError",
    "DialogueExpertRoutingModel", "build_dialogue_expert_routing_model",
    "build_dialogue_expert_routing_model_from_courses",
    "encode_dialogue_expert_routing_model",
    "load_dialogue_expert_routing_model",
]


if __name__ == "__main__":
    raise SystemExit(main())
