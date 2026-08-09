"""W-02 morphology successor V2 的 train-only 边缘 lemma 编辑原型。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

from pure_integer_ai.experiments.ph2_d03_contract_core import canonical_json_bytes
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_model import (
    W02CandidatePrediction,
    W02MorphologyCandidate,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    open_w02_candidate_predictor,
    read_w02_candidate_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor import (
    W02_MORPH_FEATURE_WEIGHTS,
    W02_MORPH_MAX_QUERIED_SPANS_PER_OBSERVATION,
    W02MorphologySuccessorIndex,
    W02MorphologySuccessorPrediction,
    w02_morphology_features,
)
from pure_integer_ai.experiments.ph2_dataset_contract import ObservationRecord


W02_MORPH_SUCCESSOR_V2_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V2-PUBLIC-PROTOTYPE")
W02_MORPH_V2_EDGE_RULES = (
    "DROP_PREFIX_1", "DROP_PREFIX_2", "DROP_PREFIX_3",
    "DROP_SUFFIX_1", "DROP_SUFFIX_2", "DROP_SUFFIX_3",
)
W02_MORPH_V2_MAX_EDGE_CANDIDATES_PER_SPAN = 8
W02_MORPH_V2_MAX_EDGE_CANDIDATES_PER_OBSERVATION = 4_096
W02_MORPH_V2_MAX_LOGIC_OPERATIONS_PER_OBSERVATION = 200_000
W02_MORPH_V2_MAX_RANKING_CACHE_ENTRIES = 200_000

EditCombo = tuple[str, str, str]
MorphologyFeature = tuple[str, str]


# object-model: exception
class W02MorphologySuccessorV2Error(RuntimeError):
    """V2 edge-edit 输入、索引、候选或资源合同发生漂移。"""


def _hash_value(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def w02_morphology_edge_rule(form: str, lemma: str) -> str | None:
    """只归纳 lemma 位于 form 边缘且删除长度不超过 3 的规则。"""
    if not isinstance(form, str) or not isinstance(lemma, str) or not form or not lemma:
        raise W02MorphologySuccessorV2Error("V2 form/lemma 不得为空")
    if len(lemma) >= len(form):
        return None
    removed = len(form) - len(lemma)
    if form.endswith(lemma) and 1 <= removed <= 3:
        return f"DROP_PREFIX_{removed}"
    if form.startswith(lemma) and 1 <= removed <= 3:
        return f"DROP_SUFFIX_{removed}"
    return None


def _apply_edge_rule(form: str, rule: str) -> str:
    if rule not in W02_MORPH_V2_EDGE_RULES:
        raise W02MorphologySuccessorV2Error("V2 edge rule 未注册")
    side, count_text = rule.rsplit("_", 1)
    count = int(count_text)
    if len(form) <= count:
        raise W02MorphologySuccessorV2Error("V2 edge rule 产生空 lemma")
    return form[count:] if side == "DROP_PREFIX" else form[:-count]


def _semantic_rows(
        dataset_keys: tuple[tuple[int, ...], ...],
        global_counts: dict[EditCombo, int],
        feature_counts: dict[MorphologyFeature, dict[EditCombo, int]],
        ) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"dataset_key": list(key), "row_kind": "DATASET_ROUTE"}
        for key in dataset_keys
    ]
    rows.extend({
        "count": count, "feats_json": combo[2],
        "lemma_rule": combo[0], "row_kind": "GLOBAL_EDGE_COMBO",
        "upos": combo[1],
    } for combo, count in sorted(global_counts.items()))
    for feature, counts in sorted(feature_counts.items()):
        rows.extend({
            "count": count, "feats_json": combo[2],
            "feature_kind": feature[0], "feature_value": feature[1],
            "lemma_rule": combo[0], "row_kind": "LOCAL_EDGE_COMBO",
            "upos": combo[1],
        } for combo, count in sorted(counts.items()))
    return rows


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorV2Index:
    """由 train Candidate 非 identity 词形形成的边缘编辑统计索引。"""

    dataset_keys: tuple[tuple[int, ...], ...]
    global_counts: dict[EditCombo, int]
    feature_counts: dict[MorphologyFeature, dict[EditCombo, int]]
    max_form_length: int
    accepted_lexeme_rows: int
    accepted_support_count: int
    unsupported_lexeme_rows: int
    unsupported_support_count: int
    logic_operations: int
    semantic_sha256: str
    row_count: int

    def __post_init__(self) -> None:
        if (not self.dataset_keys
                or tuple(sorted(set(self.dataset_keys))) != self.dataset_keys
                or not self.global_counts or not self.feature_counts):
            raise W02MorphologySuccessorV2Error("V2 edge index 为空或未规范")
        if any(combo[0] not in W02_MORPH_V2_EDGE_RULES
               for combo in self.global_counts):
            raise W02MorphologySuccessorV2Error("V2 edge rule identity 漂移")
        for name in (
                "max_form_length", "accepted_lexeme_rows",
                "accepted_support_count", "logic_operations", "row_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise W02MorphologySuccessorV2Error(f"V2 {name} 必须为正整数")
        for name in ("unsupported_lexeme_rows", "unsupported_support_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise W02MorphologySuccessorV2Error(f"V2 {name} 非法")
        rows = _semantic_rows(
            self.dataset_keys, self.global_counts, self.feature_counts)
        if len(rows) != self.row_count or _hash_value(rows) != self.semantic_sha256:
            raise W02MorphologySuccessorV2Error("V2 edge semantic identity 漂移")

    def semantic_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(_semantic_rows(
            self.dataset_keys, self.global_counts, self.feature_counts))


# object-model: lifecycle; owner=evaluation; cleanup=scope-end
@dataclass(slots=True)
class W02MorphologySuccessorV2Cache:
    """一次评估内的 edge ranking cache。"""

    values: dict[tuple[MorphologyFeature, ...], tuple[tuple[EditCombo, int], ...]]
    hit_count: int = 0
    miss_count: int = 0

    @classmethod
    def empty(cls) -> "W02MorphologySuccessorV2Cache":
        return cls({})

    def close(self) -> None:
        self.values.clear()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorV2Prediction:
    """V1 保留结果加 edge-edit 候选和确定性资源计数。"""

    prediction: W02CandidatePrediction
    edge_candidate_count: int
    logic_operations: int
    semantic_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, W02CandidatePrediction):
            raise W02MorphologySuccessorV2Error("V2 prediction 类型错误")
        if (type(self.edge_candidate_count) is not int
                or self.edge_candidate_count < 0
                or type(self.logic_operations) is not int
                or self.logic_operations < 0
                or len(self.semantic_sha256) != 64):
            raise W02MorphologySuccessorV2Error("V2 prediction 计数或 SHA 非法")


def build_w02_morphology_successor_v2_from_counts(
        *,
        v1_index: W02MorphologySuccessorIndex,
        lexeme_counts: Iterable[tuple[str, str, str, str, int]],
        ) -> W02MorphologySuccessorV2Index:
    """从 Candidate 聚合词形中学习 edge-edit；identity 仍完全交给 V1。"""
    if not isinstance(v1_index, W02MorphologySuccessorIndex):
        raise TypeError("V2 parent index 类型错误")
    global_counts: Counter[EditCombo] = Counter()
    local_counts: dict[MorphologyFeature, Counter[EditCombo]] = {}
    accepted_rows = 0
    accepted_support = 0
    unsupported_rows = 0
    unsupported_support = 0
    operations = 0
    seen: set[tuple[str, str, str, str]] = set()
    for raw in lexeme_counts:
        if (not isinstance(raw, tuple) or len(raw) != 5
                or any(not isinstance(value, str) for value in raw[:4])
                or type(raw[4]) is not int or raw[4] <= 0):
            raise W02MorphologySuccessorV2Error("V2 lexeme count row 非法")
        form, lemma, upos, feats_json, support = raw
        identity = (form, lemma, upos, feats_json)
        if identity in seen:
            raise W02MorphologySuccessorV2Error("V2 lexeme count row 重复")
        seen.add(identity)
        if lemma == form or lemma == form.casefold():
            continue
        rule = w02_morphology_edge_rule(form, lemma)
        operations += support * (len(form) + len(lemma) + 4)
        if rule is None:
            unsupported_rows += 1
            unsupported_support += support
            continue
        combo = (rule, upos, feats_json)
        global_counts[combo] += support
        for feature in w02_morphology_features(form):
            local_counts.setdefault(feature, Counter())[combo] += support
        accepted_rows += 1
        accepted_support += support
    if not seen or not global_counts or accepted_rows <= 0:
        raise W02MorphologySuccessorV2Error("V2 edge train evidence 不闭合")
    frozen_local = {feature: dict(counts)
                    for feature, counts in local_counts.items()}
    frozen_global = dict(global_counts)
    rows = _semantic_rows(v1_index.dataset_keys, frozen_global, frozen_local)
    return W02MorphologySuccessorV2Index(
        v1_index.dataset_keys, frozen_global, frozen_local,
        v1_index.max_form_length, accepted_rows, accepted_support,
        unsupported_rows, unsupported_support, operations, _hash_value(rows),
        len(rows))


def derive_w02_morphology_successor_v2_from_candidate(
        candidate_artifact_root: str | Path,
        v1_index: W02MorphologySuccessorIndex,
        ) -> W02MorphologySuccessorV2Index:
    """只读 sealed Candidate lexemes，派生与聚合入口相同的 V2 索引。"""
    root = Path(candidate_artifact_root).resolve()
    read_w02_candidate_artifact(root)
    with open_w02_candidate_predictor(root) as predictor:
        rows = tuple(predictor.connection.execute(
            "SELECT form,lemma,upos,feats_json,support_count FROM lexemes "
            "ORDER BY form,lemma,upos,feats_json"))
    return build_w02_morphology_successor_v2_from_counts(
        v1_index=v1_index,
        lexeme_counts=tuple(
            (str(row[0]), str(row[1]), str(row[2]), str(row[3]), int(row[4]))
            for row in rows))


def _ranked_edge_combos(
        index: W02MorphologySuccessorV2Index,
        form: str,
        cache: W02MorphologySuccessorV2Cache,
        ) -> tuple[tuple[tuple[EditCombo, int], ...], int]:
    active = w02_morphology_features(form)
    cached = cache.values.get(active)
    if cached is not None:
        cache.hit_count += 1
        return cached, 1
    pool: set[EditCombo] = set()
    for feature in active:
        pool.update(index.feature_counts.get(feature, ()))
    scored = []
    for combo in pool:
        score = sum(
            W02_MORPH_FEATURE_WEIGHTS[feature[0]]
            * index.feature_counts.get(feature, {}).get(combo, 0)
            for feature in active)
        if score > 0:
            scored.append((combo, score))
    scored.sort(key=lambda item: (
        -item[1], -index.global_counts.get(item[0], 0), item[0]))
    if len(cache.values) >= W02_MORPH_V2_MAX_RANKING_CACHE_ENTRIES:
        raise W02MorphologySuccessorV2Error("V2 ranking cache resource stop")
    result = tuple(scored)
    cache.values[active] = result
    cache.miss_count += 1
    return result, len(active) + len(pool) * (len(active) + 2) + len(result)


def predict_w02_morphology_successor_v2(
        index: W02MorphologySuccessorV2Index,
        observation: ObservationRecord,
        v1: W02MorphologySuccessorPrediction,
        *,
        requested_spans: tuple[tuple[int, int], ...],
        cache: W02MorphologySuccessorV2Cache | None = None,
        ) -> W02MorphologySuccessorV2Prediction:
    """逐 span 在完整 V1 结果上追加最多 8 个 edge-edit 候选。"""
    if (not isinstance(index, W02MorphologySuccessorV2Index)
            or not isinstance(observation, ObservationRecord)
            or not isinstance(v1, W02MorphologySuccessorPrediction)):
        raise TypeError("V2 prediction 输入类型错误")
    if (v1.prediction.observation_key != observation.stable_key.components
            or not isinstance(requested_spans, tuple)
            or len(requested_spans) > W02_MORPH_MAX_QUERIED_SPANS_PER_OBSERVATION
            or tuple(sorted(set(requested_spans))) != requested_spans):
        raise W02MorphologySuccessorV2Error("V2 observation/span identity 漂移")
    if observation.dataset_key.components not in index.dataset_keys:
        return W02MorphologySuccessorV2Prediction(
            v1.prediction, 0, 0, index.semantic_sha256)
    ranking_cache = cache or W02MorphologySuccessorV2Cache.empty()
    prediction = v1.prediction
    surface = prediction.generation.surface
    boundaries = set(prediction.boundary_lattice)
    existing = {
        (item.start, item.end, item.form, item.lemma, item.upos, item.feats_json): item
        for item in prediction.morphology_candidates
    }
    generated = 0
    operations = 0
    for start, end in requested_spans:
        if (type(start) is not int or type(end) is not int
                or start not in boundaries or end not in boundaries
                or start < 0 or end <= start or end > len(surface)
                or end - start > index.max_form_length):
            raise W02MorphologySuccessorV2Error("V2 requested span 越界")
        form = surface[start:end]
        ranked, rank_operations = _ranked_edge_combos(
            index, form, ranking_cache)
        operations += rank_operations + len(form) + 4
        for combo, score in ranked[:W02_MORPH_V2_MAX_EDGE_CANDIDATES_PER_SPAN]:
            removed = int(combo[0].rsplit("_", 1)[1])
            if len(form) <= removed:
                operations += 1
                continue
            lemma = _apply_edge_rule(form, combo[0])
            key = (start, end, form, lemma, combo[1], combo[2])
            if key in existing:
                continue
            existing[key] = W02MorphologyCandidate(
                start, end, form, lemma, combo[1], combo[2], score)
            generated += 1
            operations += 1
            if generated > W02_MORPH_V2_MAX_EDGE_CANDIDATES_PER_OBSERVATION:
                raise W02MorphologySuccessorV2Error(
                    "V2 observation candidate resource stop")
        if operations > W02_MORPH_V2_MAX_LOGIC_OPERATIONS_PER_OBSERVATION:
            raise W02MorphologySuccessorV2Error(
                "V2 observation logic resource stop")
    result = W02CandidatePrediction(
        prediction.observation_key, prediction.status, prediction.generation,
        prediction.boundary_lattice, prediction.unicode_units,
        tuple(sorted(existing.values(), key=lambda item: (
            item.start, item.end, item.form, item.lemma, item.upos,
            item.feats_json, item.support_count))), prediction.capabilities)
    return W02MorphologySuccessorV2Prediction(
        result, generated, operations, index.semantic_sha256)


__all__ = [
    "W02_MORPH_SUCCESSOR_V2_VERSION", "W02_MORPH_V2_EDGE_RULES",
    "W02_MORPH_V2_MAX_EDGE_CANDIDATES_PER_SPAN",
    "W02MorphologySuccessorV2Cache", "W02MorphologySuccessorV2Error",
    "W02MorphologySuccessorV2Index", "W02MorphologySuccessorV2Prediction",
    "build_w02_morphology_successor_v2_from_counts",
    "derive_w02_morphology_successor_v2_from_candidate",
    "predict_w02_morphology_successor_v2", "w02_morphology_edge_rule",
]
