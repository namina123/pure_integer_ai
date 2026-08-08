"""W-02 dev FAIL 后的 train-only 有界词形候选归纳原型。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Iterable
import unicodedata

from pure_integer_ai.experiments.ph2_d03_contract_core import canonical_json_bytes
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_model import (
    W02_EVIDENCE_UD,
    W02CandidatePrediction,
    W02MorphologyCandidate,
    learn_w02_training_pair,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    TeacherEvidenceRecord,
)


W02_MORPH_SUCCESSOR_VERSION = "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V1"
W02_MORPH_FEATURE_KINDS = (
    "LENGTH_BUCKET",
    "SCRIPT_CLASS",
    "PREFIX_1",
    "SUFFIX_1",
    "PREFIX_2",
    "SUFFIX_2",
)
W02_MORPH_FEATURE_WEIGHTS = {
    "LENGTH_BUCKET": 4,
    "SCRIPT_CLASS": 2,
    "PREFIX_1": 8,
    "SUFFIX_1": 8,
    "PREFIX_2": 16,
    "SUFFIX_2": 16,
}
W02_MORPH_DEFAULT_FEATURE_KINDS = ("LENGTH_BUCKET", "SCRIPT_CLASS")
W02_MORPH_LEMMA_RULES = ("CASEFOLD", "IDENTITY")
W02_MORPH_MAX_CANDIDATES_PER_SPAN = 20
W02_MORPH_MAX_QUERIED_SPANS_PER_OBSERVATION = 512
W02_MORPH_MAX_CANDIDATES_PER_OBSERVATION = (
    W02_MORPH_MAX_CANDIDATES_PER_SPAN
    * W02_MORPH_MAX_QUERIED_SPANS_PER_OBSERVATION
)
W02_MORPH_MAX_LOGIC_OPERATIONS_PER_OBSERVATION = 300_000
W02_MORPH_MAX_RANKING_CACHE_ENTRIES = 200_000

MorphologyCombo = tuple[str, str, str]
MorphologyFeature = tuple[str, str]


# object-model: exception
class W02MorphologySuccessorError(RuntimeError):
    """形态归纳输入、规则或预测违反 successor 合同。"""


# object-model: exception
class W02MorphologySuccessorStop(W02MorphologySuccessorError):
    """单 observation 候选或逻辑操作超过预注册硬上界。"""


def _hash_value(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _script_class(form: str) -> str:
    """把表层折成确定性 Unicode category 大类集合。"""
    return "".join(sorted({unicodedata.category(char)[0] for char in form}))


def w02_morphology_features(form: str) -> tuple[MorphologyFeature, ...]:
    """返回固定六类局部特征，空字符串不得进入学习。"""
    if not isinstance(form, str) or not form:
        raise W02MorphologySuccessorError("morphology form 不得为空")
    return (
        ("LENGTH_BUCKET", str(min(len(form), 6))),
        ("SCRIPT_CLASS", _script_class(form)),
        ("PREFIX_1", form[:1]),
        ("SUFFIX_1", form[-1:]),
        ("PREFIX_2", form[:2]),
        ("SUFFIX_2", form[-2:]),
    )


def w02_morphology_lemma_rule(form: str, lemma: str) -> str | None:
    """只接受能从输入表层确定重建的 lemma 变换。"""
    if lemma == form:
        return "IDENTITY"
    if lemma == form.casefold():
        return "CASEFOLD"
    return None


def _apply_lemma_rule(form: str, rule: str) -> str:
    """将已注册 lemma 变换应用于未见表层。"""
    if rule == "IDENTITY":
        return form
    if rule == "CASEFOLD":
        return form.casefold()
    raise W02MorphologySuccessorError("lemma rule 未注册")


def _semantic_rows(
        dataset_keys: tuple[tuple[int, ...], ...],
        global_counts: dict[MorphologyCombo, int],
        feature_counts: dict[MorphologyFeature, dict[MorphologyCombo, int]],
        ) -> list[dict[str, object]]:
    """将内存计数投影为稳定行，供摘要和后续 artifact 共用。"""
    rows: list[dict[str, object]] = [
        {"dataset_key": list(key), "row_kind": "DATASET_ROUTE"}
        for key in dataset_keys
    ]
    rows.extend({
        "count": count,
        "feats_json": combo[2],
        "lemma_rule": combo[0],
        "row_kind": "GLOBAL_COMBO",
        "upos": combo[1],
    } for combo, count in sorted(global_counts.items()))
    for feature, counts in sorted(feature_counts.items()):
        rows.extend({
            "count": count,
            "feats_json": combo[2],
            "feature_kind": feature[0],
            "feature_value": feature[1],
            "lemma_rule": combo[0],
            "row_kind": "LOCAL_COMBO",
            "upos": combo[1],
        } for combo, count in sorted(counts.items()))
    return rows


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorIndex:
    """由 train UD Evidence 形成的有界局部统计索引。"""

    dataset_keys: tuple[tuple[int, ...], ...]
    global_counts: dict[MorphologyCombo, int]
    feature_counts: dict[MorphologyFeature, dict[MorphologyCombo, int]]
    max_form_length: int
    training_pair_count: int
    morphology_observation_count: int
    morphology_token_count: int
    logic_operations: int
    semantic_sha256: str
    row_count: int

    def __post_init__(self) -> None:
        if (not self.dataset_keys
                or tuple(sorted(set(self.dataset_keys))) != self.dataset_keys):
            raise W02MorphologySuccessorError("successor dataset routes 非法")
        if not self.global_counts or not self.feature_counts:
            raise W02MorphologySuccessorError("successor morphology 规则为空")
        if any(kind not in W02_MORPH_FEATURE_KINDS
               for kind, _ in self.feature_counts):
            raise W02MorphologySuccessorError("successor feature kind 漂移")
        if any(type(value) is not int or value <= 0 for value in (
                self.max_form_length, self.training_pair_count,
                self.morphology_observation_count, self.morphology_token_count,
                self.logic_operations, self.row_count)):
            raise W02MorphologySuccessorError("successor 计数必须为正整数")
        if (not isinstance(self.semantic_sha256, str)
                or len(self.semantic_sha256) != 64):
            raise W02MorphologySuccessorError("successor semantic SHA 非法")
        rows = _semantic_rows(
            self.dataset_keys, self.global_counts, self.feature_counts)
        if len(rows) != self.row_count or _hash_value(rows) != self.semantic_sha256:
            raise W02MorphologySuccessorError("successor semantic identity 漂移")

    def semantic_rows(self) -> tuple[dict[str, object], ...]:
        """返回可持久化的规范规则行副本。"""
        return tuple(_semantic_rows(
            self.dataset_keys, self.global_counts, self.feature_counts))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorPrediction:
    """基础预测、归纳候选和资源计数的不可变结果。"""

    prediction: W02CandidatePrediction
    generalized_candidate_count: int
    logic_operations: int
    successor_semantic_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, W02CandidatePrediction):
            raise W02MorphologySuccessorError("successor prediction 类型错误")
        if (type(self.generalized_candidate_count) is not int
                or self.generalized_candidate_count < 0
                or type(self.logic_operations) is not int
                or self.logic_operations < 0):
            raise W02MorphologySuccessorError("successor prediction 计数非法")
        if len(self.successor_semantic_sha256) != 64:
            raise W02MorphologySuccessorError("successor prediction SHA 非法")


# object-model: lifecycle; owner=evaluation; cleanup=scope-end
@dataclass(slots=True)
class W02MorphologyRankingCache:
    """在一次评估内复用同 feature identity 的整数排序。"""

    values: dict[
        tuple[tuple[MorphologyFeature, ...], tuple[str, ...]],
        tuple[tuple[MorphologyCombo, int], ...],
    ]
    hit_count: int = 0
    miss_count: int = 0

    @classmethod
    def empty(cls) -> "W02MorphologyRankingCache":
        """建立空的有界评估期缓存。"""
        return cls({})

    def close(self) -> None:
        """结束评估时释放全部派生排序，不改变 Candidate。"""
        self.values.clear()


def learn_w02_morphology_successor(
        pairs: Iterable[tuple[ObservationRecord, TeacherEvidenceRecord]],
        *,
        max_training_pairs: int = 51_200,
        max_logic_operations: int = 9_000_000,
        ) -> W02MorphologySuccessorIndex:
    """从 train pair 学局部词形计数；非 UD Evidence 只计输入不产规则。"""
    if (type(max_training_pairs) is not int or max_training_pairs <= 0
            or type(max_logic_operations) is not int or max_logic_operations <= 0):
        raise W02MorphologySuccessorError("successor training budget 非法")
    dataset_keys: set[tuple[int, ...]] = set()
    lexeme_counts: Counter[tuple[str, str, str, str]] = Counter()
    pair_count = 0
    morphology_count = 0
    for pair in pairs:
        if (not isinstance(pair, tuple) or len(pair) != 2
                or not isinstance(pair[0], ObservationRecord)
                or not isinstance(pair[1], TeacherEvidenceRecord)):
            raise W02MorphologySuccessorError("successor train pair 类型错误")
        if pair_count >= max_training_pairs:
            raise W02MorphologySuccessorStop("successor training pair resource stop")
        observation, evidence = pair
        pair_count += 1
        expected = evidence.typed_evidence.to_value()
        if "morphology" not in expected:
            continue
        delta = learn_w02_training_pair(observation, evidence)
        if delta.evidence_mode != W02_EVIDENCE_UD or not delta.lexemes:
            raise W02MorphologySuccessorError("successor UD Evidence 漂移")
        dataset_keys.add(observation.dataset_key.components)
        morphology_count += 1
        for lexeme in delta.lexemes:
            lexeme_counts[
                (lexeme.form, lexeme.lemma, lexeme.upos, lexeme.feats_json)
            ] += 1
    if pair_count <= 0 or morphology_count <= 0 or not lexeme_counts:
        raise W02MorphologySuccessorError("successor train Evidence 不闭合")
    index = build_w02_morphology_successor_from_counts(
        dataset_keys=tuple(sorted(dataset_keys)),
        lexeme_counts=tuple(
            (*key, count) for key, count in sorted(lexeme_counts.items())),
        training_pair_count=pair_count,
        morphology_observation_count=morphology_count,
    )
    if index.logic_operations > max_logic_operations:
        raise W02MorphologySuccessorStop("successor training logic resource stop")
    return index


def build_w02_morphology_successor_from_counts(
        *,
        dataset_keys: tuple[tuple[int, ...], ...],
        lexeme_counts: Iterable[tuple[str, str, str, str, int]],
        training_pair_count: int,
        morphology_observation_count: int,
        ) -> W02MorphologySuccessorIndex:
    """从已聚合 train 词形次数重建与逐 pair 学习相同的索引。"""
    if (not isinstance(dataset_keys, tuple)
            or tuple(sorted(set(dataset_keys))) != dataset_keys
            or any(not key or any(type(item) is not int or item <= 0 for item in key)
                   for key in dataset_keys)):
        raise W02MorphologySuccessorError("successor aggregate dataset routes 非法")
    if (type(training_pair_count) is not int or training_pair_count <= 0
            or type(morphology_observation_count) is not int
            or morphology_observation_count <= 0
            or morphology_observation_count > training_pair_count):
        raise W02MorphologySuccessorError("successor aggregate observation 计数非法")
    global_counts: Counter[MorphologyCombo] = Counter()
    feature_counts: dict[MorphologyFeature, Counter[MorphologyCombo]] = {}
    seen_lexemes: set[tuple[str, str, str, str]] = set()
    token_count = 0
    max_form_length = 0
    operations = training_pair_count - morphology_observation_count
    for raw in lexeme_counts:
        if (not isinstance(raw, tuple) or len(raw) != 5
                or any(not isinstance(item, str) for item in raw[:4])
                or not raw[0] or not raw[2] or not raw[3]
                or type(raw[4]) is not int or raw[4] <= 0):
            raise W02MorphologySuccessorError("successor aggregate lexeme row 非法")
        form, lemma, upos, feats_json, support_count = raw
        key = (form, lemma, upos, feats_json)
        if key in seen_lexemes:
            raise W02MorphologySuccessorError("successor aggregate lexeme row 重复")
        seen_lexemes.add(key)
        lemma_rule = w02_morphology_lemma_rule(form, lemma)
        if lemma_rule is None:
            operations += support_count
            continue
        combo = (lemma_rule, upos, feats_json)
        global_counts[combo] += support_count
        for feature in w02_morphology_features(form):
            feature_counts.setdefault(feature, Counter())[combo] += support_count
        token_count += support_count
        max_form_length = max(max_form_length, len(form))
        operations += support_count * (
            len(form) + len(W02_MORPH_FEATURE_KINDS) + 8)
    if not seen_lexemes or not global_counts or not feature_counts or token_count <= 0:
        raise W02MorphologySuccessorError("successor aggregate lexeme 计数不闭合")
    frozen_global = dict(global_counts)
    frozen_features = {
        feature: dict(counts) for feature, counts in feature_counts.items()
    }
    rows = _semantic_rows(dataset_keys, frozen_global, frozen_features)
    return W02MorphologySuccessorIndex(
        dataset_keys,
        frozen_global,
        frozen_features,
        max_form_length,
        training_pair_count,
        morphology_observation_count,
        token_count,
        operations,
        _hash_value(rows),
        len(rows),
    )


def _ranked_combos(
        index: W02MorphologySuccessorIndex,
        form: str,
        enabled_features: tuple[str, ...],
        cache: W02MorphologyRankingCache,
        ) -> tuple[tuple[tuple[MorphologyCombo, int], ...], int]:
    """在局部 feature union 内按固定整数分和全局支持排序。"""
    if (not enabled_features
            or any(kind not in W02_MORPH_FEATURE_KINDS for kind in enabled_features)
            or len(set(enabled_features)) != len(enabled_features)):
        raise W02MorphologySuccessorError("successor enabled features 非法")
    if not isinstance(cache, W02MorphologyRankingCache):
        raise TypeError("successor ranking cache 类型错误")
    active = tuple(
        feature for feature in w02_morphology_features(form)
        if feature[0] in enabled_features)
    pool: set[MorphologyCombo] = set()
    for feature in active:
        pool.update(index.feature_counts.get(feature, ()))
    if (len(pool) < W02_MORPH_MAX_CANDIDATES_PER_SPAN
            and enabled_features == W02_MORPH_DEFAULT_FEATURE_KINDS):
        fallback = tuple(
            feature for feature in w02_morphology_features(form)
            if feature[0] not in enabled_features)
        for feature in fallback:
            pool.update(index.feature_counts.get(feature, ()))
            if len(pool) >= W02_MORPH_MAX_CANDIDATES_PER_SPAN:
                break
        active = tuple(w02_morphology_features(form))
    cache_key = (active, enabled_features)
    cached = cache.values.get(cache_key)
    if cached is not None:
        cache.hit_count += 1
        return cached, 1
    scored = []
    for combo in pool:
        score = sum(
            W02_MORPH_FEATURE_WEIGHTS[feature[0]]
            * index.feature_counts.get(feature, {}).get(combo, 0)
            for feature in active
        )
        if score > 0:
            scored.append((combo, score))
    scored.sort(key=lambda item: (
        -item[1], -index.global_counts.get(item[0], 0), item[0]))
    if len(cache.values) >= W02_MORPH_MAX_RANKING_CACHE_ENTRIES:
        raise W02MorphologySuccessorStop("successor ranking cache resource stop")
    result = tuple(scored)
    cache.values[cache_key] = result
    cache.miss_count += 1
    operations = len(active) + len(pool) * (len(active) + 2) + len(scored)
    return result, operations


def predict_w02_morphology_successor(
        index: W02MorphologySuccessorIndex,
        observation: ObservationRecord,
        base: W02CandidatePrediction,
        *,
        requested_spans: tuple[tuple[int, int], ...] = (),
        enabled_features: tuple[str, ...] = W02_MORPH_DEFAULT_FEATURE_KINDS,
        candidate_limit: int = W02_MORPH_MAX_CANDIDATES_PER_SPAN,
        ranking_cache: W02MorphologyRankingCache | None = None,
        ) -> W02MorphologySuccessorPrediction:
    """保留基础预测，并只为 consumer 请求的 span 增加有界候选。"""
    if not isinstance(index, W02MorphologySuccessorIndex):
        raise TypeError("successor index 类型错误")
    if not isinstance(observation, ObservationRecord):
        raise TypeError("successor observation 类型错误")
    if not isinstance(base, W02CandidatePrediction):
        raise TypeError("successor base prediction 类型错误")
    if base.observation_key != observation.stable_key.components:
        raise W02MorphologySuccessorError("successor base/observation 身份漂移")
    if (type(candidate_limit) is not int or candidate_limit <= 0
            or candidate_limit > W02_MORPH_MAX_CANDIDATES_PER_SPAN):
        raise W02MorphologySuccessorError("successor candidate limit 非法")
    if (not isinstance(requested_spans, tuple)
            or len(requested_spans) > W02_MORPH_MAX_QUERIED_SPANS_PER_OBSERVATION
            or tuple(sorted(set(requested_spans))) != requested_spans):
        raise W02MorphologySuccessorError("successor requested spans 非法")
    if observation.dataset_key.components not in index.dataset_keys:
        return W02MorphologySuccessorPrediction(
            base, 0, 0, index.semantic_sha256)
    cache = ranking_cache or W02MorphologyRankingCache.empty()
    surface = base.generation.surface
    existing = {
        (item.start, item.end, item.form, item.lemma, item.upos, item.feats_json): item
        for item in base.morphology_candidates
    }
    generated = 0
    operations = 0
    boundary_points = set(base.boundary_lattice)
    for start, end in requested_spans:
        if (type(start) is not int or type(end) is not int
                or start not in boundary_points or end not in boundary_points
                or start < 0 or end <= start or end > len(surface)
                or end - start > index.max_form_length):
            raise W02MorphologySuccessorError("successor requested span 越界")
        form = surface[start:end]
        ranked, rank_operations = _ranked_combos(
            index, form, enabled_features, cache)
        operations += rank_operations + len(form) + 4
        for combo, score in ranked[:candidate_limit]:
            lemma = _apply_lemma_rule(form, combo[0])
            key = (start, end, form, lemma, combo[1], combo[2])
            if key in existing:
                continue
            existing[key] = W02MorphologyCandidate(
                start, end, form, lemma, combo[1], combo[2], score)
            generated += 1
            operations += 1
            if generated > W02_MORPH_MAX_CANDIDATES_PER_OBSERVATION:
                raise W02MorphologySuccessorStop(
                    "successor observation candidate resource stop")
        if operations > W02_MORPH_MAX_LOGIC_OPERATIONS_PER_OBSERVATION:
            raise W02MorphologySuccessorStop(
                "successor observation logic resource stop")
    morphology = tuple(sorted(existing.values(), key=lambda item: (
        item.start, item.end, item.form, item.lemma, item.upos,
        item.feats_json, item.support_count)))
    prediction = W02CandidatePrediction(
        base.observation_key,
        base.status,
        base.generation,
        base.boundary_lattice,
        base.unicode_units,
        morphology,
        base.capabilities,
    )
    return W02MorphologySuccessorPrediction(
        prediction, generated, operations, index.semantic_sha256)


__all__ = [
    "W02_MORPH_FEATURE_KINDS",
    "W02_MORPH_MAX_CANDIDATES_PER_SPAN",
    "W02MorphologySuccessorError",
    "W02MorphologySuccessorIndex",
    "W02MorphologySuccessorPrediction",
    "W02MorphologyRankingCache",
    "W02MorphologySuccessorStop",
    "build_w02_morphology_successor_from_counts",
    "learn_w02_morphology_successor",
    "predict_w02_morphology_successor",
    "w02_morphology_features",
    "w02_morphology_lemma_rule",
]
