"""W-02 append-only language-conditioned morphology overlay.

The frozen Candidate/V1/V2/V3 chain learned morphology from modern Chinese.
This module adds a separate language-scoped lexeme and bounded backoff index;
it never mutates or broadens any frozen parent index.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Iterable

from pure_integer_ai.experiments.ph2_d03_contract_core import canonical_json_bytes
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_model import (
    W02CandidatePrediction,
    W02MorphologyCandidate,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor import (
    W02_MORPH_FEATURE_KINDS,
    W02_MORPH_FEATURE_WEIGHTS,
    W02_MORPH_MAX_QUERIED_SPANS_PER_OBSERVATION,
    w02_morphology_features,
    w02_morphology_lemma_rule,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_route import (
    W02MorphologySuccessorV3Prediction,
)
from pure_integer_ai.experiments.ph2_dataset_contract import ObservationRecord


W02_MORPH_SUCCESSOR_V4_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V4-LANGUAGE-OVERLAY-V1")
W02_MORPH_V4_EXACT_CANDIDATE_LIMIT = 16
W02_MORPH_V4_BACKOFF_CANDIDATE_LIMIT = 8
W02_MORPH_V4_MAX_CANDIDATES_PER_OBSERVATION = 8_192
W02_MORPH_V4_MAX_LOGIC_OPERATIONS_PER_OBSERVATION = 300_000

ExactCombo = tuple[str, str, str]
BackoffCombo = tuple[str, str, str]
LanguageFeature = tuple[str, tuple[str, str]]


# object-model: exception
class W02MorphologySuccessorV4Error(RuntimeError):
    """Language overlay input, identity, route, or resource drifted."""


def _hash_value(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _nonempty(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise W02MorphologySuccessorV4Error(f"{where} must be non-empty text")
    return value


def _semantic_rows(
        languages: tuple[str, ...],
        exact_counts: dict[tuple[str, str], dict[ExactCombo, int]],
        backoff_counts: dict[LanguageFeature, dict[BackoffCombo, int]],
        ) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {"language": language, "row_kind": "LANGUAGE_ROUTE"}
        for language in languages
    ]
    for (language, form), counts in sorted(exact_counts.items()):
        rows.extend({
            "count": count,
            "feats_json": combo[2],
            "form": form,
            "language": language,
            "lemma": combo[0],
            "row_kind": "EXACT_LEXEME",
            "upos": combo[1],
        } for combo, count in sorted(counts.items()))
    for (language, feature), counts in sorted(backoff_counts.items()):
        rows.extend({
            "count": count,
            "feats_json": combo[2],
            "feature_kind": feature[0],
            "feature_value": feature[1],
            "language": language,
            "lemma_rule": combo[0],
            "row_kind": "LANGUAGE_BACKOFF",
            "upos": combo[1],
        } for combo, count in sorted(counts.items()))
    return rows


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorV4Index:
    """Language-isolated exact lexemes plus bounded productive backoff."""

    languages: tuple[str, ...]
    exact_counts: dict[tuple[str, str], dict[ExactCombo, int]]
    backoff_counts: dict[LanguageFeature, dict[BackoffCombo, int]]
    max_form_length: int
    training_token_count: int
    exact_lexeme_row_count: int
    backoff_lexeme_row_count: int
    logic_operations: int
    semantic_sha256: str
    row_count: int

    def __post_init__(self) -> None:
        if (not self.languages
                or tuple(sorted(set(self.languages))) != self.languages
                or not self.exact_counts or not self.backoff_counts):
            raise W02MorphologySuccessorV4Error(
                "V4 language overlay is empty or non-canonical")
        if any(language not in self.languages or not form
               for language, form in self.exact_counts):
            raise W02MorphologySuccessorV4Error(
                "V4 exact language/form identity drifted")
        if any(language not in self.languages
               or feature[0] not in W02_MORPH_FEATURE_KINDS
               for language, feature in self.backoff_counts):
            raise W02MorphologySuccessorV4Error(
                "V4 backoff language/feature identity drifted")
        for counts in (*self.exact_counts.values(),
                       *self.backoff_counts.values()):
            if not counts or any(type(count) is not int or count <= 0
                                 for count in counts.values()):
                raise W02MorphologySuccessorV4Error(
                    "V4 support counts must be positive integers")
        for name in (
                "max_form_length", "training_token_count",
                "exact_lexeme_row_count", "backoff_lexeme_row_count",
                "logic_operations", "row_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise W02MorphologySuccessorV4Error(
                    "V4 aggregate counts must be positive integers")
        rows = _semantic_rows(
            self.languages, self.exact_counts, self.backoff_counts)
        if (len(rows) != self.row_count
                or _hash_value(rows) != self.semantic_sha256):
            raise W02MorphologySuccessorV4Error(
                "V4 semantic identity drifted")

    def semantic_rows(self) -> tuple[dict[str, object], ...]:
        """Return canonical rows for append-only artifact publication."""
        return tuple(_semantic_rows(
            self.languages, self.exact_counts, self.backoff_counts))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorV4Ranking:
    """One span's bounded candidates and their evidence mode."""

    candidates: tuple[tuple[str, str, str, int], ...]
    evidence_mode: str
    logic_operations: int

    def __post_init__(self) -> None:
        if self.evidence_mode not in {"EXACT_LEXEME", "LANGUAGE_BACKOFF", "NONE"}:
            raise W02MorphologySuccessorV4Error(
                "V4 ranking evidence mode is not registered")
        if (type(self.logic_operations) is not int
                or self.logic_operations < 0):
            raise W02MorphologySuccessorV4Error(
                "V4 ranking operations are invalid")
        limit = (W02_MORPH_V4_EXACT_CANDIDATE_LIMIT
                 if self.evidence_mode == "EXACT_LEXEME"
                 else W02_MORPH_V4_BACKOFF_CANDIDATE_LIMIT)
        if len(self.candidates) > limit:
            raise W02MorphologySuccessorV4Error(
                "V4 ranking candidate limit drifted")
        if any(not lemma or not upos or not feats
               or type(score) is not int or score <= 0
               for lemma, upos, feats, score in self.candidates):
            raise W02MorphologySuccessorV4Error(
                "V4 ranking candidate is invalid")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorV4Prediction:
    """V3 prediction extended only by the matching language overlay."""

    prediction: W02CandidatePrediction
    exact_candidate_count: int
    backoff_candidate_count: int
    queried_span_count: int
    logic_operations: int
    overlay_semantic_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, W02CandidatePrediction):
            raise W02MorphologySuccessorV4Error(
                "V4 prediction payload type drifted")
        if any(type(value) is not int or value < 0 for value in (
                self.exact_candidate_count, self.backoff_candidate_count,
                self.queried_span_count, self.logic_operations)):
            raise W02MorphologySuccessorV4Error(
                "V4 prediction counts are invalid")
        if len(self.overlay_semantic_sha256) != 64:
            raise W02MorphologySuccessorV4Error(
                "V4 prediction semantic SHA is invalid")


def build_w02_morphology_successor_v4_from_counts(
        rows: Iterable[tuple[str, str, str, str, str, int]],
        *,
        max_rows: int = 100_000,
        max_logic_operations: int = 30_000_000,
        ) -> W02MorphologySuccessorV4Index:
    """Build a language-conditioned index from aggregated public lexemes."""
    if (type(max_rows) is not int or max_rows <= 0
            or type(max_logic_operations) is not int
            or max_logic_operations <= 0):
        raise W02MorphologySuccessorV4Error(
            "V4 training budget is invalid")
    exact: dict[tuple[str, str], Counter[ExactCombo]] = {}
    backoff: dict[LanguageFeature, Counter[BackoffCombo]] = {}
    languages: set[str] = set()
    seen: set[tuple[str, str, str, str, str]] = set()
    training_tokens = 0
    exact_rows = 0
    backoff_rows = 0
    max_form_length = 0
    operations = 0
    for raw in rows:
        if len(seen) >= max_rows:
            raise W02MorphologySuccessorV4Error(
                "V4 training row resource stop")
        if (not isinstance(raw, tuple) or len(raw) != 6
                or any(not isinstance(value, str) for value in raw[:5])
                or type(raw[5]) is not int or raw[5] <= 0):
            raise W02MorphologySuccessorV4Error(
                "V4 aggregated lexeme row is invalid")
        language = _nonempty(raw[0], where="V4 language")
        form = _nonempty(raw[1], where="V4 form")
        lemma = _nonempty(raw[2], where="V4 lemma")
        upos = _nonempty(raw[3], where="V4 UPOS")
        feats_json = _nonempty(raw[4], where="V4 FEATS")
        support = raw[5]
        identity = (language, form, lemma, upos, feats_json)
        if identity in seen:
            raise W02MorphologySuccessorV4Error(
                "V4 aggregated lexeme row is duplicated")
        seen.add(identity)
        languages.add(language)
        exact.setdefault((language, form), Counter())[
            (lemma, upos, feats_json)] += support
        exact_rows += 1
        training_tokens += support
        max_form_length = max(max_form_length, len(form))
        operations += support * (len(form) + len(lemma) + 5)
        lemma_rule = w02_morphology_lemma_rule(form, lemma)
        if lemma_rule is None:
            continue
        combo = (lemma_rule, upos, feats_json)
        backoff_rows += 1
        for feature in w02_morphology_features(form):
            backoff.setdefault((language, feature), Counter())[combo] += support
            operations += support + 1
        if operations > max_logic_operations:
            raise W02MorphologySuccessorV4Error(
                "V4 training logic resource stop")
    if not seen or not exact or not backoff or training_tokens <= 0:
        raise W02MorphologySuccessorV4Error(
            "V4 training evidence is not closed")
    frozen_exact = {key: dict(counts) for key, counts in exact.items()}
    frozen_backoff = {key: dict(counts) for key, counts in backoff.items()}
    frozen_languages = tuple(sorted(languages))
    semantic_rows = _semantic_rows(
        frozen_languages, frozen_exact, frozen_backoff)
    return W02MorphologySuccessorV4Index(
        frozen_languages,
        frozen_exact,
        frozen_backoff,
        max_form_length,
        training_tokens,
        exact_rows,
        backoff_rows,
        operations,
        _hash_value(semantic_rows),
        len(semantic_rows),
    )


def rank_w02_morphology_successor_v4(
        index: W02MorphologySuccessorV4Index,
        language: str,
        form: str,
        ) -> W02MorphologySuccessorV4Ranking:
    """Rank exact lexemes first; use language backoff only for unseen forms."""
    if not isinstance(index, W02MorphologySuccessorV4Index):
        raise TypeError("V4 index type drifted")
    _nonempty(language, where="V4 prediction language")
    _nonempty(form, where="V4 prediction form")
    if language not in index.languages or len(form) > index.max_form_length:
        return W02MorphologySuccessorV4Ranking((), "NONE", 1)
    exact = index.exact_counts.get((language, form))
    if exact is not None:
        ranked = sorted(exact.items(), key=lambda item: (-item[1], item[0]))
        candidates = tuple(
            (combo[0], combo[1], combo[2], support * 64)
            for combo, support in ranked[:W02_MORPH_V4_EXACT_CANDIDATE_LIMIT]
        )
        return W02MorphologySuccessorV4Ranking(
            candidates, "EXACT_LEXEME", len(ranked) + len(form) + 3)
    features = w02_morphology_features(form)
    pool: set[BackoffCombo] = set()
    for feature in features:
        pool.update(index.backoff_counts.get((language, feature), ()))
    scored: list[tuple[BackoffCombo, int]] = []
    for combo in pool:
        score = sum(
            W02_MORPH_FEATURE_WEIGHTS[feature[0]]
            * index.backoff_counts.get((language, feature), {}).get(combo, 0)
            for feature in features)
        if score > 0:
            scored.append((combo, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    candidates = []
    for combo, score in scored[:W02_MORPH_V4_BACKOFF_CANDIDATE_LIMIT]:
        if combo[0] == "IDENTITY":
            lemma = form
        elif combo[0] == "CASEFOLD":
            lemma = form.casefold()
        else:
            raise W02MorphologySuccessorV4Error(
                "V4 backoff lemma rule is not registered")
        candidates.append((lemma, combo[1], combo[2], score))
    mode = "LANGUAGE_BACKOFF" if candidates else "NONE"
    operations = len(features) + len(pool) * (len(features) + 2) + len(scored)
    return W02MorphologySuccessorV4Ranking(
        tuple(candidates), mode, operations)


def predict_w02_morphology_successor_v4(
        index: W02MorphologySuccessorV4Index,
        observation: ObservationRecord,
        v3: W02MorphologySuccessorV3Prediction,
        *,
        requested_spans: tuple[tuple[int, int], ...],
        ) -> W02MorphologySuccessorV4Prediction:
    """Append bounded candidates without altering the frozen V3 prediction."""
    if (not isinstance(index, W02MorphologySuccessorV4Index)
            or not isinstance(observation, ObservationRecord)
            or not isinstance(v3, W02MorphologySuccessorV3Prediction)):
        raise TypeError("V4 prediction input type drifted")
    base = v3.v2.prediction
    if base.observation_key != observation.stable_key.components:
        raise W02MorphologySuccessorV4Error(
            "V4 observation/V3 identity drifted")
    if (not isinstance(requested_spans, tuple)
            or tuple(sorted(set(requested_spans))) != requested_spans
            or len(requested_spans) > W02_MORPH_MAX_QUERIED_SPANS_PER_OBSERVATION):
        raise W02MorphologySuccessorV4Error(
            "V4 requested spans are invalid")
    if observation.language not in index.languages:
        return W02MorphologySuccessorV4Prediction(
            base, 0, 0, 0, 1, index.semantic_sha256)
    if v3.route_authorized != 1:
        raise W02MorphologySuccessorV4Error(
            "V4 language overlay requires an authorized source route")
    surface = base.generation.surface
    boundaries = set(base.boundary_lattice)
    existing = {
        (item.start, item.end, item.form, item.lemma, item.upos, item.feats_json): item
        for item in base.morphology_candidates
    }
    exact_count = 0
    backoff_count = 0
    operations = 0
    for start, end in requested_spans:
        if (type(start) is not int or type(end) is not int
                or start not in boundaries or end not in boundaries
                or start < 0 or end <= start or end > len(surface)):
            raise W02MorphologySuccessorV4Error(
                "V4 requested span is out of bounds")
        form = surface[start:end]
        ranking = rank_w02_morphology_successor_v4(
            index, observation.language, form)
        operations += ranking.logic_operations + len(form) + 3
        for lemma, upos, feats_json, score in ranking.candidates:
            key = (start, end, form, lemma, upos, feats_json)
            if key in existing:
                continue
            existing[key] = W02MorphologyCandidate(
                start, end, form, lemma, upos, feats_json, score)
            if ranking.evidence_mode == "EXACT_LEXEME":
                exact_count += 1
            elif ranking.evidence_mode == "LANGUAGE_BACKOFF":
                backoff_count += 1
            operations += 1
            if exact_count + backoff_count > W02_MORPH_V4_MAX_CANDIDATES_PER_OBSERVATION:
                raise W02MorphologySuccessorV4Error(
                    "V4 observation candidate resource stop")
        if operations > W02_MORPH_V4_MAX_LOGIC_OPERATIONS_PER_OBSERVATION:
            raise W02MorphologySuccessorV4Error(
                "V4 observation logic resource stop")
    prediction = W02CandidatePrediction(
        base.observation_key,
        base.status,
        base.generation,
        base.boundary_lattice,
        base.unicode_units,
        tuple(sorted(existing.values(), key=lambda item: (
            item.start, item.end, item.form, item.lemma, item.upos,
            item.feats_json, item.support_count))),
        base.capabilities,
    )
    return W02MorphologySuccessorV4Prediction(
        prediction,
        exact_count,
        backoff_count,
        len(requested_spans),
        operations,
        index.semantic_sha256,
    )


__all__ = [
    "W02_MORPH_SUCCESSOR_V4_VERSION",
    "W02_MORPH_V4_BACKOFF_CANDIDATE_LIMIT",
    "W02_MORPH_V4_EXACT_CANDIDATE_LIMIT",
    "W02MorphologySuccessorV4Error",
    "W02MorphologySuccessorV4Index",
    "W02MorphologySuccessorV4Prediction",
    "W02MorphologySuccessorV4Ranking",
    "build_w02_morphology_successor_v4_from_counts",
    "predict_w02_morphology_successor_v4",
    "rank_w02_morphology_successor_v4",
]
