"""Bounded, incremental recall over persisted broad-dialogue turns.

The index is a disposable runtime projection.  It learns the shape of a recall
request only from an observed dialogue transition whose answer exactly replays
the preceding unanswered user statement.  Language surfaces stay in the
checkpoint; the policy stores only integer scalar n-gram evidence and ordinals.
"""
from __future__ import annotations

from pure_integer_ai.experiments.conversation_broad_dialogue_persistence import (
    PersistentBroadDialogueRecovery,
)
from pure_integer_ai.experiments.conversation_broad_qa_runtime import DialogueTurn


_MAX_RECALLED_STATEMENTS = 2
_MIN_CONTENT_FEATURE_WIDTH = 2
_MIN_RECALL_SHAPE_SIMILARITY_PERMILLE = 200
_MIN_RECALL_SHAPE_SHARED_FEATURES = 2
_MEMORY_CANDIDATE_STATUSES = frozenset({"UNKNOWN", "CLARIFY"})


# object-model: derived_cache; representation=runtime; interop=broad-dialogue-recall-v1
class BroadDialogueMemoryRecallIndex:
    """Incremental turn/replay index with no vocabulary or language table."""

    __slots__ = (
        "_turn_by_ordinal", "_features_by_ordinal", "_feature_ordinals",
        "_memory_ordinals", "_recall_query_ordinals",
        "_operator_feature_counts", "_last_ordinal",
    )

    def __init__(self, turns: tuple[DialogueTurn, ...] = ()) -> None:
        if (not isinstance(turns, tuple)
                or any(not isinstance(item, DialogueTurn) for item in turns)):
            raise TypeError("memory recall turns 必须是 DialogueTurn tuple")
        self._turn_by_ordinal: dict[int, DialogueTurn] = {}
        self._features_by_ordinal: dict[
            int, frozenset[tuple[int, ...]]] = {}
        self._feature_ordinals: dict[tuple[int, ...], list[int]] = {}
        self._memory_ordinals: set[int] = set()
        self._recall_query_ordinals: set[int] = set()
        self._operator_feature_counts: dict[tuple[int, ...], int] = {}
        self._last_ordinal = -1
        for turn in turns:
            self.append(turn)

    @staticmethod
    def _features(value: str) -> frozenset[tuple[int, ...]]:
        return PersistentBroadDialogueRecovery._recall_features(value)

    @staticmethod
    def _is_replay_pair(previous: DialogueTurn,
                        current: DialogueTurn) -> bool:
        return (
            previous.status in _MEMORY_CANDIDATE_STATUSES
            and previous.answer is None
            and current.status == "ANSWER"
            and current.answer is not None
            and current.answer.strip() == previous.question.strip()
        )

    def _learn_replay_pair(self, previous: DialogueTurn,
                           current: DialogueTurn) -> None:
        """Learn recall-operation evidence, never the remembered surface."""
        if not self._is_replay_pair(previous, current):
            return
        self._memory_ordinals.add(previous.ordinal)
        self._recall_query_ordinals.add(current.ordinal)
        previous_features = self._features_by_ordinal[previous.ordinal]
        current_features = self._features_by_ordinal[current.ordinal]
        for feature in current_features.difference(previous_features):
            if len(feature) < _MIN_CONTENT_FEATURE_WIDTH:
                continue
            self._operator_feature_counts[feature] = (
                self._operator_feature_counts.get(feature, 0) + 1)

    def append(self, turn: DialogueTurn) -> None:
        """Append one monotonic turn in work proportional to its own surface."""
        if not isinstance(turn, DialogueTurn):
            raise TypeError("memory recall turn 类型错误")
        existing = self._turn_by_ordinal.get(turn.ordinal)
        if existing is not None:
            if existing.turn_key == turn.turn_key:
                return
            raise ValueError("memory recall ordinal identity 漂移")
        if turn.ordinal <= self._last_ordinal:
            raise ValueError("memory recall turn 必须按 ordinal 递增")
        previous = self._turn_by_ordinal.get(self._last_ordinal)
        features = self._features(turn.question)
        self._turn_by_ordinal[turn.ordinal] = turn
        self._features_by_ordinal[turn.ordinal] = features
        for feature in features:
            self._feature_ordinals.setdefault(feature, []).append(turn.ordinal)
        self._last_ordinal = turn.ordinal
        if previous is not None:
            self._learn_replay_pair(previous, turn)

    def query_relevant_turns(
            self, question: str, *, limit: int = 4,
            minimum_similarity_permille: int = 500,
            ) -> tuple[DialogueTurn, ...]:
        """Return bounded relevant history without scanning all prior turns."""
        if type(limit) is not int or limit <= 0:
            raise ValueError("memory recall limit 必须为正整数")
        if (type(minimum_similarity_permille) is not int
                or not 0 <= minimum_similarity_permille <= 1000):
            raise ValueError("memory recall similarity 必须是 0..1000 整数")
        query = self._features(question)
        candidates = {
            ordinal
            for feature in query
            for ordinal in self._feature_ordinals.get(feature, ())
        }
        ranked: list[tuple[int, int, int, DialogueTurn]] = []
        for ordinal in candidates:
            turn = self._turn_by_ordinal[ordinal]
            if turn.question.strip() == question.strip():
                continue
            candidate = self._features_by_ordinal[ordinal]
            overlap = len(query.intersection(candidate))
            if overlap <= 0:
                continue
            score = (2000 * overlap) // (len(query) + len(candidate))
            if score >= minimum_similarity_permille:
                ranked.append((score, overlap, ordinal, turn))
        ranked.sort(key=lambda item: (-item[0], -item[1], -item[2]))
        return tuple(item[3] for item in ranked[:limit])

    def _has_learned_recall_shape(
            self, query: frozenset[tuple[int, ...]],
            ) -> bool:
        for ordinal in self._recall_query_ordinals:
            prototype = self._features_by_ordinal[ordinal]
            overlap = query.intersection(prototype)
            shared = sum(
                len(feature) >= _MIN_CONTENT_FEATURE_WIDTH
                for feature in overlap)
            similarity = (2000 * len(overlap)) // (
                len(query) + len(prototype))
            if (shared >= _MIN_RECALL_SHAPE_SHARED_FEATURES
                    and similarity
                    >= _MIN_RECALL_SHAPE_SIMILARITY_PERMILLE):
                return True
        return False

    def _immediate_bootstrap_candidate(
            self, query: frozenset[tuple[int, ...]], question: str,
            ) -> DialogueTurn | None:
        """Bootstrap a new recall form only against the immediately prior turn."""
        previous = self._turn_by_ordinal.get(self._last_ordinal)
        if (previous is None
                or previous.status not in _MEMORY_CANDIDATE_STATUSES
                or previous.answer is not None
                or previous.question.strip() == question.strip()):
            return None
        overlap = query.intersection(
            self._features_by_ordinal[previous.ordinal])
        if not any(len(feature) >= _MIN_CONTENT_FEATURE_WIDTH
                   for feature in overlap):
            return None
        return previous

    def recall(self, question: str) -> str | None:
        """Return persisted statement text only for an evidenced recall shape."""
        if type(question) is not str or not question.strip():
            return None
        query = self._features(question)
        learned_shape = self._has_learned_recall_shape(query)
        immediate = (
            self._immediate_bootstrap_candidate(query, question)
            if not self._memory_ordinals or learned_shape else None
        )
        if immediate is not None:
            return immediate.question
        if not learned_shape:
            return None

        operator_features = frozenset(self._operator_feature_counts)
        content = frozenset(
            feature for feature in query
            if len(feature) >= _MIN_CONTENT_FEATURE_WIDTH
            and feature not in operator_features
        )
        candidates = {
            ordinal
            for feature in content
            for ordinal in self._feature_ordinals.get(feature, ())
            if ordinal in self._memory_ordinals
        }
        ranked: list[tuple[int, int, int, DialogueTurn]] = []
        for ordinal in candidates:
            turn = self._turn_by_ordinal[ordinal]
            overlap = content.intersection(
                self._features_by_ordinal[ordinal])
            if not overlap:
                continue
            score = sum(
                len(feature) * 1000
                // max(1, len(self._feature_ordinals.get(feature, ())))
                for feature in overlap
            )
            ranked.append((score, len(overlap), ordinal, turn))
        ranked.sort(key=lambda item: (-item[0], -item[1], -item[2]))
        if not ranked:
            return None
        best = ranked[0][0]
        selected: list[str] = []
        for score, _overlap, _ordinal, turn in ranked:
            if score * 4 < best or turn.question in selected:
                continue
            selected.append(turn.question)
            if len(selected) >= _MAX_RECALLED_STATEMENTS:
                break
        return "\n".join(selected) if selected else None


__all__ = ["BroadDialogueMemoryRecallIndex"]
