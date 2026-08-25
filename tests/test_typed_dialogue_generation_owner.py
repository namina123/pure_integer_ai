"""公开对话 typed generation owner 的真实纵切回归。"""
from __future__ import annotations

from pure_integer_ai.config import gates
from pure_integer_ai.cognition.understanding.occurrence_index import OccurrenceIndex
from pure_integer_ai.cognition.understanding.span_index import SpanIndex
from pure_integer_ai.experiments.conversation_training_pack import (
    load_dialogue_training_pack,
)
from pure_integer_ai.experiments.corpus_identity import assign_corpus_source_refs
from pure_integer_ai.experiments.generation_production_runtime import (
    install_production_generation_runtime,
)
from pure_integer_ai.experiments.language_semantic_runtime import (
    install_language_semantic_course_runtime,
)
from pure_integer_ai.experiments.round_runtime import DefaultRoundRunner
from pure_integer_ai.experiments.run_conversation_training import (
    _dialogue_semantic_protocol,
    default_course_paths,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.typed_dialogue_generation_owner import (
    TypedDialogueGenerationRuntimeFactory,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.training.stages import STAGE3_REWARD


def test_public_adoption_payload_reaches_unicode_renderer() -> None:
    """唯一 typed adoption claim 经过正式 S-02 与 G-00..G-03 形成输出。"""
    pack = load_dialogue_training_pack(default_course_paths("."))
    items = pack.training_items()
    assign_corpus_source_refs(items, source_namespace=pack.pack_sha256)
    item = [
        value for value in items
        if value.payload_kind == "GenerationAdoptionPostcheckQuery"
    ][2]

    backend = DictBackend()
    previous = gates.TRAINING_MODE
    try:
        ctx = make_train_context(backend)
        semantic, occurrence_protocol, span_protocol = _dialogue_semantic_protocol()
        occurrence = OccurrenceIndex(
            ctx.graph_ontology,
            ctx.scoped_identity_store,
            occurrence_protocol,
        )
        span = SpanIndex(
            ctx.graph_ontology,
            ctx.scoped_identity_store,
            span_protocol.span_protocol,
            occurrence,
        )
        ctx.occurrence_index = occurrence
        ctx.span_index = span
        install_language_semantic_course_runtime(ctx, semantic)
        install_production_generation_runtime(
            ctx,
            TypedDialogueGenerationRuntimeFactory.from_project_root("."),
        )
        gates.TRAINING_MODE = True
        result = DefaultRoundRunner().run_round_full(
            ctx, item, STAGE3_REWARD, 1)
        assert result.typed_episode is not None
        assert result.typed_episode.generation_complete is True
        assert result.output is not None
        assert result.output.execution is not None
        assert result.output.execution.rendered is not None
        assert result.output.execution.rendered.units
        assert result.output.postcheck is not None
        assert result.output.postcheck.complete is True
    finally:
        gates.TRAINING_MODE = previous
        backend.close()
