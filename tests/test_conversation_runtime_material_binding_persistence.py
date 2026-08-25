from pathlib import Path

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE, SourceRef, VersionBundle,
)
from pure_integer_ai.cognition.shared.learning_input_capsule import RuntimeMemoryState
from pure_integer_ai.cognition.shared.scope_identity import session_scope
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    RawPropositionQualification,
)
from pure_integer_ai.experiments.conversation_runtime_material_binding_persistence import (
    RuntimeMaterialBindingPersistenceError,
    decode_runtime_material_response_bindings,
    encode_runtime_material_response_bindings,
    load_runtime_material_response_provider,
    persist_runtime_material_response_bindings,
)
from pure_integer_ai.experiments.conversation_runtime_material_ingest import (
    ingest_runtime_material,
)
from pure_integer_ai.experiments.conversation_runtime_material_language import (
    observe_runtime_material_language,
)
from pure_integer_ai.experiments.conversation_runtime_material_response import (
    RuntimeMaterialResponseSpec,
    build_runtime_material_response_provider,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.k_run_boundary import create_new_run_root
from pure_integer_ai.storage.source_record import SourceRecordMetadata


def test_runtime_response_binding_integer_ledger_roundtrip(tmp_path: Path):
    backend = DictBackend()
    bootstrap(backend)
    ctx = make_train_context(backend, companion=True)
    repository = ctx.memory_read_intake.source_intake.repository
    companion = ctx.memory_read_intake.source_intake.companion
    source = SourceRef(91, 9901, 0, GLOBAL_OWNER_SCOPE, VersionBundle())
    scope = session_scope(9901, source=source)
    ingest = ingest_runtime_material(
        RuntimeMemoryState(scope.stable_key()), source=source, scope=scope,
        raw_text="夜间模式会降低屏幕亮度。长按菜单键可打开设置。",
        source_records=repository,
        metadata=SourceRecordMetadata(
            "CC0-1.0", 1, companion.identity.type_hash,
            companion.identity.name_hash, 4),
        source_intake=ctx.memory_read_intake.source_intake,
        version_key=(1, 9901), authority_key=(7, 9901),
    )
    observation = observe_runtime_material_language(
        ctx, ingest, observation_id="obs-9901", context_id="ctx-9901",
        family_id="family-ledger", source_namespace="runtime-ledger",
    )
    relation = observation.relation_candidates[0]
    qualification = RawPropositionQualification(
        "qual-9901", relation.proposition.proposition_id,
        observation.raw_observation.observation_id,
        observation.raw_observation.source_id,
        observation.raw_observation.context_id,
        observation.raw_observation.family_id,
        observation.raw_observation.source_namespace,
        observation.raw_observation.split, "SUPPORTED", "ledger-authority",
        tuple(item.evidence_id for item in relation.evidence), "runtime-owner",
    )
    provider = build_runtime_material_response_provider(
        (RuntimeMaterialResponseSpec(
            observation, qualification,
            "夜间模式与设置的先后关系是什么？",
            source_title="手册", source_url="https://example.invalid/manual"),),
        source_records=repository,
    )
    payload = encode_runtime_material_response_bindings(provider)
    assert decode_runtime_material_response_bindings(payload)[0].question == (
        "夜间模式与设置的先后关系是什么？")
    root = create_new_run_root(tmp_path / "runtime-binding", require_k_drive=False)
    path = persist_runtime_material_response_bindings(root, provider)
    assert path.is_file()
    restored = load_runtime_material_response_provider(
        root.path, source_records=repository, observations=(observation,),
        require_k_drive=False,
    )
    assert restored.response("夜间模式与设置的先后关系是什么？") == provider.response(
        "夜间模式与设置的先后关系是什么？")

    with __import__("pytest").raises(RuntimeMaterialBindingPersistenceError):
        load_runtime_material_response_provider(
            root.path, source_records=repository, observations=(),
            require_k_drive=False,
        )
