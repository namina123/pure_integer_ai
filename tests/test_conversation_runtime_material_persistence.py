from pathlib import Path

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.learning_input_capsule import RuntimeMemoryState
from pure_integer_ai.cognition.shared.scope_identity import session_scope
from pure_integer_ai.experiments.conversation_runtime_material_ingest import ingest_runtime_material
from pure_integer_ai.experiments.conversation_runtime_material_language import (
    observe_runtime_material_language,
)
from pure_integer_ai.experiments.conversation_runtime_material_persistence import (
    RuntimeMaterialPersistenceError,
    load_runtime_material_runtime,
    persist_runtime_material_observation,
    rebuild_runtime_material_observations,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.k_run_boundary import create_new_run_root
from pure_integer_ai.storage.source_record import SourceRecordMetadata


def test_runtime_material_event_and_observation_ledger_rebuilds_without_manual_observation(
        tmp_path: Path) -> None:
    backend = DictBackend()
    bootstrap(backend)
    ctx = make_train_context(backend, companion=True)
    repository = ctx.memory_read_intake.source_intake.repository
    companion = ctx.memory_read_intake.source_intake.companion
    source = SourceRef(91, 8811, 0, GLOBAL_OWNER_SCOPE, VersionBundle())
    scope = session_scope(8811, source=source)
    ingest = ingest_runtime_material(
        RuntimeMemoryState(scope.stable_key()), source=source, scope=scope,
        raw_text="夜间模式会降低屏幕亮度。长按菜单键可打开设置。",
        source_records=repository,
        metadata=SourceRecordMetadata(
            "CC0-1.0", 1, companion.identity.type_hash,
            companion.identity.name_hash, 4),
        source_intake=ctx.memory_read_intake.source_intake,
        version_key=(1, 8811), authority_key=(7, 8811),
    )
    observed = observe_runtime_material_language(
        ctx, ingest, observation_id="obs-8811", context_id="ctx-8811",
        family_id="family-manual", source_namespace="runtime-ledger",
    )
    root = create_new_run_root(tmp_path / "runtime-ledger", require_k_drive=False)
    persist_runtime_material_observation(root, observed)

    recovery = load_runtime_material_runtime(
        root.path, source_records=repository, require_k_drive=False,
    )
    assert recovery.runtime_state == ingest.memory_after
    assert len(recovery.observations) == 1
    rebuilt = rebuild_runtime_material_observations(
        ctx, recovery, source_records=repository,
    )
    assert len(rebuilt) == 1
    assert rebuilt[0].stable_key() == observed.stable_key()
    assert rebuilt[0].raw_observation.canonical_record() == (
        observed.raw_observation.canonical_record())


def test_runtime_material_event_ledger_is_append_only_without_cumulative_state(
        tmp_path: Path) -> None:
    backend = DictBackend()
    bootstrap(backend)
    ctx = make_train_context(backend, companion=True)
    repository = ctx.memory_read_intake.source_intake.repository
    companion = ctx.memory_read_intake.source_intake.companion
    source = SourceRef(91, 8813, 0, GLOBAL_OWNER_SCOPE, VersionBundle())
    scope = session_scope(8813, source=source)
    metadata = SourceRecordMetadata(
        "CC0-1.0", 1, companion.identity.type_hash,
        companion.identity.name_hash, 4)
    first_ingest = ingest_runtime_material(
        RuntimeMemoryState(scope.stable_key()), source=source, scope=scope,
        raw_text="第一份资料。第二句。", source_records=repository,
        metadata=metadata, source_intake=ctx.memory_read_intake.source_intake,
        version_key=(1, 8813, 1), authority_key=(7, 8813),
    )
    first = observe_runtime_material_language(
        ctx, first_ingest, observation_id="obs-8813-1", context_id="ctx-8813-1",
        family_id="family-ledger", source_namespace="runtime-ledger",
    )
    second_ingest = ingest_runtime_material(
        first_ingest.memory_after, source=source, scope=scope,
        raw_text="第一份资料。第二句。", source_records=repository,
        metadata=metadata, source_intake=ctx.memory_read_intake.source_intake,
        version_key=(1, 8813, 2), authority_key=(7, 8813),
    )
    second = observe_runtime_material_language(
        ctx, second_ingest, observation_id="obs-8813-2", context_id="ctx-8813-2",
        family_id="family-ledger", source_namespace="runtime-ledger",
    )
    root = create_new_run_root(tmp_path / "runtime-linear-ledger", require_k_drive=False)
    first_event, _ = persist_runtime_material_observation(root, first)
    second_event, _ = persist_runtime_material_observation(root, second)
    assert second_event.stat().st_size < first_event.stat().st_size * 2
    recovery = load_runtime_material_runtime(
        root.path, source_records=repository, require_k_drive=False,
    )
    assert recovery.runtime_state == second_ingest.memory_after
    assert len(recovery.observations) == 2


def test_runtime_material_ledger_supports_multiple_source_scopes(
        tmp_path: Path) -> None:
    backend = DictBackend()
    bootstrap(backend)
    ctx = make_train_context(backend, companion=True)
    repository = ctx.memory_read_intake.source_intake.repository
    companion = ctx.memory_read_intake.source_intake.companion
    metadata = SourceRecordMetadata(
        "CC0-1.0", 1, companion.identity.type_hash,
        companion.identity.name_hash, 4)

    def observed(source_id: int, ordinal: int):
        source = SourceRef(source_id=source_id, source_kind=91,
                           document_id=0, owner=GLOBAL_OWNER_SCOPE,
                           versions=VersionBundle())
        scope = session_scope(source_id, source=source)
        ingest = ingest_runtime_material(
            RuntimeMemoryState(scope.stable_key()), source=source, scope=scope,
            raw_text="资料甲。资料乙。", source_records=repository,
            metadata=metadata, source_intake=ctx.memory_read_intake.source_intake,
            version_key=(1, source_id), authority_key=(7, source_id),
        )
        return observe_runtime_material_language(
            ctx, ingest, observation_id=f"obs-881{ordinal}",
            context_id=f"ctx-881{ordinal}", family_id="family-ledger",
            source_namespace="runtime-ledger",
        )

    first = observed(8814, 4)
    second = observed(8815, 5)
    root = create_new_run_root(tmp_path / "runtime-multi-scope", require_k_drive=False)
    persist_runtime_material_observation(root, first)
    persist_runtime_material_observation(root, second)
    recovery = load_runtime_material_runtime(
        root.path, source_records=repository, require_k_drive=False,
    )
    assert len(recovery.runtime_states) == 2
    rebuilt = rebuild_runtime_material_observations(
        ctx, recovery, source_records=repository,
    )
    assert tuple(item.stable_key() for item in rebuilt) == tuple(
        sorted((first.stable_key(), second.stable_key())))


def test_runtime_material_ledger_rejects_missing_source_record(tmp_path: Path) -> None:
    backend = DictBackend()
    bootstrap(backend)
    ctx = make_train_context(backend, companion=True)
    repository = ctx.memory_read_intake.source_intake.repository
    companion = ctx.memory_read_intake.source_intake.companion
    source = SourceRef(91, 8812, 0, GLOBAL_OWNER_SCOPE, VersionBundle())
    scope = session_scope(8812, source=source)
    ingest = ingest_runtime_material(
        RuntimeMemoryState(scope.stable_key()), source=source, scope=scope,
        raw_text="资料。下一步。", source_records=repository,
        metadata=SourceRecordMetadata(
            "CC0-1.0", 1, companion.identity.type_hash,
            companion.identity.name_hash, 4),
        source_intake=ctx.memory_read_intake.source_intake,
        version_key=(1, 8812), authority_key=(7, 8812),
    )
    observed = observe_runtime_material_language(
        ctx, ingest, observation_id="obs-8812", context_id="ctx-8812",
        family_id="family-manual", source_namespace="runtime-ledger",
    )
    root = create_new_run_root(tmp_path / "runtime-ledger-missing", require_k_drive=False)
    persist_runtime_material_observation(root, observed)
    # A separate empty repository models a Runtime SQLite opened without the
    # required append-only SourceRecord row; production repositories are never
    # mutated to create this condition.
    missing_backend = DictBackend()
    bootstrap(missing_backend)
    missing_repository = type(repository)(missing_backend)
    try:
        load_runtime_material_runtime(
            root.path, source_records=missing_repository, require_k_drive=False,
        )
    except RuntimeMaterialPersistenceError:
        pass
    else:
        raise AssertionError("missing SourceRecord must fail closed")
