"""U-04 首个切片：文件来源、图桥、候选竞争和 clone 隔离。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.language_signal import (
    read_language_signal_catalog,
)
from pure_integer_ai.cognition.shared.modal_primitives import (
    MODAL_KIND_BOX_NECESSITY,
    MODAL_KIND_BOX_POSSIBILITY,
)
from pure_integer_ai.cognition.shared.action_primitives import (
    ACTION_GENERATE,
    INTENT_COMMAND_MOOD,
    ensure_action_primitives,
)
from pure_integer_ai.cognition.understanding.cue_words import (
    ARITH_EQUALS_CUE,
    CAUSES_CUE_FORWARD,
    EXISTENTIAL_CUE,
    _COND_ELSE,
    _COND_IF,
    _COND_THEN,
    arith_op_of,
    action_intent_of,
    collect_action_intent_concepts,
    collect_action_intent_word_decisions,
    comparison_op_of,
    cond_keyword_of,
    cue_type_of,
    is_negation_cue,
    is_action_intent_cue,
    is_property_attr_marker,
    is_property_possess_cue,
    is_property_value_copula,
    is_similar_cue,
    modal_op_of,
)
from pure_integer_ai.cognition.shared.types import (
    DOMAIN_TEXT,
    InputPayload,
    INTENT_COMMAND,
    IntentType,
    LANG_ZH,
    MODALITY_LANGUAGE,
    Segment,
    STAGE_TRAINING,
    TERMINAL_REACHED_SINK,
)
from pure_integer_ai.cognition.process.dag_path import _intent_override
from pure_integer_ai.cognition.understanding.modification_direction import (
    head_preference,
)
from pure_integer_ai.cognition.understanding.observe import ObservePipeline
from pure_integer_ai.cognition.understanding.refers_to import (
    resolve_pronoun_signal,
)
from pure_integer_ai.crosscut.integer.compare import CMP_GT
from pure_integer_ai.numeric.symbol_domain import OPCODE_ADD
from pure_integer_ai.config import gates
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.evaluation_isolation import (
    isolated_evaluation,
)
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig,
    formal_train,
    make_train_context,
)
from pure_integer_ai.experiments.language_observation import (
    _split_item_to_segments,
)
from pure_integer_ai.experiments.language_signal_intake import (
    build_language_signal_runtime,
)
from pure_integer_ai.experiments.round_runtime import (
    _build_space_ctx,
    _resolve_emergent_excluded_refs,
)
from pure_integer_ai.experiments.round_runtime import (
    _collect_action_seed_candidates,
    _feed_action_experience,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT, SOURCE_DERIVED
from pure_integer_ai.storage.edge_types import (
    EDGE_PROPERTY,
    EDGE_RELATION_SIGNAL,
)
from pure_integer_ai.storage.experience_count import (
    pack_ctx_code,
    read_experience_count,
)
from pure_integer_ai.cognition.understanding.intent_classify import classify_intent
from pure_integer_ai.cognition.understanding.word_concept_signal import (
    bootstrap_action_signals,
    bootstrap_word_concept_signals,
)
from pure_integer_ai.storage.telemetry import collect_backend_telemetry


def _seed_payload() -> dict:
    """构造包含同表层竞争候选的最小来源 seed。"""
    return {
        "schema_version": 1,
        "source_kind": 9901,
        "versions": [1, 2, 3, 4],
        "unicode_family_key": [99011],
        "branch_keys": {"1": [99012]},
        "relation_keys": {
            "branch_inventory": [99013],
            "branch_atom": [99014],
            "atom_representation": [99015],
            "atom_instruction": [99016],
        },
        "entries": [
            {"language": 1, "surface": "甲", "atom_key": [1], "instruction_key": [7]},
            {"language": 1, "surface": "甲", "atom_key": [2], "instruction_key": [8]},
            {"language": 1, "surface": "乙", "atom_key": [3], "instruction_key": [7]},
        ],
    }


def _write_seed(tmp_path: Path) -> Path:
    """写入测试 seed 文件并返回路径。"""
    path = tmp_path / "language-signal.json"
    path.write_text(
        json.dumps(_seed_payload(), ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _write_payload(tmp_path: Path, payload: dict, name: str) -> Path:
    """把指定语言信号 payload 写成独立测试来源。"""
    path = tmp_path / name
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _scaled_seed_payload(entry_count: int) -> dict:
    """构造指定规模且每个表层只有一个候选的性能 seed。"""
    payload = _seed_payload()
    payload["entries"] = [
        {
            "language": 1,
            "surface": f"词{ordinal}",
            "atom_key": [ordinal + 1],
            "instruction_key": [ordinal + 1001],
        }
        for ordinal in range(entry_count)
    ]
    return payload


def _integer_signal_payload() -> dict:
    """构造关系、算术、比较和条件各自具有独立指令的 seed。"""
    payload = _seed_payload()
    payload["entries"] = [
        {"language": 1, "surface": "图因", "atom_key": [31],
         "instruction_key": [41]},
        {"language": 1, "surface": "图加", "atom_key": [32],
         "instruction_key": [42]},
        {"language": 1, "surface": "图等", "atom_key": [33],
         "instruction_key": [43]},
        {"language": 1, "surface": "图大", "atom_key": [34],
         "instruction_key": [44]},
        {"language": 1, "surface": "图若", "atom_key": [35],
         "instruction_key": [45]},
        {"language": 1, "surface": "图则", "atom_key": [36],
         "instruction_key": [46]},
        {"language": 1, "surface": "图否", "atom_key": [37],
         "instruction_key": [47]},
    ]
    return payload


def _property_signal_payload() -> dict:
    """构造属性标记、值系词、领属和存在量化的独立图指令。"""
    payload = _seed_payload()
    payload["entries"] = [
        {"language": 1, "surface": "图的", "atom_key": [61],
         "instruction_key": [81]},
        {"language": 1, "surface": "图是", "atom_key": [62],
         "instruction_key": [82]},
        {"language": 1, "surface": "图有", "atom_key": [63],
         "instruction_key": [83]},
        {"language": 1, "surface": "图存在", "atom_key": [64],
         "instruction_key": [84]},
        {"language": 1, "surface": "乙", "atom_key": [65],
         "instruction_key": [7]},
    ]
    return payload


def _action_signal_payload() -> dict:
    """构造命令 mood、生成动作和旧字面冲突的图指令。"""
    payload = _seed_payload()
    payload["entries"] = [
        {"language": 1, "surface": "图请", "atom_key": [91],
         "instruction_key": [111]},
        {"language": 1, "surface": "图生成", "atom_key": [92],
         "instruction_key": [112]},
        {"language": 1, "surface": "生成", "atom_key": [93],
         "instruction_key": [112]},
        {"language": 1, "surface": "生成", "atom_key": [94],
         "instruction_key": [199]},
    ]
    return payload


def _pronoun_signal_payload() -> dict:
    """构造含多特征代词 profile 的来源化图 seed。"""
    payload = _seed_payload()
    payload["branch_keys"]["2"] = [99017]
    payload["entries"] = [
        {"language": 1, "surface": "图他", "atom_key": [121],
         "instruction_key": [121]},
        {"language": 1, "surface": "图她", "atom_key": [122],
         "instruction_key": [122]},
        {"language": 2, "surface": "图he", "atom_key": [123],
         "instruction_key": [121]},
    ]
    return payload


def _similar_signal_payload() -> dict:
    """构造图相似 cue、旧字面冲突和未绑定候选。"""
    payload = _seed_payload()
    payload["entries"] = [
        {"language": 1, "surface": "图似", "atom_key": [131],
         "instruction_key": [131]},
        {"language": 1, "surface": "像", "atom_key": [132],
         "instruction_key": [131]},
        {"language": 1, "surface": "像", "atom_key": [133],
         "instruction_key": [999]},
    ]
    return payload


def _lookup_operations(tmp_path: Path, entry_count: int) -> tuple[int, int]:
    """安装指定规模 seed，并返回全目录查询的后端调用数与读取行数。"""
    path = tmp_path / f"scale-{entry_count}.json"
    path.write_text(
        json.dumps(_scaled_seed_payload(entry_count), ensure_ascii=False),
        encoding="utf-8",
    )
    ctx = make_train_context(DictBackend())
    runtime = build_language_signal_runtime(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        seed_path=path,
    )
    runtime.install()
    with collect_backend_telemetry() as collector:
        for ordinal in range(entry_count):
            assert len(runtime.lookup(f"词{ordinal}", language=1)) == 1
    operations = collector.operation_snapshot()
    calls = sum(stats[0] for (operation, _), stats in operations.items()
                if operation == "select")
    rows = sum(stats[1] for (operation, _), stats in operations.items()
               if operation == "select")
    return calls, rows


def test_catalog_derives_distinct_source_refs_from_file_content(tmp_path):
    """文件内容变化必须改变 SourceRef 来源身份，行序保留 fragment 身份。"""
    path = _write_seed(tmp_path)
    catalog = read_language_signal_catalog(path)
    assert len(catalog.entries) == 3
    assert catalog.entries[0].source.document_id == 0
    assert len({entry.source.source_id for entry in catalog.entries}) == 1
    first_digest = catalog.content_sha256
    path.write_text(path.read_text(encoding="utf-8").replace("乙", "丙"),
                    encoding="utf-8")
    changed = read_language_signal_catalog(path)
    assert changed.content_sha256 != first_digest
    assert changed.entries[0].source.source_id != catalog.entries[0].source.source_id


def test_runtime_materializes_graph_and_returns_all_competing_candidates(tmp_path):
    """同 surface 的两个 instruction 候选都应被返回，不能按稳定序私选。"""
    ctx = make_train_context(DictBackend())
    runtime = build_language_signal_runtime(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        seed_path=_write_seed(tmp_path),
    )
    report = runtime.install()
    assert report.entry_count == 3
    assert report.statement_count == 12
    candidates = runtime.lookup("甲", language=1)
    assert len(candidates) == 2
    assert {ctx.graph_ontology.identity_of(item.instruction).components[-1]
            for item in candidates} == {7, 8}
    assert all(item.source.source_kind == 9901 for item in candidates)


def test_runtime_instruction_match_preserves_three_states(tmp_path):
    """图无证据、候选一致和候选冲突必须保持三个不同结果。"""
    ctx = make_train_context(DictBackend())
    runtime = build_language_signal_runtime(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        seed_path=_write_seed(tmp_path),
    )
    runtime.install()
    assert runtime.matches_instruction(
        "未知", language=1, instruction_key=(7,)) is None
    assert runtime.matches_instruction(
        "乙", language=1, instruction_key=(7,)) is True
    assert runtime.matches_instruction(
        "甲", language=1, instruction_key=(7,)) is False
    assert runtime.resolve_instruction(
        "未知", language=1).has_evidence is False
    assert runtime.resolve_instruction(
        "乙", language=1).instruction_key == (7,)
    assert runtime.resolve_instruction(
        "甲", language=1).instruction_key is None


def test_runtime_rejects_missing_branch_and_malformed_duplicate(tmp_path):
    """未知语言和重复 seed 必须 fail closed。"""
    ctx = make_train_context(DictBackend())
    runtime = build_language_signal_runtime(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        seed_path=_write_seed(tmp_path),
    )
    runtime.install()
    assert runtime.lookup("甲", language=2) == ()
    payload = _seed_payload()
    payload["entries"].append(payload["entries"][0])
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="seed entry 重复"):
        read_language_signal_catalog(duplicate)

    payload = _seed_payload()
    payload["relation_keys"]["branch_atom"] = [99013]
    collision = tmp_path / "collision.json"
    collision.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="relation_keys 必须两两不同"):
        read_language_signal_catalog(collision)


def test_runtime_clone_isolated_and_recoverable(tmp_path):
    """评测 clone 必须重建图 runtime，宿主新增 statement 不得被 clone 共享。"""
    ctx = make_train_context(DictBackend())
    runtime = build_language_signal_runtime(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        seed_path=_write_seed(tmp_path),
    )
    ctx.language_signal_runtime = runtime
    action_refs = ensure_action_primitives(
        ctx.concept_index, ctx.backend, space_id=ctx.space_id)
    ctx.language_property_attr_instruction_key = (81,)
    ctx.language_property_value_instruction_key = (82,)
    ctx.language_property_possess_instruction_key = (83,)
    ctx.language_similar_instruction_key = (131,)
    ctx.language_pronoun_instruction_bindings = (
        ((121,), (701, 702)),)
    ctx.language_action_instruction_bindings = (((7,), ACTION_GENERATE),)
    ctx.language_action_primitive_refs = tuple(sorted(action_refs.items()))
    ctx.language_modality_instruction_bindings = (
        ((7,), MODAL_KIND_BOX_NECESSITY),)
    ctx.language_arithmetic_instruction_bindings = (((7,), OPCODE_ADD),)
    ctx.language_comparison_instruction_bindings = (((7,), CMP_GT),)
    ctx.language_condition_instruction_bindings = (((7,), _COND_IF),)
    ctx.language_cue_instruction_bindings = (
        ((7,), CAUSES_CUE_FORWARD),)
    runtime.install()
    before = ctx.backend.snapshot()
    with isolated_evaluation(ctx, label="u04-language-signal") as cloned:
        assert cloned.language_signal_runtime is not runtime
        assert cloned.language_property_attr_instruction_key == (81,)
        assert cloned.language_property_value_instruction_key == (82,)
        assert cloned.language_property_possess_instruction_key == (83,)
        assert cloned.language_similar_instruction_key == (131,)
        assert cloned.language_pronoun_instruction_bindings == (
            ((121,), (701, 702)),)
        assert cloned.language_action_instruction_bindings == (
            ((7,), ACTION_GENERATE),)
        assert cloned.language_action_primitive_refs == tuple(
            sorted(action_refs.items()))
        assert cloned.language_modality_instruction_bindings == (
            ((7,), MODAL_KIND_BOX_NECESSITY),)
        assert cloned.language_arithmetic_instruction_bindings == (
            ((7,), OPCODE_ADD),)
        assert cloned.language_comparison_instruction_bindings == (
            ((7,), CMP_GT),)
        assert cloned.language_condition_instruction_bindings == (
            ((7,), _COND_IF),)
        assert cloned.language_cue_instruction_bindings == (
            ((7,), CAUSES_CUE_FORWARD),)
        assert len(cloned.language_signal_runtime.lookup("甲", language=1)) == 2
        cloned.language_signal_runtime.install()
        assert cloned.backend.snapshot() == before
    assert ctx.backend.snapshot() == before


def test_graph_pronoun_profile_drives_multi_feature_occurrence_edges(tmp_path):
    """图代词 profile 必须保留多特征，并由同次解析写入多条 PROPERTY 边。"""
    ctx = make_train_context(DictBackend())
    runtime = build_language_signal_runtime(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        seed_path=_write_payload(
            tmp_path, _pronoun_signal_payload(), "pronoun-signals.json"),
    )
    runtime.install()
    bindings = (((121,), (701, 702)), ((122,), (703,)))
    signal = resolve_pronoun_signal(
        "图他", lang=LANG_ZH, language_signal_runtime=runtime,
        pronoun_instruction_bindings=bindings,
        language_signal_compatibility_enabled=False,
    )
    assert signal.is_pronoun is True
    assert signal.feature_keys == (701, 702)
    assert resolve_pronoun_signal(
        "他", lang=LANG_ZH, language_signal_runtime=runtime,
        pronoun_instruction_bindings=bindings,
        language_signal_compatibility_enabled=False,
    ).is_pronoun is False

    pipeline = ObservePipeline(
        _build_space_ctx(ctx),
        concept_index=ctx.concept_index,
        language_signal_runtime=runtime,
        pronoun_instruction_bindings=bindings,
        language_signal_compatibility_enabled=False,
    )
    pipeline.observe(InputPayload(
        segments=[Segment(
            seg_id=0,
            modality=MODALITY_LANGUAGE,
            lang=LANG_ZH,
            domain=DOMAIN_TEXT,
            tokens=["图他"],
        )],
        source=SOURCE_BARE_TEXT,
        stage=STAGE_TRAINING,
        modality=MODALITY_LANGUAGE,
        lang=LANG_ZH,
        domain=DOMAIN_TEXT,
    ))
    mem_sid = ctx.space_id
    feature_edges = ctx.backend.select(
        "edge",
        where={
            "edge_type": EDGE_PROPERTY,
            "space_id_from": mem_sid,
            "source": SOURCE_DERIVED,
        },
    )
    assert len(feature_edges) == 2
    assert {row["local_id_to"] for row in feature_edges} == {
        ctx.concept_index.lookup(701, mem_sid)[1],
        ctx.concept_index.lookup(702, mem_sid)[1],
    }


def test_graph_pronoun_conflict_and_unbound_block_legacy_fallback(tmp_path):
    """图冲突或图未绑定时，旧代词字面不得重新激活代词语义。"""
    payload = _pronoun_signal_payload()
    payload["entries"].extend([
        {"language": 1, "surface": "图冲突", "atom_key": [124],
         "instruction_key": [121]},
        {"language": 1, "surface": "图冲突", "atom_key": [125],
         "instruction_key": [999]},
    ])
    ctx = make_train_context(DictBackend())
    runtime = build_language_signal_runtime(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        seed_path=_write_payload(
            tmp_path, payload, "pronoun-conflict.json"),
    )
    runtime.install()
    bindings = (((121,), (701, 702)),)
    assert resolve_pronoun_signal(
        "图冲突", lang=LANG_ZH, language_signal_runtime=runtime,
        pronoun_instruction_bindings=bindings,
        language_signal_compatibility_enabled=True,
    ).is_pronoun is False
    assert resolve_pronoun_signal(
        "他", lang=LANG_ZH, language_signal_runtime=runtime,
        pronoun_instruction_bindings=bindings,
        language_signal_compatibility_enabled=True,
    ).is_pronoun is True
    assert resolve_pronoun_signal(
        "图他", lang=LANG_ZH, language_signal_runtime=runtime,
        pronoun_instruction_bindings=(),
        language_signal_compatibility_enabled=True,
    ).is_pronoun is False


def test_graph_similar_cue_drives_split_and_blocks_legacy_fallback(tmp_path):
    """图相似作用须驱动正式分段，冲突和未绑定不得回退 D:11。"""
    ctx = make_train_context(DictBackend())
    bootstrap_word_concept_signals(
        ctx.concept_index, ctx.edge_store, ctx.backend,
        space_id=ctx.space_id, langs={LANG_ZH})
    runtime = build_language_signal_runtime(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        seed_path=_write_payload(
            tmp_path, _similar_signal_payload(), "similar-signals.json"),
    )
    runtime.install()
    saved_readback = gates.EMERGENT_RELATION_CUE_READBACK_MODE
    saved_extract = gates.CUE_EXTRACTOR_MODE
    gates.EMERGENT_RELATION_CUE_READBACK_MODE = True
    gates.CUE_EXTRACTOR_MODE = True
    try:
        assert is_similar_cue(
            "图似", LANG_ZH,
            backend=ctx.backend, edge_store=ctx.edge_store,
            space_id=ctx.space_id, concept_index=ctx.concept_index,
            language_signal_runtime=runtime,
            similar_instruction_key=(131,),
            language_signal_compatibility_enabled=False,
        ) is True
        segments = _split_item_to_segments(
            CollectedItem(tokens=["猫", "图似", "老虎"]),
            backend=ctx.backend, edge_store=ctx.edge_store,
            space_id=ctx.space_id, concept_index=ctx.concept_index,
            language_signal_runtime=runtime,
            similar_instruction_key=(131,),
            language_signal_compatibility_enabled=False,
        )
        assert segments[0].similar_claims == [(0, 2)]
        assert is_similar_cue(
            "像", LANG_ZH,
            backend=ctx.backend, edge_store=ctx.edge_store,
            space_id=ctx.space_id, concept_index=ctx.concept_index,
            language_signal_runtime=runtime,
            similar_instruction_key=(131,),
            language_signal_compatibility_enabled=True,
        ) is False
        assert is_similar_cue(
            "图似", LANG_ZH,
            backend=ctx.backend, edge_store=ctx.edge_store,
            space_id=ctx.space_id, concept_index=ctx.concept_index,
            language_signal_runtime=runtime,
            similar_instruction_key=None,
            language_signal_compatibility_enabled=True,
        ) is False
        assert is_similar_cue(
            "像", LANG_ZH,
            backend=ctx.backend, edge_store=ctx.edge_store,
            space_id=ctx.space_id, concept_index=ctx.concept_index,
            language_signal_compatibility_enabled=False,
        ) is False
    finally:
        gates.EMERGENT_RELATION_CUE_READBACK_MODE = saved_readback
        gates.CUE_EXTRACTOR_MODE = saved_extract


def test_graph_negation_drives_production_property_and_clone(tmp_path):
    """图中一致否定候选必须驱动生产分段，评测 clone 保持相同只读结果。"""
    ctx = make_train_context(DictBackend())
    runtime = build_language_signal_runtime(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        seed_path=_write_payload(
            tmp_path, _property_signal_payload(), "negation-property.json"),
    )
    ctx.language_signal_runtime = runtime
    ctx.language_property_attr_instruction_key = (81,)
    ctx.language_property_value_instruction_key = (82,)
    ctx.language_property_possess_instruction_key = (83,)
    ctx.language_negation_instruction_key = (7,)
    ctx.language_signal_compatibility_enabled = False
    runtime.install()
    item = CollectedItem(tokens=["猫", "图的", "颜色", "乙", "图是", "黑"])
    saved_prop = gates.PROPOSITION_MODE
    saved_neg = gates.NEGATION_MODE
    gates.PROPOSITION_MODE = True
    gates.NEGATION_MODE = True
    try:
        segments = _split_item_to_segments(
            item,
            backend=ctx.backend,
            edge_store=ctx.edge_store,
            space_id=ctx.space_id,
                concept_index=ctx.concept_index,
                language_signal_runtime=ctx.language_signal_runtime,
                property_attr_instruction_key=(
                    ctx.language_property_attr_instruction_key),
                property_value_instruction_key=(
                    ctx.language_property_value_instruction_key),
                property_possess_instruction_key=(
                    ctx.language_property_possess_instruction_key),
                negation_instruction_key=ctx.language_negation_instruction_key,
            language_signal_compatibility_enabled=(
                ctx.language_signal_compatibility_enabled),
        )
        assert segments[0].property_claims == [
            (0, 2, 5, 0, 1, 0, 1, 1)]
        before = ctx.backend.snapshot()
        with isolated_evaluation(ctx, label="u04-negation-consumer") as cloned:
            cloned_segments = _split_item_to_segments(
                item,
                backend=cloned.backend,
                edge_store=cloned.edge_store,
                space_id=cloned.space_id,
                concept_index=cloned.concept_index,
                language_signal_runtime=cloned.language_signal_runtime,
                property_attr_instruction_key=(
                    cloned.language_property_attr_instruction_key),
                property_value_instruction_key=(
                    cloned.language_property_value_instruction_key),
                property_possess_instruction_key=(
                    cloned.language_property_possess_instruction_key),
                negation_instruction_key=(
                    cloned.language_negation_instruction_key),
                language_signal_compatibility_enabled=(
                    cloned.language_signal_compatibility_enabled),
            )
            assert cloned_segments[0].property_claims == [
                (0, 2, 5, 0, 1, 0, 1, 1)]
        assert ctx.backend.snapshot() == before
    finally:
        gates.PROPOSITION_MODE = saved_prop
        gates.NEGATION_MODE = saved_neg


def test_negation_compatibility_closure_and_graph_conflict(tmp_path):
    """关闭兼容后旧字面失效，图中混合候选也不得被旧词表覆盖。"""
    ctx = make_train_context(DictBackend())
    assert is_negation_cue(
        "不", 1, language_signal_compatibility_enabled=False) is False

    payload = _seed_payload()
    payload["entries"] = [
        {"language": 1, "surface": "不", "atom_key": [11],
         "instruction_key": [7]},
        {"language": 1, "surface": "不", "atom_key": [12],
         "instruction_key": [8]},
    ]
    runtime = build_language_signal_runtime(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        seed_path=_write_payload(tmp_path, payload, "conflict.json"),
    )
    runtime.install()
    assert is_negation_cue(
        "不",
        1,
        language_signal_runtime=runtime,
        negation_instruction_key=(7,),
        language_signal_compatibility_enabled=True,
    ) is False


def test_graph_modality_drives_property_and_blocks_legacy_conflict(tmp_path):
    """一致图指令驱动情态值，混合指令不得被旧情态词表覆盖。"""
    ctx = make_train_context(DictBackend())
    runtime = build_language_signal_runtime(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        seed_path=_write_payload(
            tmp_path, _property_signal_payload(), "modality-property.json"),
    )
    runtime.install()
    bindings = (((7,), MODAL_KIND_BOX_NECESSITY),)
    assert modal_op_of(
        "乙",
        1,
        language_signal_runtime=runtime,
        modality_instruction_bindings=bindings,
        language_signal_compatibility_enabled=False,
    ) == MODAL_KIND_BOX_NECESSITY
    assert modal_op_of(
        "必然", 1, language_signal_compatibility_enabled=False) is None

    item = CollectedItem(tokens=["猫", "图的", "颜色", "乙", "图是", "黑"])
    saved_prop = gates.PROPOSITION_MODE
    saved_modal = gates.MODALITY_MODE
    gates.PROPOSITION_MODE = True
    gates.MODALITY_MODE = True
    try:
        segments = _split_item_to_segments(
            item,
            backend=ctx.backend,
            edge_store=ctx.edge_store,
            space_id=ctx.space_id,
                concept_index=ctx.concept_index,
                language_signal_runtime=runtime,
                property_attr_instruction_key=(81,),
                property_value_instruction_key=(82,),
                property_possess_instruction_key=(83,),
                modality_instruction_bindings=bindings,
            language_signal_compatibility_enabled=False,
        )
        assert segments[0].property_claims == [
            (0, 2, 5, 0, 0, MODAL_KIND_BOX_NECESSITY, 1, 1)]
    finally:
        gates.PROPOSITION_MODE = saved_prop
        gates.MODALITY_MODE = saved_modal

    conflict_payload = _seed_payload()
    conflict_payload["entries"] = [
        {"language": 1, "surface": "必然", "atom_key": [21],
         "instruction_key": [7]},
        {"language": 1, "surface": "必然", "atom_key": [22],
         "instruction_key": [8]},
    ]
    conflict_ctx = make_train_context(DictBackend())
    conflict_runtime = build_language_signal_runtime(
        backend=conflict_ctx.backend,
        concept_index=conflict_ctx.concept_index,
        ontology=conflict_ctx.graph_ontology,
        seed_path=_write_payload(
            tmp_path, conflict_payload, "modal-conflict.json"),
    )
    conflict_runtime.install()
    assert modal_op_of(
        "必然",
        1,
        language_signal_runtime=conflict_runtime,
        modality_instruction_bindings=(
            ((7,), MODAL_KIND_BOX_NECESSITY),
            ((8,), MODAL_KIND_BOX_POSSIBILITY),
        ),
        language_signal_compatibility_enabled=True,
    ) is None


def test_graph_integer_bindings_drive_production_extractors(tmp_path):
    """关系、算术和比较图指令必须驱动正式分段，条件槽保持独立绑定。"""
    ctx = make_train_context(DictBackend())
    runtime = build_language_signal_runtime(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        seed_path=_write_payload(
            tmp_path, _integer_signal_payload(), "integer-signals.json"),
    )
    runtime.install()
    cue_bindings = (
        ((41,), CAUSES_CUE_FORWARD),
        ((43,), ARITH_EQUALS_CUE),
    )
    arithmetic_bindings = (((42,), OPCODE_ADD),)
    comparison_bindings = (((44,), CMP_GT),)
    condition_bindings = (
        ((45,), _COND_IF),
        ((46,), _COND_THEN),
        ((47,), _COND_ELSE),
    )
    saved = gates.CUE_EXTRACTOR_MODE
    gates.CUE_EXTRACTOR_MODE = True
    try:
        causal = _split_item_to_segments(
            CollectedItem(tokens=["雨", "图因", "湿"]),
            language_signal_runtime=runtime,
            cue_instruction_bindings=cue_bindings,
            language_signal_compatibility_enabled=False,
        )
        assert causal[0].cue_based_causal_pairs == [(0, 2)]
        numeric = _split_item_to_segments(
            CollectedItem(tokens=["2", "图加", "3", "图等", "5"]),
            language_signal_runtime=runtime,
            cue_instruction_bindings=cue_bindings,
            arithmetic_instruction_bindings=arithmetic_bindings,
            language_signal_compatibility_enabled=False,
        )
        assert numeric[0].numeric_claims == [(2, OPCODE_ADD, 3, 5)]
        comparison = _split_item_to_segments(
            CollectedItem(tokens=["5", "图大", "3"]),
            language_signal_runtime=runtime,
            comparison_instruction_bindings=comparison_bindings,
            language_signal_compatibility_enabled=False,
        )
        assert comparison[0].comparison_claims == [(5, CMP_GT, 3)]
    finally:
        gates.CUE_EXTRACTOR_MODE = saved

    assert cond_keyword_of(
        "图若", 1,
        language_signal_runtime=runtime,
        condition_instruction_bindings=condition_bindings,
        language_signal_compatibility_enabled=False,
    ) == _COND_IF
    assert cond_keyword_of(
        "图则", 1,
        language_signal_runtime=runtime,
        condition_instruction_bindings=condition_bindings,
        language_signal_compatibility_enabled=False,
    ) == _COND_THEN
    assert cond_keyword_of(
        "图否", 1,
        language_signal_runtime=runtime,
        condition_instruction_bindings=condition_bindings,
        language_signal_compatibility_enabled=False,
    ) == _COND_ELSE


def test_integer_binding_compatibility_closure_and_conflict(tmp_path):
    """关闭兼容后旧整数 cue 失效，图冲突不得被旧算术词覆盖。"""
    assert arith_op_of(
        "加", 1, language_signal_compatibility_enabled=False) is None
    assert comparison_op_of(
        "大于", 1, language_signal_compatibility_enabled=False) is None
    assert cue_type_of(
        "因此", 1, language_signal_compatibility_enabled=False) is None
    assert cond_keyword_of(
        "如果", 1, language_signal_compatibility_enabled=False) is None

    payload = _seed_payload()
    payload["entries"] = [
        {"language": 1, "surface": "加", "atom_key": [51],
         "instruction_key": [42]},
        {"language": 1, "surface": "加", "atom_key": [52],
         "instruction_key": [99]},
    ]
    ctx = make_train_context(DictBackend())
    runtime = build_language_signal_runtime(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        seed_path=_write_payload(tmp_path, payload, "arith-conflict.json"),
    )
    runtime.install()
    assert arith_op_of(
        "加",
        1,
        language_signal_runtime=runtime,
        arithmetic_instruction_bindings=(((42,), OPCODE_ADD),),
        language_signal_compatibility_enabled=True,
    ) is None


def test_graph_property_signals_drive_all_production_consumers(tmp_path):
    """属性三作用须驱动命题、存在窗口和 observe 修饰统计。"""
    ctx = make_train_context(DictBackend())
    runtime = build_language_signal_runtime(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        seed_path=_write_payload(
            tmp_path, _property_signal_payload(), "property-signals.json"),
    )
    runtime.install()
    attr_key = (81,)
    value_key = (82,)
    possess_key = (83,)
    assert is_property_attr_marker(
        "图的", 1,
        language_signal_runtime=runtime,
        property_attr_instruction_key=attr_key,
        language_signal_compatibility_enabled=False,
    ) is True
    assert is_property_value_copula(
        "图是", 1,
        language_signal_runtime=runtime,
        property_value_instruction_key=value_key,
        language_signal_compatibility_enabled=False,
    ) is True
    assert is_property_possess_cue(
        "图有", 1,
        language_signal_runtime=runtime,
        property_possess_instruction_key=possess_key,
        language_signal_compatibility_enabled=False,
    ) is True
    assert is_property_attr_marker(
        "的", 1, language_signal_compatibility_enabled=False) is False
    assert is_property_value_copula(
        "是", 1, language_signal_compatibility_enabled=False) is False
    assert is_property_possess_cue(
        "有", 1, language_signal_compatibility_enabled=False) is False

    saved_cue = gates.CUE_EXTRACTOR_MODE
    saved_proposition = gates.PROPOSITION_MODE
    gates.CUE_EXTRACTOR_MODE = True
    gates.PROPOSITION_MODE = True
    try:
        property_segments = _split_item_to_segments(
            CollectedItem(tokens=["猫", "图的", "颜色", "图是", "黑"]),
            language_signal_runtime=runtime,
            property_attr_instruction_key=attr_key,
            property_value_instruction_key=value_key,
            property_possess_instruction_key=possess_key,
            language_signal_compatibility_enabled=False,
        )
        assert property_segments[0].property_claims == [
            (0, 2, 4, 0, 0, 0, 1, 1)]
        possess_segments = _split_item_to_segments(
            CollectedItem(tokens=["猫", "图有", "尾巴"]),
            language_signal_runtime=runtime,
            property_attr_instruction_key=attr_key,
            property_value_instruction_key=value_key,
            property_possess_instruction_key=possess_key,
            language_signal_compatibility_enabled=False,
        )
        assert possess_segments[0].property_claims == [
            (0, -1, 2, 0, 0, 0, 1, 1)]
        existential_segments = _split_item_to_segments(
            CollectedItem(tokens=["图存在", "猫", "图是", "动物"]),
            language_signal_runtime=runtime,
            property_value_instruction_key=value_key,
            cue_instruction_bindings=(((84,), EXISTENTIAL_CUE),),
            language_signal_compatibility_enabled=False,
        )
        assert existential_segments[0].existential_claims == [(1, 3)]
    finally:
        gates.CUE_EXTRACTOR_MODE = saved_cue
        gates.PROPOSITION_MODE = saved_proposition

    pipeline = ObservePipeline(
        _build_space_ctx(ctx),
        concept_index=ctx.concept_index,
        language_signal_runtime=runtime,
        property_attr_instruction_key=attr_key,
        language_signal_compatibility_enabled=False,
    )
    pipeline.observe(InputPayload(
        segments=[Segment(
            seg_id=0,
            modality=MODALITY_LANGUAGE,
            lang=LANG_ZH,
            domain=DOMAIN_TEXT,
            tokens=["红色", "图的", "苹果"],
        )],
        source=SOURCE_BARE_TEXT,
        stage=STAGE_TRAINING,
        modality=MODALITY_LANGUAGE,
        lang=LANG_ZH,
        domain=DOMAIN_TEXT,
    ))
    head_ref = ctx.concept_index.lookup("苹果", ctx.space_id)
    modifier_ref = ctx.concept_index.lookup("红色", ctx.space_id)
    assert head_ref is not None
    assert modifier_ref is not None
    assert head_preference(ctx.backend, head_ref) == (1, 0)
    assert head_preference(ctx.backend, modifier_ref) == (0, 1)


def test_property_graph_conflict_blocks_legacy_literal(tmp_path):
    """属性字面存在混合图候选时，兼容词表不得覆盖冲突。"""
    payload = _seed_payload()
    payload["entries"] = [
        {"language": 1, "surface": "的", "atom_key": [71],
         "instruction_key": [81]},
        {"language": 1, "surface": "的", "atom_key": [72],
         "instruction_key": [99]},
    ]
    ctx = make_train_context(DictBackend())
    runtime = build_language_signal_runtime(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        seed_path=_write_payload(tmp_path, payload, "property-conflict.json"),
    )
    runtime.install()
    assert is_property_attr_marker(
        "的",
        1,
        language_signal_runtime=runtime,
        property_attr_instruction_key=(81,),
        language_signal_compatibility_enabled=True,
    ) is False


def test_graph_evidence_without_consumer_binding_never_falls_back(tmp_path):
    """图已有证据但 consumer 未绑定时，所有旧字面源都必须 fail closed。"""
    payload = _seed_payload()
    payload["entries"] = [
        {"language": 1, "surface": "不", "atom_key": [81],
         "instruction_key": [101]},
        {"language": 1, "surface": "的", "atom_key": [82],
         "instruction_key": [102]},
        {"language": 1, "surface": "是", "atom_key": [83],
         "instruction_key": [103]},
        {"language": 1, "surface": "有", "atom_key": [84],
         "instruction_key": [104]},
        {"language": 1, "surface": "加", "atom_key": [85],
         "instruction_key": [105]},
        {"language": 1, "surface": "必然", "atom_key": [86],
         "instruction_key": [106]},
        {"language": 1, "surface": "大于", "atom_key": [87],
         "instruction_key": [107]},
        {"language": 1, "surface": "如果", "atom_key": [88],
         "instruction_key": [108]},
        {"language": 1, "surface": "所以", "atom_key": [89],
         "instruction_key": [109]},
    ]
    ctx = make_train_context(DictBackend())
    runtime = build_language_signal_runtime(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        seed_path=_write_payload(tmp_path, payload, "unbound-signals.json"),
    )
    runtime.install()
    assert is_negation_cue(
        "不", 1, language_signal_runtime=runtime,
        language_signal_compatibility_enabled=True) is False
    assert is_property_attr_marker(
        "的", 1, language_signal_runtime=runtime,
        language_signal_compatibility_enabled=True) is False
    assert is_property_value_copula(
        "是", 1, language_signal_runtime=runtime,
        language_signal_compatibility_enabled=True) is False
    assert is_property_possess_cue(
        "有", 1, language_signal_runtime=runtime,
        language_signal_compatibility_enabled=True) is False
    assert arith_op_of(
        "加", 1, language_signal_runtime=runtime,
        language_signal_compatibility_enabled=True) is None
    assert modal_op_of(
        "必然", 1, language_signal_runtime=runtime,
        language_signal_compatibility_enabled=True) is None
    assert comparison_op_of(
        "大于", 1, language_signal_runtime=runtime,
        language_signal_compatibility_enabled=True) is None
    assert cond_keyword_of(
        "如果", 1, language_signal_runtime=runtime,
        language_signal_compatibility_enabled=True) is None
    assert cue_type_of(
        "所以", 1, language_signal_runtime=runtime,
        language_signal_compatibility_enabled=True) is None


def test_graph_action_intent_preserves_kind_ref_and_all_consumers(tmp_path):
    """动作图信号须贯通分类、对象收集、经验、seed 和 dag_path 覆写。"""
    ctx = make_train_context(DictBackend())
    bootstrap_action_signals(
        ctx.concept_index,
        ctx.edge_store,
        ctx.backend,
        space_id=ctx.space_id,
        langs={LANG_ZH},
    )
    action_refs = ensure_action_primitives(
        ctx.concept_index, ctx.backend, space_id=ctx.space_id)
    primitive_refs = tuple(sorted(action_refs.items()))
    runtime = build_language_signal_runtime(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        seed_path=_write_payload(
            tmp_path, _action_signal_payload(), "action-signals.json"),
    )
    runtime.install()
    bindings = (
        ((111,), INTENT_COMMAND_MOOD),
        ((112,), ACTION_GENERATE),
    )
    for token in ("图请", "图生成", "生成"):
        ctx.concept_index.ensure(token, space_id=ctx.space_id)
    segments = [Segment(
        seg_id=0,
        modality=MODALITY_LANGUAGE,
        lang=LANG_ZH,
        domain=DOMAIN_TEXT,
        tokens=["图请", "图生成", "生成"],
    )]
    assert action_intent_of(
        "图生成", LANG_ZH,
        language_signal_runtime=runtime,
        action_instruction_bindings=bindings,
        language_signal_compatibility_enabled=False,
    ) == ACTION_GENERATE
    assert is_action_intent_cue(
        "图请", LANG_ZH,
        language_signal_runtime=runtime,
        action_instruction_bindings=bindings,
        language_signal_compatibility_enabled=False,
    ) is True
    assert is_action_intent_cue(
        "生成", LANG_ZH,
        backend=ctx.backend,
        edge_store=ctx.edge_store,
        space_id=ctx.space_id,
        concept_index=ctx.concept_index,
        language_signal_runtime=runtime,
        action_instruction_bindings=bindings,
        language_signal_compatibility_enabled=True,
    ) is False

    saved_command = gates.INTENT_COMMAND_MODE
    gates.INTENT_COMMAND_MODE = True
    try:
        intent = classify_intent(
            None,
            segments,
            backend=ctx.backend,
            edge_store=ctx.edge_store,
            space_id=ctx.space_id,
            concept_index=ctx.concept_index,
            language_signal_runtime=runtime,
            action_instruction_bindings=bindings,
            language_signal_compatibility_enabled=False,
        )
    finally:
        gates.INTENT_COMMAND_MODE = saved_command
    assert intent.type == INTENT_COMMAND

    targets = collect_action_intent_concepts(
        segments,
        backend=ctx.backend,
        edge_store=ctx.edge_store,
        space_id=ctx.space_id,
        concept_index=ctx.concept_index,
        language_signal_runtime=runtime,
        action_instruction_bindings=bindings,
        action_primitive_refs=primitive_refs,
        language_signal_compatibility_enabled=True,
    )
    assert targets == [
        (action_refs[INTENT_COMMAND_MOOD], INTENT_COMMAND_MOOD),
        (action_refs[ACTION_GENERATE], ACTION_GENERATE),
    ]

    decisions = collect_action_intent_word_decisions(
        segments,
        space_id=ctx.space_id,
        concept_index=ctx.concept_index,
        language_signal_runtime=runtime,
        action_instruction_bindings=bindings,
        language_signal_compatibility_enabled=True,
    )
    graph_word_ref = ctx.concept_index.lookup("图生成", ctx.space_id)
    conflict_word_ref = ctx.concept_index.lookup("生成", ctx.space_id)
    assert graph_word_ref is not None
    assert conflict_word_ref is not None
    assert decisions[graph_word_ref] is True
    assert decisions[conflict_word_ref] is False
    ctx.work_memory.action_intent_word_decisions = decisions
    saved_override = gates.ACTION_INTENT_OVERRIDE_MODE
    gates.ACTION_INTENT_OVERRIDE_MODE = True
    try:
        assert _intent_override(
            graph_word_ref,
            IntentType(type=INTENT_COMMAND),
            ctx.work_memory,
            backend=ctx.backend,
            edge_store=ctx.edge_store,
        ) == 1
        assert _intent_override(
            conflict_word_ref,
            IntentType(type=INTENT_COMMAND),
            ctx.work_memory,
            backend=ctx.backend,
            edge_store=ctx.edge_store,
        ) == 0
    finally:
        gates.ACTION_INTENT_OVERRIDE_MODE = saved_override

    command_ctx = pack_ctx_code(
        DOMAIN_TEXT, MODALITY_LANGUAGE, 0, INTENT_COMMAND)
    seeds = _collect_action_seed_candidates(
        segments=segments,
        backend=ctx.backend,
        edge_store=ctx.edge_store,
        space_id=ctx.space_id,
        concept_index=ctx.concept_index,
        intent_type=INTENT_COMMAND,
        ctx_code=command_ctx,
        language_signal_runtime=runtime,
        action_instruction_bindings=bindings,
        action_primitive_refs=primitive_refs,
        language_signal_compatibility_enabled=True,
    )
    assert graph_word_ref in seeds
    assert conflict_word_ref not in seeds

    saved_feed = gates.ACTION_EXPERIENCE_FEED_MODE
    gates.ACTION_EXPERIENCE_FEED_MODE = True
    try:
        _feed_action_experience(
            backend=ctx.backend,
            edge_store=ctx.edge_store,
            space_id=ctx.space_id,
            concept_index=ctx.concept_index,
            language_signal_runtime=runtime,
            action_instruction_bindings=bindings,
            action_primitive_refs=primitive_refs,
            language_signal_compatibility_enabled=True,
            segments=segments,
            domain=DOMAIN_TEXT,
            modality=MODALITY_LANGUAGE,
            intent_type=INTENT_COMMAND,
            reward=1,
            terminal=TERMINAL_REACHED_SINK,
        )
    finally:
        gates.ACTION_EXPERIENCE_FEED_MODE = saved_feed
    assert read_experience_count(
        ctx.backend,
        action_refs[ACTION_GENERATE],
        ctx_code=command_ctx,
    ) == (0, 1, 1)


def test_lookup_backend_operations_grow_no_faster_than_catalog_size(tmp_path):
    """目录规模翻倍时，全目录 lookup 的后端调用和读取行数不得超线性。"""
    small_calls, small_rows = _lookup_operations(tmp_path, 16)
    large_calls, large_rows = _lookup_operations(tmp_path, 32)
    assert large_calls <= small_calls * 2
    assert large_rows <= small_rows * 2


def test_graph_only_formal_boot_skips_static_d11_and_keeps_action_primitives(
        tmp_path):
    """关闭兼容后正式 boot 不写旧 D:11，但图动作绑定仍有一等原语。"""
    seed_path = _write_payload(
        tmp_path, _action_signal_payload(), "graph-only-formal.json")
    backend = DictBackend()
    formal_train(
        FormalTrainConfig(
            run_dir=str(tmp_path / "runs"),
            run_id="u04-graph-only-boot",
            rounds_per_stage=1,
            active_training_stages=(),
            persist_graph_dump=False,
            language_signal_seed_path=str(seed_path),
            language_action_instruction_bindings=(
                ((111,), INTENT_COMMAND_MOOD),
                ((112,), ACTION_GENERATE),
            ),
            language_signal_compatibility_enabled=False,
        ),
        [CollectedItem(tokens=["不", "必然", "加", "导致", "生成"])],
        backend=backend,
    )
    assert backend.select(
        "edge", where={"edge_type": EDGE_RELATION_SIGNAL}) == []
    restored = make_train_context(backend)
    assert restored.concept_index.lookup(
        "__ACTION_GENERATE__", restored.space_id) is not None


def test_emergent_exclusion_prefers_catalog_and_never_scans_learned_d11(
        tmp_path):
    """图目录应排除已知 signal，关闭兼容后旧表和学习 D:11 均不得混入。"""
    ctx = make_train_context(DictBackend())
    runtime = build_language_signal_runtime(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        seed_path=_write_payload(
            tmp_path, _integer_signal_payload(), "emergent-signals.json"),
    )
    runtime.install()
    ctx.language_signal_runtime = runtime
    graph_ref = ctx.concept_index.ensure("图因", space_id=ctx.space_id)
    legacy_ref = ctx.concept_index.ensure("导致", space_id=ctx.space_id)
    learned_ref = ctx.concept_index.ensure("自学连接", space_id=ctx.space_id)
    relation_refs = bootstrap_word_concept_signals(
        ctx.concept_index, ctx.edge_store, ctx.backend,
        space_id=ctx.space_id, langs=set())
    assert relation_refs == 0
    from pure_integer_ai.cognition.shared.relation_primitives import (
        REL_CAUSES,
        ensure_relation_primitives,
    )
    from pure_integer_ai.cognition.understanding.word_concept_signal import (
        record_word_concept,
    )
    targets = ensure_relation_primitives(
        ctx.concept_index, ctx.backend, space_id=ctx.space_id)
    record_word_concept(
        ctx.concept_index, ctx.edge_store, "自学连接", targets[REL_CAUSES],
        space_id=ctx.space_id)

    ctx.language_signal_compatibility_enabled = False
    excluded = _resolve_emergent_excluded_refs(ctx, LANG_ZH)
    assert graph_ref in excluded
    assert legacy_ref not in excluded
    assert learned_ref not in excluded

    ctx.language_signal_compatibility_enabled = True
    compatible = _resolve_emergent_excluded_refs(ctx, LANG_ZH)
    assert graph_ref in compatible
    assert legacy_ref in compatible
    assert learned_ref not in compatible
