"""DLG-RAW-06 source-bound slot catalog 的 DLG-RAW-07 closure 专项。"""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from shutil import copy2

import pytest

from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PublicFrameCatalog,
    load_public_frame_catalog_from_closure,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeError,
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_reference_catalog import (
    load_public_reference_frame_catalog_from_closure,
)
from pure_integer_ai.experiments.conversation_public_response_act_catalog import (
    load_public_response_act_frame_catalog_from_closure,
    merge_public_frame_catalogs,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1,
    public_source_payload_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    DLG_RAW_REJECT_CONSTRUCTION_MISS,
    DLG_RAW_REJECT_LEXICAL_MISS,
    DLG_RAW_REJECT_SOURCE_CONFLICT,
    encode_utf8_v1,
)
from pure_integer_ai.experiments.conversation_source_bound_slot_catalog import (
    SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V1,
    SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V2,
    SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1,
    SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V2,
    SourceBoundSlotCompositionResolution,
    SourceBoundSlotCompositionError,
    _find_all_u8_subsequence_v1,
    _u64_count,
    load_source_bound_slot_composition_catalog,
    load_source_bound_slot_composition_catalog_from_closure,
    portable_integer_record_bytes,
    portable_sha256_v1,
    resolve_source_bound_slot_composition,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    canonical_json_line,
    parse_canonical_json_bytes,
)


_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_CATALOG_SOURCE = (
    _ROOT / "src/pure_integer_ai/experiments/"
    "conversation_source_bound_slot_catalog.py"
)
_BASE_MANIFEST = b"data/ph2/dlg_raw_public_frame_v1.jsonl.sample"
_RESPONSE_ACT_MANIFEST = b"data/ph2/dlg_raw_public_response_act_frame_v2.jsonl.sample"
_DERIVED_MANIFEST = b"data/ph2/dlg_raw_public_derived_frame_v3.jsonl.sample"
_CONTEXTUAL_MANIFEST = (
    b"data/ph2/dlg_raw_public_contextual_ellipsis_frame_v4.jsonl.sample")


def _closure_at(resource_root: Path):
    """仅在测试 host adapter 处读取物理资源，core 均只接收 closure。"""
    return load_public_source_payload_closure_from_root(resource_root)


def _copy_public_dialogue_resources(target: Path) -> Path:
    """复制完整 27 项 public payload registry，构造独立物理 closure 根。"""
    for logical_key in PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1:
        relative_path = logical_key.decode("ascii")
        destination = target / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(_ROOT / relative_path, destination)
    return target


def _manifest_path(resource_root: Path) -> Path:
    """只供测试改写独立 transport，production core 不接收该物理路径。"""
    return resource_root / SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V1.decode("ascii")


@pytest.fixture(scope="module")
def closure():
    """建立当前冻结 27 项资源的 host closure。"""
    return _closure_at(_ROOT)


@pytest.fixture(scope="module")
def base_catalog(closure) -> PublicFrameCatalog:
    """由同一 closure 加载 V1 base catalog，不触及 held-out/private。"""
    return load_public_frame_catalog_from_closure(closure)


def _load_slot(closure, base_catalog: PublicFrameCatalog):
    """以 V1 同时作为最小 active catalog 加载 source-bound 组件。"""
    return load_source_bound_slot_composition_catalog_from_closure(
        closure,
        base_catalog,
        base_catalog,
    )


def _load_slot_v2(
        closure,
        base_catalog: PublicFrameCatalog,
        active_catalog: PublicFrameCatalog,
        ):
    """V2 必须绑定完整 active catalog，不能退回 V1 base-only 语义。"""
    return load_source_bound_slot_composition_catalog_from_closure(
        closure,
        base_catalog,
        active_catalog,
        catalog_logical_key=SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V2,
    )


def _assembled_scalars(catalog) -> tuple[int, ...]:
    """只从独立 entity/suffix scalar 片段组装测试输入，不写完整问句。"""
    family = catalog.families[0]
    binding = catalog.bindings[0]
    return (*family.prefix_scalars, *binding.entity_scalars,
            *family.suffix_scalars)


def _v2_surface(
        catalog,
        binding_key: str,
        expected_text: str,
        ) -> tuple[int, ...]:
    """从 V2 的真实 binding/family 交叉组装给定 surface，不借完整问句旁路。"""
    binding = next(item for item in catalog.bindings
                   if item.binding_key == binding_key)
    expected = tuple(ord(character) for character in expected_text)
    surfaces = tuple(
        (*family.prefix_scalars, *binding.entity_scalars,
         *family.suffix_scalars)
        for family in catalog.families
        if (*family.prefix_scalars, *binding.entity_scalars,
            *family.suffix_scalars) == expected
    )
    assert surfaces == (expected,)
    return expected


def _merged_active_catalog(closure, base_catalog: PublicFrameCatalog) -> PublicFrameCatalog:
    """按 terminal 的实际资源集合构造合并 active exact catalog。"""
    response_act = load_public_response_act_frame_catalog_from_closure(
        closure, _RESPONSE_ACT_MANIFEST)
    derived = load_public_response_act_frame_catalog_from_closure(
        closure, _DERIVED_MANIFEST)
    contextual = load_public_response_act_frame_catalog_from_closure(
        closure, _CONTEXTUAL_MANIFEST)
    reference = load_public_reference_frame_catalog_from_closure(closure)
    return merge_public_frame_catalogs(
        base_catalog, response_act, derived, contextual, reference)


def _read_manifest(path: Path) -> dict[str, object]:
    """只接受一行 canonical JSONL，避免测试将宽松 JSON 当作合法 manifest。"""
    payload = path.read_bytes()
    assert payload.endswith(b"\n")
    return parse_canonical_json_bytes(payload[:-1], require_object=True)


def _write_manifest(path: Path, record: dict[str, object]) -> None:
    """使用生产同一 canonical JSON 编码回写独立 transport。"""
    path.write_bytes(canonical_json_line(record))


def _source_record(manifest: dict[str, object], record_id: str) -> dict[str, object]:
    """按公开 record id 定位临时 source metadata，不依赖 list 插入顺序。"""
    records = manifest["source_records"]
    assert isinstance(records, list)
    return next(item for item in records
                if isinstance(item, dict) and item["record_id"] == record_id)


def test_unique_composition_builds_ingress_ready_dynamic_catalog(
        closure,
        base_catalog: PublicFrameCatalog) -> None:
    """未登记完整 surface 必须由独立 closure segment 组成完整动态 frame。"""
    catalog = _load_slot(closure, base_catalog)
    assert catalog.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1
    assert catalog.manifest_logical_key == (
        SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V1.decode("ascii"))
    input_scalars = _assembled_scalars(catalog)
    result = resolve_source_bound_slot_composition(
        catalog, base_catalog, base_catalog, input_scalars, closure)

    assert result.result_code == DLG_RAW_ACCEPT
    assert result.accepted is True
    assert result.matched_frame_count == 1
    assert result.frame is not None
    assert result.public_frame_catalog is not None
    assert result.frame.surface_scalars == input_scalars
    assert result.public_frame_catalog.matching_frames(input_scalars) == (result.frame,)
    assert result.frame.question.target == base_catalog.frames[0].question.target
    assert result.frame.recipe == base_catalog.frames[0].recipe
    assert result.frame.question.trace_prefix != base_catalog.frames[0].question.trace_prefix
    assert len({route.atom.stable_key() for route in result.frame.routes}) == len(result.frame.routes)
    assert all(type(value) is int for value in result.canonical_record())

    complete_bytes = bytes(encode_utf8_v1(input_scalars))
    assert complete_bytes not in closure.payload_for(SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V1)
    assert all(
        complete_bytes not in closure.payload_for(logical_key)
        for logical_key in (
            b"data/ph2/dlg_raw_public_slot_entity_v1_a.txt.sample",
            b"data/ph2/dlg_raw_public_slot_entity_v1_b.txt.sample",
            b"data/ph2/dlg_raw_public_slot_family_v1_a.txt.sample",
            b"data/ph2/dlg_raw_public_slot_family_v1_b.txt.sample",
        ))


def test_unknown_and_static_surface_fail_closed_without_dynamic_frame(
        closure,
        base_catalog: PublicFrameCatalog) -> None:
    """未知 slot 与 active static hit 均不能被误报为动态组合。"""
    catalog = _load_slot(closure, base_catalog)
    input_scalars = _assembled_scalars(catalog)
    unknown = (*input_scalars[:-len(catalog.families[0].suffix_scalars)],
               input_scalars[-len(catalog.families[0].suffix_scalars)] + 1,
               *input_scalars[-len(catalog.families[0].suffix_scalars) + 1:])

    miss = resolve_source_bound_slot_composition(
        catalog, base_catalog, base_catalog, unknown, closure)
    static_hit = resolve_source_bound_slot_composition(
        catalog, base_catalog, base_catalog,
        base_catalog.frames[0].surface_scalars, closure)

    assert miss.result_code == DLG_RAW_REJECT_LEXICAL_MISS
    assert miss.frame is miss.public_frame_catalog is None
    assert static_hit.result_code == DLG_RAW_REJECT_LEXICAL_MISS
    assert static_hit.frame is static_hit.public_frame_catalog is None


def test_rebuild_is_canonical_and_family_witnesses_are_distinct_entities(
        closure,
        base_catalog: PublicFrameCatalog) -> None:
    """同一 closure 两次加载/解析必须逐整数相同，并保留两个实体观察。"""
    first = _load_slot(closure, base_catalog)
    second = _load_slot(closure, base_catalog)
    first_input = _assembled_scalars(first)
    second_input = _assembled_scalars(second)
    first_result = resolve_source_bound_slot_composition(
        first, base_catalog, base_catalog, first_input, closure)
    second_result = resolve_source_bound_slot_composition(
        second, base_catalog, base_catalog, second_input, closure)

    assert first.canonical_record() == second.canonical_record()
    assert first_result.canonical_record() == second_result.canonical_record()
    assert len({entity for entity, _ in first.families[0].witnesses}) == 2
    assert len({source.source.stable_key()
                for _, source in first.families[0].witnesses}) == 2
    assert len({source.source.stable_key()
                for source in first.bindings[0].witnesses}) == 2


def test_equal_payloads_in_two_physical_roots_preserve_catalog_and_resolution(
        tmp_path: Path,
        closure,
        base_catalog: PublicFrameCatalog) -> None:
    """路径、mtime 和复制根不能影响 source-bound identity 或动态 frame。"""
    first = _load_slot(closure, base_catalog)
    copied_root = _copy_public_dialogue_resources(tmp_path / "copied-public")
    copied_closure = _closure_at(copied_root)
    copied_base = load_public_frame_catalog_from_closure(copied_closure)
    second = _load_slot(copied_closure, copied_base)

    first_result = resolve_source_bound_slot_composition(
        first, base_catalog, base_catalog, _assembled_scalars(first), closure)
    second_result = resolve_source_bound_slot_composition(
        second, copied_base, copied_base, _assembled_scalars(second), copied_closure)

    assert first.source_payload_closure_identity == second.source_payload_closure_identity
    assert first.canonical_record() == second.canonical_record()
    assert first_result.canonical_record() == second_result.canonical_record()


def test_base_v1_and_merged_active_catalog_are_distinct_valid_inputs(
        closure,
        base_catalog: PublicFrameCatalog) -> None:
    """组合锁 V1 base recipe，但可对 terminal 使用的 merged exact catalog 做 collision。"""
    active_catalog = _merged_active_catalog(closure, base_catalog)
    assert active_catalog.source_sha256 != base_catalog.source_sha256
    catalog = load_source_bound_slot_composition_catalog_from_closure(
        closure,
        base_catalog,
        active_catalog,
    )
    result = resolve_source_bound_slot_composition(
        catalog, base_catalog, active_catalog, _assembled_scalars(catalog), closure)

    assert result.result_code == DLG_RAW_ACCEPT
    assert result.frame is not None
    assert result.frame.question.target == base_catalog.frames[0].question.target


def test_v2_requires_merged_active_catalog_and_binds_existing_clarify_frame(
        closure,
        base_catalog: PublicFrameCatalog) -> None:
    """V2 alias 必须锁定 active 中现有 CLARIFY frame，而不是复制新回答模板。"""
    active_catalog = _merged_active_catalog(closure, base_catalog)
    with pytest.raises(SourceBoundSlotCompositionError, match="catalog SHA"):
        _load_slot_v2(closure, base_catalog, base_catalog)

    catalog = _load_slot_v2(closure, base_catalog, active_catalog)
    assert catalog.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V2
    assert catalog.manifest_logical_key == (
        SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V2.decode("ascii"))
    surface = _v2_surface(
        catalog,
        "star-bridge-project-site-v2",
        "星桥项目的试验地点在哪里？",
    )
    assert not active_catalog.matching_frames(surface)
    result = resolve_source_bound_slot_composition(
        catalog, base_catalog, active_catalog, surface, closure)
    clarify = next(item for item in active_catalog.frames
                   if item.frame_key == "dlg-raw-public-v2-clarify-site")

    assert result.result_code == DLG_RAW_ACCEPT
    assert result.accepted
    assert result.matched_frame_count == 1
    assert result.frame is not None
    assert result.public_frame_catalog is not None
    assert result.frame.frame_key != clarify.frame_key
    assert result.frame.question.target == clarify.question.target
    assert result.frame.recipe == clarify.recipe
    assert result.frame.raw_line_sha256 != clarify.raw_line_sha256
    assert result.public_frame_catalog.frames == (result.frame,)


def test_v2_explicit_counterevidence_returns_source_conflict_without_frame(
        closure,
        base_catalog: PublicFrameCatalog) -> None:
    """同一 alias 的正反 relation 同存必须停在 code 13，绝不生成动态 frame。"""
    active_catalog = _merged_active_catalog(closure, base_catalog)
    catalog = _load_slot_v2(closure, base_catalog, active_catalog)
    surface = _v2_surface(
        catalog,
        "north-east-side-passage-conflict-v2",
        "北川站东侧通道何时启用？",
    )
    binding = next(item for item in catalog.bindings
                   if item.binding_key == "north-east-side-passage-conflict-v2")
    assert len(binding.witnesses) == 2
    assert len(binding.negative_witnesses) == 1
    assert not active_catalog.matching_frames(surface)

    result = resolve_source_bound_slot_composition(
        catalog, base_catalog, active_catalog, surface, closure)

    assert result.result_code == DLG_RAW_REJECT_SOURCE_CONFLICT
    assert result.accepted is False
    assert result.matched_frame_count == 1
    assert result.frame is None
    assert result.public_frame_catalog is None


def test_portable_sha_framing_has_fixed_cross_language_golden_vector() -> None:
    """port 可用这个 domain、record 和 exact bytes/SHA 验证 framing。"""
    record = (0, 1, 255, 256)
    assert portable_integer_record_bytes(record, label="golden") == bytes.fromhex(
        "0000000000000004000000000000000100000000000000000101"
        "0000000000000001ff00000000000000020100")
    assert bytes(portable_sha256_v1(
        b"PURE-INTEGER-AI/DLG-RAW-06/GOLDEN/V1",
        ((), record, (65001, 60, 1)),
    )).hex() == "7e5ac8eebc8b6fd961bc940f628fb15e83e4ae45210ed24c4c26539ebb08f379"


def test_portable_framing_rejects_u64_overflow_before_host_encoding() -> None:
    """u64 count 边界必须有明确协议拒绝，不泄露 Python OverflowError。"""
    assert _u64_count((1 << 64) - 1, label="maximum") == (1 << 64) - 1
    with pytest.raises(SourceBoundSlotCompositionError, match="u64 count"):
        _u64_count(1 << 64, label="overflow")


def test_relation_scan_is_explicitly_overlap_aware() -> None:
    """关系 scan 的重叠规则由整数循环定义，不能继承 bytes.count 语义。"""
    assert _find_all_u8_subsequence_v1(b"aaaa", (0x61, 0x61)) == (0, 1, 2)


def test_loader_rejects_closure_source_sha_drift_and_resolver_returns_code_9(
        tmp_path: Path,
        base_catalog: PublicFrameCatalog) -> None:
    """内容漂移必须经重建 closure 显式进入，loader/resolve 均 fail closed。"""
    root = _copy_public_dialogue_resources(tmp_path / "before-load")
    source = root / "data/ph2/dlg_raw_public_slot_entity_v1_a.txt.sample"
    payload = source.read_bytes()
    source.write_bytes(b"X" + payload[1:])
    drifted_closure = _closure_at(root)
    with pytest.raises(SourceBoundSlotCompositionError, match="raw SHA-256 漂移"):
        load_source_bound_slot_composition_catalog_from_closure(
            drifted_closure, base_catalog, base_catalog)

    root = _copy_public_dialogue_resources(tmp_path / "after-load")
    baseline_closure = _closure_at(root)
    copied_base = load_public_frame_catalog_from_closure(baseline_closure)
    catalog = _load_slot(baseline_closure, copied_base)
    source = root / "data/ph2/dlg_raw_public_slot_entity_v1_b.txt.sample"
    payload = source.read_bytes()
    source.write_bytes(b"X" + payload[1:])
    drifted_closure = _closure_at(root)
    result = resolve_source_bound_slot_composition(
        catalog, copied_base, copied_base, _assembled_scalars(catalog), drifted_closure)

    assert result.result_code == DLG_RAW_REJECT_CONSTRUCTION_MISS
    assert result.frame is not None
    assert result.public_frame_catalog is not None
    assert result.public_frame_catalog.frames == (result.frame,)


def test_resolver_rejects_changed_unrelated_closure_payload_before_source_use(
        tmp_path: Path,
        closure,
        base_catalog: PublicFrameCatalog) -> None:
    """catalog binding 覆盖完整 27 项 closure，非 slot payload 漂移也不许混用。"""
    catalog = _load_slot(closure, base_catalog)
    root = _copy_public_dialogue_resources(tmp_path / "unrelated-drift")
    unrelated = root / "data/ph2/dlg_raw_public_response_act_lexical_v2_a.txt.sample"
    payload = unrelated.read_bytes()
    unrelated.write_bytes(b"X" + payload[1:])
    drifted_closure = _closure_at(root)

    result = resolve_source_bound_slot_composition(
        catalog, base_catalog, base_catalog, _assembled_scalars(catalog), drifted_closure)

    assert result.result_code == DLG_RAW_REJECT_CONSTRUCTION_MISS
    assert result.frame is not None
    assert result.public_frame_catalog is not None


def test_loader_rejects_static_and_cross_pair_surface_collision(
        tmp_path: Path,
        base_catalog: PublicFrameCatalog) -> None:
    """任意 static 或 family x binding 相同 surface 均必须在 closure load 时拒绝。"""
    root = _copy_public_dialogue_resources(tmp_path / "static")
    manifest_path = _manifest_path(root)
    manifest = _read_manifest(manifest_path)
    family = manifest["families"][0]
    assert isinstance(family, dict)
    suffix = family["suffix"]
    assert isinstance(suffix, dict)
    suffix_scalars = tuple(suffix["scalars"])
    static_entity = base_catalog.frames[0].surface_scalars[:-len(suffix_scalars)]
    static_entity_bytes = bytes(encode_utf8_v1(static_entity))
    binding = manifest["bindings"][0]
    assert isinstance(binding, dict)
    binding["entity"] = {
        "scalars": list(static_entity),
        "utf8_hex": static_entity_bytes.hex(),
    }
    for record_id in binding["entity_witness_record_ids"]:
        record = _source_record(manifest, record_id)
        relative_path = record["relative_path"]
        assert isinstance(relative_path, str)
        source_path = root / relative_path
        relation_bytes = static_entity_bytes + b"=" + static_entity_bytes
        source_path.write_bytes(relation_bytes + b"\n")
        record["raw_sha256"] = public_source_payload_sha256_v1(
            source_path.read_bytes()).hex()
        record["span"] = [len(static_entity_bytes) + 1, len(relation_bytes)]
        record["span_utf8_hex"] = static_entity_bytes.hex()
    _write_manifest(manifest_path, manifest)
    with pytest.raises(SourceBoundSlotCompositionError, match="static frame 冲突"):
        load_source_bound_slot_composition_catalog_from_closure(
            _closure_at(root), base_catalog, base_catalog)

    root = _copy_public_dialogue_resources(tmp_path / "cross-pair")
    manifest_path = _manifest_path(root)
    manifest = _read_manifest(manifest_path)
    binding = manifest["bindings"][0]
    assert isinstance(binding, dict)
    duplicate = deepcopy(binding)
    duplicate["binding_key"] = "north-east-side-entrance-time-v2"
    bindings = manifest["bindings"]
    assert isinstance(bindings, list)
    bindings.append(duplicate)
    _write_manifest(manifest_path, manifest)
    with pytest.raises(SourceBoundSlotCompositionError, match="重复 surface"):
        load_source_bound_slot_composition_catalog_from_closure(
            _closure_at(root), base_catalog, base_catalog)


def test_loader_rejects_base_catalog_and_nonempty_prefix_drift(
        tmp_path: Path,
        closure,
        base_catalog: PublicFrameCatalog) -> None:
    """binding 锁 V1 base catalog，V1 family 也不得偷换未证实 prefix。"""
    drifted_base = PublicFrameCatalog((0,) * 32, base_catalog.frames)
    with pytest.raises(SourceBoundSlotCompositionError, match="base catalog SHA 漂移"):
        load_source_bound_slot_composition_catalog_from_closure(
            closure, drifted_base, drifted_base)

    root = _copy_public_dialogue_resources(tmp_path / "prefix")
    manifest_path = _manifest_path(root)
    manifest = _read_manifest(manifest_path)
    family = manifest["families"][0]
    assert isinstance(family, dict)
    family["prefix"] = {"scalars": [65], "utf8_hex": "41"}
    _write_manifest(manifest_path, manifest)
    with pytest.raises(SourceBoundSlotCompositionError, match="只支持空 prefix"):
        load_source_bound_slot_composition_catalog_from_closure(
            _closure_at(root), base_catalog, base_catalog)


def test_loader_rejects_alias_relation_drift_even_with_updated_source_sha(
        tmp_path: Path,
        base_catalog: PublicFrameCatalog) -> None:
    """更新 manifest SHA 也不能把无 ``=`` 的 alias 误认作 base entity。"""
    root = _copy_public_dialogue_resources(tmp_path / "relation")
    manifest_path = _manifest_path(root)
    manifest = _read_manifest(manifest_path)
    source = root / "data/ph2/dlg_raw_public_slot_entity_v1_a.txt.sample"
    payload = source.read_bytes()
    assert payload.count(b"=") == 1
    source.write_bytes(payload.replace(b"=", b"-", 1))
    record = _source_record(manifest, "entity-a-north-east-side-entrance")
    record["raw_sha256"] = public_source_payload_sha256_v1(
        source.read_bytes()).hex()
    _write_manifest(manifest_path, manifest)

    with pytest.raises(SourceBoundSlotCompositionError, match="等价观察"):
        load_source_bound_slot_composition_catalog_from_closure(
            _closure_at(root), base_catalog, base_catalog)


def test_loader_rejects_duplicate_alias_relation_with_updated_source_sha(
        tmp_path: Path,
        base_catalog: PublicFrameCatalog) -> None:
    """两个同向 ``base=alias`` 观察仍然歧义，不能由宿主计数规则选一条。"""
    root = _copy_public_dialogue_resources(tmp_path / "duplicate-relation")
    manifest_path = _manifest_path(root)
    manifest = _read_manifest(manifest_path)
    source = root / "data/ph2/dlg_raw_public_slot_entity_v1_a.txt.sample"
    payload = source.read_bytes()
    source.write_bytes(payload + payload)
    record = _source_record(manifest, "entity-a-north-east-side-entrance")
    record["raw_sha256"] = public_source_payload_sha256_v1(
        source.read_bytes()).hex()
    _write_manifest(manifest_path, manifest)

    with pytest.raises(SourceBoundSlotCompositionError, match="等价观察"):
        load_source_bound_slot_composition_catalog_from_closure(
            _closure_at(root), base_catalog, base_catalog)


def test_legacy_loader_name_explicitly_rejects_path_style_input(
        closure,
        base_catalog: PublicFrameCatalog) -> None:
    """旧名称只能承接 closure，不能让路径式 production caller 静默存活。"""
    assert load_source_bound_slot_composition_catalog(
        closure, base_catalog, base_catalog).canonical_record() == (
        _load_slot(closure, base_catalog).canonical_record())
    with pytest.raises(SourceBoundSlotCompositionError, match="路径式 source-bound slot catalog 加载已废止"):
        load_source_bound_slot_composition_catalog(
            object(), base_catalog, base_catalog)  # type: ignore[arg-type]


def test_portable_structs_reject_bool_float_and_mutable_scalar_state(
        closure,
        base_catalog: PublicFrameCatalog) -> None:
    """record 边界不得让 Python bool/float 或可变 list 成为可运输状态。"""
    active_catalog = _merged_active_catalog(closure, base_catalog)
    catalog = _load_slot_v2(closure, base_catalog, active_catalog)
    surface = _v2_surface(
        catalog,
        "star-bridge-project-site-v2",
        "星桥项目的试验地点在哪里？",
    )
    resolved = resolve_source_bound_slot_composition(
        catalog, base_catalog, active_catalog, surface, closure)
    assert resolved.accepted

    with pytest.raises(SourceBoundSlotCompositionError, match="catalog schema"):
        replace(catalog, catalog_schema=True)
    with pytest.raises(SourceBoundSlotCompositionError, match="result code"):
        SourceBoundSlotCompositionResolution(
            False, 1, surface, catalog)
    with pytest.raises(SourceBoundSlotCompositionError, match="matched frame count"):
        SourceBoundSlotCompositionResolution(
            DLG_RAW_ACCEPT, True, surface, catalog,
            resolved.frame, resolved.public_frame_catalog)
    with pytest.raises(SourceBoundSlotCompositionError, match="不可变 scalar tuple"):
        SourceBoundSlotCompositionResolution(
            DLG_RAW_ACCEPT, 1, list(surface), catalog,  # type: ignore[arg-type]
            resolved.frame, resolved.public_frame_catalog)
    family = catalog.families[0]
    mutable_witnesses = (
        (list(family.witnesses[0][0]), family.witnesses[0][1]),
        *family.witnesses[1:],
    )
    with pytest.raises(SourceBoundSlotCompositionError, match="不可变 scalar tuple"):
        replace(family, witnesses=mutable_witnesses)  # type: ignore[arg-type]

    runtime = build_public_dialogue_runtime_v1(closure)
    with pytest.raises(PublicDialogueRuntimeError, match="protocol"):
        replace(runtime, protocol_revision=1.0)


def test_pure_source_bound_slot_catalog_ast_has_no_physical_filesystem_boundary() -> None:
    """core 不得引入 pathlib、物理读取、旧 root 或 path-based source 语义。"""
    source = _SOURCE_CATALOG_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = tuple(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ) + tuple(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert "pathlib" not in imported_modules
    assert "repository_root" not in source
    assert "read_bytes" not in source
    assert "PurePosixPath" not in source
    assert "Path(" not in source
