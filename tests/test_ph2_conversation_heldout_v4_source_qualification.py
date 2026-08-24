"""DLG-05 v4 来源声明的无标签、test-only transport 专项。"""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_source_qualification as qualification_module
from pure_integer_ai.experiments.conversation_heldout_v4_bundle import (
    ConversationHeldOutV4DependencyBinding,
    unicode_scalars,
)
from pure_integer_ai.experiments.conversation_heldout_v4_candidate_runtime import (
    V4_RUNTIME_FAMILY_KEY,
    build_v4_synthetic_runtime_fixture,
)
from pure_integer_ai.experiments.conversation_heldout_v4_external_input_capsule import (
    ConversationHeldOutV4ExternalInputCapsule,
    ConversationHeldOutV4ExternalProducer,
    read_v4_external_input_capsule,
    write_v4_external_input_capsule,
)
from pure_integer_ai.experiments.conversation_heldout_v4_source_qualification import (
    ConversationHeldOutV4QualificationTrustAnchor,
    ConversationHeldOutV4SourceQualificationError,
    V4_CLAIM_CONSISTENT_UNQUALIFIED_STATUS,
    V4_SOURCE_QUALIFICATION_AUDIT_SCOPE,
    V4_SELECTION_RULE,
    V4_SOURCE_QUALIFICATION_MANIFEST_SCHEMA,
    V4_SOURCE_QUALIFICATION_RECEIPT_KIND,
    V4_SOURCE_QUALIFICATION_RECEIPT_SCHEMA,
    V4_TEST_ONLY_QUALIFICATION_STATUS,
    V4_TRAINING_COVERAGE,
    V4_TRAINING_PROVENANCE_MANIFEST_SCHEMA,
    V4_TRAINING_PROVENANCE_KIND,
    V4_TRAINING_PROVENANCE_SCHEMA,
    V4_TRAINING_UNIVERSE_NONEMPTY,
    audit_v4_independent_source_qualification,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.ph2_dataset_core import canonical_json_bytes
from pure_integer_ai.storage.integer_codec import encode_integer_tuple


_NAMESPACE = (20260821, 405, 504)


def _sha256(value: bytes) -> tuple[int, ...]:
    """为 test-only transport 重算完整 SHA-256。"""
    return tuple(hashlib.sha256(value).digest())


def _hex(value: bytes | tuple[int, ...]) -> str:
    """把测试 payload 或 digest 规范导出为小写 SHA-256。"""
    payload = value if isinstance(value, bytes) else bytes(value)
    return hashlib.sha256(payload).hexdigest() if isinstance(value, bytes) else payload.hex()


def _key_sha256(value: tuple[int, ...]) -> str:
    """按生产合同对完整整数 stable key 求 SHA。"""
    return hashlib.sha256(encode_integer_tuple(value)).hexdigest()


def _scalar_sha256(value: tuple[int, ...]) -> str:
    """按生产合同对 URI/许可/归属的 UTF-8 scalar 文本求 SHA。"""
    return hashlib.sha256("".join(chr(item) for item in value).encode("utf-8")).hexdigest()


def _test_transport_draft() -> ConversationHeldOutV4ExternalInputCapsule:
    """构造 synthetic-derived 临时 transport，明确不声称真实独立来源。"""
    fixture = build_v4_synthetic_runtime_fixture()
    source_input = fixture.inputs[0]
    source_record = source_input.source_records[0]
    raw_text = source_record.raw_text + "。" * 96
    external_record = replace(
        source_record,
        raw_text_scalars=unicode_scalars(raw_text),
        content_sha256=_sha256(raw_text.encode("utf-8")),
        source_uri_scalars=unicode_scalars(
            "https://example.invalid/dlg05-v4-test-source"),
    )
    external_input = replace(
        source_input,
        source_records=(external_record,),
        evidence_plans=tuple(sorted(
            source_input.evidence_plans,
            key=lambda item: item.target.stable_key(),
        )),
    )
    return ConversationHeldOutV4ExternalInputCapsule(
        V4_RUNTIME_FAMILY_KEY,
        ConversationHeldOutV4ExternalProducer(
            ProtocolKey((*_NAMESPACE, 1))),
        ConversationHeldOutV4DependencyBinding(
            fixture.dependencies.artifact_sha256,
            fixture.dependencies.inventory_sha256,
            fixture.dependencies.document_sha256,
        ),
        (external_input,),
    )


def _write_closure(root, *, body_name, ints_name, artifact_kind, schema, document):
    """仅为 test-only transport 生成三文件 canonical 闭包；生产模块没有对应 writer。"""
    root.mkdir()
    body = canonical_json_bytes(document)
    ints = encode_integer_tuple(tuple(body))
    manifest = canonical_json_bytes({
        "artifact_kind": artifact_kind,
        "files": {
            body_name: {
                "sha256": hashlib.sha256(body).hexdigest(),
                "size": len(body),
            },
            ints_name: {
                "sha256": hashlib.sha256(ints).hexdigest(),
                "size": len(ints),
            },
        },
        "files_scope": "PRE_MANIFEST_PAYLOAD_FILES_ONLY",
        "format_version": 1,
        "schema": schema,
    })
    (root / body_name).write_bytes(body)
    (root / ints_name).write_bytes(ints)
    (root / "manifest.json").write_bytes(manifest)
    return body, manifest


def _capsule_binding(capsule, record):
    """独立构造 receipt 必须绑定的 R02 capsule 身份。"""
    turn_key = []
    for runtime_input in capsule.inputs:
        key = runtime_input.stable_key()
        turn_key.extend((len(key), *key))
    dependencies = capsule.dependencies
    return {
        "dependencies": {
            "artifact_sha256": bytes(dependencies.artifact_sha256).hex(),
            "document_sha256": bytes(dependencies.document_sha256).hex(),
            "inventory_sha256": bytes(dependencies.inventory_sha256).hex(),
        },
        "family_key": list(V4_RUNTIME_FAMILY_KEY.components),
        "input_stable_key_sha256": _key_sha256(capsule.stable_key()),
        "manifest_sha256": bytes(capsule.manifest_sha256).hex(),
        "origin": capsule.origin,
        "producer_declaration_sha256": hashlib.sha256(
            capsule.external_producer_declaration.encode("utf-8")).hexdigest(),
        "producer_key": list(capsule.external_producer_key.components),
        "source_count": 1,
        "turn_count": len(capsule.inputs),
        "turn_table_sha256": _key_sha256(tuple(turn_key)),
    }


def _source_identity(record, lineage):
    """导出 receipt/训练表共用的完整三重来源身份。"""
    return {
        "content_sha256": bytes(record.content_sha256).hex(),
        "lineage_key": list(lineage.components),
        "source_ref_key": list(record.source.stable_key()),
    }


def _snapshot(*roots):
    """记录所有 test artifact 字节，证明 audit 不写入任何 root。"""
    return {
        (root.name, path.relative_to(root).as_posix()): path.read_bytes()
        for root in roots for path in root.rglob("*") if path.is_file()
    }


def _materials(
        tmp_path, *, overlap: str | None = None, authority_conflict: bool = False,
        anchor_selector_conflict: bool = False,
        missing_license_evidence: bool = False, bad_coverage: bool = False,
        ):
    """建立完整但明确 test-only 的 source/qualification/training 三根输入。"""
    source_root = write_v4_external_input_capsule(
        tmp_path / "source", _test_transport_draft(), require_k_drive=False)
    capsule = read_v4_external_input_capsule(source_root, require_k_drive=False)
    record = capsule.inputs[0].source_records[0]
    population_lineage = ProtocolKey((*_NAMESPACE, 30))
    training_source = record.source.from_stable_key(tuple(
        (*record.source.stable_key()[:1], record.source.source_id + 1000,
         *record.source.stable_key()[2:])))
    training_content = _sha256(b"independent training source")
    training_lineage = ProtocolKey((*_NAMESPACE, 31))
    if overlap == "source_ref":
        training_source = record.source
    elif overlap == "content":
        training_content = record.content_sha256
    elif overlap == "lineage":
        training_lineage = population_lineage
    elif overlap is not None:
        raise AssertionError("unknown overlap")
    training_authority = ProtocolKey((*_NAMESPACE, 13))
    training_document = {
        "artifact_kind": V4_TRAINING_PROVENANCE_KIND,
        "coverage": ("PARTIAL" if bad_coverage else V4_TRAINING_COVERAGE),
        "entries": [{
            "content_sha256": bytes(training_content).hex(),
            "lineage_key": list(training_lineage.components),
            "source_ref_key": list(training_source.stable_key()),
        }],
        "format_version": 1,
        "inventory_authority_key": list(training_authority.components),
        "inventory_scope_sha256": hashlib.sha256(
            b"test training scope").hexdigest(),
        "schema": V4_TRAINING_PROVENANCE_SCHEMA,
        "source_count": 1,
        "training_universe": V4_TRAINING_UNIVERSE_NONEMPTY,
    }
    training_root = tmp_path / "training"
    _training_body, training_manifest = _write_closure(
        training_root,
        body_name="training_provenance.json",
        ints_name="training_provenance.canonical.ints",
        artifact_kind=V4_TRAINING_PROVENANCE_KIND,
        schema=V4_TRAINING_PROVENANCE_MANIFEST_SCHEMA,
        document=training_document,
    )
    population = [_source_identity(record, population_lineage)]
    population_sha = hashlib.sha256(canonical_json_bytes({
        "population": population,
    })).hexdigest()
    producer = capsule.external_producer_key
    selector = ProtocolKey((*_NAMESPACE, 11))
    qualifier = producer if authority_conflict else ProtocolKey((*_NAMESPACE, 12))
    license_record = {
        "allowed": 1,
        "attribution_scalars_sha256": _scalar_sha256(record.attribution_scalars),
        "content_sha256": bytes(record.content_sha256).hex(),
        "license_scalars_sha256": _scalar_sha256(record.license_scalars),
        "lineage_key": list(population_lineage.components),
        "official_license_evidence_sha256": hashlib.sha256(
            b"test official license evidence").hexdigest(),
        "official_license_evidence_uri_sha256": hashlib.sha256(
            b"https://example.invalid/license-evidence").hexdigest(),
        "source_record_stable_key_sha256": _key_sha256(record.stable_key()),
        "source_ref_key": list(record.source.stable_key()),
        "source_uri_sha256": _scalar_sha256(record.source_uri_scalars),
    }
    if missing_license_evidence:
        del license_record["official_license_evidence_sha256"]
    qualification_document = {
        "artifact_kind": V4_SOURCE_QUALIFICATION_RECEIPT_KIND,
        "candidate_result_read_count": 0,
        "capsule": _capsule_binding(capsule, record),
        "format_version": 1,
        "independent_authorities": {
            "producer_equals_qualifier": 0,
            "producer_equals_selector": 0,
            "producer_equals_training_inventory_authority": 0,
            "producer_key": list(producer.components),
            "qualifier_equals_training_inventory_authority": 0,
            "qualifier_key": list(qualifier.components),
            "selector_equals_qualifier": 0,
            "selector_equals_training_inventory_authority": 0,
            "selector_key": list(selector.components),
            "training_inventory_authority_key": list(training_authority.components),
        },
        "label_or_formal_read_count": 0,
        "license_review": {
            "policy_sha256": hashlib.sha256(b"test license policy").hexdigest(),
            "records": [license_record],
            "reviewed_source_count": 1,
        },
        "qualification_status": V4_CLAIM_CONSISTENT_UNQUALIFIED_STATUS,
        "schema": V4_SOURCE_QUALIFICATION_RECEIPT_SCHEMA,
        "selection": {
            "population": population,
            "population_count": 1,
            "population_roster_sha256": population_sha,
            "selected_source_count": 1,
            "selected_source_table_sha256": population_sha,
            "selection_policy_sha256": hashlib.sha256(
                b"test selection policy").hexdigest(),
            "selection_rule": V4_SELECTION_RULE,
        },
        "training_disjointness": {
            "content_sha256_intersection_count": 0,
            "coverage": V4_TRAINING_COVERAGE,
            "exclusion_rule": "WHOLE_SOURCE_AND_DERIVATIVES",
            "lineage_intersection_count": 0,
            "source_ref_intersection_count": 0,
            "training_inventory_manifest_sha256": hashlib.sha256(
                training_manifest).hexdigest(),
        },
    }
    qualification_root = tmp_path / "qualification"
    receipt, _qualification_manifest = _write_closure(
        qualification_root,
        body_name="qualification_receipt.json",
        ints_name="qualification.canonical.ints",
        artifact_kind=V4_SOURCE_QUALIFICATION_RECEIPT_KIND,
        schema=V4_SOURCE_QUALIFICATION_MANIFEST_SCHEMA,
        document=qualification_document,
    )
    anchor = ConversationHeldOutV4QualificationTrustAnchor(
        qualifier,
        training_authority,
        _sha256(receipt),
        _sha256(training_manifest),
        selector if anchor_selector_conflict else ProtocolKey((*_NAMESPACE, 14)),
    )
    return source_root, qualification_root, training_root, anchor


def test_qualification_test_transport_is_read_only_and_explicitly_downgraded(tmp_path):
    """完整临时 transport 只验证机械声明闭环，绝不产生真实 external 资格。"""
    source_root, qualification_root, training_root, anchor = _materials(tmp_path)
    before = _snapshot(source_root, qualification_root, training_root)

    result = audit_v4_independent_source_qualification(
        source_root, qualification_root, training_root, anchor,
        require_k_drive=False)

    assert result.qualification_status == V4_TEST_ONLY_QUALIFICATION_STATUS
    assert result.audit_scope == V4_SOURCE_QUALIFICATION_AUDIT_SCOPE
    assert result.source_count == result.turn_count == 1
    assert _snapshot(source_root, qualification_root, training_root) == before
    assert not hasattr(qualification_module, "write_v4_source_qualification")


def test_qualification_requires_k_drive_by_default(tmp_path):
    """临时 D 盘 root 不能借默认入口绕过生产 transport 的 K 盘边界。"""
    source_root, qualification_root, training_root, anchor = _materials(tmp_path)

    with pytest.raises(ConversationHeldOutV4SourceQualificationError, match="K 盘"):
        audit_v4_independent_source_qualification(
            source_root, qualification_root, training_root, anchor)


@pytest.mark.parametrize("overlap", ["source_ref", "content", "lineage"])
def test_qualification_rejects_each_training_disjointness_overlap(tmp_path, overlap):
    """SourceRef、内容和 receipt 自述 lineage 任一交集都必须阻断机械审计。"""
    source_root, qualification_root, training_root, anchor = _materials(
        tmp_path, overlap=overlap)

    with pytest.raises(ConversationHeldOutV4SourceQualificationError, match="intersection"):
        audit_v4_independent_source_qualification(
            source_root, qualification_root, training_root, anchor,
            require_k_drive=False)


@pytest.mark.parametrize("option", [
    "authority_conflict", "anchor_selector_conflict", "missing_license_evidence",
    "bad_coverage",
])
def test_qualification_rejects_inconsistent_or_incomplete_external_declarations(
        tmp_path, option):
    """冲突 key、缺失证据摘要或非全量声明不得被 receipt 文字掩盖。"""
    source_root, qualification_root, training_root, anchor = _materials(
        tmp_path, **{option: True})

    with pytest.raises(ConversationHeldOutV4SourceQualificationError):
        audit_v4_independent_source_qualification(
            source_root, qualification_root, training_root, anchor,
            require_k_drive=False)


def test_qualification_rejects_anchor_mismatch_extra_files_and_nested_roots(tmp_path):
    """锚点 SHA、精确三文件闭包和三根物理隔离均是不可绕过的前置条件。"""
    source_root, qualification_root, training_root, anchor = _materials(tmp_path)
    wrong_anchor = replace(anchor, qualification_receipt_sha256=(0,) * 32)
    with pytest.raises(ConversationHeldOutV4SourceQualificationError, match="trust anchor"):
        audit_v4_independent_source_qualification(
            source_root, qualification_root, training_root, wrong_anchor,
            require_k_drive=False)

    (qualification_root / "extra.txt").write_text("test", encoding="utf-8")
    with pytest.raises(ConversationHeldOutV4SourceQualificationError, match="闭包"):
        audit_v4_independent_source_qualification(
            source_root, qualification_root, training_root, anchor,
            require_k_drive=False)

    with pytest.raises(ConversationHeldOutV4SourceQualificationError, match="不得重叠"):
        audit_v4_independent_source_qualification(
            source_root, source_root, training_root, anchor,
            require_k_drive=False)


def test_qualification_never_upgrades_a_plain_anchor_to_external_qualification():
    """K 盘模式也只返回未资格化声明一致状态，不能凭普通结构体认证第三方。"""
    assert qualification_module._audit_status_for_transport(
        require_k_drive=True) == V4_CLAIM_CONSISTENT_UNQUALIFIED_STATUS
    assert "UNQUALIFIED" in V4_CLAIM_CONSISTENT_UNQUALIFIED_STATUS


@pytest.mark.parametrize(("limit_name", "limit_value", "message"), [
    ("V4_MAX_TRAINING_PROVENANCE_ENTRIES", 0, "条目容量"),
    ("V4_MAX_TRAINING_PROVENANCE_PAYLOAD_BYTES", 1, "超过小型 transport 容量"),
])
def test_qualification_fails_closed_above_bounded_transport_capacity(
        tmp_path, monkeypatch, limit_name, limit_value, message):
    """全量训练 provenance 尚无流式实现，超过当前小型 transport 预算必须拒绝。"""
    source_root, qualification_root, training_root, anchor = _materials(tmp_path)
    monkeypatch.setattr(qualification_module, limit_name, limit_value)
    if limit_name == "V4_MAX_TRAINING_PROVENANCE_PAYLOAD_BYTES":
        training_body = training_root / "training_provenance.json"
        original_read_bytes = Path.read_bytes

        def guarded_read_bytes(path):
            """确保超限 training body 在容量拒绝前不会进入完整读取。"""
            if path == training_body:
                raise AssertionError("超限 training provenance 不得先 read_bytes")
            return original_read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    with pytest.raises(ConversationHeldOutV4SourceQualificationError, match=message):
        audit_v4_independent_source_qualification(
            source_root, qualification_root, training_root, anchor,
            require_k_drive=False)


def test_qualification_module_keeps_read_only_dependency_boundary():
    """来源声明审计只可依赖 capsule reader 与数据合同，不能接回 runtime 或 owner/formal。"""
    source_path = Path(qualification_module.__file__)
    tree = ast.parse(source_path.read_bytes(), filename=str(source_path))
    allowed_local_imports = {
        "pure_integer_ai.cognition.shared.identity": {"SourceRef"},
        "pure_integer_ai.experiments.conversation_heldout_v4_candidate_runtime": {
            "ConversationHeldOutV4RuntimeSourceCapsule",
            "V4_RUNTIME_FAMILY_KEY",
            "V4_RUNTIME_SOURCE_ORIGIN_EXTERNAL",
        },
        "pure_integer_ai.experiments.conversation_heldout_v4_external_input_capsule": {
            "ConversationHeldOutV4ExternalCapsuleError",
            "read_v4_external_input_capsule",
        },
        "pure_integer_ai.experiments.evaluation_protocol": {"ProtocolKey"},
        "pure_integer_ai.experiments.ph2_dataset_core": {
            "DatasetContractError", "canonical_json_bytes", "parse_canonical_json_bytes",
        },
        "pure_integer_ai.storage.integer_codec": {
            "IntegerCodecError", "decode_integer_tuple", "encode_integer_tuple",
        },
    }
    dynamic_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("pure_integer_ai")
                       for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
                "pure_integer_ai"):
            assert node.level == 0
            assert node.module in allowed_local_imports
            assert {alias.name for alias in node.names} == allowed_local_imports[node.module]
            assert all(alias.name != "*" for alias in node.names)
        elif isinstance(node, ast.Call):
            if ((isinstance(node.func, ast.Name) and node.func.id == "__import__")
                    or (isinstance(node.func, ast.Attribute)
                        and node.func.attr == "import_module")):
                dynamic_imports.append(node.lineno)
    assert dynamic_imports == []
    for forbidden_name in (
            "run_v4_candidate_runtime", "write_v4_runtime_artifact",
            "audit_v4_runtime_artifact", "read_v4_owner_metadata"):
        assert not hasattr(qualification_module, forbidden_name)
