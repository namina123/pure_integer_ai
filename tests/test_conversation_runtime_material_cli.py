from pathlib import Path

import json
import pytest

from pure_integer_ai.experiments.conversation_runtime_material_cli import (
    _read_document_manifest,
    RuntimeMaterialDocumentRequest,
    _read_binding_requests,
    RuntimeMaterialBindingRequest,
    RuntimeMaterialCliError,
    build_runtime_material_run,
)
from pure_integer_ai.experiments.conversation_runtime_material_persistence import (
    load_runtime_material_runtime,
    open_runtime_material_sqlite,
    rebuild_runtime_material_observations,
)
from pure_integer_ai.experiments.conversation_runtime_material_binding_persistence import (
    load_runtime_material_response_provider,
)


def test_cli_builder_publishes_recoverable_runtime_run(tmp_path: Path) -> None:
    material = tmp_path / "manual.txt"
    material.write_text(
        "夜间模式会降低屏幕亮度。长按菜单键可打开设置。",
        encoding="utf-8",
    )
    # Production output is K-only; tests use the explicit non-production
    # transport so no temporary run is left on the training disk.
    output = tmp_path / "cli-test-runtime"
    root, database = build_runtime_material_run(
        material_file=material,
        output_root=output,
        source_kind=93,
        source_id=8801,
        document_id=0,
        scope_id=8801,
        license_id="CC0-1.0",
        batch_id=1,
        authority_key=(7, 8801),
        version_key=(1, 8801),
        question="夜间模式与设置的先后关系是什么？",
        qualification_state="SUPPORTED",
        reason_id="explicit-test-authority",
        source_title="测试手册",
        source_url="https://example.invalid/manual",
        require_k_drive=False,
    )
    runtime = open_runtime_material_sqlite(database, require_k_drive=False)
    try:
        recovery = load_runtime_material_runtime(
            root, source_records=runtime.source_records, require_k_drive=False)
        observations = rebuild_runtime_material_observations(
            runtime.context, recovery, source_records=runtime.source_records)
        provider = load_runtime_material_response_provider(
            root,
            source_records=runtime.source_records,
            observations=observations,
            require_k_drive=False,
        )
        answer = provider.response("夜间模式与设置的先后关系是什么？")
        assert answer is not None
        assert answer[0] == "ANSWER"
        assert "长按菜单键可打开设置" in answer[1]
        followup = provider.response_followup(
            "它与设置有什么关系？", "测试手册")
        assert followup is not None
        assert followup[0] == "ANSWER"
        assert "夜间模式会降低屏幕亮度" in followup[1]
        assert provider.response_followup("普通问题？", "测试手册") is None
        assert provider.response_followup(
            "它与设置有什么关系？", "未登记手册")[0] == "CLARIFY"
        related = provider.response_related(
            "夜间模式和设置是什么关系？", "测试手册")
        assert related is not None
        assert related[0] == "ANSWER"
        assert provider.response_related("设置怎么打开？", "测试手册") is None
    finally:
        runtime.close()


def test_cli_rejects_utf8_bom_with_actionable_error(tmp_path: Path) -> None:
    material = tmp_path / "bom-manual.txt"
    material.write_bytes(b"\xef\xbb\xbf" + "夜间模式会降低屏幕亮度。".encode("utf-8"))
    try:
        build_runtime_material_run(
            material_file=material,
            output_root=tmp_path / "bom-runtime",
            source_kind=93,
            source_id=8802,
            document_id=0,
            scope_id=8802,
            license_id="CC0-1.0",
            batch_id=1,
            authority_key=(7, 8802),
            version_key=(1, 8802),
            question="夜间模式是什么？",
            qualification_state="SUPPORTED",
            reason_id="explicit-test-authority",
            require_k_drive=False,
        )
    except RuntimeMaterialCliError as error:
        assert "无 BOM" in str(error)
    else:
        raise AssertionError("BOM 资料必须在 CLI 边界被拒绝")


def test_cli_builder_publishes_multiple_explicit_bindings(tmp_path: Path) -> None:
    material = tmp_path / "multi-manual.txt"
    material.write_text(
        "夜间模式会降低屏幕亮度。长按菜单键可打开设置。",
        encoding="utf-8",
    )
    output = tmp_path / "multi-cli-runtime"
    root, database = build_runtime_material_run(
        material_file=material,
        output_root=output,
        source_kind=93,
        source_id=8803,
        document_id=0,
        scope_id=8803,
        license_id="CC0-1.0",
        batch_id=1,
        authority_key=(7, 8803),
        version_key=(1, 8803),
        binding_requests=(
            RuntimeMaterialBindingRequest(
                "夜间模式与设置的先后关系是什么？",
                "SUPPORTED", "explicit-test-authority"),
            RuntimeMaterialBindingRequest(
                "夜间模式如何影响屏幕？",
                "SUPPORTED", "explicit-test-authority"),
        ),
        source_title="多问题手册",
        source_url="https://example.invalid/multi-manual",
        require_k_drive=False,
    )
    runtime = open_runtime_material_sqlite(database, require_k_drive=False)
    try:
        recovery = load_runtime_material_runtime(
            root, source_records=runtime.source_records, require_k_drive=False)
        observations = rebuild_runtime_material_observations(
            runtime.context, recovery, source_records=runtime.source_records)
        provider = load_runtime_material_response_provider(
            root,
            source_records=runtime.source_records,
            observations=observations,
            require_k_drive=False,
        )
        assert provider.response(
            "夜间模式与设置的先后关系是什么？")[0] == "ANSWER"
        assert provider.response("夜间模式如何影响屏幕？")[0] == "ANSWER"
        assert provider.response_followup(
            "它如何影响屏幕？", "多问题手册")[0] == "CLARIFY"
    finally:
        runtime.close()


def test_binding_file_parser_is_strict_and_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "bindings.jsonl"
    path.write_text(
        '{"question":"问题一？","qualification_state":"SUPPORTED",'
        '"reason_id":"authority","relation_index":0}\n'
        '{"question":"问题二？","qualification_state":"UNKNOWN",'
        '"reason_id":"insufficient","relation_index":1}\n',
        encoding="utf-8",
    )
    requests = _read_binding_requests(path, require_k_drive=False)
    assert tuple(item.question for item in requests) == ("问题一？", "问题二？")
    assert requests[1].relation_index == 1

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        '{"question":"问题一？","qualification_state":"SUPPORTED",'
        '"reason_id":"authority"}\n'
        '{"question":"问题一？","qualification_state":"SUPPORTED",'
        '"reason_id":"authority"}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeMaterialCliError, match="重复"):
        _read_binding_requests(duplicate, require_k_drive=False)


def test_document_manifest_parser_keeps_sources_and_scopes_separate(
        tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("资料一。", encoding="utf-8")
    second.write_text("资料二。", encoding="utf-8")
    manifest = tmp_path / "materials.jsonl"
    rows = (
        {
            "material_file": str(first), "source_kind": 100,
            "source_id": 1, "document_id": 0, "scope_id": 11,
            "license_id": "CC0-1.0", "batch_id": 1,
            "authority_key": [7, 1], "version_key": [1, 1],
            "bindings": [{"question": "资料一是什么？",
                          "qualification_state": "SUPPORTED",
                          "reason_id": "authority"}],
        },
        {
            "material_file": str(second), "source_kind": 100,
            "source_id": 2, "document_id": 0, "scope_id": 12,
            "license_id": "CC0-1.0", "batch_id": 1,
            "authority_key": [7, 2], "version_key": [1, 2],
            "bindings": [{"question": "资料二是什么？",
                          "qualification_state": "UNKNOWN",
                          "reason_id": "insufficient"}],
        },
    )
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    parsed = _read_document_manifest(manifest, require_k_drive=False)
    assert all(isinstance(item, RuntimeMaterialDocumentRequest)
               for item in parsed)
    assert tuple(item.scope_id for item in parsed) == (11, 12)
    assert tuple(item.source_id for item in parsed) == (1, 2)


def test_relation_index_isolates_runtime_answer_surface(tmp_path: Path) -> None:
    material = tmp_path / "steps.txt"
    material.write_text(
        "第一步打开菜单。第二步选择设置。第三步保存。",
        encoding="utf-8",
    )
    root, database = build_runtime_material_run(
        material_file=material,
        output_root=tmp_path / "steps-runtime",
        source_kind=93,
        source_id=8804,
        document_id=0,
        scope_id=8804,
        license_id="CC0-1.0",
        batch_id=1,
        authority_key=(7, 8804),
        version_key=(1, 8804),
        binding_requests=(
            RuntimeMaterialBindingRequest(
                "前两步是什么？", "SUPPORTED", "authority", 0),
            RuntimeMaterialBindingRequest(
                "后两步是什么？", "SUPPORTED", "authority", 1),
            RuntimeMaterialBindingRequest(
                "全部步骤是什么？", "SUPPORTED", "authority", 0),
            RuntimeMaterialBindingRequest(
                "全部步骤是什么？", "SUPPORTED", "authority", 1),
        ),
        source_title="步骤手册",
        require_k_drive=False,
    )
    runtime = open_runtime_material_sqlite(database, require_k_drive=False)
    try:
        recovery = load_runtime_material_runtime(
            root, source_records=runtime.source_records, require_k_drive=False)
        observations = rebuild_runtime_material_observations(
            runtime.context, recovery, source_records=runtime.source_records)
        provider = load_runtime_material_response_provider(
            root, source_records=runtime.source_records,
            observations=observations, require_k_drive=False)
        first = provider.response("前两步是什么？")
        second = provider.response("后两步是什么？")
        combined = provider.response("全部步骤是什么？")
        assert first is not None and second is not None and combined is not None
        assert first[0] == second[0] == "ANSWER"
        assert first[1] == "第一步打开菜单。第二步选择设置。"
        assert second[1] == "第二步选择设置。第三步保存。"
        assert combined[1] == "第一步打开菜单。第二步选择设置。第三步保存。"
    finally:
        runtime.close()
