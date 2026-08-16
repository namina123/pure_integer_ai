"""Recovery-v10 precision-first candidate 的纯合成回归。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_tokens,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_candidate import (
    V8_CANDIDATE_RULE_COUNTS,
    compile_normalization_recovery_v8_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_training_records import (
    V8_TRAIN_FAMILIES,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_precision_candidate import (
    compile_normalization_recovery_v10_precision_candidate,
    compile_normalization_recovery_v10_precision_candidate_v2,
    derive_normalization_recovery_v10_precision_preflight,
    derive_normalization_recovery_v10_precision_source_loso_audit,
    derive_normalization_recovery_v10_precision_training_audit,
    derive_normalization_recovery_v10_precision_v2_preflight,
    derive_normalization_recovery_v10_precision_v2_training_audit,
    execute_normalization_recovery_v10_precision_candidate_batch,
    execute_normalization_recovery_v10_precision_candidate_v2_batch,
    profile_normalization_recovery_v10_precision_candidate_v2_batch,
    reference_normalization_recovery_v10_precision_candidate_batch,
    reference_normalization_recovery_v10_precision_candidate_v2_batch,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v10_precision_feasibility as feasibility,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _sha(label: str) -> str:
    """返回测试 identity。"""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _directions(
        *, kind: str, identity: str, semantic: dict[str, object],
        support_three: bool,
        ) -> list[dict[str, object]]:
    """构造满足 v8 candidate LOSO 去重合同的方向记录。"""
    held = V8_TRAIN_FAMILIES if support_three else V8_TRAIN_FAMILIES[:1]
    support = (
        list(V8_TRAIN_FAMILIES)
        if not support_three else None)
    values = []
    for family in held:
        train = (
            [item for item in V8_TRAIN_FAMILIES if item != family]
            if support_three else list(V8_TRAIN_FAMILIES[1:]))
        values.append({
            "authorization_id": identity,
            "candidate_id": _sha(identity + " candidate"),
            "held_out_family": family,
            "production_enabled": 0,
            "rule_kind": kind,
            "train_support_families": train,
            **semantic,
        })
    if support is not None:
        assert values[0]["train_support_families"] == support[1:]
    return values


def _base_candidate() -> dict[str, object]:
    """构造固定 inventory 数量的 v8 disabled base candidate。"""
    outputs: dict[str, tuple[dict[str, object], ...]] = {}
    orthographic = []
    for index in range(V8_CANDIDATE_RULE_COUNTS["orthographic_rules"]):
        orthographic.extend(_directions(
            kind="ORTHOGRAPHIC_ATOM", identity=_sha(f"ortho-{index}"),
            semantic={
                "input_atom": chr(0x3400 + index),
                "output_atom": chr(0x3500 + index),
            }, support_three=True))
    lexical = []
    for index in range(V8_CANDIDATE_RULE_COUNTS["source_conditioned_rules"]):
        lexical.extend(_directions(
            kind="SOURCE_CONDITIONED_LEXICAL_ATOM",
            identity=_sha(f"lexical-{index}"),
            semantic={
                "input_text": f"词{index}",
                "official_source_text": f"Source {index}",
                "output_text": f"詞{index}",
            }, support_three=index < 2))
    structures = []
    for index in range(V8_CANDIDATE_RULE_COUNTS["structure_obligations"]):
        structures.extend(_directions(
            kind="LAYOUT_MORPHOLOGY_OBLIGATION",
            identity=_sha(f"structure-{index}"),
            semantic={"structure_tokens": [f"TOKEN:{index}"]},
            support_three=True))
    identities = []
    for index in range(V8_CANDIDATE_RULE_COUNTS["identity_veto_rules"]):
        candidate_id = _sha(f"identity-{index}")
        identities.append({
            "candidate_id": candidate_id,
            "held_out_family": V8_TRAIN_FAMILIES[0],
            "input_text": f"恒等{index}",
            "output_text": f"恒等{index}",
            "production_enabled": 0,
            "rule_kind": "IDENTITY_VETO",
            "train_support_families": list(V8_TRAIN_FAMILIES[1:]),
        })
    outputs["orthographic-rules.jsonl"] = tuple(orthographic)
    outputs["source-conditioned-lexical-rules.jsonl"] = tuple(lexical)
    outputs["layout-morphology-obligations.jsonl"] = tuple(structures)
    outputs["identity-veto-rules.jsonl"] = tuple(identities)
    return compile_normalization_recovery_v8_candidate(
        rule_pack_manifest={
            "manifest_sha256": _sha("rule-pack"),
            "mastery_claimed": 0,
            "production_enabled": 0,
            "status": "FAMILY_LOSO_FROZEN_NOT_FORMAL_NOT_DEPLOYED",
        },
        rule_outputs=outputs,
        training_audit_manifest_sha256=_sha("audit"),
        evaluation_commitment_manifest_sha256=_sha("commitment"),
    )


def _control(
        label: str, input_text: str, outputs: tuple[str, ...], *,
        families: tuple[str, ...],
        ) -> dict[str, object]:
    """构造一条 exact-input control。"""
    values = []
    for output in outputs:
        values.append({
            "output_text": output,
            "support_families": list(families),
            "support_family_count": len(families),
        })
    return {
        "candidate_id": _sha(label),
        "input_text": input_text,
        "outputs": values,
        "support_families": list(families),
        "support_family_count": len(families),
    }


def _candidate() -> dict[str, object]:
    """构造覆盖 identity、整串正字与 source 条件的 v10 candidate。"""
    controls = (
        _control(
            "identity", "檔名", ("檔名",),
            families=(V8_TRAIN_FAMILIES[0],)),
        _control(
            "identity-safe", "版本", ("版本",),
            families=V8_TRAIN_FAMILIES[:2]),
        _control(
            "orthographic", "檔案", ("档案",),
            families=V8_TRAIN_FAMILIES[:2]),
        _control(
            "lexical-trap", "儲存", ("保存",),
            families=V8_TRAIN_FAMILIES[:2]),
        _control(
            "single-family", "啟動", ("启动",),
            families=(V8_TRAIN_FAMILIES[0],)),
    )
    safety = (
        {
            "input_text": "檔名",
            "official_source_text": "Name",
            "output_text": "檔名",
            "source_family": V8_TRAIN_FAMILIES[0],
        },
        {
            "input_text": "檔名",
            "official_source_text": "File name",
            "output_text": "文件名",
            "source_family": V8_TRAIN_FAMILIES[1],
        },
        {
            "input_text": "版本",
            "official_source_text": "Version",
            "output_text": "版本",
            "source_family": V8_TRAIN_FAMILIES[0],
        },
        {
            "input_text": "版本",
            "official_source_text": "Version",
            "output_text": "版本",
            "source_family": V8_TRAIN_FAMILIES[1],
        },
        {
            "input_text": "檔案",
            "official_source_text": "Files",
            "output_text": "档案",
            "source_family": V8_TRAIN_FAMILIES[0],
        },
        {
            "input_text": "檔案",
            "official_source_text": "File",
            "output_text": "档案",
            "source_family": V8_TRAIN_FAMILIES[1],
        },
        {
            "input_text": "儲存",
            "official_source_text": "Save",
            "output_text": "保存",
            "source_family": V8_TRAIN_FAMILIES[0],
        },
        {
            "input_text": "啟動",
            "official_source_text": "Start",
            "output_text": "启动",
            "source_family": V8_TRAIN_FAMILIES[0],
        },
        {
            "input_text": "词0",
            "official_source_text": "Source 0",
            "output_text": "詞0",
            "source_family": V8_TRAIN_FAMILIES[0],
        },
        {
            "input_text": "词1",
            "official_source_text": "Source 1",
            "output_text": "詞1",
            "source_family": V8_TRAIN_FAMILIES[1],
        },
    )
    return compile_normalization_recovery_v10_precision_candidate(
        base_candidate=_base_candidate(),
        exact_input_controls=controls,
        safety_observations=safety,
        training_protocol_manifest_sha256=_sha("protocol"),
        observation_pack_manifest_sha256=_sha("observations"),
        opencc_routes={
            "檔": "档",
            "儲": "储",
            "啟": "启",
        },
        opencc_source_pack_manifest_sha256=_sha("opencc"),
    )


def _candidate_v2() -> dict[str, object]:
    """以相同 TRAIN material 构造 source-only commit 的 v2 候选。"""
    predecessor = _candidate()
    controls = (
        _control(
            "identity", "檔名", ("檔名",),
            families=(V8_TRAIN_FAMILIES[0],)),
        _control(
            "identity-safe", "版本", ("版本",),
            families=V8_TRAIN_FAMILIES[:2]),
        _control(
            "orthographic", "檔案", ("档案",),
            families=V8_TRAIN_FAMILIES[:2]),
        _control(
            "lexical-trap", "儲存", ("保存",),
            families=V8_TRAIN_FAMILIES[:2]),
        _control(
            "single-family", "啟動", ("启动",),
            families=(V8_TRAIN_FAMILIES[0],)),
    )
    safety = (
        {"input_text": "檔名", "official_source_text": "Name",
         "output_text": "檔名", "source_family": V8_TRAIN_FAMILIES[0]},
        {"input_text": "檔名", "official_source_text": "File name",
         "output_text": "文件名", "source_family": V8_TRAIN_FAMILIES[1]},
        {"input_text": "版本", "official_source_text": "Version",
         "output_text": "版本", "source_family": V8_TRAIN_FAMILIES[0]},
        {"input_text": "版本", "official_source_text": "Version",
         "output_text": "版本", "source_family": V8_TRAIN_FAMILIES[1]},
        {"input_text": "檔案", "official_source_text": "Files",
         "output_text": "档案", "source_family": V8_TRAIN_FAMILIES[0]},
        {"input_text": "檔案", "official_source_text": "File",
         "output_text": "档案", "source_family": V8_TRAIN_FAMILIES[1]},
        {"input_text": "儲存", "official_source_text": "Save",
         "output_text": "保存", "source_family": V8_TRAIN_FAMILIES[0]},
        {"input_text": "啟動", "official_source_text": "Start",
         "output_text": "启动", "source_family": V8_TRAIN_FAMILIES[0]},
        {"input_text": "词0", "official_source_text": "Source 0",
         "output_text": "詞0", "source_family": V8_TRAIN_FAMILIES[0]},
        {"input_text": "词1", "official_source_text": "Source 1",
         "output_text": "詞1", "source_family": V8_TRAIN_FAMILIES[1]},
    )
    candidate = compile_normalization_recovery_v10_precision_candidate_v2(
        base_candidate=_base_candidate(),
        exact_input_controls=controls,
        safety_observations=safety,
        training_protocol_manifest_sha256=_sha("protocol"),
        observation_pack_manifest_sha256=_sha("observations"),
        opencc_routes={"檔": "档", "儲": "储", "啟": "启"},
        opencc_source_pack_manifest_sha256=_sha("opencc"),
    )
    assert candidate["predecessor_candidate_program_sha256"] == predecessor[
        "candidate_program_sha256"]
    return candidate


def _query(input_text: str, source: str) -> dict[str, object]:
    """构造结构自洽 query。"""
    return {
        "input_text": input_text,
        "official_source_text": source,
        "structure_tokens": list(localization_structure_tokens(input_text)),
    }


def test_precision_candidate_only_commits_three_safe_routes() -> None:
    """identity 必须前置，且不得恢复局部字符组合。"""
    candidate = _candidate()
    assert candidate["rule_counts"] == {
        "identity_veto_rules": 1,
        "orthographic_whole_input_rules": 1,
        "source_conditioned_rules": 2,
    }
    queries = (
        _query("版本", "Any source"),
        _query("词0", "Source 0"),
        _query("檔案", "Files"),
        _query("新檔案", "New file"),
        _query("儲存", "Save"),
        _query("啟動", "Start"),
        _query("词0", "Different source"),
    )
    indexed = execute_normalization_recovery_v10_precision_candidate_batch(
        candidate, queries)
    reference = (
        reference_normalization_recovery_v10_precision_candidate_batch(
            candidate, queries))
    assert indexed == reference
    assert [(item["route_kind"], item["output_text"]) for item in indexed] == [
        ("IDENTITY_VETO", "版本"),
        ("SOURCE_CONDITIONED_LEXICAL_ATOM", "詞0"),
        ("ORTHOGRAPHIC_WHOLE_INPUT", "档案"),
        ("UNKNOWN", ""),
        ("UNKNOWN", ""),
        ("UNKNOWN", ""),
        ("UNKNOWN", ""),
    ]


def test_precision_preflight_and_training_audit_are_zero_wrong() -> None:
    """预检和 TRAIN audit 必须有 changed exact 且 WRONG 为零。"""
    candidate = _candidate()
    preflight = derive_normalization_recovery_v10_precision_preflight(candidate)
    assert preflight["failure_count"] == 0
    assert preflight["indexed_reference_mismatch_count"] == 0
    audit = derive_normalization_recovery_v10_precision_training_audit(
        candidate,
        (
            {
                **_query("版本", "Version"),
                "expected_output": "版本",
                "source_family": V8_TRAIN_FAMILIES[0],
            },
            {
                **_query("词0", "Source 0"),
                "expected_output": "詞0",
                "source_family": V8_TRAIN_FAMILIES[1],
            },
            {
                **_query("檔案", "Files"),
                "expected_output": "档案",
                "source_family": V8_TRAIN_FAMILIES[2],
            },
            {
                **_query("新檔案", "New file"),
                "expected_output": "新档案",
                "source_family": V8_TRAIN_FAMILIES[2],
            },
        ),
    )
    assert audit["training_outcome"] == (
        "PASS_ZERO_WRONG_NONZERO_CHANGED_EXACT")
    assert audit["outcomes"] == {
        "EXACT": 3,
        "UNKNOWN": 1,
        "WRONG": 0,
        "MISMATCH": 0,
    }


def test_precision_candidate_rejects_duplicate_control_input() -> None:
    """同步重复 control 不得形成不确定优先级。"""
    item = _control(
        "duplicate", "檔案", ("档案",),
        families=V8_TRAIN_FAMILIES[:2])
    with pytest.raises(BroadQaExternalDataError, match="重复"):
        compile_normalization_recovery_v10_precision_candidate(
            base_candidate=_base_candidate(),
            exact_input_controls=(item, item),
            safety_observations=({
                "input_text": "檔案",
                "official_source_text": "Files",
                "output_text": "档案",
                "source_family": V8_TRAIN_FAMILIES[0],
            },),
            training_protocol_manifest_sha256=_sha("protocol"),
            observation_pack_manifest_sha256=_sha("observations"),
            opencc_routes={"檔": "档"},
            opencc_source_pack_manifest_sha256=_sha("opencc"),
        )


def test_precision_candidate_tamper_is_rejected() -> None:
    """修改库存但保留旧 program SHA 必须失败关闭。"""
    candidate = _candidate()
    candidate["inventories"]["orthographic_whole_input_rules"][0][
        "output_text"] = "错误"
    with pytest.raises(BroadQaExternalDataError, match="program 漂移"):
        execute_normalization_recovery_v10_precision_candidate_batch(
            candidate, (_query("檔案", "Files"),))


def test_precision_v2_only_source_route_can_commit() -> None:
    """identity 和 orthographic 只保留 trace，不能冒充正确答案。"""
    candidate = _candidate_v2()
    queries = (
        _query("版本", "Version"),
        _query("词0", "Source 0"),
        _query("檔案", "Files"),
        _query("未知", "Unknown"),
    )
    indexed = execute_normalization_recovery_v10_precision_candidate_v2_batch(
        candidate, queries)
    reference = (
        reference_normalization_recovery_v10_precision_candidate_v2_batch(
            candidate, queries))
    assert indexed == reference
    assert [(item["behavior"], item["route_kind"], item["output_text"])
            for item in indexed] == [
        ("UNKNOWN", "IDENTITY_VETO_NONCOMMITTING", ""),
        ("EXACT", "SOURCE_CONDITIONED_LEXICAL_ATOM", "詞0"),
        ("UNKNOWN", "ORTHOGRAPHIC_WHOLE_INPUT_HYPOTHESIS", ""),
        ("UNKNOWN", "UNKNOWN", ""),
    ]
    preflight = derive_normalization_recovery_v10_precision_v2_preflight(
        candidate)
    assert preflight["failure_count"] == 0
    audit = derive_normalization_recovery_v10_precision_v2_training_audit(
        candidate,
        tuple({
            **query,
            "expected_output": output,
            "source_family": V8_TRAIN_FAMILIES[index % 3],
        } for index, (query, output) in enumerate(zip(
            queries, ("版本", "詞0", "档案", "未知")))),
    )
    assert audit["outcomes"] == {
        "EXACT": 1, "UNKNOWN": 3, "WRONG": 0, "MISMATCH": 0}


def test_precision_v2_profile_keeps_single_validation_batch_semantics() -> None:
    """profile 的 indexed/reference 结果必须一致且只产生整数耗时。"""
    candidate = _candidate_v2()
    queries = (
        _query("版本", "Version"),
        _query("词0", "Source 0"),
        _query("檔案", "Files"),
        _query("未知", "Unknown"),
    )

    def clock():
        clock.value += 10
        return clock.value

    clock.value = 0
    indexed, indexed_durations = (
        profile_normalization_recovery_v10_precision_candidate_v2_batch(
            candidate, queries, indexed=True, clock_ns=clock))
    reference, reference_durations = (
        profile_normalization_recovery_v10_precision_candidate_v2_batch(
            candidate, queries, indexed=False, clock_ns=clock))
    assert indexed == reference
    assert indexed_durations == (10, 10, 10, 10)
    assert reference_durations == indexed_durations


def test_precision_source_loso_rebuilds_without_held_output() -> None:
    """每方向只用另外两家重建 source rule，且 held WRONG 必须为零。"""
    observations = tuple({
        "input_text": f"词{rule_index}",
        "official_source_text": f"Source {rule_index}",
        "output_text": f"詞{rule_index}",
        "source_family": family,
    } for rule_index in range(2) for family in V8_TRAIN_FAMILIES)
    audit = derive_normalization_recovery_v10_precision_source_loso_audit(
        base_candidate=_base_candidate(),
        safety_observations=observations,
    )
    assert audit["status"] == "PASS_ZERO_WRONG_NONZERO_EXACT"
    assert audit["outcomes"] == {
        "EXACT": 6, "UNKNOWN": 0, "WRONG": 0}
    assert all(item["source_rule_count"] == 2
               for item in audit["directions"])


def _write_artifact(
        root: Path, files: dict[str, tuple[bytes, str, int | None]], *,
        summary: dict[str, object] | None = None,
        ) -> str:
    """写入供 publisher 测试使用的最小冻结 artifact。"""
    root.mkdir()
    records = []
    for name, (payload, role, count) in files.items():
        (root / name).write_bytes(payload)
        record = {
            "bytes": len(payload),
            "relative_path": name,
            "role": role,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if count is not None:
            record["record_count"] = count
        records.append(record)
    manifest = {"files": records}
    if summary is not None:
        manifest["summary"] = summary
    payload = canonical_json_line(manifest)
    (root / "manifest.json").write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _observation(
        input_text: str, output_text: str, source: str, family: str,
        ) -> dict[str, object]:
    """构造 publisher 所需的最小统一 Observation。"""
    return {
        "official_source_text": source,
        "source_family": family,
        "zh_hans": {"translation": output_text},
        "zh_hant": {"translation": input_text},
    }


def test_precision_feasibility_publish_strict_read_and_duplicate_guard(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """feasibility 必须可重派生回读且拒绝覆盖。"""
    base = _base_candidate()
    base_sha = _write_artifact(
        tmp_path / "base",
        {"candidate-program.json": (
            canonical_json_line(base), "BASE_CANDIDATE", None)},
    )
    controls = (
        _control(
            "identity-safe", "版本", ("版本",),
            families=V8_TRAIN_FAMILIES[:2]),
        _control(
            "orthographic", "檔案", ("档案",),
            families=V8_TRAIN_FAMILIES[:2]),
    )
    control_payload = b"".join(canonical_json_line(item) for item in controls)
    protocol_sha = _write_artifact(
        tmp_path / "protocol",
        {"exact-input-control.jsonl": (
            control_payload, "EXACT_INPUT_CONTROL", len(controls))},
    )
    observations = {
        "qbittorrent-observations.jsonl": (
            _observation(
                "版本", "版本", "Version", V8_TRAIN_FAMILIES[0]),
            _observation(
                "檔案", "档案", "Files", V8_TRAIN_FAMILIES[0]),
            _observation(
                "词0", "詞0", "Source 0", V8_TRAIN_FAMILIES[0]),
            _observation(
                "词1", "詞1", "Source 1", V8_TRAIN_FAMILIES[0]),
        ),
        "stellarium-observations.jsonl": (
            _observation(
                "版本", "版本", "Version", V8_TRAIN_FAMILIES[1]),
            _observation(
                "檔案", "档案", "File", V8_TRAIN_FAMILIES[1]),
            _observation(
                "词1", "詞1", "Source 1", V8_TRAIN_FAMILIES[1]),
            _observation(
                "词0", "詞0", "Source 0", V8_TRAIN_FAMILIES[1]),
        ),
        "keepassxc-observations.jsonl": (
            _observation(
                "词0", "詞0", "Source 0", V8_TRAIN_FAMILIES[2]),
            _observation(
                "词1", "詞1", "Source 1", V8_TRAIN_FAMILIES[2]),
        ),
    }
    observation_files = {
        name: (
            b"".join(canonical_json_line(item) for item in values),
            "OBSERVATIONS", len(values),
        )
        for name, values in observations.items()
    }
    observation_sha = _write_artifact(
        tmp_path / "observations", observation_files,
        summary={"observation_count": sum(
            len(values) for values in observations.values())},
    )
    opencc = tmp_path / "opencc"
    opencc.mkdir()
    opencc_payload = canonical_json_line({"source": "synthetic"})
    (opencc / "manifest.json").write_bytes(opencc_payload)
    opencc_sha = hashlib.sha256(opencc_payload).hexdigest()
    monkeypatch.setattr(
        feasibility, "_require_k_root", lambda value: Path(value).resolve())
    monkeypatch.setattr(
        feasibility, "read_opencc_unique_t2s_routes",
        lambda _root: ({"檔": "档"}, {
            "ambiguous_route_count": 0,
            "line_count": 1,
            "unique_route_count": 1,
        }),
    )
    target = tmp_path / "published"
    manifest = feasibility.publish_normalization_recovery_v10_precision_feasibility(
        run_root=tmp_path,
        base_candidate_dir=tmp_path / "base",
        expected_base_candidate_manifest_sha256=base_sha,
        protocol_dir=tmp_path / "protocol",
        expected_protocol_manifest_sha256=protocol_sha,
        observation_dir=tmp_path / "observations",
        expected_observation_manifest_sha256=observation_sha,
        opencc_source_pack_dir=opencc,
        expected_opencc_source_manifest_sha256=opencc_sha,
        target_dir=target,
    )
    stored, candidate, preflight, audit = (
        feasibility.read_normalization_recovery_v10_precision_feasibility(
            target,
            base_candidate_dir=tmp_path / "base",
            expected_base_candidate_manifest_sha256=base_sha,
            protocol_dir=tmp_path / "protocol",
            expected_protocol_manifest_sha256=protocol_sha,
            observation_dir=tmp_path / "observations",
            expected_observation_manifest_sha256=observation_sha,
            opencc_source_pack_dir=opencc,
            expected_opencc_source_manifest_sha256=opencc_sha,
            expected_feasibility_manifest_sha256=manifest["manifest_sha256"],
        ))
    assert stored["manifest_sha256"] == manifest["manifest_sha256"]
    assert candidate["rule_counts"] == {
        "identity_veto_rules": 1,
        "orthographic_whole_input_rules": 1,
        "source_conditioned_rules": 2,
    }
    assert preflight["failure_count"] == 0
    assert audit["outcomes"]["WRONG"] == 0
    target_v2 = tmp_path / "published-v2"
    manifest_v2 = (
        feasibility.publish_normalization_recovery_v10_precision_feasibility_v2(
            run_root=tmp_path,
            predecessor_feasibility_dir=target,
            expected_predecessor_feasibility_manifest_sha256=manifest[
                "manifest_sha256"],
            base_candidate_dir=tmp_path / "base",
            expected_base_candidate_manifest_sha256=base_sha,
            protocol_dir=tmp_path / "protocol",
            expected_protocol_manifest_sha256=protocol_sha,
            observation_dir=tmp_path / "observations",
            expected_observation_manifest_sha256=observation_sha,
            opencc_source_pack_dir=opencc,
            expected_opencc_source_manifest_sha256=opencc_sha,
            target_dir=target_v2,
        ))
    stored_v2, candidate_v2, preflight_v2, audit_v2, loso_v2 = (
        feasibility.read_normalization_recovery_v10_precision_feasibility_v2(
            target_v2,
            predecessor_feasibility_dir=target,
            expected_predecessor_feasibility_manifest_sha256=manifest[
                "manifest_sha256"],
            base_candidate_dir=tmp_path / "base",
            expected_base_candidate_manifest_sha256=base_sha,
            protocol_dir=tmp_path / "protocol",
            expected_protocol_manifest_sha256=protocol_sha,
            observation_dir=tmp_path / "observations",
            expected_observation_manifest_sha256=observation_sha,
            opencc_source_pack_dir=opencc,
            expected_opencc_source_manifest_sha256=opencc_sha,
            expected_feasibility_manifest_sha256=manifest_v2[
                "manifest_sha256"],
        ))
    assert stored_v2["manifest_sha256"] == manifest_v2["manifest_sha256"]
    assert candidate_v2["candidate_program_sha256"] == manifest_v2[
        "candidate_program_sha256"]
    assert preflight_v2["failure_count"] == 0
    assert audit_v2["outcomes"]["WRONG"] == 0
    assert loso_v2["status"] == "PASS_ZERO_WRONG_NONZERO_EXACT"
    with pytest.raises(BroadQaExternalDataError, match="path 非法"):
        feasibility.publish_normalization_recovery_v10_precision_feasibility(
            run_root=tmp_path,
            base_candidate_dir=tmp_path / "base",
            expected_base_candidate_manifest_sha256=base_sha,
            protocol_dir=tmp_path / "protocol",
            expected_protocol_manifest_sha256=protocol_sha,
            observation_dir=tmp_path / "observations",
            expected_observation_manifest_sha256=observation_sha,
            opencc_source_pack_dir=opencc,
            expected_opencc_source_manifest_sha256=opencc_sha,
            target_dir=target,
        )
