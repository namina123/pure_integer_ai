"""normalization candidate clone、ICU evaluator、freeze 与单次 runner 测试。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_candidate_clone import (
    NormalizationCandidateCloneProgram,
    compile_normalization_candidate_clone,
    execute_normalization_candidate_clone,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_contrastive_protocol import (
    publish_normalization_contrastive_protocol,
    read_normalization_contrastive_protocol,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_development_learner_v1 import (
    read_normalization_development_learner_v1,
    run_normalization_development_learner_v1,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_development_learner_v1 as learner_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_icu_evaluation_family import (
    build_normalization_icu_evaluation_family_freeze,
    publish_normalization_icu_evaluation_family_freeze,
    read_normalization_icu_evaluation_family_freeze,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_icu_evaluation_family as family_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_icu_evaluation_protocol import (
    normalization_icu_evaluation_split,
    publish_normalization_icu_evaluation_protocol,
    read_normalization_icu_evaluation_protocol,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_icu_evaluation_runner import (
    run_normalization_icu_formal_evaluation,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_icu_evaluation_runner as runner_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_icu_evaluator import (
    evaluate_normalization_icu_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_icu_source_pack import (
    parse_normalization_icu_source,
    publish_normalization_icu_source_pack,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_icu_source_pack as icu_source_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_rule_pack_v3 import (
    publish_normalization_rule_pack_v3,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_source_pack import (
    publish_normalization_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _run_id(label: str) -> str:
    """返回 deterministic learner run identity。"""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _repository_root() -> Path:
    """返回当前测试文件对应的公开仓库根。"""
    return Path(__file__).resolve().parents[1]


def _publish_icu_source(
        root: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        learned_input: int,
        learned_output: int,
        ) -> tuple[Path, Path]:
    """发布含一条 learned mapping 和足量 identity 的独立 synthetic ICU family。"""
    selected = None
    for ordinal in range(1_000):
        line = (
            f"{chr(learned_output)}↔{chr(learned_input)}; # {ordinal}\n"
        ).encode("utf-8")
        payload = (b"\xef\xbb\xbf# synthetic independent source\n"
                   + "$Digits = [一二] ;\n".encode("utf-8") + line)
        _, rules, _ = parse_normalization_icu_source(payload)
        if normalization_icu_evaluation_split(
                rules[0]["statement_sha256"]) == "EVALUATION":
            selected = line
            break
    assert selected is not None
    lines = [
        b"\xef\xbb\xbf# synthetic independent source\n",
        "$Digits = [一二] ;\n".encode("utf-8"),
        selected,
    ]
    used = {learned_input}
    for ordinal in range(80):
        codepoint = 0x3400 + ordinal
        if codepoint in used:
            continue
        character = chr(codepoint)
        lines.append(f"{character}↔{character};\n".encode("utf-8"))
    rule_payload = b"".join(lines)
    license_payload = (
        "UNICODE LICENSE V3\n"
        "SPDX-License-Identifier: Unicode-3.0\n"
    ).encode("utf-8")
    monkeypatch.setattr(
        icu_source_module, "NORMALIZATION_ICU_RULE_BYTES", len(rule_payload))
    monkeypatch.setattr(
        icu_source_module,
        "NORMALIZATION_ICU_RULE_SHA256",
        hashlib.sha256(rule_payload).hexdigest(),
    )
    monkeypatch.setattr(
        icu_source_module,
        "NORMALIZATION_ICU_LICENSE_BYTES",
        len(license_payload),
    )
    monkeypatch.setattr(
        icu_source_module,
        "NORMALIZATION_ICU_LICENSE_SHA256",
        hashlib.sha256(license_payload).hexdigest(),
    )
    inputs = root / "icu-inputs"
    inputs.mkdir()
    rule_path = inputs / "Hans_Hant.txt"
    license_path = inputs / "LICENSE"
    rule_path.write_bytes(rule_payload)
    license_path.write_bytes(license_payload)
    source = root / "icu-source-pack"
    publish_normalization_icu_source_pack(
        run_root=root,
        rule_source_path=rule_path,
        license_source_path=license_path,
        target_dir=source,
    )
    protocol = root / "icu-evaluation-protocol"
    publish_normalization_icu_evaluation_protocol(
        run_root=root,
        source_pack_dir=source,
        target_dir=protocol,
    )
    return source, protocol


@pytest.fixture(scope="module")
def normalization_artifacts(tmp_path_factory: pytest.TempPathFactory):
    """一次构造真实 OpenCC learner/pack 与独立 synthetic ICU protocol。"""
    root = tmp_path_factory.mktemp("normalization-icu-evaluator")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        learner_module,
        "_require_k_run_root",
        lambda value: Path(value).resolve(),
    )
    source = root / "normalization-source-pack"
    publish_normalization_source_pack(run_root=root, target_dir=source)
    contrastive = root / "normalization-contrastive-protocol"
    publish_normalization_contrastive_protocol(
        run_root=root,
        source_pack_dir=source,
        target_dir=contrastive,
    )
    fresh = root / "learner-fresh"
    resumed = root / "learner-resumed"
    run_normalization_development_learner_v1(
        run_root=root,
        source_pack_dir=source,
        contrastive_protocol_dir=contrastive,
        run_dir=fresh,
        run_id=_run_id("evaluator-fresh"),
        mode="fresh",
        checkpoint_interval=1_024,
    )
    run_normalization_development_learner_v1(
        run_root=root,
        source_pack_dir=source,
        contrastive_protocol_dir=contrastive,
        run_dir=resumed,
        run_id=_run_id("evaluator-resumed"),
        mode="fresh",
        checkpoint_interval=1_024,
        stop_after=4_200,
    )
    run_normalization_development_learner_v1(
        run_root=root,
        source_pack_dir=source,
        contrastive_protocol_dir=contrastive,
        run_dir=resumed,
        run_id=_run_id("evaluator-resumed"),
        mode="resume",
        checkpoint_interval=1_024,
    )
    fresh_manifest, accepted, rejected = (
        read_normalization_development_learner_v1(
            fresh,
            source_pack_dir=source,
            contrastive_protocol_dir=contrastive,
        ))
    resumed_manifest, resumed_accepted, resumed_rejected = (
        read_normalization_development_learner_v1(
            resumed,
            source_pack_dir=source,
            contrastive_protocol_dir=contrastive,
        ))
    assert accepted == resumed_accepted
    assert rejected == resumed_rejected
    assert fresh_manifest["result_sha256"] == resumed_manifest["result_sha256"]
    pack = root / "normalization-rule-pack-v3"
    pack_manifest = publish_normalization_rule_pack_v3(
        source_pack_dir=source,
        contrastive_protocol_dir=contrastive,
        fresh_accepted_rules=accepted,
        fresh_rejected_trials=rejected,
        resumed_accepted_rules=resumed_accepted,
        resumed_rejected_trials=resumed_rejected,
        target_dir=pack,
        fresh_checkpoint_chain_path=fresh / "checkpoints.jsonl",
        resumed_checkpoint_chain_path=resumed / "checkpoints.jsonl",
    )
    _, _, trials = read_normalization_contrastive_protocol(
        contrastive, source_pack_dir=source)
    program = compile_normalization_candidate_clone(
        rule_pack_manifest_sha256=pack_manifest["manifest_sha256"],
        accepted_rules=accepted,
        rejected_trials=rejected,
        contrastive_trials=trials,
    )
    icu_source, icu_protocol = _publish_icu_source(
        root,
        monkeypatch,
        learned_input=program.rules[0].input_codepoint,
        learned_output=program.rules[0].output_codepoint,
    )
    yield {
        "root": root,
        "source": source,
        "contrastive": contrastive,
        "fresh": fresh,
        "resumed": resumed,
        "pack": pack,
        "accepted": accepted,
        "rejected": rejected,
        "trials": trials,
        "program": program,
        "icu_source": icu_source,
        "icu_protocol": icu_protocol,
    }
    monkeypatch.undo()


def _evaluation_inputs(artifacts):
    """严格回读 synthetic ICU evaluation，不读取 frozen real family。"""
    manifest, evaluation, reserve = read_normalization_icu_evaluation_protocol(
        artifacts["icu_protocol"], source_pack_dir=artifacts["icu_source"])
    assert evaluation and reserve
    return manifest, evaluation


def test_candidate_clone_executes_all_source_support_and_refute_offsets(
        normalization_artifacts,
        ) -> None:
    """真实 v3 pack 可编译为 3 条规则和 80 个 exact-context defeater。"""
    artifacts = normalization_artifacts
    program = artifacts["program"]
    accepted = artifacts["accepted"]
    rejected = artifacts["rejected"]
    trials = {item["trial_id"]: item for item in artifacts["trials"]}
    assert len(program.rules) == 3
    assert sum(len(item.defeaters) for item in program.rules) == 80
    for rule in accepted:
        for evidence in rule.evidence_commitments:
            trial = trials[evidence.trial_id]
            result = execute_normalization_candidate_clone(
                program, trial["phrase_source"])
            assert result.steps[evidence.source_codepoint_offset].output_codepoint == (
                evidence.observed_output_codepoint)
    for rejected_trial in rejected:
        trial = trials[rejected_trial.trial_id]
        result = execute_normalization_candidate_clone(
            program, trial["phrase_source"])
        step = result.steps[trial["source_codepoint_offset"]]
        assert step.output_codepoint == trial["observed_output_codepoint"]
        assert rejected_trial.trial_id in step.defeater_trial_ids
    assert program.production_enabled == 0


def test_compiler_rejects_synchronized_trial_observation_drift(
        normalization_artifacts,
        ) -> None:
    """trial 自报 observation 漂移时不得绕过 Evidence/source commitment。"""
    artifacts = normalization_artifacts
    rejected = artifacts["rejected"][0]
    tampered = []
    for trial in artifacts["trials"]:
        if trial["trial_id"] == rejected.trial_id:
            replacement = dict(trial)
            replacement["observed_output_codepoint"] = (
                replacement["observed_output_codepoint"] + 1)
            tampered.append(replacement)
        else:
            tampered.append(trial)
    with pytest.raises(BroadQaExternalDataError, match="Evidence/source 漂移"):
        compile_normalization_candidate_clone(
            rule_pack_manifest_sha256=(
                artifacts["program"].rule_pack_manifest_sha256),
            accepted_rules=artifacts["accepted"],
            rejected_trials=artifacts["rejected"],
            contrastive_trials=tuple(tampered),
        )


def test_synthetic_icu_evaluator_has_real_pass_fail_ne_and_label_firewall(
        normalization_artifacts,
        ) -> None:
    """四维全过才 PASS；冲突为 FAIL，缺 clone 为 NE，未知 label 失败关闭。"""
    artifacts = normalization_artifacts
    manifest, evaluation = _evaluation_inputs(artifacts)
    report = evaluate_normalization_icu_candidate(
        protocol_manifest=manifest,
        evaluation_records=evaluation,
        accepted_rules=artifacts["accepted"],
        rejected_trials=artifacts["rejected"],
        contrastive_trials=artifacts["trials"],
        program=artifacts["program"],
    )
    assert report.overall_outcome == "PASS"
    assert [item.outcome for item in report.dimensions] == [
        "PASS", "PASS", "PASS", "PASS"]
    assert report.production_enabled == 0
    assert report.reserve_label_read_count == 0

    learned_input = chr(artifacts["program"].rules[0].input_codepoint)
    changed = []
    for record in evaluation:
        if record["input_text"] == learned_input:
            changed.append({**record, "expected_output": learned_input})
        else:
            changed.append(record)
    failed = evaluate_normalization_icu_candidate(
        protocol_manifest=manifest,
        evaluation_records=tuple(changed),
        accepted_rules=artifacts["accepted"],
        rejected_trials=artifacts["rejected"],
        contrastive_trials=artifacts["trials"],
        program=artifacts["program"],
    )
    assert failed.overall_outcome == "FAIL"
    assert failed.dimensions[0].outcome == "FAIL"

    absent = evaluate_normalization_icu_candidate(
        protocol_manifest=manifest,
        evaluation_records=evaluation,
        accepted_rules=artifacts["accepted"],
        rejected_trials=artifacts["rejected"],
        contrastive_trials=artifacts["trials"],
        program=None,
    )
    assert absent.overall_outcome == "NE"
    assert [item.outcome for item in absent.dimensions] == [
        "NE", "NE", "NE", "NE"]

    leaked = ({**evaluation[0], "reserve_label": "forbidden"}, *evaluation[1:])
    with pytest.raises(BroadQaExternalDataError, match="reserve/未知字段"):
        evaluate_normalization_icu_candidate(
            protocol_manifest=manifest,
            evaluation_records=leaked,
            accepted_rules=artifacts["accepted"],
            rejected_trials=artifacts["rejected"],
            contrastive_trials=artifacts["trials"],
            program=artifacts["program"],
        )


def _freeze_arguments(artifacts) -> dict[str, object]:
    """返回 synthetic family 的完整 live identity 参数。"""
    return {
        "repository_root": _repository_root(),
        "icu_source_pack_dir": artifacts["icu_source"],
        "evaluation_protocol_dir": artifacts["icu_protocol"],
        "normalization_source_pack_dir": artifacts["source"],
        "contrastive_protocol_dir": artifacts["contrastive"],
        "rule_pack_dir": artifacts["pack"],
        "fresh_learner_dir": artifacts["fresh"],
        "resumed_learner_dir": artifacts["resumed"],
    }


def test_family_freeze_reads_no_evaluation_payload_and_binds_live_code(
        normalization_artifacts,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """freeze 只绑定 payload identity，并可重算 candidate clone 与 live code。"""
    artifacts = normalization_artifacts
    arguments = _freeze_arguments(artifacts)
    built = build_normalization_icu_evaluation_family_freeze(**arguments)
    assert built["evaluation_payload_read_count"] == 0
    assert built["reserve_payload_read_count"] == 0
    assert built["evaluation_run_count"] == 0
    assert built["candidate_freeze"]["candidate_clone_sha256"] == (
        artifacts["program"].sha256())
    assert len(built["code_files"]) == 5

    monkeypatch.setattr(
        family_module,
        "require_normalization_k_run_root",
        lambda value: Path(value).resolve(),
    )
    target = artifacts["root"] / "synthetic-family-freeze"
    report = publish_normalization_icu_evaluation_family_freeze(
        run_root=artifacts["root"],
        target_dir=target,
        **arguments,
    )
    reread = read_normalization_icu_evaluation_family_freeze(
        target, **arguments)
    assert reread["manifest_sha256"] == report["manifest_sha256"]
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_icu_evaluation_family_freeze(
            run_root=artifacts["root"],
            target_dir=target,
            **arguments,
        )

    path = target / "family-freeze.json"
    value = json.loads(path.read_bytes())
    value["candidate_freeze_sha256"] = "f" * 64
    path.write_bytes(canonical_json_line(value))
    with pytest.raises(BroadQaExternalDataError, match="live identity 漂移"):
        read_normalization_icu_evaluation_family_freeze(target, **arguments)


def test_family_freeze_does_not_open_evaluation_or_reserve_payload(
        normalization_artifacts,
        ) -> None:
    """freeze 只读取 manifest；损坏 payload 仍可构造同一 family identity。"""
    artifacts = normalization_artifacts
    arguments = _freeze_arguments(artifacts)
    before = build_normalization_icu_evaluation_family_freeze(**arguments)
    evaluation_path = artifacts["icu_protocol"] / "evaluation.inventory.jsonl"
    reserve_path = artifacts["icu_protocol"] / "reserve.identity.jsonl"
    evaluation_payload = evaluation_path.read_bytes()
    reserve_payload = reserve_path.read_bytes()
    try:
        evaluation_path.write_bytes(b"not-jsonl\n")
        reserve_path.write_bytes(b"not-jsonl\n")
        after = build_normalization_icu_evaluation_family_freeze(**arguments)
    finally:
        evaluation_path.write_bytes(evaluation_payload)
        reserve_path.write_bytes(reserve_payload)
    assert after == before
    assert after["evaluation_payload_read_count"] == 0
    assert after["reserve_payload_read_count"] == 0


def test_synthetic_formal_runner_is_single_run_and_keeps_production_disabled(
        normalization_artifacts,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """synthetic runner 先消费 guard，再发布唯一报告并拒绝第二次运行。"""
    artifacts = normalization_artifacts
    arguments = _freeze_arguments(artifacts)
    family = artifacts["root"] / "runner-family-freeze"
    family.mkdir()
    freeze = build_normalization_icu_evaluation_family_freeze(**arguments)
    (family / "family-freeze.json").write_bytes(canonical_json_line(freeze))
    monkeypatch.setattr(
        runner_module,
        "require_normalization_k_run_root",
        lambda value: Path(value).resolve(),
    )
    publication = artifacts["root"] / "synthetic-formal-publication"
    report = run_normalization_icu_formal_evaluation(
        run_root=artifacts["root"],
        family_freeze_dir=family,
        publication_dir=publication,
        **arguments,
    )
    assert report["overall_outcome"] == "PASS"
    assert report["production_enabled"] == 0
    assert report["mastery_claimed"] == 0
    assert report["receipt_published"] == 0
    assert (publication / "run-000001.guard.json").is_file()
    assert (publication / "run-000001.report.json").is_file()
    assert not (publication / "run-000001.failure.json").exists()
    with pytest.raises(BroadQaExternalDataError, match="已消费"):
        run_normalization_icu_formal_evaluation(
            run_root=artifacts["root"],
            family_freeze_dir=family,
            publication_dir=publication,
            **arguments,
        )


def test_formal_guard_seals_failure_after_payload_drift(
        normalization_artifacts,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """guard 后 payload 漂移必须封存 NE failure，且不能恢复成第二次正式运行。"""
    artifacts = normalization_artifacts
    arguments = _freeze_arguments(artifacts)
    family = artifacts["root"] / "failure-family-freeze"
    family.mkdir()
    freeze = build_normalization_icu_evaluation_family_freeze(**arguments)
    (family / "family-freeze.json").write_bytes(canonical_json_line(freeze))
    monkeypatch.setattr(
        runner_module,
        "require_normalization_k_run_root",
        lambda value: Path(value).resolve(),
    )
    publication = artifacts["root"] / "synthetic-failure-publication"
    inventory_path = artifacts["icu_protocol"] / "evaluation.inventory.jsonl"
    payload = inventory_path.read_bytes()
    try:
        inventory_path.write_bytes(b"not-jsonl\n")
        with pytest.raises(BroadQaExternalDataError):
            run_normalization_icu_formal_evaluation(
                run_root=artifacts["root"],
                family_freeze_dir=family,
                publication_dir=publication,
                **arguments,
            )
    finally:
        inventory_path.write_bytes(payload)
    assert (publication / "run-000001.guard.json").is_file()
    failure_path = publication / "run-000001.failure.json"
    assert failure_path.is_file()
    failure = json.loads(failure_path.read_bytes())
    assert failure["evaluation_run_count"] == 1
    assert failure["status"] == "NE_NO_RECEIPT"
    assert failure["production_enabled"] == 0
    with pytest.raises(BroadQaExternalDataError, match="已消费"):
        run_normalization_icu_formal_evaluation(
            run_root=artifacts["root"],
            family_freeze_dir=family,
            publication_dir=publication,
            **arguments,
        )


def test_program_constructor_rejects_public_production_enablement(
        normalization_artifacts,
        ) -> None:
    """candidate clone 永远不能借 formal evaluator 打开公开 production gate。"""
    program = normalization_artifacts["program"]
    with pytest.raises(BroadQaExternalDataError, match="错误启用"):
        replace(program, production_enabled=1)
    assert isinstance(program, NormalizationCandidateCloneProgram)
