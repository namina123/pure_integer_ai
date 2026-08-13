"""normalization v3 来源化 Evidence、记录分账和 pack 严格回读测试。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.hypothesis import EvidenceRecord
from pure_integer_ai.cognition.shared.identity import (
    concept_identity,
    minimal_instruction_identity,
    structure_concept_identity,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_contrastive_protocol import (
    NORMALIZATION_CONTRASTIVE_FAMILY,
    publish_normalization_contrastive_protocol,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_evidence_v3 import (
    normalization_evidence_commitment_from_records,
    normalization_evidence_payload,
    read_normalization_training_provenance,
    validate_normalization_evidence_commitment,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_rule_identity_v3 import (
    normalization_context_defeater,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_rule_pack_v3 import (
    BroadQaNormalizationAcceptedRuleV3,
    BroadQaNormalizationRejectedTrialV3,
    BroadQaNormalizationRuleCandidateV3,
    parse_normalization_accepted_rule_v3,
    parse_normalization_rejected_trial_v3,
    publish_normalization_rule_pack_v3,
    read_normalization_rule_pack_v3,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_source_pack import (
    publish_normalization_source_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_learning_checkpoint import (
    advance_source_inference_learning_checkpoint,
    append_source_inference_learning_checkpoint,
    initial_source_inference_learning_checkpoint,
    parse_source_inference_learning_checkpoint,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


def _training_sources(tmp_path: Path):
    """发布测试来源并选择同一 candidate 的真实正反 trial。"""
    source_pack = tmp_path / "normalization-source-pack"
    publish_normalization_source_pack(
        run_root=tmp_path,
        target_dir=source_pack,
    )
    protocol = tmp_path / "normalization-contrastive-protocol"
    publish_normalization_contrastive_protocol(
        run_root=tmp_path,
        source_pack_dir=source_pack,
        target_dir=protocol,
    )
    source_manifest, protocol_manifest, candidates, trials, item_ids = (
        read_normalization_training_provenance(
            source_pack_dir=source_pack,
            contrastive_protocol_dir=protocol,
        ))
    trials_by_candidate: dict[str, dict[str, dict[str, object]]] = {}
    for trial in trials:
        trials_by_candidate.setdefault(trial["candidate_id"], {})[
            trial["qualification_kind"]] = trial
    candidate = next(
        item for item in candidates
        if set(trials_by_candidate.get(item["candidate_id"], {}))
        == {"SOURCE_REPLAY_SUPPORT", "SOURCE_REPLAY_REFUTE"}
    )
    return (
        source_pack,
        protocol,
        source_manifest,
        protocol_manifest,
        candidate,
        trials_by_candidate[candidate["candidate_id"]],
        item_ids,
        len(candidates),
        len(trials),
    )


def _records(tmp_path: Path):
    """构造一条 accepted SUPPORT 与独立 context REFUTE。"""
    (
        source_pack,
        protocol,
        source_manifest,
        protocol_manifest,
        source_candidate,
        trials,
        item_ids,
        candidate_count,
        trial_count,
    ) = _training_sources(tmp_path)
    rejected_trial = trials["SOURCE_REPLAY_REFUTE"]
    defeater = normalization_context_defeater(rejected_trial["trial_id"])
    candidate = BroadQaNormalizationRuleCandidateV3(
        protocol_manifest["manifest_sha256"],
        source_manifest["manifest_sha256"],
        source_candidate["candidate_id"],
        source_candidate["input_codepoint"],
        source_candidate["output_codepoint"],
        minimal_instruction_identity((817045, 1)),
        1,
        structure_concept_identity((817046, 1)),
        "FORWARD",
        source_candidate["application_domain"],
        (defeater,),
    )
    support = normalization_evidence_commitment_from_records(
        contrastive_protocol_manifest_sha256=(
            protocol_manifest["manifest_sha256"]),
        source_pack_manifest_sha256=source_manifest["manifest_sha256"],
        candidate=source_candidate,
        trial=trials["SOURCE_REPLAY_SUPPORT"],
        hypothesis=candidate.hypothesis(),
    )
    refute = normalization_evidence_commitment_from_records(
        contrastive_protocol_manifest_sha256=(
            protocol_manifest["manifest_sha256"]),
        source_pack_manifest_sha256=source_manifest["manifest_sha256"],
        candidate=source_candidate,
        trial=rejected_trial,
        hypothesis=candidate.trial_hypothesis(rejected_trial["trial_id"]),
    )
    rejected = BroadQaNormalizationRejectedTrialV3(
        candidate,
        rejected_trial["trial_id"],
        defeater,
        hashlib.sha256(canonical_json_bytes(
            candidate.to_dict())).hexdigest(),
        (refute,),
    )
    accepted = BroadQaNormalizationAcceptedRuleV3(
        candidate,
        (support,),
        (rejected.sha256(),),
    )
    return (
        source_pack,
        protocol,
        accepted,
        rejected,
        item_ids,
        candidate_count,
        trial_count,
    )


def _complete_chain(
        root: Path,
        *,
        protocol_sha: str,
        item_ids: tuple[str, ...],
        evidence_count: int,
        rule_count: int,
        suffix: str,
        ) -> Path:
    """形成覆盖完整 TRAIN_SOURCE 的独立 COMPLETE checkpoint 链。"""
    initial = initial_source_inference_learning_checkpoint(
        run_id=hashlib.sha256(suffix.encode("utf-8")).hexdigest(),
        protocol_manifest_sha256=protocol_sha,
        operator_family=NORMALIZATION_CONTRASTIVE_FAMILY,
        training_item_ids=item_ids,
    )
    completed = advance_source_inference_learning_checkpoint(
        initial,
        training_item_ids=item_ids,
        processed_item_ids=item_ids,
        evidence_candidate_count=evidence_count,
        rule_candidate_count=rule_count,
        complete=True,
    )
    path = root / f"{suffix}.checkpoints.jsonl"
    append_source_inference_learning_checkpoint(path, initial)
    append_source_inference_learning_checkpoint(path, completed)
    return path


def _published_pack(tmp_path: Path):
    """发布一个有双独立完成链的 normalization v3 测试 pack。"""
    (
        source_pack,
        protocol,
        accepted,
        rejected,
        item_ids,
        candidate_count,
        trial_count,
    ) = _records(tmp_path)
    protocol_sha = accepted.candidate.contrastive_protocol_manifest_sha256
    fresh = _complete_chain(
        tmp_path,
        protocol_sha=protocol_sha,
        item_ids=item_ids,
        evidence_count=2,
        rule_count=2,
        suffix="normalization-fresh",
    )
    resumed = _complete_chain(
        tmp_path,
        protocol_sha=protocol_sha,
        item_ids=item_ids,
        evidence_count=2,
        rule_count=2,
        suffix="normalization-resumed",
    )
    target = tmp_path / "normalization-rule-pack-v3"
    report = publish_normalization_rule_pack_v3(
        source_pack_dir=source_pack,
        contrastive_protocol_dir=protocol,
        fresh_accepted_rules=(accepted,),
        fresh_rejected_trials=(rejected,),
        resumed_accepted_rules=(accepted,),
        resumed_rejected_trials=(rejected,),
        target_dir=target,
        fresh_checkpoint_chain_path=fresh,
        resumed_checkpoint_chain_path=resumed,
    )
    return (
        source_pack,
        protocol,
        target,
        fresh,
        resumed,
        report,
        accepted,
        rejected,
        candidate_count,
        trial_count,
    )


def test_candidate_hypothesis_and_context_defeater_bind_exact_identity(
        tmp_path: Path,
        ) -> None:
    """mapping 坐标不得共享 hypothesis，trial 也不得借用任意 defeater。"""
    _, _, accepted, rejected, *_ = _records(tmp_path)
    candidate = accepted.candidate
    other = replace(
        candidate,
        mapping_candidate_id="f" * 64,
        output_codepoint=candidate.output_codepoint + 1,
    )
    assert candidate.hypothesis() != other.hypothesis()
    assert (candidate.hypothesis().competition_key
            == other.hypothesis().competition_key)
    assert rejected.context_defeater == normalization_context_defeater(
        rejected.trial_id)
    arbitrary = concept_identity((817047, 1))
    broad_candidate = replace(
        candidate,
        defeaters=tuple(sorted(
            (rejected.context_defeater, arbitrary),
            key=lambda item: item.stable_key(),
        )),
    )
    with pytest.raises(BroadQaExternalDataError, match="defeater/kind"):
        replace(
            rejected,
            candidate=broad_candidate,
            context_defeater=arbitrary,
        )


def test_rule_pack_round_trip_replays_sources_and_complete_chains(
        tmp_path: Path,
        ) -> None:
    """合法 pack 可回读，记录、来源、双链和禁用态逐项闭合。"""
    (
        source_pack,
        protocol,
        target,
        fresh,
        resumed,
        report,
        accepted,
        rejected,
        candidate_count,
        trial_count,
    ) = _published_pack(tmp_path)
    assert parse_normalization_accepted_rule_v3(
        accepted.canonical_bytes()) == accepted
    assert parse_normalization_rejected_trial_v3(
        rejected.canonical_bytes()) == rejected
    manifest, accepted_records, rejected_records = (
        read_normalization_rule_pack_v3(
            target,
            source_pack_dir=source_pack,
            contrastive_protocol_dir=protocol,
            fresh_checkpoint_chain_path=fresh,
            resumed_checkpoint_chain_path=resumed,
        ))
    assert manifest["manifest_sha256"] == report["manifest_sha256"]
    assert accepted_records == (accepted,)
    assert rejected_records == (rejected,)
    assert manifest["production_enabled"] == 0
    assert manifest["runtime_state"] == "LEARNED_PACK_DISABLED"
    assert manifest["training_item_count"] == candidate_count + trial_count
    assert manifest["fresh_run_id"] != manifest["resumed_run_id"]
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_rule_pack_v3(
            source_pack_dir=source_pack,
            contrastive_protocol_dir=protocol,
            fresh_accepted_rules=(accepted,),
            fresh_rejected_trials=(rejected,),
            resumed_accepted_rules=(accepted,),
            resumed_rejected_trials=(rejected,),
            target_dir=target,
            fresh_checkpoint_chain_path=fresh,
            resumed_checkpoint_chain_path=resumed,
        )


def test_pack_rejects_resume_drift_reused_chain_and_runtime_enablement(
        tmp_path: Path,
        ) -> None:
    """fresh/resume 漂移、链复用和生产/identity dispatch 均失败关闭。"""
    (
        source_pack,
        protocol,
        accepted,
        rejected,
        item_ids,
        *_rest,
    ) = _records(tmp_path)
    fresh = _complete_chain(
        tmp_path,
        protocol_sha=accepted.candidate.contrastive_protocol_manifest_sha256,
        item_ids=item_ids,
        evidence_count=2,
        rule_count=2,
        suffix="guard-fresh",
    )
    resumed = _complete_chain(
        tmp_path,
        protocol_sha=accepted.candidate.contrastive_protocol_manifest_sha256,
        item_ids=item_ids,
        evidence_count=2,
        rule_count=2,
        suffix="guard-resumed",
    )
    with pytest.raises(BroadQaExternalDataError, match="字节不等价"):
        publish_normalization_rule_pack_v3(
            source_pack_dir=source_pack,
            contrastive_protocol_dir=protocol,
            fresh_accepted_rules=(accepted,),
            fresh_rejected_trials=(rejected,),
            resumed_accepted_rules=(replace(
                accepted,
                rejection_record_sha256s=("e" * 64,),
            ),),
            resumed_rejected_trials=(rejected,),
            target_dir=tmp_path / "resume-drift",
            fresh_checkpoint_chain_path=fresh,
            resumed_checkpoint_chain_path=resumed,
        )
    with pytest.raises(BroadQaExternalDataError, match="必须独立"):
        publish_normalization_rule_pack_v3(
            source_pack_dir=source_pack,
            contrastive_protocol_dir=protocol,
            fresh_accepted_rules=(accepted,),
            fresh_rejected_trials=(rejected,),
            resumed_accepted_rules=(accepted,),
            resumed_rejected_trials=(rejected,),
            target_dir=tmp_path / "same-chain",
            fresh_checkpoint_chain_path=fresh,
            resumed_checkpoint_chain_path=fresh,
        )
    with pytest.raises(BroadQaExternalDataError, match="生产"):
        replace(accepted, production_enabled=1)
    with pytest.raises(BroadQaExternalDataError, match="identity dispatch"):
        replace(accepted, identity_dispatch=1)


def test_pack_requires_exact_defeater_and_rejection_ledger(
        tmp_path: Path,
        ) -> None:
    """candidate 声明的 defeater 必须逐项对应实际 rejected trial。"""
    (
        source_pack,
        protocol,
        accepted,
        rejected,
        item_ids,
        *_rest,
    ) = _records(tmp_path)
    extra_defeater = concept_identity((817047, 2))
    forged_candidate = replace(
        accepted.candidate,
        defeaters=tuple(sorted(
            (*accepted.candidate.defeaters, extra_defeater),
            key=lambda item: item.stable_key(),
        )),
    )
    forged_rejected = replace(rejected, candidate=forged_candidate)
    forged_accepted = replace(
        accepted,
        candidate=forged_candidate,
        rejection_record_sha256s=(forged_rejected.sha256(),),
    )
    fresh = _complete_chain(
        tmp_path,
        protocol_sha=(
            accepted.candidate.contrastive_protocol_manifest_sha256),
        item_ids=item_ids,
        evidence_count=2,
        rule_count=2,
        suffix="extra-defeater-fresh",
    )
    resumed = _complete_chain(
        tmp_path,
        protocol_sha=(
            accepted.candidate.contrastive_protocol_manifest_sha256),
        item_ids=item_ids,
        evidence_count=2,
        rule_count=2,
        suffix="extra-defeater-resumed",
    )
    with pytest.raises(BroadQaExternalDataError, match="ledger 未精确闭合"):
        publish_normalization_rule_pack_v3(
            source_pack_dir=source_pack,
            contrastive_protocol_dir=protocol,
            fresh_accepted_rules=(forged_accepted,),
            fresh_rejected_trials=(forged_rejected,),
            resumed_accepted_rules=(forged_accepted,),
            resumed_rejected_trials=(forged_rejected,),
            target_dir=tmp_path / "extra-defeater-pack",
            fresh_checkpoint_chain_path=fresh,
            resumed_checkpoint_chain_path=resumed,
        )


def test_pack_replays_physical_source_after_synchronized_record_tamper(
        tmp_path: Path,
        ) -> None:
    """同步改 Evidence、record 和 manifest/result 仍不能绕过物理来源重放。"""
    (
        source_pack,
        protocol,
        target,
        fresh,
        resumed,
        _report,
        accepted,
        _rejected,
        *_counts,
    ) = _published_pack(tmp_path)
    commitment = accepted.evidence_commitments[0]
    evidence = EvidenceRecord.from_stable_key(commitment.evidence_key)
    forged_offset = commitment.source_codepoint_offset + 1
    forged_evidence = replace(
        evidence,
        payload=normalization_evidence_payload(
            candidate_id=commitment.candidate_id,
            trial_id=commitment.trial_id,
            source_codepoint_offset=forged_offset,
            input_codepoint=commitment.input_codepoint,
            candidate_output_codepoint=(
                commitment.candidate_output_codepoint),
            observed_output_codepoint=commitment.observed_output_codepoint,
        ),
    )
    forged_commitment = replace(
        commitment,
        source_codepoint_offset=forged_offset,
        evidence_key=forged_evidence.stable_key(),
    )
    forged_rule = replace(
        accepted,
        evidence_commitments=(forged_commitment,),
    )
    accepted_payload = forged_rule.canonical_bytes()
    (target / "accepted-rules.jsonl").write_bytes(accepted_payload)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["accepted_records_bytes"] = len(accepted_payload)
    manifest["accepted_records_sha256"] = hashlib.sha256(
        accepted_payload).hexdigest()
    manifest["fresh_result_sha256"] = "d" * 64
    manifest["resumed_result_sha256"] = "d" * 64
    manifest_path.write_bytes(canonical_json_line(manifest))
    with pytest.raises(BroadQaExternalDataError, match="source replay"):
        read_normalization_rule_pack_v3(
            target,
            source_pack_dir=source_pack,
            contrastive_protocol_dir=protocol,
            fresh_checkpoint_chain_path=fresh,
            resumed_checkpoint_chain_path=resumed,
        )


def test_rejected_trial_refuses_wrong_trial_defeater_and_stance(
        tmp_path: Path,
        ) -> None:
    """错误 trial defeater 或把 REFUTE 改写为 SUPPORT 都不能形成记录。"""
    _, _, _accepted, rejected, *_ = _records(tmp_path)
    with pytest.raises(BroadQaExternalDataError, match="defeater/kind"):
        replace(
            rejected,
            context_defeater=normalization_context_defeater("c" * 64),
        )
    commitment = rejected.evidence_commitments[0]
    evidence = EvidenceRecord.from_stable_key(commitment.evidence_key)
    with pytest.raises(BroadQaExternalDataError, match="Evidence event"):
        replace(
            rejected,
            evidence_commitments=(replace(
                commitment,
                evidence_key=replace(evidence, stance=1).stable_key(),
            ),),
        )


def test_evidence_replay_rejects_wrong_candidate_trial_pair(
        tmp_path: Path,
        ) -> None:
    """真实但不匹配的 candidate/trial 身份不能拼成来源 Evidence。"""
    (
        _source_pack,
        _protocol,
        source_manifest,
        protocol_manifest,
        candidate,
        trials,
        *_rest,
    ) = _training_sources(tmp_path)
    _candidates, all_trials = read_normalization_training_provenance(
        source_pack_dir=tmp_path / "normalization-source-pack",
        contrastive_protocol_dir=(
            tmp_path / "normalization-contrastive-protocol"),
    )[2:4]
    wrong_trial = next(
        trial for trial in all_trials
        if trial["candidate_id"] != candidate["candidate_id"]
    )
    with pytest.raises(BroadQaExternalDataError, match="identity"):
        normalization_evidence_commitment_from_records(
            contrastive_protocol_manifest_sha256=(
                protocol_manifest["manifest_sha256"]),
            source_pack_manifest_sha256=source_manifest["manifest_sha256"],
            candidate=candidate,
            trial=wrong_trial,
            hypothesis=BroadQaNormalizationRuleCandidateV3(
                protocol_manifest["manifest_sha256"],
                source_manifest["manifest_sha256"],
                candidate["candidate_id"],
                candidate["input_codepoint"],
                candidate["output_codepoint"],
                minimal_instruction_identity((817045, 1)),
                1,
                structure_concept_identity((817046, 1)),
                "FORWARD",
                candidate["application_domain"],
                (normalization_context_defeater(
                    trials["SOURCE_REPLAY_REFUTE"]["trial_id"]),),
            ).hypothesis(),
        )


def test_evidence_replay_rejects_physical_line_span_drift(
        tmp_path: Path,
        ) -> None:
    """规范记录中的合法形状 span 也必须与冻结物理字节完全一致。"""
    (
        source_pack,
        protocol,
        accepted,
        _rejected,
        *_rest,
    ) = _records(tmp_path)
    commitment = accepted.evidence_commitments[0]
    drifted_source = replace(
        commitment.candidate_source,
        byte_start=commitment.candidate_source.byte_start + 1,
        byte_end=commitment.candidate_source.byte_end + 1,
    )
    drifted = replace(commitment, candidate_source=drifted_source)
    (
        source_manifest,
        protocol_manifest,
        candidates,
        trials,
        _item_ids,
    ) = read_normalization_training_provenance(
        source_pack_dir=source_pack,
        contrastive_protocol_dir=protocol,
    )
    with pytest.raises(BroadQaExternalDataError, match="source replay"):
        validate_normalization_evidence_commitment(
            drifted,
            protocol_manifest_sha256=protocol_manifest["manifest_sha256"],
            source_pack_manifest_sha256=source_manifest["manifest_sha256"],
            candidate_by_id={item["candidate_id"]: item for item in candidates},
            trial_by_id={item["trial_id"]: item for item in trials},
            expected_hypothesis=accepted.candidate.hypothesis(),
            expected_qualification="SOURCE_REPLAY_SUPPORT",
        )


def test_rule_pack_rejects_checkpoint_chain_pointer_drift(
        tmp_path: Path,
        ) -> None:
    """checkpoint 内容保持规范时，前驱 SHA 指针漂移仍须失败关闭。"""
    (
        source_pack,
        protocol,
        target,
        fresh,
        resumed,
        *_rest,
    ) = _published_pack(tmp_path)
    lines = fresh.read_bytes().splitlines(keepends=True)
    terminal = parse_source_inference_learning_checkpoint(lines[-1])
    lines[-1] = replace(
        terminal,
        previous_checkpoint_sha256="a" * 64,
    ).canonical_bytes()
    fresh.write_bytes(b"".join(lines))
    with pytest.raises(BroadQaExternalDataError, match="chain"):
        read_normalization_rule_pack_v3(
            target,
            source_pack_dir=source_pack,
            contrastive_protocol_dir=protocol,
            fresh_checkpoint_chain_path=fresh,
            resumed_checkpoint_chain_path=resumed,
        )


def test_rule_pack_recomputes_every_checkpoint_training_prefix(
        tmp_path: Path,
        ) -> None:
    """同步修正后继指针也不能掩盖中间 revision 的 TRAIN 前缀漂移。"""
    (
        source_pack,
        protocol,
        accepted,
        rejected,
        item_ids,
        *_rest,
    ) = _records(tmp_path)
    protocol_sha = accepted.candidate.contrastive_protocol_manifest_sha256
    initial = initial_source_inference_learning_checkpoint(
        run_id=hashlib.sha256(b"prefix-drift-fresh").hexdigest(),
        protocol_manifest_sha256=protocol_sha,
        operator_family=NORMALIZATION_CONTRASTIVE_FAMILY,
        training_item_ids=item_ids,
    )
    midpoint = len(item_ids) // 2
    middle = advance_source_inference_learning_checkpoint(
        initial,
        training_item_ids=item_ids,
        processed_item_ids=item_ids[:midpoint],
        evidence_candidate_count=1,
        rule_candidate_count=1,
    )
    complete = advance_source_inference_learning_checkpoint(
        middle,
        training_item_ids=item_ids,
        processed_item_ids=item_ids,
        evidence_candidate_count=2,
        rule_candidate_count=2,
        complete=True,
    )
    drifted_middle = replace(
        middle,
        processed_item_prefix_sha256="b" * 64,
    )
    synchronized_complete = replace(
        complete,
        previous_checkpoint_sha256=drifted_middle.sha256(),
    )
    fresh = tmp_path / "prefix-drift-fresh.checkpoints.jsonl"
    fresh.write_bytes(b"".join((
        initial.canonical_bytes(),
        drifted_middle.canonical_bytes(),
        synchronized_complete.canonical_bytes(),
    )))
    resumed = _complete_chain(
        tmp_path,
        protocol_sha=protocol_sha,
        item_ids=item_ids,
        evidence_count=2,
        rule_count=2,
        suffix="prefix-drift-resumed",
    )
    with pytest.raises(BroadQaExternalDataError, match="TRAIN prefix"):
        publish_normalization_rule_pack_v3(
            source_pack_dir=source_pack,
            contrastive_protocol_dir=protocol,
            fresh_accepted_rules=(accepted,),
            fresh_rejected_trials=(rejected,),
            resumed_accepted_rules=(accepted,),
            resumed_rejected_trials=(rejected,),
            target_dir=tmp_path / "prefix-drift-pack",
            fresh_checkpoint_chain_path=fresh,
            resumed_checkpoint_chain_path=resumed,
        )
