"""来源归纳 learning protocol、checkpoint 与 rule pack 的边界测试。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    structure_concept_identity,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
    normalize_external_text,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_contract import (
    source_inference_rule_hypothesis_key,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_learning_checkpoint import (
    advance_source_inference_learning_checkpoint,
    append_source_inference_learning_checkpoint,
    initial_source_inference_learning_checkpoint,
    parse_source_inference_learning_checkpoint,
    read_source_inference_learning_chain,
    require_source_inference_fresh_resume_equivalence,
    source_inference_learning_result_sha256,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_learning_protocol import (
    SOURCE_INFERENCE_BLOCKED_FAMILIES,
    SOURCE_INFERENCE_LEARNING_FAMILIES,
    publish_source_inference_learning_protocol,
    read_source_inference_learning_protocol,
    read_source_inference_learning_slice,
    source_inference_learning_split,
    source_inference_protocol_scope,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_rule_pack import (
    BroadQaSourceInferenceLearnedRule,
    BroadQaSourceInferenceRuleEvidenceCommitment,
    SOURCE_INFERENCE_EVIDENCE_REASON_KEYS,
    SOURCE_INFERENCE_RULE_APPLICATION_DOMAINS,
    SOURCE_INFERENCE_TERMINAL_DOCUMENT_SOURCE_KIND,
    parse_source_inference_learned_rule,
    publish_source_inference_rule_pack,
    read_source_inference_rule_pack,
    source_inference_qualification_input_sha256,
    source_inference_rule_pack_result_sha256,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_training_census import (
    SOURCE_INFERENCE_TRAINING_CENSUS_KIND,
    SOURCE_INFERENCE_TRAINING_CENSUS_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_training_dossier import (
    SOURCE_INFERENCE_TRAINING_DOSSIER_KIND,
    SOURCE_INFERENCE_TRAINING_DOSSIER_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


_ALL_FAMILIES = (
    SOURCE_INFERENCE_LEARNING_FAMILIES + SOURCE_INFERENCE_BLOCKED_FAMILIES)


def _sha(path: Path) -> str:
    """返回测试 artifact 的 SHA-256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _item_ids() -> tuple[str, ...]:
    """确定性寻找覆盖三种 split 且 TRAIN 至少四条的测试 identity。"""
    values = []
    counts = {"TRAIN": 0, "VALIDATION": 0, "RESERVE": 0}
    ordinal = 1
    while (counts["TRAIN"] < 4 or counts["VALIDATION"] < 2
           or counts["RESERVE"] < 2):
        item_id = format(ordinal, "064x")
        split = source_inference_learning_split(item_id)
        if ((split == "TRAIN" and counts[split] < 4)
                or (split != "TRAIN" and counts[split] < 2)):
            values.append(item_id)
            counts[split] += 1
        ordinal += 1
    return tuple(values)


def _dossier_record(item_id: str, ordinal: int) -> dict[str, object]:
    """构造严格 reader 可回读的训练 dossier record。"""
    title = f"协议示例页{ordinal}"
    question = f"协议问题{ordinal}？"
    context = f"旧上下文{ordinal}"
    wikitext = f"甲{ordinal}、乙{ordinal}。"
    passage = wikitext
    return {
        "format_version": 1,
        "item_id": item_id,
        "record_kind": SOURCE_INFERENCE_TRAINING_DOSSIER_RECORD_KIND,
        "roster_commitment": {
            "question_sha256": hashlib.sha256(
                question.encode("utf-8")).hexdigest(),
            "source_alignment_status": "SOURCE_ALIGNED",
            "title_key": title,
        },
        "terminal_source": {
            "attribution": "Wikipedia contributors",
            "contributor": {"id": ordinal},
            "license_id": "CC-BY-SA-4.0",
            "page_id": 1000 + ordinal,
            "passages": [{
                "ordinal": 1,
                "raw_end": len(passage),
                "raw_sha256": hashlib.sha256(
                    passage.encode("utf-8")).hexdigest(),
                "raw_start": 0,
                "section_title": "",
                "text": passage,
                "text_sha256": hashlib.sha256(
                    passage.encode("utf-8")).hexdigest(),
            }],
            "plain_text": passage,
            "plain_text_sha256": hashlib.sha256(
                passage.encode("utf-8")).hexdigest(),
            "revision_id": 2000 + ordinal,
            "revision_timestamp": "2026-07-01T00:00:00Z",
            "snapshot_id": "zhwiki-20260701",
            "source_url": "https://example.test/page",
            "title": title,
            "wikitext": wikitext,
            "wikitext_sha256": hashlib.sha256(
                wikitext.encode("utf-8")).hexdigest(),
        },
        "training_assignment": "EXTRACTIVE_REFERENCE",
        "training_source": {
            "context": context,
            "context_sha256": hashlib.sha256(
                context.encode("utf-8")).hexdigest(),
            "gold_answers": [f"乙{ordinal}"],
            "license_id": "CC-BY-SA-4.0",
            "question": question,
            "source_key": "CMRC2018" if ordinal % 2 else "DRCD",
            "source_partition": "train",
            "source_question_id": f"q-{ordinal}",
            "source_revision": "revision",
            "title": title,
            "upstream_url": "https://example.test/source",
        },
    }


def _protocol_inputs(
        root: Path,
        *,
        mechanical_reason: str = "TEST_MECHANICAL_SIGNAL",
        ):
    """发布协议所需的 dossier/census 测试输入。"""
    item_ids = _item_ids()
    dossier_records = tuple(
        _dossier_record(item_id, ordinal)
        for ordinal, item_id in enumerate(item_ids, start=1))
    dossier = root / "training.dossier.jsonl"
    dossier.write_bytes(b"".join(
        canonical_json_line(item) for item in dossier_records))
    dossier_manifest = root / "dossier.manifest.json"
    dossier_manifest.write_bytes(canonical_json_line({
        "artifact_kind": SOURCE_INFERENCE_TRAINING_DOSSIER_KIND,
        "dossier_bytes": dossier.stat().st_size,
        "dossier_record_count": len(dossier_records),
        "dossier_sha256": _sha(dossier),
        "format_version": 1,
        "learner_read_count": 0,
        "rules_written": 0,
        "semantic_labels_written": 0,
        "status": "MATERIALIZED_UNREAD_UNLEARNED",
    }))

    train_index = 0
    census_records = []
    for record in dossier_records:
        split = source_inference_learning_split(record["item_id"])
        if split == "TRAIN":
            train_index += 1
        for family in _ALL_FAMILIES:
            if family in SOURCE_INFERENCE_LEARNING_FAMILIES:
                if split == "TRAIN":
                    state = (
                        "MECHANICAL_SUPPORT_SIGNAL"
                        if train_index % 2 else "MECHANICAL_COUNTER_SIGNAL")
                else:
                    state = "UNDETERMINED"
            else:
                state = "UNDETERMINED"
            census_records.append({
                "format_version": 1,
                "item_id": record["item_id"],
                "mechanical_reason": mechanical_reason,
                "mechanical_signal_state": state,
                "operator_family": family,
                "record_kind": SOURCE_INFERENCE_TRAINING_CENSUS_RECORD_KIND,
                "rules_written": 0,
                "semantic_label_written": 0,
                "source_key": record["training_source"]["source_key"],
                "training_assignment": record["training_assignment"],
            })
    census = root / "operator-census.records.jsonl"
    census.write_bytes(b"".join(
        canonical_json_line(item) for item in census_records))
    census_manifest = root / "census.manifest.json"
    census_manifest.write_bytes(canonical_json_line({
        "artifact_kind": SOURCE_INFERENCE_TRAINING_CENSUS_KIND,
        "dossier_manifest_sha256": _sha(dossier_manifest),
        "dossier_sha256": _sha(dossier),
        "format_version": 1,
        "item_count": len(dossier_records),
        "learner_read_count": 0,
        "operator_preassigned_count": 0,
        "record_count": len(census_records),
        "records_sha256": _sha(census),
        "rules_written": 0,
        "semantic_labels_written": 0,
        "status": "MECHANICAL_CENSUS_ONLY_NOT_LEARNED",
    }))
    return dossier_manifest, dossier, census_manifest, census, dossier_records


def _publish_protocol(root: Path) -> tuple[Path, tuple[dict[str, object], ...]]:
    """发布并返回一个覆盖三 split 的测试协议。"""
    inputs = _protocol_inputs(root)
    target = root / "learning-protocol"
    publish_source_inference_learning_protocol(
        run_root=root,
        dossier_manifest_path=inputs[0],
        dossier_path=inputs[1],
        census_manifest_path=inputs[2],
        census_records_path=inputs[3],
        target_dir=target,
    )
    return target, inputs[4]


def test_protocol_freezes_physical_slices_and_reserve_identity_only(
        tmp_path: Path,
        ) -> None:
    """全局 split 唯一，两个角色只能读各自物理切片，RESERVE 无 payload。"""
    target, records = _publish_protocol(tmp_path)
    manifest = read_source_inference_learning_protocol(
        target / "manifest.json")
    expected_counts = {
        split: sum(source_inference_learning_split(item["item_id"]) == split
                   for item in records)
        for split in ("TRAIN", "VALIDATION", "RESERVE")
    }
    assert manifest["split_item_counts"] == expected_counts
    assert manifest["reserve_payload_published"] == 0
    learner_dossier, learner_census = read_source_inference_learning_slice(
        protocol_dir=target,
        access_role="LEARNER",
        operator_family="NORMALIZATION_EQUIVALENCE",
    )
    evaluator_dossier, evaluator_census = read_source_inference_learning_slice(
        protocol_dir=target,
        access_role="EVALUATOR",
        operator_family="NORMALIZATION_EQUIVALENCE",
    )
    assert all(source_inference_learning_split(item["item_id"]) == "TRAIN"
               for item in learner_dossier)
    assert all(source_inference_learning_split(item["item_id"]) == "VALIDATION"
               for item in evaluator_dossier)
    assert {item["item_id"] for item in learner_dossier}.isdisjoint(
        item["item_id"] for item in evaluator_dossier)
    assert len(learner_dossier) == len(learner_census)
    assert len(evaluator_dossier) == len(evaluator_census)
    reserve_lines = (
        target / "reserve" / "reserve.identity.jsonl").read_text(
            encoding="utf-8").splitlines()
    assert all(set(json.loads(line)) == {
        "format_version", "item_id", "record_kind", "split"}
        for line in reserve_lines)


def test_protocol_rejects_unknown_fields_tamper_overwrite_and_bad_access(
        tmp_path: Path,
        ) -> None:
    """manifest/切片篡改、覆盖、未知角色或 family 均失败关闭。"""
    target, _ = _publish_protocol(tmp_path)
    train_dossier = target / "learner" / "train.dossier.jsonl"
    original_dossier = train_dossier.read_bytes()
    train_dossier.write_bytes(original_dossier + b"\n")
    with pytest.raises(BroadQaExternalDataError, match="commitment 漂移"):
        read_source_inference_learning_slice(
            protocol_dir=target,
            access_role="LEARNER",
            operator_family="NORMALIZATION_EQUIVALENCE",
        )
    train_dossier.write_bytes(original_dossier)

    manifest_path = target / "manifest.json"
    value = json.loads(manifest_path.read_bytes())
    value["reserve_payload_published"] = False
    manifest_path.write_bytes(canonical_json_line(value))
    with pytest.raises(BroadQaExternalDataError, match="manifest 漂移"):
        read_source_inference_learning_protocol(manifest_path)
    value["reserve_payload_published"] = 0
    value["mechanical_signal_usage"]["candidate_routing_allowed"] = True
    manifest_path.write_bytes(canonical_json_line(value))
    with pytest.raises(BroadQaExternalDataError, match="manifest 漂移"):
        read_source_inference_learning_protocol(manifest_path)
    value["mechanical_signal_usage"]["candidate_routing_allowed"] = 1
    value["unexpected"] = 1
    manifest_path.write_bytes(canonical_json_line(value))
    with pytest.raises(BroadQaExternalDataError, match="manifest 漂移"):
        read_source_inference_learning_protocol(manifest_path)

    other_root = tmp_path / "other"
    other_root.mkdir()
    inputs = _protocol_inputs(other_root)
    exists = other_root / "exists"
    exists.mkdir()
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_source_inference_learning_protocol(
            run_root=other_root,
            dossier_manifest_path=inputs[0],
            dossier_path=inputs[1],
            census_manifest_path=inputs[2],
            census_records_path=inputs[3],
            target_dir=exists,
        )
    with pytest.raises(BroadQaExternalDataError, match="access role"):
        read_source_inference_learning_slice(
            protocol_dir=target,
            access_role="RESERVE",
            operator_family="NORMALIZATION_EQUIVALENCE",
        )
    with pytest.raises(BroadQaExternalDataError, match="未启用"):
        read_source_inference_learning_slice(
            protocol_dir=target,
            access_role="LEARNER",
            operator_family="PARENTHETICAL_EXPANSION",
        )


def test_checkpoint_chain_and_fresh_resume_equivalence_are_strict(
        tmp_path: Path,
        ) -> None:
    """checkpoint 连续追加、规范回读和 fresh/resume 等价均为硬门。"""
    run_id = "1" * 64
    protocol_sha = "2" * 64
    items = ("3" * 64, "4" * 64)
    initial = initial_source_inference_learning_checkpoint(
        run_id=run_id,
        protocol_manifest_sha256=protocol_sha,
        operator_family="NORMALIZATION_EQUIVALENCE",
        training_item_ids=items,
    )
    advanced = advance_source_inference_learning_checkpoint(
        initial,
        training_item_ids=items,
        processed_item_ids=items[:1],
        evidence_candidate_count=1,
        rule_candidate_count=0,
    )
    completed = advance_source_inference_learning_checkpoint(
        advanced,
        training_item_ids=items,
        processed_item_ids=items,
        evidence_candidate_count=2,
        rule_candidate_count=1,
        complete=True,
    )
    chain = tmp_path / "checkpoints.jsonl"
    for checkpoint in (initial, advanced, completed):
        append_source_inference_learning_checkpoint(chain, checkpoint)
    assert read_source_inference_learning_chain(chain) == (
        initial, advanced, completed)
    assert parse_source_inference_learning_checkpoint(
        completed.canonical_bytes()) == completed
    result = source_inference_learning_result_sha256(
        protocol_manifest_sha256=protocol_sha,
        operator_family="NORMALIZATION_EQUIVALENCE",
        processed_item_ids=items,
        evidence_record_sha256s=("5" * 64, "6" * 64),
        rule_record_sha256s=("7" * 64,),
    )
    require_source_inference_fresh_resume_equivalence(result, result)
    with pytest.raises(BroadQaExternalDataError, match="不等价"):
        require_source_inference_fresh_resume_equivalence(result, "8" * 64)
    with pytest.raises(BroadQaExternalDataError, match="冻结 TRAIN 有序前缀"):
        advance_source_inference_learning_checkpoint(
            advanced,
            training_item_ids=items,
            processed_item_ids=("9" * 64, items[1]),
            evidence_candidate_count=2,
            rule_candidate_count=1,
        )
    with pytest.raises(BroadQaExternalDataError, match="完整 TRAIN"):
        advance_source_inference_learning_checkpoint(
            advanced,
            training_item_ids=items,
            processed_item_ids=items[:1],
            evidence_candidate_count=2,
            rule_candidate_count=1,
            complete=True,
        )
    with pytest.raises(BroadQaExternalDataError, match="TRAIN identity 漂移"):
        advance_source_inference_learning_checkpoint(
            advanced,
            training_item_ids=(items[0], "a" * 64),
            processed_item_ids=items[:1],
            evidence_candidate_count=2,
            rule_candidate_count=1,
        )

    drift = replace(completed, run_id="a" * 64)
    with pytest.raises(BroadQaExternalDataError, match="运行身份漂移"):
        append_source_inference_learning_checkpoint(chain, drift)
    drift = replace(completed, operator_family="SOURCE_SPAN_SELECTION")
    with pytest.raises(BroadQaExternalDataError, match="运行身份漂移"):
        append_source_inference_learning_checkpoint(chain, drift)

    lines = chain.read_bytes().splitlines(keepends=True)
    tampered = json.loads(lines[1])
    tampered["previous_checkpoint_sha256"] = "9" * 64
    chain.write_bytes(lines[0] + canonical_json_line(tampered) + lines[2])
    with pytest.raises(BroadQaExternalDataError, match="断裂"):
        read_source_inference_learning_chain(chain)
    with pytest.raises(BroadQaExternalDataError, match="字段漂移"):
        parse_source_inference_learning_checkpoint(
            canonical_json_line({**initial.to_dict(), "extra": 1}))


def _commitment(
        *,
        dossier: dict[str, object],
        routing_state: str,
        qualification_kind: str,
        evidence_id: int,
        operator_family: str,
        hypothesis,
        ) -> BroadQaSourceInferenceRuleEvidenceCommitment:
    """从物理 dossier passage 构造来源化规则 Evidence 承诺。"""
    terminal = dossier["terminal_source"]
    passage = terminal["passages"][0]
    gold = dossier["training_source"]["gold_answers"][0]
    support_start = terminal["wikitext"].find(
        gold, passage["raw_start"], passage["raw_end"])
    assert support_start >= 0
    if qualification_kind == "REPLAYED_CANDIDATE_SUPPORT":
        candidate_start = support_start
        candidate_end = support_start + len(gold)
    else:
        candidate_start = passage["raw_start"]
        candidate_end = support_start
        assert candidate_start < candidate_end
    candidate_raw = terminal["wikitext"][candidate_start:candidate_end]
    expected_sha = hashlib.sha256(
        normalize_external_text(gold).encode("utf-8")).hexdigest()
    observed_sha = hashlib.sha256(
        normalize_external_text(candidate_raw).encode("utf-8")).hexdigest()
    source = SourceRef(
        SOURCE_INFERENCE_TERMINAL_DOCUMENT_SOURCE_KIND,
        terminal["page_id"],
        terminal["revision_id"],
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )
    evidence = EvidenceRecord(
        evidence_id,
        hypothesis,
        (EVIDENCE_SUPPORT if qualification_kind == "REPLAYED_CANDIDATE_SUPPORT"
         else EVIDENCE_REFUTE),
        SOURCE_INFERENCE_EVIDENCE_REASON_KEYS[
            operator_family][qualification_kind],
        source,
        evidence_id,
    )
    return BroadQaSourceInferenceRuleEvidenceCommitment(
        dossier["item_id"],
        dossier["training_source"]["source_key"],
        source,
        terminal["page_id"],
        terminal["revision_id"],
        passage["ordinal"],
        passage["raw_start"],
        passage["raw_end"],
        passage["raw_sha256"],
        candidate_start,
        candidate_end,
        hashlib.sha256(candidate_raw.encode("utf-8")).hexdigest(),
        routing_state,
        qualification_kind,
        source_inference_qualification_input_sha256(operator_family, dossier),
        expected_sha,
        observed_sha,
        evidence.stable_key(),
    )


def _rule(
        protocol: Path,
        records: tuple[dict[str, object], ...],
        *,
        validation_support: bool = False,
        ) -> BroadQaSourceInferenceLearnedRule:
    """构造同时带支持和反驳 Evidence 的单 family 测试规则。"""
    manifest = read_source_inference_learning_protocol(
        protocol / "manifest.json")
    protocol_sha = manifest["manifest_sha256"]
    operator = minimal_instruction_identity((817034, 1))
    schema = structure_concept_identity((817035, 1))
    hypothesis = source_inference_rule_hypothesis_key(
        operator,
        schema,
        "FORWARD",
        1,
        source_inference_protocol_scope(protocol_sha),
    )
    train = [item for item in records
             if source_inference_learning_split(item["item_id"]) == "TRAIN"]
    validation = [item for item in records
                  if source_inference_learning_split(item["item_id"])
                  == "VALIDATION"]
    support_record = validation[0] if validation_support else train[1]
    support = _commitment(
        dossier=support_record,
        routing_state=("UNDETERMINED" if validation_support
                       else "MECHANICAL_COUNTER_SIGNAL"),
        qualification_kind="REPLAYED_CANDIDATE_SUPPORT",
        evidence_id=1,
        operator_family="NORMALIZATION_EQUIVALENCE",
        hypothesis=hypothesis,
    )
    counter = _commitment(
        dossier=train[0],
        routing_state="MECHANICAL_SUPPORT_SIGNAL",
        qualification_kind="REPLAYED_CANDIDATE_REFUTE",
        evidence_id=2,
        operator_family="NORMALIZATION_EQUIVALENCE",
        hypothesis=hypothesis,
    )
    commitments = tuple(sorted(
        (support, counter), key=lambda item: item.evidence_key))
    return BroadQaSourceInferenceLearnedRule(
        protocol_sha,
        "NORMALIZATION_EQUIVALENCE",
        operator,
        1,
        schema,
        "FORWARD",
        SOURCE_INFERENCE_RULE_APPLICATION_DOMAINS[
            "NORMALIZATION_EQUIVALENCE"],
        (concept_identity((817036, 1)),),
        commitments,
    )


def _completed_chain(
        protocol: Path,
        rule: BroadQaSourceInferenceLearnedRule,
        *,
        suffix: str,
        ) -> tuple[Path, str]:
    """为测试规则形成绑定完整 TRAIN 顺序的 append-only 完成链。"""
    manifest = read_source_inference_learning_protocol(
        protocol / "manifest.json")
    dossier, _ = read_source_inference_learning_slice(
        protocol_dir=protocol,
        access_role="LEARNER",
        operator_family=rule.operator_family,
    )
    training_item_ids = tuple(item["item_id"] for item in dossier)
    initial = initial_source_inference_learning_checkpoint(
        run_id=hashlib.sha256(suffix.encode("utf-8")).hexdigest(),
        protocol_manifest_sha256=manifest["manifest_sha256"],
        operator_family=rule.operator_family,
        training_item_ids=training_item_ids,
    )
    completed = advance_source_inference_learning_checkpoint(
        initial,
        training_item_ids=training_item_ids,
        processed_item_ids=training_item_ids,
        evidence_candidate_count=len(rule.evidence_commitments),
        rule_candidate_count=1,
        complete=True,
    )
    chain = protocol.parent / f"{suffix}.checkpoints.jsonl"
    append_source_inference_learning_checkpoint(chain, initial)
    append_source_inference_learning_checkpoint(chain, completed)
    result = source_inference_rule_pack_result_sha256(
        protocol_manifest_sha256=manifest["manifest_sha256"],
        operator_family=rule.operator_family,
        training_item_ids=training_item_ids,
        rules=(rule,),
    )
    return chain, result


def _two_completed_chains(
        protocol: Path,
        rule: BroadQaSourceInferenceLearnedRule,
        *,
        suffix: str,
        ) -> tuple[Path, Path]:
    """构造彼此独立的 fresh/resume 完成链。"""
    fresh, _ = _completed_chain(protocol, rule, suffix=f"{suffix}-fresh")
    resumed, _ = _completed_chain(protocol, rule, suffix=f"{suffix}-resumed")
    return fresh, resumed


def test_rule_pack_round_trip_binds_protocol_evidence_and_train_spans(
        tmp_path: Path,
        ) -> None:
    """合法 pack 保留协议 scope、正反 Evidence、TRAIN span 且默认禁用。"""
    protocol, records = _publish_protocol(tmp_path)
    rule = _rule(protocol, records)
    fresh_chain, resumed_chain = _two_completed_chains(
        protocol, rule, suffix="round-trip")
    restored = parse_source_inference_learned_rule(rule.canonical_bytes())
    assert restored == rule
    assert restored.production_enabled == 0
    assert restored.runtime_state == "LEARNED_PACK_DISABLED"
    assert {
        item.routing_signal_state: item.qualification_kind
        for item in restored.evidence_commitments
    } == {
        "MECHANICAL_COUNTER_SIGNAL": "REPLAYED_CANDIDATE_SUPPORT",
        "MECHANICAL_SUPPORT_SIGNAL": "REPLAYED_CANDIDATE_REFUTE",
    }
    target = tmp_path / "rule-pack"
    report = publish_source_inference_rule_pack(
        protocol_dir=protocol,
        operator_family="NORMALIZATION_EQUIVALENCE",
        fresh_rules=(rule,),
        resumed_rules=(rule,),
        target_dir=target,
        fresh_checkpoint_chain_path=fresh_chain,
        resumed_checkpoint_chain_path=resumed_chain,
    )
    manifest, rules = read_source_inference_rule_pack(target)
    assert report["records_sha256"] == manifest["records_sha256"]
    assert rules == (rule,)
    assert manifest["status"] == "FROZEN_NOT_EVALUATED_NOT_DEPLOYED"
    assert manifest["fresh_checkpoint_chain_sha256"] == _sha(fresh_chain)
    assert manifest["resumed_checkpoint_chain_sha256"] == _sha(resumed_chain)
    assert manifest["fresh_run_id"] != manifest["resumed_run_id"]
    assert manifest["training_item_count"] == 4
    with pytest.raises(BroadQaExternalDataError, match="必须独立"):
        publish_source_inference_rule_pack(
            protocol_dir=protocol,
            operator_family="NORMALIZATION_EQUIVALENCE",
            fresh_rules=(rule,),
            resumed_rules=(rule,),
            target_dir=tmp_path / "same-chain",
            fresh_checkpoint_chain_path=fresh_chain,
            resumed_checkpoint_chain_path=fresh_chain,
        )

    refute = next(item for item in rule.evidence_commitments
                   if item.qualification_kind == "REPLAYED_CANDIDATE_REFUTE")
    forged_refute = replace(
        refute, qualification_observed_sha256="e" * 64)
    forged_rule = replace(rule, evidence_commitments=tuple(sorted(
        (forged_refute,) + tuple(
            item for item in rule.evidence_commitments if item != refute),
        key=lambda item: item.evidence_key,
    )))
    with pytest.raises(BroadQaExternalDataError, match="commitment 漂移"):
        publish_source_inference_rule_pack(
            protocol_dir=protocol,
            operator_family="NORMALIZATION_EQUIVALENCE",
            fresh_rules=(forged_rule,),
            resumed_rules=(forged_rule,),
            target_dir=tmp_path / "forged-replay",
            fresh_checkpoint_chain_path=fresh_chain,
            resumed_checkpoint_chain_path=resumed_chain,
        )


def test_rule_pack_rejects_protocol_drift_dispatch_and_validation_leakage(
        tmp_path: Path,
        ) -> None:
    """错误 protocol、逐题 dispatch 与 VALIDATION Evidence 均不能发布。"""
    protocol, records = _publish_protocol(tmp_path)
    rule = _rule(protocol, records)
    fresh_chain, resumed_chain = _two_completed_chains(
        protocol, rule, suffix="rejections")
    with pytest.raises(BroadQaExternalDataError, match="字节不等价"):
        publish_source_inference_rule_pack(
            protocol_dir=protocol,
            operator_family="NORMALIZATION_EQUIVALENCE",
            fresh_rules=(rule,),
            resumed_rules=(replace(
                rule, defeaters=(concept_identity((817036, 2)),)),),
            target_dir=tmp_path / "bad-result",
            fresh_checkpoint_chain_path=fresh_chain,
            resumed_checkpoint_chain_path=resumed_chain,
        )
    with pytest.raises(BroadQaExternalDataError, match="dispatch"):
        replace(rule, item_identity_dispatch=1)
    with pytest.raises(BroadQaExternalDataError, match="协议来源"):
        replace(rule, protocol_manifest_sha256="c" * 64)
    with pytest.raises(BroadQaExternalDataError, match="family/protocol"):
        publish_source_inference_rule_pack(
            protocol_dir=protocol,
            operator_family="SOURCE_SPAN_SELECTION",
            fresh_rules=(rule,),
            resumed_rules=(rule,),
            target_dir=tmp_path / "cross-family",
            fresh_checkpoint_chain_path=fresh_chain,
            resumed_checkpoint_chain_path=resumed_chain,
        )

    other_root = tmp_path / "other-protocol"
    other_root.mkdir()
    inputs = _protocol_inputs(other_root, mechanical_reason="OTHER_SIGNAL")
    other_protocol = other_root / "learning-protocol"
    publish_source_inference_learning_protocol(
        run_root=other_root,
        dossier_manifest_path=inputs[0],
        dossier_path=inputs[1],
        census_manifest_path=inputs[2],
        census_records_path=inputs[3],
        target_dir=other_protocol,
    )
    other_rule = _rule(other_protocol, inputs[4])
    with pytest.raises(BroadQaExternalDataError, match="family/protocol"):
        publish_source_inference_rule_pack(
            protocol_dir=protocol,
            operator_family="NORMALIZATION_EQUIVALENCE",
            fresh_rules=(other_rule,),
            resumed_rules=(other_rule,),
            target_dir=tmp_path / "wrong-protocol",
            fresh_checkpoint_chain_path=fresh_chain,
            resumed_checkpoint_chain_path=resumed_chain,
        )

    validation_rule = _rule(protocol, records, validation_support=True)
    validation_fresh_chain, validation_resumed_chain = _two_completed_chains(
        protocol, validation_rule, suffix="validation-leak")
    with pytest.raises(BroadQaExternalDataError, match="非 TRAIN"):
        publish_source_inference_rule_pack(
            protocol_dir=protocol,
            operator_family="NORMALIZATION_EQUIVALENCE",
            fresh_rules=(validation_rule,),
            resumed_rules=(validation_rule,),
            target_dir=tmp_path / "validation-leak",
            fresh_checkpoint_chain_path=validation_fresh_chain,
            resumed_checkpoint_chain_path=validation_resumed_chain,
        )
    with pytest.raises(BroadQaExternalDataError, match="正例和反例"):
        replace(rule, evidence_commitments=(rule.evidence_commitments[0],))
    with pytest.raises(BroadQaExternalDataError, match="字段漂移"):
        parse_source_inference_learned_rule(canonical_json_line({
            **rule.to_dict(), "unexpected": 1,
        }))
