"""V-00 独立评测数据协议、泄漏守卫和统一 probe API 测试。"""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.types import (
    DOMAIN_TEXT,
    LANG_ZH,
    MODALITY_LANGUAGE,
)
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.evaluation_protocol import (
    CanonicalIdentity,
    EvaluationAssignment,
    EvaluationLeakageError,
    EvaluationPlan,
    EvaluationProtocol,
    EvaluationProtocolError,
    EvaluationStatePollutionError,
    ProbeOutcome,
    ProtocolKey,
    build_evaluation_report,
    evaluate_probe,
    make_evaluation_data_identity,
    read_evaluation_plan,
    write_evaluation_plan,
)
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig,
    formal_train,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT


def _key(value: int) -> ProtocolKey:
    """构造测试使用的单分量注入协议键。"""
    return ProtocolKey((value,))


def _protocol(*, full_coverage: bool = True) -> EvaluationProtocol:
    """构造不依赖实现内置枚举的 split、维度和对抗协议。"""
    dimensions = tuple(_key(100 + index) for index in range(
        6 if full_coverage else 1))
    adversarial = tuple(_key(200 + index) for index in range(
        9 if full_coverage else 1))
    return EvaluationProtocol(
        version=7,
        training_split=_key(10),
        development_split=_key(11),
        held_out_split=_key(12),
        adversarial_split=_key(13),
        external_split=_key(14),
        statistical_evidence=_key(20),
        external_evidence=_key(21),
        required_dimensions=dimensions,
        required_adversarial_kinds=adversarial,
    )


def _item(text: str, *, source_id: int,
          document_id: int = 0) -> CollectedItem:
    """构造携带显式真实来源位置的语言评测项。"""
    return CollectedItem(
        tokens=[text],
        raw_text=text,
        source=SOURCE_BARE_TEXT,
        modality=MODALITY_LANGUAGE,
        lang=LANG_ZH,
        domain=DOMAIN_TEXT,
        source_ref=SourceRef(
            SOURCE_BARE_TEXT,
            source_id,
            document_id,
            GLOBAL_OWNER_SCOPE,
            VersionBundle(),
        ),
    )


def _assignment(
        item: CollectedItem, *,
        protocol: EvaluationProtocol,
        split: ProtocolKey,
        probe_kind: ProtocolKey | None,
        provenance: object,
        dedup: object | None = None,
        expected: object | None = None,
        ) -> EvaluationAssignment:
    """用完整内容和显式簇键构造一条测试 ledger 记录。"""
    return EvaluationAssignment(
        identity=make_evaluation_data_identity(
            item,
            dedup_cluster=("dedup", item.raw_text) if dedup is None else dedup,
            provenance_cluster=provenance,
        ),
        split=split,
        probe_kind=probe_kind,
        dimensions=(
            () if split == protocol.training_split
            else protocol.required_dimensions
        ),
        expected_outcome=(
            None if expected is None
            else CanonicalIdentity.from_value(expected)
        ),
    )


def _complete_plan(*, full_coverage: bool = True
                   ) -> tuple[EvaluationPlan, list[CollectedItem]]:
    """构造覆盖五类 split、全部维度和全部对抗种类的完整计划。"""
    protocol = _protocol(full_coverage=full_coverage)
    training = _item("训练输入", source_id=1)
    development = _item("开发输入", source_id=2)
    held_out = _item("留出输入", source_id=3)
    external = _item("外部输入", source_id=4)
    adversarial_items = [
        _item(f"对抗输入{index}", source_id=3, document_id=index + 1)
        for index in range(len(protocol.required_adversarial_kinds))
    ]
    assignments = [
        _assignment(
            training,
            protocol=protocol,
            split=protocol.training_split,
            probe_kind=None,
            provenance=("source", 1),
        ),
        _assignment(
            development,
            protocol=protocol,
            split=protocol.development_split,
            probe_kind=_key(300),
            provenance=("source", 2),
            expected=("label", 1),
        ),
        _assignment(
            held_out,
            protocol=protocol,
            split=protocol.held_out_split,
            probe_kind=_key(301),
            provenance=("source", 3),
            expected=("label", 1),
        ),
    ]
    assignments.extend(
        _assignment(
            item,
            protocol=protocol,
            split=protocol.adversarial_split,
            probe_kind=kind,
            provenance=("source", 3),
            expected=("label", index),
        )
        for index, (item, kind) in enumerate(zip(
            adversarial_items,
            protocol.required_adversarial_kinds,
        ))
    )
    assignments.append(_assignment(
        external,
        protocol=protocol,
        split=protocol.external_split,
        probe_kind=_key(302),
        provenance=("source", 4),
        expected=("label", 1),
    ))
    items = [
        training,
        development,
        held_out,
        *adversarial_items,
        external,
    ]
    return EvaluationPlan(protocol, tuple(assignments)), items


def test_complete_plan_partitions_all_splits_and_preserves_input_order():
    """完整计划按全身份分区，且同一 split 内保持调用方输入顺序。"""
    plan, items = _complete_plan()
    reordered = [items[0], items[1], items[2], *reversed(items[3:-1]), items[-1]]
    partition = plan.partition(reordered)
    assert partition.items(plan.protocol.training_split) == (items[0],)
    assert partition.items(plan.protocol.development_split) == (items[1],)
    assert partition.items(plan.protocol.held_out_split) == (items[2],)
    assert partition.items(plan.protocol.adversarial_split) == tuple(
        reversed(items[3:-1]))
    assert partition.items(plan.protocol.external_split) == (items[-1],)
    assert len(partition.non_training_items()) == len(items) - 1


def test_plan_manifest_round_trip_is_canonical_and_tamper_evident(tmp_path):
    """完整 ledger 可逐字节复现，载荷摘要损坏和路径覆盖均 fail closed。"""
    plan, _items = _complete_plan()
    path = tmp_path / "evaluation-plan.json"
    assert write_evaluation_plan(plan, path) == path
    first_bytes = path.read_bytes()
    assert write_evaluation_plan(plan, path) == path
    assert path.read_bytes() == first_bytes
    restored = read_evaluation_plan(path)
    assert restored == plan
    assert restored.sha256() == plan.sha256()

    changed = EvaluationPlan(
        replace(plan.protocol, version=plan.protocol.version + 1),
        plan.assignments,
    )
    with pytest.raises(EvaluationProtocolError, match="新版本路径"):
        write_evaluation_plan(changed, path)

    value = json.loads(path.read_text(encoding="utf-8"))
    value["plan"]["assignments"][0]["identity"]["content"]["sha256"] = (
        "0" * 64)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluationProtocolError, match="规范身份 SHA-256"):
        read_evaluation_plan(path)


def test_exact_content_cannot_cross_training_and_held_out():
    """不同 SourceRef 不能掩盖训练与留出之间的完整内容重复。"""
    plan, items = _complete_plan()
    duplicate = replace(items[2], raw_text=items[0].raw_text,
                        tokens=list(items[0].tokens))
    replacement = _assignment(
        duplicate,
        protocol=plan.protocol,
        split=plan.protocol.held_out_split,
        probe_kind=_key(301),
        provenance=("source", 3),
    )
    assignments = list(plan.assignments)
    assignments[2] = replacement
    with pytest.raises(EvaluationLeakageError, match="完整内容"):
        EvaluationPlan(plan.protocol, tuple(assignments))


def test_dedup_cluster_cannot_cross_training_and_held_out():
    """表层不同但同 dedup cluster 的改写不得跨训练与留出。"""
    plan, _items = _complete_plan()
    assignments = list(plan.assignments)
    assignments[2] = replace(
        assignments[2],
        identity=replace(
            assignments[2].identity,
            dedup_cluster=assignments[0].identity.dedup_cluster,
        ),
    )
    with pytest.raises(EvaluationLeakageError, match="dedup cluster"):
        EvaluationPlan(plan.protocol, tuple(assignments))


def test_same_source_rewrite_cannot_cross_training_and_evaluation():
    """内容不同的同源改写仍由 provenance cluster 识别并拒绝。"""
    plan, _items = _complete_plan()
    assignments = list(plan.assignments)
    assignments[2] = replace(
        assignments[2],
        identity=replace(
            assignments[2].identity,
            provenance_cluster=assignments[0].identity.provenance_cluster,
        ),
    )
    with pytest.raises(EvaluationLeakageError, match="同源改写"):
        EvaluationPlan(plan.protocol, tuple(assignments))


def test_external_source_cannot_overlap_statistical_splits():
    """EXTERNAL 来源簇不得与 held-out 等纯统计 split 复用。"""
    plan, _items = _complete_plan()
    assignments = list(plan.assignments)
    assignments[-1] = replace(
        assignments[-1],
        identity=replace(
            assignments[-1].identity,
            provenance_cluster=assignments[2].identity.provenance_cluster,
        ),
    )
    with pytest.raises(EvaluationLeakageError, match="EXTERNAL"):
        EvaluationPlan(plan.protocol, tuple(assignments))


def test_conflicting_expected_outcomes_for_same_input_are_rejected():
    """同 split 相同输入的标签变化必须显式冲突，不能作为两条独立真值。"""
    plan, items = _complete_plan()
    duplicate = replace(
        items[1],
        source_ref=SourceRef(
            SOURCE_BARE_TEXT,
            22,
            0,
            GLOBAL_OWNER_SCOPE,
            VersionBundle(),
        ),
    )
    conflicting = _assignment(
        duplicate,
        protocol=plan.protocol,
        split=plan.protocol.development_split,
        probe_kind=_key(300),
        provenance=("source", 22),
        expected=("label", 2),
    )
    with pytest.raises(EvaluationProtocolError, match="冲突预期"):
        EvaluationPlan(plan.protocol, (*plan.assignments, conflicting))


def test_probe_api_detects_host_state_pollution():
    """统一 probe API 返回分维结果，并在宿主状态变化时 fail closed。"""
    plan, _items = _complete_plan()
    assignment = plan.assignments[2]
    dimension = plan.protocol.required_dimensions[0]
    state = {"counter": [1]}
    observation = evaluate_probe(
        plan,
        assignment,
        dimension,
        lambda: ProbeOutcome(True, value=7),
        state_reader=lambda: state,
    )
    assert observation.outcome.passed is True
    assert observation.evidence == plan.protocol.statistical_evidence

    def polluting_evaluator() -> ProbeOutcome:
        """模拟绕过评测沙箱写入宿主状态的错误 evaluator。"""
        state["counter"].append(2)
        return ProbeOutcome(True)

    with pytest.raises(EvaluationStatePollutionError, match="宿主状态"):
        evaluate_probe(
            plan,
            assignment,
            dimension,
            polluting_evaluator,
            state_reader=lambda: state,
        )


def test_report_separates_dimensions_external_and_not_evaluated():
    """报告按维度和证据分账，未运行项保持 NE 且不存在综合分数。"""
    plan, _items = _complete_plan()
    dimension = plan.protocol.required_dimensions[0]
    held_out = plan.assignments[2]
    external = plan.assignments[-1]
    observations = [
        evaluate_probe(
            plan, held_out, dimension, lambda: ProbeOutcome(True)),
        evaluate_probe(
            plan, external, dimension, lambda: ProbeOutcome(False)),
    ]
    report = build_evaluation_report(plan, observations)
    rows = [
        row for row in report.measurements
        if row.dimension == dimension
    ]
    assert {row.evidence for row in rows} == {
        plan.protocol.statistical_evidence,
        plan.protocol.external_evidence,
    }
    statistical = next(
        row for row in rows
        if row.evidence == plan.protocol.statistical_evidence)
    external_row = next(
        row for row in rows
        if row.evidence == plan.protocol.external_evidence)
    assert statistical.passed == 1 and statistical.not_evaluated > 0
    assert external_row.failed == 1
    assert not hasattr(report, "score")
    assert not hasattr(report, "passed")

    mixed = replace(
        observations[-1],
        evidence=plan.protocol.statistical_evidence,
    )
    with pytest.raises(EvaluationProtocolError, match="混账"):
        build_evaluation_report(plan, [mixed])


def test_formal_train_excludes_every_non_training_split(tmp_path):
    """正式入口只训练 training，开发、留出、对抗和 EXTERNAL 全部计入 withheld。"""
    plan, items = _complete_plan(full_coverage=False)
    seen_splits = []

    def evaluator(eval_ctx, item, assignment, dimension):
        """在独立沙箱写局部节点，并返回注入维度的确定通过结果。"""
        seen_splits.append(assignment.split)
        eval_ctx.concept_index.ensure(
            f"v00-eval-{item.raw_text}-{dimension.stable_key()}",
            space_id=eval_ctx.space_id,
        )
        return ProbeOutcome(True)

    result = formal_train(
        FormalTrainConfig(
            run_dir=str(tmp_path),
            run_id="v00-strict",
            rounds_per_stage=1,
            active_training_stages=(),
            persist_graph_dump=False,
            evaluation_plan=plan,
            evaluation_probe_evaluator=evaluator,
        ),
        items,
        backend=DictBackend(),
    )
    assert result.evaluation_plan is plan
    assert result.execution.input_items == len(items)
    assert result.execution.training_items == 1
    assert result.execution.probe_items == len(items) - 1
    assert result.evaluation_strictly_isolated is True
    assert result.evaluation_plan_path is not None
    assert read_evaluation_plan(result.evaluation_plan_path) == plan
    assert result.evaluation_plan_sha256 == plan.sha256()
    assert result.probe_set is not None
    assert result.probe_set.version == plan.protocol.version
    assert set(seen_splits) == {
        plan.protocol.development_split,
        plan.protocol.held_out_split,
        plan.protocol.adversarial_split,
        plan.protocol.external_split,
    }
    assert result.evaluation_report is not None
    assert all(
        row.failed == 0 and row.not_evaluated == 0
        for row in result.evaluation_report.measurements
    )
