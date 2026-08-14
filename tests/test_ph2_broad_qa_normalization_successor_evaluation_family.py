"""normalization successor family freeze 与唯一 formal runner 测试。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_successor_evaluation_family as family_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_successor_evaluation_runner as runner_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_evaluation_family import (
    build_normalization_successor_evaluation_family_freeze,
    publish_normalization_successor_evaluation_family_freeze,
    read_normalization_successor_evaluation_family_freeze,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_evaluation_runner import (
    run_normalization_successor_formal_evaluation,
)


def _sha(value: str) -> str:
    """构造 synthetic identity。"""
    return hashlib.sha256(value.encode()).hexdigest()


def _repository_root() -> Path:
    """返回当前测试对应的公开仓库根。"""
    return Path(__file__).resolve().parents[1]


def _arguments(tmp_path: Path) -> dict[str, object]:
    """创建 family/runner 所需的全部物理目录与外部 identity。"""
    values: dict[str, object] = {
        "repository_root": _repository_root(),
        "expected_evaluation_protocol_manifest_sha256": _sha("evaluation"),
        "expected_training_protocol_manifest_sha256": _sha("training"),
        "expected_fresh_learner_manifest_sha256": _sha("fresh"),
        "expected_resumed_learner_manifest_sha256": _sha("resumed"),
        "expected_rule_pack_manifest_sha256": _sha("pack"),
    }
    for name in (
            "evaluation_protocol_dir", "training_protocol_dir",
            "fresh_learner_dir", "resumed_learner_dir", "rule_pack_dir"):
        path = tmp_path / name
        path.mkdir()
        values[name] = path
    return values


def _install_family_readers(
        monkeypatch: pytest.MonkeyPatch,
        arguments: dict[str, object],
        *,
        state: dict[str, object] | None = None,
        ) -> dict[str, object]:
    """安装不读取 evaluation/reserve payload 的 deterministic fake lineage。"""
    current = state if state is not None else {}
    current.setdefault(
        "pack_sha", arguments["expected_rule_pack_manifest_sha256"])
    training_sha = str(arguments[
        "expected_training_protocol_manifest_sha256"])
    fresh_sha = str(arguments["expected_fresh_learner_manifest_sha256"])
    resumed_sha = str(arguments["expected_resumed_learner_manifest_sha256"])
    semantic_sha = _sha("semantic")
    fresh = {
        "checkpoint_chain_sha256": _sha("fresh-chain"),
        "checkpoint_terminal_sha256": _sha("fresh-terminal"),
        "manifest_sha256": fresh_sha,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "protocol_manifest_sha256": training_sha,
        "resume_markers": {"record_count": 0},
        "run_id": _sha("fresh-run"),
        "runtime_state": "LEARNED_PACK_DISABLED",
        "semantic_result_sha256": semantic_sha,
    }
    resumed = {
        "checkpoint_chain_sha256": _sha("resumed-chain"),
        "checkpoint_terminal_sha256": _sha("resumed-terminal"),
        "manifest_sha256": resumed_sha,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "protocol_manifest_sha256": training_sha,
        "resume_markers": {"record_count": 1},
        "run_id": _sha("resumed-run"),
        "runtime_state": "LEARNED_PACK_DISABLED",
        "semantic_result_sha256": semantic_sha,
    }
    outputs = {"consensus-rules.jsonl": ({"value": 1},)}

    def evaluation_reader(path, *, expected_manifest_sha256):
        """只返回已经冻结的 manifest identity。"""
        assert expected_manifest_sha256 == arguments[
            "expected_evaluation_protocol_manifest_sha256"]
        return {
            "evaluation_inventory": {
                "bytes": 10,
                "record_count": 1,
                "relative_path": "evaluation.inventory.jsonl",
                "role": "EVALUATION_WITH_LABELS",
                "sha256": _sha("inventory"),
            },
            "manifest_sha256": expected_manifest_sha256,
            "reserve_identity": {
                "bytes": 10,
                "record_count": 1,
                "relative_path": "reserve.identity.jsonl",
                "role": "RESERVE_IDENTITY_WITHOUT_LABELS",
                "sha256": _sha("reserve"),
            },
            "source_pack_manifest_sha256": _sha("evaluation-source"),
        }

    def training_reader(path, *, expected_manifest_sha256):
        """返回冻结 TRAIN protocol 的最小 identity。"""
        assert expected_manifest_sha256 == training_sha
        return ({
            "learner_contract": {"work_identity_sha256": _sha("work")},
            "manifest_sha256": training_sha,
        }, (), (), (), ())

    def learner_reader(path, **kwargs):
        """按物理目录区分 fresh 与 resumed lineage。"""
        assert kwargs["expected_protocol_manifest_sha256"] == training_sha
        manifest = fresh if Path(path).name == "fresh_learner_dir" else resumed
        return manifest, outputs

    def pack_reader(path, **kwargs):
        """返回与两条 learner lineage 一致的禁用态 pack。"""
        pack_sha = current["pack_sha"]
        return ({
            "learner_lineages": [{
                "checkpoint_chain_sha256": fresh["checkpoint_chain_sha256"],
                "checkpoint_terminal_sha256": fresh[
                    "checkpoint_terminal_sha256"],
                "learner_manifest_sha256": fresh_sha,
                "resume_marker_count": 0,
                "role": "FRESH",
                "run_id": fresh["run_id"],
            }, {
                "checkpoint_chain_sha256": resumed[
                    "checkpoint_chain_sha256"],
                "checkpoint_terminal_sha256": resumed[
                    "checkpoint_terminal_sha256"],
                "learner_manifest_sha256": resumed_sha,
                "resume_marker_count": 1,
                "role": "RESUMED",
                "run_id": resumed["run_id"],
            }],
            "manifest_sha256": pack_sha,
            "mastery_claimed": 0,
            "production_enabled": 0,
            "protocol_manifest_sha256": training_sha,
            "runtime_state": "LEARNED_PACK_DISABLED",
            "semantic_result_sha256": semantic_sha,
        }, outputs)

    program = SimpleNamespace(
        context_rules=(1, 2),
        declared_conflict_ids=(_sha("conflict"),),
        production_enabled=0,
        rule_pack_manifest_sha256=arguments[
            "expected_rule_pack_manifest_sha256"],
        sha256=lambda: _sha("candidate"),
        source_replays=(1, 2, 3),
        target_rules=(1, 2),
    )
    monkeypatch.setattr(
        family_module,
        "read_normalization_successor_evaluation_manifest_only",
        evaluation_reader,
    )
    monkeypatch.setattr(
        family_module, "read_normalization_successor_learner_input",
        training_reader)
    monkeypatch.setattr(
        family_module, "read_normalization_successor_learner", learner_reader)
    monkeypatch.setattr(
        family_module, "read_normalization_successor_rule_pack", pack_reader)
    monkeypatch.setattr(
        family_module, "compile_normalization_successor_candidate",
        lambda **kwargs: program)
    current["pack_reader"] = pack_reader
    current["outputs"] = outputs
    current["program"] = program
    return current


def test_family_freeze_reads_manifest_only_and_rejects_drift_or_overwrite(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """freeze 保持 payload 零读取，并绑定 pack、candidate 与全部 live code。"""
    arguments = _arguments(tmp_path)
    state = _install_family_readers(monkeypatch, arguments)
    built = build_normalization_successor_evaluation_family_freeze(**arguments)
    assert built["evaluation_payload_read_count"] == 0
    assert built["reserve_payload_read_count"] == 0
    assert built["evaluation_run_count"] == 0
    assert built["candidate_freeze"]["candidate_clone_sha256"] == _sha(
        "candidate")
    assert len(built["code_files"]) == 11

    monkeypatch.setattr(
        family_module, "require_normalization_successor_k_run_root",
        lambda value: Path(value).resolve())
    target = tmp_path / "family"
    report = publish_normalization_successor_evaluation_family_freeze(
        run_root=tmp_path, target_dir=target, **arguments)
    reread = read_normalization_successor_evaluation_family_freeze(
        target, **arguments)
    assert reread["manifest_sha256"] == report["manifest_sha256"]
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_successor_evaluation_family_freeze(
            run_root=tmp_path, target_dir=target, **arguments)

    state["pack_sha"] = _sha("drifted-pack")
    with pytest.raises(BroadQaExternalDataError, match="manifest identity 漂移"):
        read_normalization_successor_evaluation_family_freeze(
            target, **arguments)


@dataclass(frozen=True, slots=True)
class _SyntheticReport:
    """供 runner 边界测试使用的最小 immutable report。"""

    overall_outcome: str = "PASS"

    def to_dict(self) -> dict[str, object]:
        """导出 synthetic report。"""
        return {"overall_outcome": self.overall_outcome}

    def sha256(self) -> str:
        """返回 synthetic report identity。"""
        return _sha("synthetic-report-" + self.overall_outcome)


def _install_runner_readers(
        monkeypatch: pytest.MonkeyPatch,
        arguments: dict[str, object],
        *,
        fail_inventory: bool = False,
        ) -> None:
    """安装 runner synthetic family、evaluation、pack 与 evaluator。"""
    candidate_sha = _sha("runner-candidate")
    freeze = {
        "candidate_freeze": {"candidate_clone_sha256": candidate_sha},
        "evaluation_inventory_identity": {"sha256": _sha("inventory")},
        "evaluation_protocol_manifest_sha256": arguments[
            "expected_evaluation_protocol_manifest_sha256"],
        "family_commitment_sha256": _sha("family-commitment"),
        "manifest_sha256": _sha("family-freeze"),
    }
    monkeypatch.setattr(
        runner_module, "require_normalization_successor_k_run_root",
        lambda value: Path(value).resolve())
    monkeypatch.setattr(
        runner_module,
        "read_normalization_successor_evaluation_family_freeze",
        lambda *args, **kwargs: freeze,
    )

    def evaluation_reader(*args, **kwargs):
        """在 guard 之后返回 evaluation，或模拟 payload 漂移。"""
        if fail_inventory:
            raise BroadQaExternalDataError("synthetic inventory drift")
        return ({
            "evaluation_inventory": freeze["evaluation_inventory_identity"],
            "manifest_sha256": freeze[
                "evaluation_protocol_manifest_sha256"],
        }, ({"evaluation_id": _sha("item")},))

    monkeypatch.setattr(
        runner_module,
        "read_normalization_successor_evaluation_inventory_only",
        evaluation_reader,
    )
    pack = {
        "manifest_sha256": arguments[
            "expected_rule_pack_manifest_sha256"],
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_state": "LEARNED_PACK_DISABLED",
    }
    monkeypatch.setattr(
        runner_module, "read_normalization_successor_rule_pack",
        lambda *args, **kwargs: (pack, {}))
    monkeypatch.setattr(
        runner_module, "compile_normalization_successor_candidate",
        lambda **kwargs: SimpleNamespace(sha256=lambda: candidate_sha))
    monkeypatch.setattr(
        runner_module, "evaluate_normalization_successor_candidate",
        lambda **kwargs: _SyntheticReport())


def test_formal_runner_is_unique_and_keeps_pack_disabled(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """runner 先消费 guard，再发布唯一报告并拒绝第二次运行。"""
    arguments = _arguments(tmp_path)
    _install_runner_readers(monkeypatch, arguments)
    family = tmp_path / "runner-family"
    family.mkdir()
    publication = tmp_path / "publication"
    report = run_normalization_successor_formal_evaluation(
        run_root=tmp_path,
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
        run_normalization_successor_formal_evaluation(
            run_root=tmp_path,
            family_freeze_dir=family,
            publication_dir=publication,
            **arguments,
        )


def test_formal_guard_seals_inventory_failure_and_forbids_retry(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """guard 后 inventory 异常必须封存 NE，恢复 payload 也不得重跑。"""
    arguments = _arguments(tmp_path)
    _install_runner_readers(monkeypatch, arguments, fail_inventory=True)
    family = tmp_path / "failure-family"
    family.mkdir()
    publication = tmp_path / "failure-publication"
    with pytest.raises(BroadQaExternalDataError, match="inventory drift"):
        run_normalization_successor_formal_evaluation(
            run_root=tmp_path,
            family_freeze_dir=family,
            publication_dir=publication,
            **arguments,
        )
    failure_path = publication / "run-000001.failure.json"
    assert (publication / "run-000001.guard.json").is_file()
    assert failure_path.is_file()
    failure = json.loads(failure_path.read_bytes())
    assert failure["evaluation_run_count"] == 1
    assert failure["failure_phase"] == "GUARD_CONSUMED"
    assert failure["status"] == "NE_NO_RECEIPT"
    assert failure["production_enabled"] == 0
    with pytest.raises(BroadQaExternalDataError, match="已消费"):
        run_normalization_successor_formal_evaluation(
            run_root=tmp_path,
            family_freeze_dir=family,
            publication_dir=publication,
            **arguments,
        )
