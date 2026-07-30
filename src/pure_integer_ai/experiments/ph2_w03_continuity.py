"""W-03 所需的 W-02 retention 与两代 publication 身份。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    D03FileIdentity,
    sha1_text,
)
from pure_integer_ai.experiments.ph2_dataset_core import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w03_contract import (
    W03ContractError,
    digest_value,
    safe_relative_path,
)


FORMAL_W02_RECEIPT_PATH = (
    "formal_private_evaluator_v3_20260730_a/publication/"
    "w02_runtime_evidence_receipt.json"
)
FORMAL_W02_AGGREGATE_PATH = (
    "formal_private_evaluator_v3_20260730_a/publication/"
    "private_evaluation_aggregate.json"
)
FORMAL_W02_CANDIDATE_FREEZE_PATH = (
    "formal_candidate_v2/candidate_host_freeze_v2.json"
)
FORMAL_W02_ATTESTATION_PATH = (
    "formal_candidate_v2_publication_20260730_a/"
    "candidate_publication_attestation.json"
)
FORMAL_W02_RECEIPT_IDENTITY = D03FileIdentity(
    FORMAL_W02_RECEIPT_PATH,
    4297,
    "6b1344bfb226ea2488760987a838b4a7d4016f14831d6ed58c78b9ff0e45a2eb",
)
FORMAL_W02_CANDIDATE_FREEZE_IDENTITY = D03FileIdentity(
    FORMAL_W02_CANDIDATE_FREEZE_PATH,
    5959,
    "a280e25a01efc50f4f22ab5060be55bed193e1ca600c74050f0937969844d9ff",
)
FORMAL_W02_ATTESTATION_IDENTITY = D03FileIdentity(
    FORMAL_W02_ATTESTATION_PATH,
    7781,
    "ff19923f55fef60c29911708a956648c51930ac8f794ff66e9e5103c1bfb642e",
)
FORMAL_W02_AGGREGATE_IDENTITY = D03FileIdentity(
    FORMAL_W02_AGGREGATE_PATH,
    4811,
    "dd03d7873897cda597ba5ab0426d793aef6fade06f71852e505d67d39f9a5a94",
)
W02_HISTORICAL_PUBLICATION_HEAD = "588fa8465c6bb23cf8656d606f1d7d3e49d30056"
W02_HISTORICAL_CI_RUN_ID = 30463618992
W03_BASELINE_HEAD = "8344ef609d46f544079da9a938851c230e0a63db"
W03_BASELINE_CI_RUN_ID = 30508304244
REQUIRED_CI_JOB_NAMES = (
    "Python 3.11 on ubuntu-latest",
    "Python 3.14 on ubuntu-latest",
    "Python 3.14 on windows-latest",
    "Secret scan",
)
W02_REQUIRED_HOST_DIGEST_KEYS = (
    "artifact", "core", "cursor", "logical", "manifest", "memory", "use",
)
W02_EXECUTION_STATE = (
    ("LANGUAGE_CAPABILITY_MASTERED", 0),
    ("LANGUAGE_READINESS", 0),
    ("W02_BLOCKED_FAILED", 0),
    ("W02_RUNTIME_EVIDENCED", 1),
    ("W02_STARTED", 1),
    ("W03_STARTED", 0),
)


def _sha256(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise W03ContractError(f"{label} is not canonical SHA-256")
    return value


def _identity_matches(actual: D03FileIdentity, expected: D03FileIdentity) -> bool:
    return actual == expected


@dataclass(frozen=True, order=True)
class W03CIJobBinding:
    """把一项必需成功的 GitHub Actions job 绑定到 database id。"""

    name: str
    job_id: int

    def __post_init__(self) -> None:
        if self.name not in REQUIRED_CI_JOB_NAMES:
            raise W03ContractError("publication CI job name is not in the four-job gate")
        if type(self.job_id) is not int or self.job_id <= 0:
            raise W03ContractError("publication CI job id must be a positive strict integer")

    def to_dict(self) -> dict[str, object]:
        return {"job_id": self.job_id, "name": self.name}


@dataclass(frozen=True)
class W03PublicationBaseline:
    """冻结 reader 修复的公开 HEAD 与四项已完成 CI 门。"""

    head_sha1: str
    ci_run_id: int
    ci_jobs: tuple[W03CIJobBinding, ...]
    ci_conclusion: str = "success"
    remote_ref_match: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "head_sha1", sha1_text(
            self.head_sha1, where="W-03 publication HEAD"))
        if type(self.ci_run_id) is not int or self.ci_run_id <= 0:
            raise W03ContractError("W-03 publication CI run id is invalid")
        if (not isinstance(self.ci_jobs, tuple)
                or tuple(item.name for item in self.ci_jobs) != REQUIRED_CI_JOB_NAMES
                or len({item.job_id for item in self.ci_jobs}) != 4):
            raise W03ContractError("W-03 publication must bind the exact four CI jobs")
        if self.ci_conclusion != "success" or self.remote_ref_match != 1:
            raise W03ContractError("W-03 publication baseline is not fully green/remote-matched")

    def stable_key(self) -> tuple[int, ...]:
        return digest_value({
            "ci_conclusion": self.ci_conclusion,
            "ci_jobs": [item.to_dict() for item in self.ci_jobs],
            "ci_run_id": self.ci_run_id,
            "head_sha1": self.head_sha1,
            "remote_ref_match": self.remote_ref_match,
        })


@dataclass(frozen=True)
class W03PublicationObservation:
    """独立观测 local、tracking、remote 和 CI publication 状态。"""

    local_head_sha1: str
    tracking_head_sha1: str
    remote_head_sha1: str
    ci_run_id: int
    ci_head_sha1: str
    ci_status: str
    ci_conclusion: str
    ci_jobs: tuple[W03CIJobBinding, ...]

    def __post_init__(self) -> None:
        for name in ("local_head_sha1", "tracking_head_sha1", "remote_head_sha1",
                     "ci_head_sha1"):
            object.__setattr__(self, name, sha1_text(
                getattr(self, name), where=f"publication observation {name}"))
        if type(self.ci_run_id) is not int or self.ci_run_id <= 0:
            raise W03ContractError("observed CI run id is invalid")
        if not isinstance(self.ci_jobs, tuple):
            raise W03ContractError("observed CI jobs must be a tuple")


def validate_w03_publication_observation(
        baseline: W03PublicationBaseline,
        observation: W03PublicationObservation,
        ) -> W03PublicationObservation:
    """HEAD、tracking、remote、run 或任一 CI job 漂移时失败关闭。"""
    if not isinstance(baseline, W03PublicationBaseline):
        raise W03ContractError("publication baseline type is invalid")
    if not isinstance(observation, W03PublicationObservation):
        raise W03ContractError("publication observation type is invalid")
    if (observation.local_head_sha1 != baseline.head_sha1
            or observation.tracking_head_sha1 != baseline.head_sha1
            or observation.remote_head_sha1 != baseline.head_sha1
            or observation.ci_head_sha1 != baseline.head_sha1
            or observation.ci_run_id != baseline.ci_run_id
            or observation.ci_status != "completed"
            or observation.ci_conclusion != "success"
            or observation.ci_jobs != baseline.ci_jobs):
        raise W03ContractError("current HEAD/remote/CI publication baseline drifted")
    return observation


def _run_text(args: tuple[str, ...], repository: Path) -> str:
    result = subprocess.run(
        args,
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise W03ContractError(f"publication observation command failed: {args[0]}")
    return result.stdout.strip()


def inspect_w03_publication_observation(
        repository_root: str | Path,
        ) -> W03PublicationObservation:
    """只读观测 local、tracking、remote refs 和冻结 CI run。"""
    repository = Path(repository_root).resolve()
    if not repository.is_dir():
        raise W03ContractError("publication repository root is missing")
    baseline = formal_w03_publication_baseline()
    local = _run_text(("git", "rev-parse", "HEAD"), repository)
    tracking = _run_text(("git", "rev-parse", "origin/master"), repository)
    remote_raw = _run_text(
        ("git", "ls-remote", "--heads", "origin", "master"), repository)
    remote_parts = remote_raw.split()
    if len(remote_parts) != 2 or remote_parts[1] != "refs/heads/master":
        raise W03ContractError("remote master observation is malformed")
    raw = _run_text((
        "gh", "run", "view", str(baseline.ci_run_id), "--json",
        "databaseId,status,conclusion,headSha,jobs",
    ), repository)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise W03ContractError("publication CI observation is not JSON") from exc
    jobs = value.get("jobs") if isinstance(value, dict) else None
    if not isinstance(jobs, list):
        raise W03ContractError("publication CI jobs are missing")
    by_name = {item.get("name"): item for item in jobs if isinstance(item, dict)}
    if (set(by_name) != set(REQUIRED_CI_JOB_NAMES)
            or any(by_name[name].get("status") != "completed"
                   or by_name[name].get("conclusion") != "success"
                   for name in REQUIRED_CI_JOB_NAMES)):
        raise W03ContractError("publication CI job set is incomplete or not successful")
    return W03PublicationObservation(
        local_head_sha1=local,
        tracking_head_sha1=tracking,
        remote_head_sha1=remote_parts[0],
        ci_run_id=value.get("databaseId"),
        ci_head_sha1=value.get("headSha"),
        ci_status=str(value.get("status")),
        ci_conclusion=str(value.get("conclusion")),
        ci_jobs=tuple(W03CIJobBinding(name, by_name[name]["databaseId"])
                      for name in REQUIRED_CI_JOB_NAMES),
    )


def verify_w03_publication_host(
        repository_root: str | Path,
        ) -> W03PublicationObservation:
    """一次调用完成 W03-00C publication epoch 的观测与校验。"""
    baseline = formal_w03_publication_baseline()
    return validate_w03_publication_observation(
        baseline, inspect_w03_publication_observation(repository_root))


@dataclass(frozen=True)
class W03W02ContinuityBinding:
    """冻结完整 W-02 receipt、candidate、artifact 和结果。"""

    receipt_identity: D03FileIdentity
    candidate_freeze_identity: D03FileIdentity
    candidate_attestation_identity: D03FileIdentity
    aggregate_identity: D03FileIdentity
    historical_publication_head_sha1: str
    historical_ci_run_id: int
    historical_ci_jobs: tuple[W03CIJobBinding, ...]
    candidate_run_id: int
    formal_training_runs: int
    capability_code_identities: tuple[D03FileIdentity, ...]
    host_artifact_identities: tuple[D03FileIdentity, ...]
    host_digests: tuple[tuple[str, str], ...]
    dimension_pass_counts: tuple[int, ...]
    dimension_statuses: tuple[str, ...]
    fail_count: int
    ne_count: int
    execution_state: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        expected_identities = (
            (self.receipt_identity, FORMAL_W02_RECEIPT_IDENTITY),
            (self.candidate_freeze_identity, FORMAL_W02_CANDIDATE_FREEZE_IDENTITY),
            (self.candidate_attestation_identity, FORMAL_W02_ATTESTATION_IDENTITY),
            (self.aggregate_identity, FORMAL_W02_AGGREGATE_IDENTITY),
        )
        if any(not _identity_matches(actual, expected)
               for actual, expected in expected_identities):
            raise W03ContractError("W-02 receipt/freeze/attestation/aggregate identity drifted")
        object.__setattr__(self, "historical_publication_head_sha1", sha1_text(
            self.historical_publication_head_sha1,
            where="W-02 historical publication HEAD"))
        if (self.historical_publication_head_sha1 != W02_HISTORICAL_PUBLICATION_HEAD
                or self.historical_ci_run_id != W02_HISTORICAL_CI_RUN_ID
                or tuple(item.name for item in self.historical_ci_jobs)
                != REQUIRED_CI_JOB_NAMES
                or len({item.job_id for item in self.historical_ci_jobs}) != 4):
            raise W03ContractError("W-02 historical publication epoch drifted")
        if self.candidate_run_id != 3 or self.formal_training_runs != 1:
            raise W03ContractError("W-02 formal run identity drifted")
        for name, values, count in (
                ("capability code", self.capability_code_identities, 10),
                ("host artifact", self.host_artifact_identities, 11)):
            if (not isinstance(values, tuple) or len(values) != count
                    or any(not isinstance(item, D03FileIdentity) for item in values)
                    or len({item.relative_path for item in values}) != count):
                raise W03ContractError(f"W-02 {name} identity inventory drifted")
        if tuple(key for key, _ in self.host_digests) != W02_REQUIRED_HOST_DIGEST_KEYS:
            raise W03ContractError("W-02 seven host digest keys drifted")
        for key, value in self.host_digests:
            _sha256(value, label=f"W-02 host digest {key}")
        if (self.dimension_pass_counts != (2, 2, 4, 2, 5)
                or self.dimension_statuses != ("PASS",) * 5
                or self.fail_count != 0 or self.ne_count != 0):
            raise W03ContractError("W-02 five-way hard conjunct retention failed")
        if self.execution_state != W02_EXECUTION_STATE:
            raise W03ContractError("W-02 retained execution state drifted")

    def stable_key(self) -> tuple[int, ...]:
        return digest_value({
            "aggregate_identity": self.aggregate_identity.to_dict(),
            "candidate_attestation_identity": self.candidate_attestation_identity.to_dict(),
            "candidate_freeze_identity": self.candidate_freeze_identity.to_dict(),
            "candidate_run_id": self.candidate_run_id,
            "capability_code_identities": [
                item.to_dict() for item in self.capability_code_identities],
            "dimension_pass_counts": list(self.dimension_pass_counts),
            "dimension_statuses": list(self.dimension_statuses),
            "execution_state": dict(self.execution_state),
            "fail_count": self.fail_count,
            "formal_training_runs": self.formal_training_runs,
            "historical_ci_jobs": [item.to_dict() for item in self.historical_ci_jobs],
            "historical_ci_run_id": self.historical_ci_run_id,
            "historical_publication_head_sha1": self.historical_publication_head_sha1,
            "host_artifact_identities": [
                item.to_dict() for item in self.host_artifact_identities],
            "host_digests": dict(self.host_digests),
            "ne_count": self.ne_count,
            "receipt_identity": self.receipt_identity.to_dict(),
        })

    def base_fence_key(self) -> tuple[int, ...]:
        """不暴露 payload 地绑定不可变 W-02 run-3 base。"""
        return digest_value({
            "candidate_run_id": self.candidate_run_id,
            "host_artifact_identities": [
                item.to_dict() for item in self.host_artifact_identities],
            "host_digests": dict(self.host_digests),
            "receipt_sha256": self.receipt_identity.sha256,
        })


def formal_w03_publication_baseline() -> W03PublicationBaseline:
    """返回不可变 W03-00C publication epoch。"""
    return W03PublicationBaseline(
        W03_BASELINE_HEAD,
        W03_BASELINE_CI_RUN_ID,
        (
            W03CIJobBinding("Python 3.11 on ubuntu-latest", 90762736125),
            W03CIJobBinding("Python 3.14 on ubuntu-latest", 90762736134),
            W03CIJobBinding("Python 3.14 on windows-latest", 90762736152),
            W03CIJobBinding("Secret scan", 90762736103),
        ),
    )


def _resolve(root: Path, relative: str) -> Path:
    normalized = safe_relative_path(relative, label="W-02 continuity path")
    target = (root / Path(*PurePosixPath(normalized).parts)).resolve()
    if not target.is_relative_to(root) or not target.is_file() or target.is_symlink():
        raise W03ContractError("W-02 continuity path is missing, escaped, or a symlink")
    return target


def _read_identity(root: Path, identity: D03FileIdentity) -> tuple[bytes, dict[str, Any]]:
    path = _resolve(root, identity.relative_path)
    payload = path.read_bytes()
    if (len(payload) != identity.size_bytes
            or hashlib.sha256(payload).hexdigest() != identity.sha256):
        raise W03ContractError("W-02 continuity file identity drifted")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise W03ContractError("W-02 continuity JSON cannot be parsed") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise W03ContractError("W-02 continuity JSON is not canonical")
    return payload, value


def _inventory_identity(root: Path, prefix: str, row: Any) -> D03FileIdentity:
    if (not isinstance(row, dict)
            or set(row) != {"path", "sha256", "size_bytes"}):
        raise W03ContractError("W-02 artifact inventory row is invalid")
    relative = PurePosixPath(prefix, safe_relative_path(
        row["path"], label="W-02 artifact inventory path")).as_posix()
    identity = D03FileIdentity(relative, row["size_bytes"], row["sha256"])
    path = _resolve(root, relative)
    payload = path.read_bytes()
    if (len(payload) != identity.size_bytes
            or hashlib.sha256(payload).hexdigest() != identity.sha256):
        raise W03ContractError("W-02 host artifact identity drifted")
    return identity


def verify_formal_w02_continuity(
        repository_root: str | Path,
        artifacts_root: str | Path,
        ) -> W03W02ContinuityBinding:
    """回验当前公开代码与不可变 W-02 run-3 artifacts。"""
    repository = Path(repository_root).resolve()
    artifacts = Path(artifacts_root).resolve()
    if not repository.is_dir() or not artifacts.is_dir():
        raise W03ContractError("W-02 continuity roots are missing")
    _, receipt = _read_identity(artifacts, FORMAL_W02_RECEIPT_IDENTITY)
    _, freeze = _read_identity(artifacts, FORMAL_W02_CANDIDATE_FREEZE_IDENTITY)
    _, attestation = _read_identity(artifacts, FORMAL_W02_ATTESTATION_IDENTITY)
    _, aggregate = _read_identity(artifacts, FORMAL_W02_AGGREGATE_IDENTITY)

    candidate = receipt.get("candidate_binding")
    public = receipt.get("public_binding")
    result = receipt.get("formal_result")
    policy = receipt.get("policy")
    if (not isinstance(candidate, dict) or candidate != {
            "additional_retraining_runs": 0,
            "candidate_freeze_sha256": FORMAL_W02_CANDIDATE_FREEZE_IDENTITY.sha256,
            "candidate_publication_attestation_sha256": FORMAL_W02_ATTESTATION_IDENTITY.sha256,
            "capability_code_match_count": 10,
            "formal_training_runs": 1,
            "host_artifact_match_count": 11,
            "run_id": 3,
            }):
        raise W03ContractError("W-02 receipt candidate binding drifted")
    if (not isinstance(public, dict)
            or public.get("head_sha1") != W02_HISTORICAL_PUBLICATION_HEAD
            or public.get("ci_run_id") != W02_HISTORICAL_CI_RUN_ID
            or public.get("ci_conclusion") != "success"
            or public.get("remote_ref_match") != 1
            or public.get("worktree_clean") != 1):
        raise W03ContractError("W-02 historical publication binding drifted")
    raw_jobs = public.get("ci_jobs")
    if not isinstance(raw_jobs, list) or len(raw_jobs) != 4:
        raise W03ContractError("W-02 historical four-job CI binding is missing")
    jobs_by_name = {item.get("name"): item for item in raw_jobs if isinstance(item, dict)}
    if (set(jobs_by_name) != set(REQUIRED_CI_JOB_NAMES)
            or any(jobs_by_name[name].get("conclusion") != "success"
                   for name in REQUIRED_CI_JOB_NAMES)):
        raise W03ContractError("W-02 historical CI job status drifted")
    jobs = tuple(W03CIJobBinding(name, jobs_by_name[name]["job_id"])
                 for name in REQUIRED_CI_JOB_NAMES)
    if (not isinstance(result, dict)
            or result.get("dimension_pass_counts") != [2, 2, 4, 2, 5]
            or result.get("dimension_statuses") != ["PASS"] * 5
            or result.get("fail_count") != 0
            or result.get("ne_count") != 0
            or result.get("run_count") != 1
            or result.get("status") != "PASS"
            or result.get("failure_phase") != "NONE"
            or not isinstance(policy, dict)
            or policy.get("five_way_hard_conjunct_pass") != 1
            or policy.get("max_fail_count") != 0
            or policy.get("min_pass_numerator") != 1
            or policy.get("min_pass_denominator") != 1
            or policy.get("ne_policy") != "BLOCK"):
        raise W03ContractError("W-02 receipt hard conjunct retention failed")
    if tuple(sorted(receipt.get("execution_state", {}).items())) != W02_EXECUTION_STATE:
        raise W03ContractError("W-02 receipt execution state drifted")

    if (freeze.get("run_id") != 3
            or freeze.get("execution_state", {}).get("formal_training_runs") != 1
            or freeze.get("execution_state", {}).get("W03_STARTED") != 0
            or freeze.get("host_digests") is None):
        raise W03ContractError("W-02 candidate freeze state drifted")
    host_digests = tuple(sorted(freeze["host_digests"].items()))
    if tuple(key for key, _ in host_digests) != W02_REQUIRED_HOST_DIGEST_KEYS:
        raise W03ContractError("W-02 candidate seven-digest inventory drifted")
    artifacts_rows = freeze.get("artifact_inventory")
    if not isinstance(artifacts_rows, list) or len(artifacts_rows) != 11:
        raise W03ContractError("W-02 candidate artifact inventory is not 11 files")
    host_identities = tuple(_inventory_identity(
        artifacts, "formal_candidate_v2", row) for row in artifacts_rows)
    actual_artifacts = tuple(sorted(
        path.relative_to(artifacts / "formal_candidate_v2").as_posix()
        for path in (artifacts / "formal_candidate_v2").rglob("*")
        if path.is_file() and path.name != "candidate_host_freeze_v2.json"
    ))
    if actual_artifacts != tuple(sorted(row["path"] for row in artifacts_rows)):
        raise W03ContractError("W-02 candidate host fileset drifted")

    code_rows = freeze.get("code_inventory")
    if not isinstance(code_rows, list) or len(code_rows) != 10:
        raise W03ContractError("W-02 capability code inventory is not 10 files")
    code_identities: list[D03FileIdentity] = []
    for row in code_rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise W03ContractError("W-02 capability code row is invalid")
        relative = safe_relative_path(row["path"], label="W-02 capability code path")
        path = _resolve(repository, relative)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise W03ContractError("W-02 capability code identity drifted")
        code_identities.append(D03FileIdentity(relative, len(payload), row["sha256"]))

    attested_candidate = attestation.get("candidate_identity")
    if (not isinstance(attested_candidate, dict)
            or attested_candidate.get("candidate_freeze_sha256")
            != FORMAL_W02_CANDIDATE_FREEZE_IDENTITY.sha256
            or attested_candidate.get("host_run_id") != 3
            or attested_candidate.get("formal_training_runs_total") != 1
            or attested_candidate.get("retraining_runs_added") != 0
            or attestation.get("claims") != {
                "capability_code_unchanged": 1,
                "host_artifacts_unchanged": 1,
                "publication_identity_only_change": 1,
                "retraining_performed": 0,
            }):
        raise W03ContractError("W-02 candidate publication attestation drifted")
    dimensions = aggregate.get("dimensions")
    generation = aggregate.get("generation")
    if (not isinstance(dimensions, list) or len(dimensions) != 4
            or any(item.get("status") != "PASS" or item.get("fail_count") != 0
                   or item.get("ne_count") != 0 for item in dimensions)
            or not isinstance(generation, dict)
            or generation.get("status") != "PASS"
            or generation.get("fail_count") != 0
            or generation.get("ne_count") != 0
            or aggregate.get("status") != "PASS"
            or aggregate.get("failure_phase") != "NONE"):
        raise W03ContractError("W-02 aggregate five-way retention failed")

    return W03W02ContinuityBinding(
        receipt_identity=FORMAL_W02_RECEIPT_IDENTITY,
        candidate_freeze_identity=FORMAL_W02_CANDIDATE_FREEZE_IDENTITY,
        candidate_attestation_identity=FORMAL_W02_ATTESTATION_IDENTITY,
        aggregate_identity=FORMAL_W02_AGGREGATE_IDENTITY,
        historical_publication_head_sha1=W02_HISTORICAL_PUBLICATION_HEAD,
        historical_ci_run_id=W02_HISTORICAL_CI_RUN_ID,
        historical_ci_jobs=jobs,
        candidate_run_id=3,
        formal_training_runs=1,
        capability_code_identities=tuple(code_identities),
        host_artifact_identities=host_identities,
        host_digests=host_digests,
        dimension_pass_counts=(2, 2, 4, 2, 5),
        dimension_statuses=("PASS",) * 5,
        fail_count=0,
        ne_count=0,
        execution_state=W02_EXECUTION_STATE,
    )


__all__ = [
    "FORMAL_W02_AGGREGATE_PATH",
    "FORMAL_W02_ATTESTATION_PATH",
    "FORMAL_W02_CANDIDATE_FREEZE_PATH",
    "FORMAL_W02_RECEIPT_PATH",
    "W03CIJobBinding",
    "W03PublicationBaseline",
    "W03PublicationObservation",
    "W03W02ContinuityBinding",
    "formal_w03_publication_baseline",
    "inspect_w03_publication_observation",
    "validate_w03_publication_observation",
    "verify_formal_w02_continuity",
    "verify_w03_publication_host",
]
