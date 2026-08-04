"""Run the append-only W08-04 recovery validation with an exact test allowlist."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
REPORT_RELATIVE_PATH = (
    "data/ph2/manifests/d03_v1/w08_04_validation_recovery_v6.json"
)
REPORT_PATH = REPOSITORY_ROOT / REPORT_RELATIVE_PATH
GLOBAL_MANIFEST_RELATIVE_PATH = (
    "data/ph2/manifests/d03_v1/ph2_global_course_manifest_v1.json"
)
EXPECTED_IMPLEMENTATION_COMMIT = "d1c882e19127510a5e0417c5abc177f601f2b6ea"
EXPECTED_HASH_SEED = "0"
REPOSITORY_TEST_TEMP_ROOT = REPOSITORY_ROOT / ".pytest_tmp_safe"
INCIDENT_TEST_RELATIVE_PATH = (
    "tests/test_d02_p3ia_free_text_hierarchy_recall_runtime.py"
)
FUTURE_PACK_KEYS = (
    "AUTHORED_CC0_V1--CC0-1.0--free-text-hierarchy-recall-v1",
    "AUTHORED_CC0_V1--CC0-1.0--generation-postcheck-v1",
    "AUTHORED_CC0_V1--CC0-1.0--gg03-generation-generalization-v1",
    "AUTHORED_CC0_V1--CC0-1.0--question-answer-v1",
)
SAFE_TEST_ALLOWLIST = (
    "tests/test_w08_00_authority.py",
    "tests/test_w08_01_contract.py",
    "tests/test_w08_01_registry.py",
    "tests/test_w08_02_variation.py",
    "tests/test_w08_03_discourse.py",
    "tests/test_w08_04_recompute.py",
    "tests/test_a02_work_memory_content.py",
    "tests/test_a03_parser_revision.py",
    "tests/test_a08_memory_reparse.py",
    "tests/test_d02_md02_situation_state_adapter.py",
    "tests/test_r03_property_relation_runtime.py",
    "tests/test_r03_correction_recovery_absorption.py",
)
_PRIVATE_ROOT_MARKERS = (
    "w02_artifacts",
    "w03_artifacts",
    "w04_formal_",
    "w05_formal_",
    "w05_private_",
    "w06_formal_",
    "w06_private_",
    "w07_candidate_",
    "w07_private_",
    "w08_candidate_",
    "w08_formal_",
    "w08_private_",
)
_PUBLIC_MATERIALIZATION_RELATIVE_PATHS = (
    "ph2_dataset_artifacts",
    "ph2_p3ia_dataset_artifacts",
    "r02_storage_profile_artifacts",
    "ph2_dataset_raw/ZHWIKTIONARY_20260701/"
    "zhwiktionary-20260701.final-adapter-v1.pass-1.report.json",
    "ph2_dataset_raw/ZHWIKTIONARY_20260701/"
    "zhwiktionary-20260701.final-adapter-v1.pass-2.report.json",
)


class RecoveryValidationError(RuntimeError):
    """The recovery boundary or append-only evidence contract was violated."""


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise RecoveryValidationError("recovery path is not canonical")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise RecoveryValidationError("recovery path escapes the repository")
    return relative.as_posix()


def _normalized(path: str | os.PathLike[str]) -> str:
    normalized = os.path.normcase(os.path.abspath(os.fspath(path)))
    if normalized.startswith("\\\\?\\"):
        normalized = normalized[4:]
    return normalized


def _is_within(path: str, root: Path) -> bool:
    try:
        return os.path.commonpath((path, _normalized(root))) == _normalized(root)
    except ValueError:
        return False


def _load_future_paths() -> tuple[str, ...]:
    manifest_path = REPOSITORY_ROOT / GLOBAL_MANIFEST_RELATIVE_PATH
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryValidationError("cannot read the public D-03 inventory") from error
    by_pack = {
        item.get("pack_key"): item
        for item in manifest.get("pack_bindings", [])
        if isinstance(item, dict)
    }
    paths = {INCIDENT_TEST_RELATIVE_PATH}
    for pack_key in FUTURE_PACK_KEYS:
        binding = by_pack.get(pack_key)
        if not isinstance(binding, dict):
            raise RecoveryValidationError(f"missing future binding: {pack_key}")
        identity = binding.get("manifest_identity", {})
        paths.add(_safe_relative(identity.get("relative_path")))
        for key in (
            "source_ref_paths",
            "train_observation_paths",
            "dev_observation_paths",
            "held_out_observation_paths",
            "teacher_evidence_paths",
            "evaluator_label_paths",
        ):
            values = binding.get(key, [])
            if not isinstance(values, list):
                raise RecoveryValidationError(f"future path inventory drifted: {pack_key}")
            paths.update(_safe_relative(value) for value in values)
    return tuple(sorted(paths))


def _validate_allowlist(future_paths: tuple[str, ...]) -> tuple[Path, ...]:
    if len(SAFE_TEST_ALLOWLIST) != len(set(SAFE_TEST_ALLOWLIST)):
        raise RecoveryValidationError("safe test allowlist contains a duplicate")
    if set(SAFE_TEST_ALLOWLIST) & set(future_paths):
        raise RecoveryValidationError("future test entered the safe test allowlist")
    tests_root = REPOSITORY_ROOT / "tests"
    resolved: list[Path] = []
    for relative in SAFE_TEST_ALLOWLIST:
        canonical = _safe_relative(relative)
        path = REPOSITORY_ROOT / canonical
        if not path.is_file() or path.parent != tests_root:
            raise RecoveryValidationError(f"allowlisted test is missing or nested: {canonical}")
        resolved.append(path)
    return tuple(resolved)


def _read_head() -> str:
    head_path = REPOSITORY_ROOT / ".git" / "HEAD"
    try:
        head = head_path.read_text(encoding="ascii").strip()
    except OSError as error:
        raise RecoveryValidationError("cannot read public Git HEAD") from error
    if not head.startswith("ref: "):
        return head
    relative = _safe_relative(head[5:])
    ref_path = REPOSITORY_ROOT / ".git" / relative
    try:
        return ref_path.read_text(encoding="ascii").strip()
    except OSError as error:
        raise RecoveryValidationError("cannot resolve public Git HEAD") from error


class _AuditBoundary:
    def __init__(self, future_paths: tuple[str, ...], temp_root: Path) -> None:
        self.future_paths = {
            _normalized(REPOSITORY_ROOT / relative) for relative in future_paths
        }
        self.temp_root = temp_root
        self.system_temp_root = temp_root.parent
        self.future_payload_reads = 0
        self.private_payload_reads = 0
        self.formal_guard_consumed = 0
        self.teacher_calls = 0
        self.host_learning_writes = 0
        self.ephemeral_test_writes = 0
        self.publication_writes = 0
        self.public_materialization_writes = 0
        self.read_only_git_calls = 0
        self.blocked_network_calls = 0

    @staticmethod
    def _path_from_event(args: tuple[object, ...]) -> str | None:
        if not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return None
        try:
            return _normalized(os.fsdecode(args[0]))
        except (OSError, TypeError, ValueError):
            return None

    @staticmethod
    def _write_open(args: tuple[object, ...]) -> bool:
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        if isinstance(mode, str) and any(char in mode for char in "wax+"):
            return True
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        return isinstance(flags, int) and bool(flags & write_flags)

    def _private_payload(self, path: str) -> bool:
        if not _is_within(path, WORKSPACE_ROOT) or _is_within(path, REPOSITORY_ROOT):
            return False
        relative = os.path.relpath(path, _normalized(WORKSPACE_ROOT)).casefold()
        return any(marker in relative for marker in _PRIVATE_ROOT_MARKERS)

    @staticmethod
    def _w08_formal_guard(path: str) -> bool:
        parts = tuple(part.casefold() for part in Path(path).parts)
        has_w08_formal_root = any(
            part.startswith(("w08_candidate_", "w08_formal_", "w08_private_"))
            for part in parts
        )
        return has_w08_formal_root and any("guard" in part for part in parts)

    @staticmethod
    def _public_materialization(path: str) -> bool:
        return any(
            _is_within(path, WORKSPACE_ROOT / relative)
            for relative in _PUBLIC_MATERIALIZATION_RELATIVE_PATHS
        )

    def __call__(self, event: str, args: tuple[object, ...]) -> None:
        if event in {"socket.connect", "socket.connect_ex"}:
            self.blocked_network_calls += 1
            raise RecoveryValidationError("network access is forbidden during recovery")
        if event == "subprocess.Popen":
            command = args[1] if len(args) > 1 else ""
            if isinstance(command, (list, tuple)):
                command_line = " ".join(str(item) for item in command)
            else:
                command_line = str(command)
            cwd = args[2] if len(args) > 2 else None
            allowed = re.fullmatch(
                r"git(?:\.exe)? (?:rev-parse --verify "
                r"(?:[0-9a-f]{40}|HEAD|origin/master)|"
                r"merge-base --is-ancestor [0-9a-f]{40} [0-9a-f]{40})",
                command_line,
                flags=re.IGNORECASE,
            )
            if allowed and cwd is not None and _normalized(cwd) == _normalized(REPOSITORY_ROOT):
                self.read_only_git_calls += 1
                return
            raise RecoveryValidationError("only read-only git rev-parse is allowed")
        if event != "open":
            return
        path = self._path_from_event(args)
        if path is None:
            return
        write = self._write_open(args)
        if not write and path in self.future_paths:
            self.future_payload_reads += 1
            raise RecoveryValidationError("future W09 data/test read was blocked")
        if not write and self._private_payload(path):
            self.private_payload_reads += 1
            raise RecoveryValidationError("private/formal payload read was blocked")
        if write and self._w08_formal_guard(path):
            self.formal_guard_consumed += 1
            raise RecoveryValidationError("W08 formal guard write was blocked")
        if not write:
            return
        if path == _normalized(REPORT_PATH):
            self.publication_writes += 1
            return
        if _is_within(path, self.system_temp_root) or _is_within(
            path, REPOSITORY_TEST_TEMP_ROOT
        ):
            self.ephemeral_test_writes += 1
            return
        if self._public_materialization(path):
            self.public_materialization_writes += 1
            return
        self.host_learning_writes += 1
        raise RecoveryValidationError("non-temporary host write was blocked")


class _PytestEvidence:
    def __init__(self, allowlisted: tuple[Path, ...]) -> None:
        self.allowlisted = {_normalized(path) for path in allowlisted}
        self.collected = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.collection_errors = 0

    def pytest_collection_modifyitems(self, items: list[object]) -> None:
        collected_files = {
            _normalized(getattr(item, "path"))
            for item in items
            if getattr(item, "path", None) is not None
        }
        if not collected_files <= self.allowlisted:
            raise RecoveryValidationError("pytest collected a file outside the allowlist")
        self.collected = len(items)

    def pytest_collectreport(self, report: object) -> None:
        if getattr(report, "failed", False):
            self.collection_errors += 1

    def pytest_runtest_logreport(self, report: object) -> None:
        when = getattr(report, "when", "")
        if when == "call":
            if getattr(report, "passed", False):
                self.passed += 1
            elif getattr(report, "failed", False):
                self.failed += 1
            elif getattr(report, "skipped", False):
                self.skipped += 1
        elif when in {"setup", "teardown"} and getattr(report, "failed", False):
            self.failed += 1


def _publish_report(value: dict[str, Any]) -> str:
    payload = _canonical_bytes(value)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with REPORT_PATH.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise RecoveryValidationError("recovery report is append-only and already exists") from error
    return _sha256_bytes(payload)


def main() -> int:
    if len(sys.argv) != 1:
        raise RecoveryValidationError("recovery runner accepts no extra arguments")
    if REPORT_PATH.exists():
        raise RecoveryValidationError("recovery report is append-only and already exists")
    if Path.cwd().resolve() != REPOSITORY_ROOT:
        raise RecoveryValidationError("run recovery validation from the repository root")
    head = _read_head()
    if head != EXPECTED_IMPLEMENTATION_COMMIT:
        raise RecoveryValidationError("public HEAD is not the preserved W08-04 commit")
    if os.environ.get("PYTHONHASHSEED") != EXPECTED_HASH_SEED:
        raise RecoveryValidationError(
            "start the runner with PYTHONHASHSEED=0 so conftest does not spawn a child"
        )

    future_paths = _load_future_paths()
    allowlisted = _validate_allowlist(future_paths)
    temp_root = Path(tempfile.mkdtemp(prefix="w08-04-recovery-"))
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["TEMP"] = str(temp_root)
    os.environ["TMP"] = str(temp_root)
    tempfile.tempdir = str(temp_root)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPOSITORY_ROOT))

    boundary = _AuditBoundary(future_paths, temp_root)
    sys.addaudithook(boundary)
    evidence = _PytestEvidence(allowlisted)
    try:
        import pytest

        exit_code = int(pytest.main([
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:logging",
            "--basetemp",
            str(temp_root / "pytest"),
            *(str(path) for path in allowlisted),
        ], plugins=[evidence]))
        status = "PASS" if exit_code == 0 else "FAIL"
        report = {
            "append_only": True,
            "artifact_kind": "PH2_W08_04_VALIDATION_RECOVERY",
            "artifact_version": "PH2-W08-04-VALIDATION-RECOVERY-V6",
            "base_public_commit_sha1": head,
            "incident": {
                "invalidated_claim": "W08-00--W08-04 bounded regression: 189 passed",
                "offending_test": INCIDENT_TEST_RELATIVE_PATH,
                "previous_future_payload_reads": "NONZERO_UNCOUNTED",
                "previous_result_disposition": "INVALID_NOT_PASS_EVIDENCE",
                "w08_04_implementation_disposition": "PRESERVED",
            },
            "policy": {
                "collection_mode": "EXACT_FILE_ALLOWLIST_NO_GLOB",
                "future_path_count": len(future_paths),
                "future_path_set_sha256": _sha256_bytes(
                    _canonical_bytes({"paths": list(future_paths)})
                ),
                "test_allowlist": list(SAFE_TEST_ALLOWLIST),
            },
            "result": {
                "collection_errors": evidence.collection_errors,
                "collected_tests": evidence.collected,
                "failed_tests": evidence.failed,
                "passed_tests": evidence.passed,
                "pytest_exit_code": exit_code,
                "skipped_tests": evidence.skipped,
                "status": status,
            },
            "restricted_evidence": {
                "blocked_network_calls": boundary.blocked_network_calls,
                "ephemeral_test_writes": boundary.ephemeral_test_writes,
                "formal_guard_consumed": boundary.formal_guard_consumed,
                "future_payload_reads": boundary.future_payload_reads,
                "host_learning_writes": boundary.host_learning_writes,
                "private_payload_reads": boundary.private_payload_reads,
                "public_materialization_writes": boundary.public_materialization_writes,
                "read_only_git_calls": boundary.read_only_git_calls,
                "teacher_calls": boundary.teacher_calls,
            },
            "state": {
                "LANGUAGE_CAPABILITY_MASTERED": 0,
                "LANGUAGE_READINESS": 0,
                "OPEN_GENERATION": "NE_NOT_YET_EVALUABLE",
                "W08_STARTED": 0,
                "formal_w08_training_runs": 0,
                "next_restore_point": (
                    "W08-05 long context" if status == "PASS" else "W08-04 recovery blocked"
                ),
            },
        }
        report_sha256 = _publish_report(report)
        print(f"W08-04 recovery {status}: {evidence.passed}/{evidence.collected} passed")
        print(f"report={REPORT_RELATIVE_PATH}")
        print(f"sha256={report_sha256}")
        return 0 if status == "PASS" else 1
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
