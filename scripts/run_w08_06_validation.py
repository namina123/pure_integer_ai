"""Run append-only W08-06 validation under an exact read/write audit boundary."""
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
REPORT_RELATIVE_PATH = "data/ph2/manifests/d03_v1/w08_06_validation_v1.json"
REPORT_PATH = REPOSITORY_ROOT / REPORT_RELATIVE_PATH
GLOBAL_MANIFEST_RELATIVE_PATH = (
    "data/ph2/manifests/d03_v1/ph2_global_course_manifest_v1.json"
)
STAGE_MANIFEST_RELATIVE_PATH = (
    "data/ph2/manifests/d03_v1/stages/w08_stage_manifest_v1.json"
)
EXPECTED_BASE_COMMIT = "5f45da81b3f5a225fb517e1139eb086af2c79117"
EXPECTED_HASH_SEED = "0"
REPOSITORY_TEST_TEMP_ROOT = REPOSITORY_ROOT / ".pytest_tmp_safe"
FUTURE_PACK_KEYS = (
    "AUTHORED_CC0_V1--CC0-1.0--free-text-hierarchy-recall-v1",
    "AUTHORED_CC0_V1--CC0-1.0--generation-postcheck-v1",
    "AUTHORED_CC0_V1--CC0-1.0--gg03-generation-generalization-v1",
    "AUTHORED_CC0_V1--CC0-1.0--question-answer-v1",
)
FUTURE_SOURCE_PATHS = (
    "src/pure_integer_ai/experiments/"
    "ph2_authored_free_text_hierarchy_recall_course.py",
)
FUTURE_TEST_PATHS = (
    "tests/test_d02_p3ia_free_text_hierarchy_recall_runtime.py",
    "tests/test_r04_authorized_center_generation_absorption.py",
    "tests/test_r06_long_context_absorption.py",
)
SAFE_TEST_ALLOWLIST = (
    "tests/test_w08_00_authority.py",
    "tests/test_w08_01_contract.py",
    "tests/test_w08_01_registry.py",
    "tests/test_w08_02_variation.py",
    "tests/test_w08_03_discourse.py",
    "tests/test_w08_04_recompute.py",
    "tests/test_w08_05_long_context.py",
    "tests/test_w08_06_stage6.py",
    "tests/test_d02_md03_directional_center_adapter.py",
    "tests/test_d03_lc16_successor_overlay.py",
    "tests/test_f00_generation_postcheck.py",
    "tests/test_f00_question_answer_runtime.py",
    "tests/test_g04_generation_postcheck.py",
    "tests/test_k02_sealed_segment_paging.py",
    "tests/test_k04_memory_hot_set.py",
)
IMPLEMENTATION_FILES = (
    "scripts/run_w08_06_validation.py",
    "src/pure_integer_ai/experiments/ph2_w08_lc16.py",
    "src/pure_integer_ai/experiments/ph2_w08_open_generation.py",
    "src/pure_integer_ai/experiments/ph2_w08_open_generation_contract.py",
    "src/pure_integer_ai/experiments/ph2_w08_p3ia.py",
    "src/pure_integer_ai/experiments/ph2_w08_p3ia_contract.py",
    "src/pure_integer_ai/experiments/ph2_w08_p3ia_training.py",
    "src/pure_integer_ai/experiments/ph2_w08_stage6.py",
    "tests/test_w08_06_stage6.py",
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


class W08ValidationError(RuntimeError):
    """The W08-06 validation boundary or append-only report was violated."""


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise W08ValidationError("validation path is not canonical")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise W08ValidationError("validation path escapes the repository")
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


def _read_json(relative: str) -> dict[str, Any]:
    try:
        value = json.loads((REPOSITORY_ROOT / relative).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise W08ValidationError(f"cannot read public manifest: {relative}") from error
    if not isinstance(value, dict):
        raise W08ValidationError("public manifest is not an object")
    return value


def _binding_paths(binding: dict[str, Any], fields: tuple[str, ...]) -> set[str]:
    paths: set[str] = set()
    for field in fields:
        values = binding.get(field, [])
        if not isinstance(values, list):
            raise W08ValidationError(f"payload path inventory drifted: {field}")
        paths.update(_safe_relative(value) for value in values)
    return paths


def _load_boundaries() -> tuple[tuple[str, ...], tuple[str, ...]]:
    manifest = _read_json(GLOBAL_MANIFEST_RELATIVE_PATH)
    stage = _read_json(STAGE_MANIFEST_RELATIVE_PATH)
    by_pack = {
        item.get("pack_key"): item
        for item in manifest.get("pack_bindings", [])
        if isinstance(item, dict)
    }
    future: set[str] = set()
    all_payload_fields = (
        "source_ref_paths",
        "train_observation_paths",
        "dev_observation_paths",
        "held_out_observation_paths",
        "teacher_evidence_paths",
        "evaluator_label_paths",
    )
    for pack_key in FUTURE_PACK_KEYS:
        binding = by_pack.get(pack_key)
        if not isinstance(binding, dict):
            raise W08ValidationError(f"missing future binding: {pack_key}")
        identity = binding.get("manifest_identity", {})
        future.add(_safe_relative(identity.get("relative_path")))
        future.update(_binding_paths(binding, all_payload_fields))

    visibility = stage.get("data_visibility", {})
    private_evaluator: set[str] = set()
    restricted_fields = (
        "dev_observation_paths",
        "held_out_observation_paths",
        "evaluator_label_paths",
    )
    restricted_pack_keys = set(
        (
            *visibility.get("dev_pack_keys", []),
            *visibility.get("held_out_pack_keys", []),
            *visibility.get("evaluator_pack_keys", []),
        )
    )
    for pack_key in restricted_pack_keys:
        binding = by_pack.get(pack_key)
        if not isinstance(binding, dict):
            raise W08ValidationError(f"missing W08 restricted binding: {pack_key}")
        private_evaluator.update(_binding_paths(binding, restricted_fields))
    return tuple(sorted(future)), tuple(sorted(private_evaluator))


def _validate_file_inventory(paths: tuple[str, ...]) -> tuple[Path, ...]:
    if len(paths) != len(set(paths)):
        raise W08ValidationError("validation file inventory contains a duplicate")
    result = []
    for relative in paths:
        canonical = _safe_relative(relative)
        path = REPOSITORY_ROOT / canonical
        if not path.is_file():
            raise W08ValidationError(f"validation file is missing: {canonical}")
        result.append(path)
    return tuple(result)


def _read_head() -> str:
    head_path = REPOSITORY_ROOT / ".git" / "HEAD"
    try:
        head = head_path.read_text(encoding="ascii").strip()
    except OSError as error:
        raise W08ValidationError("cannot read public Git HEAD") from error
    if not head.startswith("ref: "):
        return head
    ref_path = REPOSITORY_ROOT / ".git" / _safe_relative(head[5:])
    try:
        return ref_path.read_text(encoding="ascii").strip()
    except OSError as error:
        raise W08ValidationError("cannot resolve public Git HEAD") from error


def _implementation_identity() -> tuple[dict[str, Any], ...]:
    result = []
    for relative, path in zip(
        IMPLEMENTATION_FILES,
        _validate_file_inventory(IMPLEMENTATION_FILES),
    ):
        payload = path.read_bytes()
        result.append(
            {
                "relative_path": relative,
                "sha256": _sha256_bytes(payload),
                "size_bytes": len(payload),
            }
        )
    return tuple(result)


class _AuditBoundary:
    def __init__(
        self,
        future_pack_paths: tuple[str, ...],
        private_evaluator_paths: tuple[str, ...],
        temp_root: Path,
    ) -> None:
        self.future_pack_paths = {
            _normalized(REPOSITORY_ROOT / relative) for relative in future_pack_paths
        }
        self.future_source_paths = {
            _normalized(REPOSITORY_ROOT / relative) for relative in FUTURE_SOURCE_PATHS
        }
        self.future_test_paths = {
            _normalized(REPOSITORY_ROOT / relative) for relative in FUTURE_TEST_PATHS
        }
        self.private_evaluator_paths = {
            _normalized(REPOSITORY_ROOT / relative)
            for relative in private_evaluator_paths
        }
        self.temp_root = temp_root
        self.system_temp_root = temp_root.parent
        self.future_pack_reads = 0
        self.future_source_reads = 0
        self.future_test_reads = 0
        self.private_evaluator_reads = 0
        self.formal_guard_writes = 0
        self.teacher_calls = 0
        self.host_learning_writes = 0
        self.memory_learning_writes = 0
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

    @staticmethod
    def _compiled_variant(path: str, source_paths: set[str]) -> bool:
        candidate = Path(path)
        if candidate.suffix.casefold() != ".pyc" or candidate.parent.name != "__pycache__":
            return False
        stem = candidate.name.split(".", 1)[0].casefold()
        return any(Path(source).stem.casefold() == stem for source in source_paths)

    def _private_root(self, path: str) -> bool:
        if not _is_within(path, WORKSPACE_ROOT) or _is_within(path, REPOSITORY_ROOT):
            return False
        relative = os.path.relpath(path, _normalized(WORKSPACE_ROOT)).casefold()
        return any(marker in relative for marker in _PRIVATE_ROOT_MARKERS)

    @staticmethod
    def _w08_formal_guard(path: str) -> bool:
        parts = tuple(part.casefold() for part in Path(path).parts)
        has_formal_root = any(
            part.startswith(("w08_candidate_", "w08_formal_", "w08_private_"))
            for part in parts
        )
        return has_formal_root and any("guard" in part for part in parts)

    @staticmethod
    def _public_materialization(path: str) -> bool:
        return any(
            _is_within(path, WORKSPACE_ROOT / relative)
            for relative in _PUBLIC_MATERIALIZATION_RELATIVE_PATHS
        )

    @staticmethod
    def _memory_learning_path(path: str) -> bool:
        lowered = path.casefold()
        return "memory_learning" in lowered or "memory-learning" in lowered

    def __call__(self, event: str, args: tuple[object, ...]) -> None:
        if event in {"socket.connect", "socket.connect_ex"}:
            self.blocked_network_calls += 1
            raise W08ValidationError("network access is forbidden during W08-06 validation")
        if event == "subprocess.Popen":
            command = args[1] if len(args) > 1 else ""
            command_line = (
                " ".join(str(item) for item in command)
                if isinstance(command, (list, tuple))
                else str(command)
            )
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
            raise W08ValidationError("only bounded read-only Git calls are allowed")
        if event != "open":
            return
        path = self._path_from_event(args)
        if path is None:
            return
        write = self._write_open(args)
        if not write and path in self.future_pack_paths:
            self.future_pack_reads += 1
            raise W08ValidationError("future pack read was blocked")
        if not write and (
            path in self.future_source_paths
            or self._compiled_variant(path, self.future_source_paths)
        ):
            self.future_source_reads += 1
            raise W08ValidationError("future source fixture read was blocked")
        if not write and (
            path in self.future_test_paths
            or self._compiled_variant(path, self.future_test_paths)
        ):
            self.future_test_reads += 1
            raise W08ValidationError("future transitive test read was blocked")
        if not write and (
            path in self.private_evaluator_paths or self._private_root(path)
        ):
            self.private_evaluator_reads += 1
            raise W08ValidationError("private/evaluator read was blocked")
        if write and self._w08_formal_guard(path):
            self.formal_guard_writes += 1
            raise W08ValidationError("W08 formal guard write was blocked")
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
        if self._memory_learning_path(path):
            self.memory_learning_writes += 1
            raise W08ValidationError("Memory-learning write was blocked")
        self.host_learning_writes += 1
        raise W08ValidationError("non-temporary host write was blocked")


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
            raise W08ValidationError("pytest collected a file outside the allowlist")
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
    try:
        with REPORT_PATH.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise W08ValidationError("W08-06 report is append-only and already exists") from error
    return _sha256_bytes(payload)


def main() -> int:
    if len(sys.argv) != 1:
        raise W08ValidationError("W08-06 validation accepts no extra arguments")
    if REPORT_PATH.exists():
        raise W08ValidationError("W08-06 report is append-only and already exists")
    if Path.cwd().resolve() != REPOSITORY_ROOT:
        raise W08ValidationError("run W08-06 validation from the repository root")
    head = _read_head()
    if head != EXPECTED_BASE_COMMIT:
        raise W08ValidationError("public HEAD is not the W08-05 base commit")
    if os.environ.get("PYTHONHASHSEED") != EXPECTED_HASH_SEED:
        raise W08ValidationError("start validation with PYTHONHASHSEED=0")

    future_pack_paths, private_evaluator_paths = _load_boundaries()
    allowlisted = _validate_file_inventory(SAFE_TEST_ALLOWLIST)
    if set(SAFE_TEST_ALLOWLIST).intersection(FUTURE_TEST_PATHS):
        raise W08ValidationError("future test entered the safe allowlist")
    implementation = _implementation_identity()
    temp_root = Path(tempfile.mkdtemp(prefix="w08-06-validation-"))
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["TEMP"] = str(temp_root)
    os.environ["TMP"] = str(temp_root)
    tempfile.tempdir = str(temp_root)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPOSITORY_ROOT))

    boundary = _AuditBoundary(future_pack_paths, private_evaluator_paths, temp_root)
    sys.addaudithook(boundary)
    evidence = _PytestEvidence(allowlisted)
    try:
        import pytest

        exit_code = int(
            pytest.main(
                [
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "-p",
                    "no:logging",
                    "--basetemp",
                    str(temp_root / "pytest"),
                    *(str(path) for path in allowlisted),
                ],
                plugins=[evidence],
            )
        )
        restricted = {
            "formal_guard_writes": boundary.formal_guard_writes,
            "future_pack_reads": boundary.future_pack_reads,
            "future_source_reads": boundary.future_source_reads,
            "future_test_reads": boundary.future_test_reads,
            "host_learning_writes": boundary.host_learning_writes,
            "memory_learning_writes": boundary.memory_learning_writes,
            "private_evaluator_reads": boundary.private_evaluator_reads,
            "teacher_calls": boundary.teacher_calls,
        }
        status = "PASS" if exit_code == 0 and not any(restricted.values()) else "FAIL"
        report = {
            "append_only": True,
            "artifact_kind": "PH2_W08_06_VALIDATION",
            "artifact_version": "PH2-W08-06-VALIDATION-V1",
            "base_public_commit_sha1": head,
            "implementation_files": list(implementation),
            "policy": {
                "collection_mode": "EXACT_FILE_ALLOWLIST_NO_GLOB",
                "future_pack_path_count": len(future_pack_paths),
                "future_pack_path_set_sha256": _sha256_bytes(
                    _canonical_bytes({"paths": list(future_pack_paths)})
                ),
                "future_source_paths": list(FUTURE_SOURCE_PATHS),
                "future_test_paths": list(FUTURE_TEST_PATHS),
                "private_evaluator_path_count": len(private_evaluator_paths),
                "private_evaluator_path_set_sha256": _sha256_bytes(
                    _canonical_bytes({"paths": list(private_evaluator_paths)})
                ),
                "test_allowlist": list(SAFE_TEST_ALLOWLIST),
            },
            "result": {
                "collected_tests": evidence.collected,
                "collection_errors": evidence.collection_errors,
                "failed_tests": evidence.failed,
                "passed_tests": evidence.passed,
                "pytest_exit_code": exit_code,
                "skipped_tests": evidence.skipped,
                "status": status,
            },
            "restricted_evidence": {
                **restricted,
                "blocked_network_calls": boundary.blocked_network_calls,
                "ephemeral_test_writes": boundary.ephemeral_test_writes,
                "public_materialization_writes": boundary.public_materialization_writes,
                "read_only_git_calls": boundary.read_only_git_calls,
            },
            "state": {
                "LANGUAGE_CAPABILITY_MASTERED": 0,
                "LANGUAGE_READINESS": 0,
                "OPEN_GENERATION": "NE_NOT_YET_EVALUABLE",
                "W08_STARTED": 0,
                "W09_STARTED": 0,
                "formal_w08_training_runs": 0,
                "next_restore_point": (
                    "W08-07 transaction/runtime"
                    if status == "PASS"
                    else "W08-06 validation recovery"
                ),
            },
        }
        report_sha256 = _publish_report(report)
        print(f"W08-06 validation {status}: {evidence.passed}/{evidence.collected} passed")
        print(f"report={REPORT_RELATIVE_PATH}")
        print(f"sha256={report_sha256}")
        return 0 if status == "PASS" else 1
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
