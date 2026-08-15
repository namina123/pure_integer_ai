"""Audacity atom-validation 的不可重跑 guard、授权冻结与独立评分。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_identifiability_audit import (
    read_normalization_recovery_v7_atom_identifiability_audit_state,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_validation_family import (
    read_audacity_atom_validation_family_freeze,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_validation_records import (
    derive_audacity_atom_validation_authorizations,
    score_audacity_atom_validation_authorizations,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_validation_source_pack import (
    materialize_audacity_atom_validation_labels_after_authorization_freeze,
    read_audacity_atom_validation_held_inputs_after_family_freeze,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


def _sha256(payload: bytes) -> str:
    """返回 guard、authorization 或 publication identity。"""
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl(
        path: Path,
        values: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """独占写入无 surface score records 并返回物理 commitment。"""
    with path.open("xb") as handle:
        for value in values:
            handle.write(canonical_json_line(value))
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "record_count": len(values),
        "relative_path": path.name,
        "sha256": _sha256(payload),
    }


def _write_json(path: Path, value: dict[str, object]) -> str:
    """独占写入一份规范 JSON object。"""
    encoded = canonical_json_line(value)
    with path.open("xb") as handle:
        handle.write(encoded)
    return _sha256(encoded)


def _require_k_root(value: str | Path) -> Path:
    """要求正式运行根为 K 盘。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "Audacity atom-validation runner root 必须是 K 盘目录")
    return root


def _overlap(left: Path, right: Path) -> bool:
    """判断 publication 与任一输入是否相同或互相包含。"""
    return (left == right or left.is_relative_to(right)
            or right.is_relative_to(left))


def run_audacity_atom_validation_once(
        *,
        run_root: str | Path,
        repository_root: str | Path,
        family_freeze_dir: str | Path,
        expected_family_manifest_sha256: str,
        source_pack_dir: str | Path,
        expected_source_manifest_sha256: str,
        atom_audit_dir: str | Path,
        expected_atom_manifest_sha256: str,
        commitment_v2_dir: str | Path,
        expected_commitment_v2_manifest_sha256: str,
        family_v1_dir: str | Path,
        expected_family_v1_manifest_sha256: str,
        family_v2_dir: str | Path,
        expected_family_v2_manifest_sha256: str,
        failed_v2_publication_dir: str | Path,
        expected_v2_guard_sha256: str,
        expected_v2_failure_sha256: str,
        training_protocol_dir: str | Path,
        variable_structure_audit_dir: str | Path,
        neutral_semantic_source_audit_dir: str | Path,
        godot_source_pack_dir: str | Path,
        libreoffice_source_pack_dir: str | Path,
        vscode_source_pack_dir: str | Path,
        thunderbird_source_pack_dir: str | Path,
        vscode_source_archive_path: str | Path,
        vscode_source_root: str | Path,
        typescript_parser_root: str | Path,
        opencc_source_pack_dir: str | Path,
        unimorph_english_dir: str | Path,
        publication_dir: str | Path,
        node_executable: str | Path = "node",
        ) -> dict[str, object]:
    """family strict-read 后消费唯一 guard，再按物理顺序运行一次。"""
    root = _require_k_root(run_root)
    publication = Path(publication_dir).resolve()
    source = Path(source_pack_dir).resolve()
    atom = Path(atom_audit_dir).resolve()
    commitment = Path(commitment_v2_dir).resolve()
    family = Path(family_freeze_dir).resolve()
    family_v1 = Path(family_v1_dir).resolve()
    family_v2 = Path(family_v2_dir).resolve()
    failed_v2 = Path(failed_v2_publication_dir).resolve()
    directory_inputs = tuple(Path(value).resolve() for value in (
        training_protocol_dir, variable_structure_audit_dir,
        neutral_semantic_source_audit_dir, godot_source_pack_dir,
        libreoffice_source_pack_dir, vscode_source_pack_dir,
        thunderbird_source_pack_dir, vscode_source_root,
        typescript_parser_root, opencc_source_pack_dir,
        unimorph_english_dir))
    archive = Path(vscode_source_archive_path).resolve()
    if (not publication.is_relative_to(root) or publication.exists()
            or any(not path.is_dir() or not path.is_relative_to(root)
                   for path in (
                       source, atom, commitment, family, family_v1, family_v2,
                       failed_v2,
                       *directory_inputs))
            or not archive.is_file() or not archive.is_relative_to(root)
            or any(_overlap(publication, path) for path in (
                source, atom, commitment, family, family_v1, family_v2,
                failed_v2,
                *directory_inputs, archive))):
        raise BroadQaExternalDataError(
            "Audacity atom-validation runner path 非法或已消费")
    family_manifest = read_audacity_atom_validation_family_freeze(
        family,
        expected_manifest_sha256=expected_family_manifest_sha256,
        repository_root=repository_root,
        source_pack_dir=source,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        atom_audit_dir=atom,
        expected_atom_manifest_sha256=expected_atom_manifest_sha256,
        commitment_v2_dir=commitment,
        expected_commitment_v2_manifest_sha256=(
            expected_commitment_v2_manifest_sha256),
        family_v1_dir=family_v1,
        expected_family_v1_manifest_sha256=(
            expected_family_v1_manifest_sha256),
        family_v2_dir=family_v2,
        expected_family_v2_manifest_sha256=(
            expected_family_v2_manifest_sha256),
        failed_v2_publication_dir=failed_v2,
        expected_v2_guard_sha256=expected_v2_guard_sha256,
        expected_v2_failure_sha256=expected_v2_failure_sha256,
    )
    expected_publication = (
        root / str(family_manifest["publication_contract"]["relative_path"])
    ).resolve()
    if publication != expected_publication:
        raise BroadQaExternalDataError(
            "Audacity atom-validation publication identity 未按 family 冻结")
    publication.mkdir()
    guard = {
        "family_commitment_sha256": family_manifest[
            "family_commitment_sha256"],
        "family_manifest_sha256": expected_family_manifest_sha256,
        "format_version": 1,
        "run_ordinal": 1,
        "status": "FORMAL_RUN_IDENTITY_CONSUMED_BEFORE_HELD_INPUT_OR_LABEL_READ",
    }
    guard_sha = _write_json(publication / "run-000001.guard.json", guard)
    try:
        atom_manifest, _atom_outputs, state = (
            read_normalization_recovery_v7_atom_identifiability_audit_state(
                atom,
                training_protocol_dir=training_protocol_dir,
                variable_structure_audit_dir=variable_structure_audit_dir,
                neutral_semantic_source_audit_dir=(
                    neutral_semantic_source_audit_dir),
                godot_source_pack_dir=godot_source_pack_dir,
                libreoffice_source_pack_dir=libreoffice_source_pack_dir,
                vscode_source_pack_dir=vscode_source_pack_dir,
                thunderbird_source_pack_dir=thunderbird_source_pack_dir,
                vscode_source_archive_path=vscode_source_archive_path,
                vscode_source_root=vscode_source_root,
                typescript_parser_root=typescript_parser_root,
                opencc_source_pack_dir=opencc_source_pack_dir,
                unimorph_english_dir=unimorph_english_dir,
                expected_manifest_sha256=expected_atom_manifest_sha256,
                node_executable=node_executable,
            ))
        source_manifest, held_inputs, held_reads = (
            read_audacity_atom_validation_held_inputs_after_family_freeze(
                source,
                expected_manifest_sha256=expected_source_manifest_sha256,
            ))
        authorizations, authorization_census = (
            derive_audacity_atom_validation_authorizations(
                observations=state["observations"],
                fragments=state["fragments"],
                plans=state["plans"],
                held_inputs=held_inputs,
                opencc_routes=state["opencc_routes"],
                morphology_by_form=state["morphology"],
            ))
        denominator_count = int(
            family_manifest["denominator"]["record_count"])
        if (len(held_inputs) != denominator_count
                or len(authorizations) != denominator_count
                or authorization_census.get("authorization_count")
                != denominator_count
                or authorization_census.get("held_label_read_count") != 0):
            raise BroadQaExternalDataError(
                "Audacity atom-validation authorization denominator 未闭合")
        authorization_commitment = _sha256(canonical_json_bytes(
            authorizations))
        authorization_publication = {
            "atom_manifest_sha256": atom_manifest["manifest_sha256"],
            "authorization_census": authorization_census,
            "authorization_commitment_sha256": authorization_commitment,
            "authorization_record_count": len(authorizations),
            "family_manifest_sha256": expected_family_manifest_sha256,
            "format_version": 1,
            "guard_sha256": guard_sha,
            "held_input_reads": held_reads,
            "held_label_read_count": 0,
            "source_manifest_sha256": source_manifest["manifest_sha256"],
            "surface_published": 0,
        }
        authorization_sha = _write_json(
            publication / "run-000001.authorization.json",
            authorization_publication)
        labels, label_reads = (
            materialize_audacity_atom_validation_labels_after_authorization_freeze(
                source,
                expected_manifest_sha256=expected_source_manifest_sha256,
                held_inputs=held_inputs,
            ))
        records, scoring = score_audacity_atom_validation_authorizations(
            authorizations,
            labels_by_pair=labels,
            expected_denominator_count=denominator_count,
        )
        score_artifact = _write_jsonl(
            publication / "run-000001.scores.jsonl", records)
        aggregate = {
            "authorization_commitment_sha256": authorization_commitment,
            "authorization_publication_sha256": authorization_sha,
            "family_manifest_sha256": expected_family_manifest_sha256,
            "format_version": 1,
            "guard_sha256": guard_sha,
            "label_reads": label_reads,
            "scoring": scoring,
            "score_artifact": score_artifact,
            "surface_published": 0,
            "validation_run_count": 1,
        }
        aggregate_sha = _write_json(
            publication / "run-000001.aggregate.json", aggregate)
        return {**aggregate, "aggregate_sha256": aggregate_sha}
    except Exception as error:
        failure = {
            "exception_message_sha256": _sha256(
                str(error).encode("utf-8")),
            "exception_type": type(error).__name__,
            "family_manifest_sha256": expected_family_manifest_sha256,
            "format_version": 1,
            "guard_sha256": guard_sha,
            "status": "FORMAL_RUN_FAILED_AFTER_GUARD_NO_RERUN",
        }
        failure_path = publication / "run-000001.failure.json"
        if not failure_path.exists():
            _write_json(failure_path, failure)
        raise


__all__ = ["run_audacity_atom_validation_once"]
