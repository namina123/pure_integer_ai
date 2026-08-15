"""测试 VS Code 官方英文 source binding feasibility。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from zipfile import ZipFile, ZipInfo

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_vscode_english_binding_audit
    as audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_neutral_source_projection_records import (
    VSCODE_SOURCE_FAMILY,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_vscode_english_binding_records import (
    BINDING_CONFLICT,
    BINDING_MISSING,
    BINDING_NOT_VSCODE,
    BINDING_UNIQUE,
    derive_vscode_english_binding_feasibility,
    inspect_vscode_typescript_source,
    run_vscode_typescript_ast_extractor,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def _sha256(payload: bytes) -> str:
    """返回测试 identity。"""
    return hashlib.sha256(payload).hexdigest()


def _source(relative_path: str) -> dict[str, object]:
    """构造 AST/source 对齐所需的内部源码记录。"""
    module = relative_path[4:].rsplit(".", 1)[0]
    digest = _sha256(relative_path.encode("utf-8"))
    return {
        "bytes": 1,
        "git_blob_sha1": "1" * 40,
        "module": module,
        "module_path_sha256": _sha256(module.encode("utf-8")),
        "record_id": digest,
        "relative_path": relative_path,
        "relative_path_sha256": digest,
        "sha256": "2" * 64,
    }


def _pair(index: int, module: str, key: str, *, main: bool = True) \
        -> dict[str, object]:
    """构造一个 eligible VS Code localization pair。"""
    return {
        "json_path": [module, key],
        "json_path_sha256": f"{index + 20:064x}",
        "pair_id": f"{index + 1:064x}",
        "training_eligible": 1,
        "translation_relative_path": (
            "translations/main.i18n.json" if main
            else "translations/extensions/example.i18n.json"),
    }


def _ast(
        bindings: list[dict[str, object]],
        *,
        files: int,
        unsupported: list[dict[str, object]] | None = None,
        diagnostics: list[dict[str, object]] | None = None,
        ) -> dict[str, object]:
    """构造严格 AST extractor 输出。"""
    unsupported = unsupported or []
    diagnostics = diagnostics or []
    return {
        "binding_key_count": len({
            (item["module"], item["key"]) for item in bindings}),
        "bindings": bindings,
        "call_count": len(bindings) + len(unsupported),
        "file_count": files,
        "parse_diagnostics": diagnostics,
        "typescript_version": "6.0.3",
        "unsupported": unsupported,
    }


def _call(
        source: dict[str, object],
        key: str,
        message: str,
        position: int,
        ) -> dict[str, object]:
    """构造一个静态 localize AST occurrence。"""
    return {
        "callee": "localize",
        "key": key,
        "message": message,
        "module": source["module"],
        "position": position,
        "relative_path": source["relative_path"],
    }


def test_derivation_separates_unique_conflict_missing_and_extension() -> None:
    """main 缺失与未选择 extension 分账，绑定不授权 transformation。"""
    source_a = _source("src/vs/a.ts")
    source_b = _source("src/vs/b.ts")
    pairs = (
        _pair(0, "vs/a", "same"),
        _pair(1, "vs/b", "conflict"),
        _pair(2, "vs/c", "missing"),
        _pair(3, "extension/example", "outside", main=False),
    )
    bindings = [
        _call(source_a, "same", "One", 1),
        _call(source_a, "same", "One", 2),
        _call(source_b, "conflict", "Left", 3),
        _call(source_b, "conflict", "Right", 4),
    ]
    proposals = ({
        "held_out_observation_id": "obs-vscode",
        "held_out_source_family": VSCODE_SOURCE_FAMILY,
        "source_pair_id": pairs[0]["pair_id"],
    }, {
        "held_out_observation_id": "obs-other",
        "held_out_source_family": "OTHER_PROJECT",
        "source_pair_id": "9" * 64,
    })
    source_records, main, proposal, census, summary = (
        derive_vscode_english_binding_feasibility(
            source_files=(source_a, source_b),
            ast_result=_ast(bindings, files=2),
            vscode_pairs=pairs,
            proposals=proposals,
        ))
    by_pair = {item["pair_id"]: item for item in main}
    assert by_pair[pairs[0]["pair_id"]]["binding_outcome"] == BINDING_UNIQUE
    assert by_pair[pairs[1]["pair_id"]]["binding_outcome"] == BINDING_CONFLICT
    assert by_pair[pairs[2]["pair_id"]]["binding_outcome"] == BINDING_MISSING
    assert all("module" not in item and "relative_path" not in item
               for item in source_records)
    assert {item["scope"]: item["outside_source_selection_count"]
            for item in census}[
                "EXTENSION_I18N_OUTSIDE_PINNED_SRC_SELECTION"] == 1
    assert summary["source_binding_authorizes_transformation"] == 0
    assert summary["proposal_binding_outcome_counts"] == {
        BINDING_NOT_VSCODE: 1, BINDING_UNIQUE: 1}
    assert all(item["source_binding_authorizes_transformation"] == 0
               for item in proposal)


def test_derivation_rejects_diagnostic_and_dynamic_localize() -> None:
    """malformed AST 与动态 message 不能被记为 missing 或 PASS。"""
    source = _source("src/vs/a.ts")
    pair = _pair(0, "vs/a", "key")
    unsupported = [{
        "position": 1,
        "reason": "DYNAMIC_OR_MISSING_MESSAGE",
        "relative_path": source["relative_path"],
    }]
    with pytest.raises(BroadQaExternalDataError, match="动态 localize"):
        derive_vscode_english_binding_feasibility(
            source_files=(source,),
            ast_result=_ast([], files=1, unsupported=unsupported),
            vscode_pairs=(pair,), proposals=())
    diagnostic = [{
        "code": 1109, "position": 1,
        "relative_path": source["relative_path"],
    }]
    with pytest.raises(BroadQaExternalDataError, match="diagnostic"):
        derive_vscode_english_binding_feasibility(
            source_files=(source,),
            ast_result=_ast([], files=1, diagnostics=diagnostic),
            vscode_pairs=(pair,), proposals=())


def _write_zip(path: Path, rows: list[tuple[str | ZipInfo, bytes]]) -> None:
    """写入小型 archive fixture。"""
    with ZipFile(path, "w") as archive:
        for name, payload in rows:
            archive.writestr(name, payload)


def test_source_archive_matches_tree_and_rejects_escape_or_symlink(
        tmp_path: Path,
        ) -> None:
    """source selection 逐字节对齐并拒绝越界、selected symlink。"""
    root = tmp_path / "source"
    path = root / "src" / "vs" / "a.ts"
    path.parent.mkdir(parents=True)
    path.write_text("localize('a', 'A');", encoding="utf-8")
    archive_path = tmp_path / "source.zip"
    _write_zip(archive_path, [
        ("fixture/", b""),
        ("fixture/src/vs/a.ts", path.read_bytes()),
    ])
    records, summary = inspect_vscode_typescript_source(
        archive_path=archive_path, source_root=root,
        archive_prefix="fixture/")
    assert len(records) == 1
    assert summary["selected_source_file_count"] == 1

    escaped = tmp_path / "escaped.zip"
    _write_zip(escaped, [("fixture/../escape.ts", b"x")])
    with pytest.raises(BroadQaExternalDataError, match="member 非法"):
        inspect_vscode_typescript_source(
            archive_path=escaped, source_root=root,
            archive_prefix="fixture/")

    symlink = ZipInfo("fixture/src/vs/link.ts")
    symlink.create_system = 3
    symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    linked = tmp_path / "linked.zip"
    _write_zip(linked, [(symlink, b"a.ts")])
    with pytest.raises(BroadQaExternalDataError, match="selected source member"):
        inspect_vscode_typescript_source(
            archive_path=linked, source_root=root,
            archive_prefix="fixture/")


def test_real_ast_extractor_reports_conflict_dynamic_and_parse_error(
        tmp_path: Path,
        ) -> None:
    """可用外部 parser fixture 时，真实 AST 覆盖三种 fail-closed 情形。"""
    parser = os.environ.get("PURE_INTEGER_AI_TYPESCRIPT_PARSER_ROOT")
    if not parser or not Path(parser).is_dir() or shutil.which("node") is None:
        pytest.skip("TypeScript parser external fixture unavailable")
    root = tmp_path / "source"
    path = root / "src" / "vs" / "a.ts"
    path.parent.mkdir(parents=True)
    path.write_text(
        "localize('same', 'Left');\n"
        "localize('same', 'Right');\n"
        "localize('dynamic', message);\n"
        "const broken = ;\n",
        encoding="utf-8",
    )
    source = _source("src/vs/a.ts")
    result = run_vscode_typescript_ast_extractor(
        source_root=root, parser_root=parser,
        source_files=(source,), node_executable="node")
    assert len(result["bindings"]) == 2
    assert len(result["unsupported"]) == 1
    assert result["parse_diagnostics"]


def test_parser_version_drift_fails_closed(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """即使物理层被隔离，npm version/integrity 漂移仍拒绝。"""
    parser = tmp_path / "parser"
    parser.mkdir()
    (parser / "package.json").write_text(
        json.dumps({"dependencies": {"@typescript/typescript6": "9.9.9"}}),
        encoding="utf-8")
    (parser / "package-lock.json").write_text(
        json.dumps({"packages": {}}), encoding="utf-8")
    monkeypatch.setattr(
        audit, "_physical_identity", lambda *_args, **_kwargs: {})
    with pytest.raises(BroadQaExternalDataError, match="version/license"):
        audit._validate_parser(parser)


def _fake_outputs() -> tuple[
        dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """构造 publisher/reader 的小型稳定输出。"""
    return {
        "source-files.jsonl": ({"record_id": "1" * 64},),
        "main-bindings.jsonl": ({"record_id": "2" * 64},),
        "proposal-bindings.jsonl": ({"record_id": "3" * 64},),
        "binding-census.jsonl": ({"record_id": "4" * 64},),
    }, {
        "audit_outcome": (
            "OFFICIAL_ENGLISH_SOURCE_BINDING_FEASIBILITY_PASS_"
            "AUTHORIZATION_NE"),
        "binding": {"source_binding_authorizes_transformation": 0},
        "source_or_message_surface_published": 0,
    }


def _audit_paths(tmp_path: Path) -> tuple[list[Path], Path, Path]:
    """创建六个目录输入、archive 与 target。"""
    directories = [tmp_path / name for name in (
        "protocol", "variable", "semantic", "vscode", "source", "parser")]
    for path in directories:
        path.mkdir()
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"archive")
    return directories, archive, tmp_path / "audit"


def _publish(
        tmp_path: Path,
        directories: list[Path],
        archive: Path,
        target: Path,
        ) -> dict[str, object]:
    """发布 synthetic binding artifact。"""
    return audit.publish_normalization_recovery_v7_vscode_english_binding_audit(
        run_root=tmp_path,
        training_protocol_dir=directories[0],
        variable_structure_audit_dir=directories[1],
        neutral_semantic_source_audit_dir=directories[2],
        vscode_source_pack_dir=directories[3],
        vscode_source_archive_path=archive,
        vscode_source_root=directories[4],
        typescript_parser_root=directories[5],
        target_dir=target,
    )


def _read(
        directories: list[Path],
        archive: Path,
        target: Path,
        sha256: str,
        ) -> tuple[dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """严格回读 synthetic binding artifact。"""
    return audit.read_normalization_recovery_v7_vscode_english_binding_audit(
        target,
        training_protocol_dir=directories[0],
        variable_structure_audit_dir=directories[1],
        neutral_semantic_source_audit_dir=directories[2],
        vscode_source_pack_dir=directories[3],
        vscode_source_archive_path=archive,
        vscode_source_root=directories[4],
        typescript_parser_root=directories[5],
        expected_manifest_sha256=sha256,
    )


def test_audit_round_trip_nonoverwrite_and_synchronized_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """artifact 不可覆盖，records+manifest 同步篡改也被重派生拒绝。"""
    monkeypatch.setattr(audit, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(audit, "_input_state", lambda **_kwargs: (
        {}, {}, {}, (), (), {}, {"archive": {}, "selection": {}}, {}))
    monkeypatch.setattr(audit, "_derive", lambda **_kwargs: _fake_outputs())
    monkeypatch.setattr(audit, "_extractor_sha256", lambda: "e" * 64)
    directories, archive, target = _audit_paths(tmp_path)
    published = _publish(tmp_path, directories, archive, target)
    manifest, outputs = _read(
        directories, archive, target, str(published["manifest_sha256"]))
    assert manifest == published
    assert outputs == _fake_outputs()[0]
    with pytest.raises(BroadQaExternalDataError, match="input/target path 非法"):
        _publish(tmp_path, directories, archive, target)

    changed = canonical_json_line({"record_id": "9" * 64})
    path = target / "main-bindings.jsonl"
    path.write_bytes(changed)
    stored = json.loads((target / "manifest.json").read_bytes())
    artifact = next(item for item in stored["files"]
                    if item["relative_path"] == path.name)
    artifact["bytes"] = len(changed)
    artifact["sha256"] = _sha256(changed)
    encoded = canonical_json_line(stored)
    (target / "manifest.json").write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="records/inputs 漂移"):
        _read(directories, archive, target, _sha256(encoded))
