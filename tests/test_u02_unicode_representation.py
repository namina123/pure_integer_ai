"""U-02 Unicode sequence、UCD 只读适配和生产接线测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_ATOM,
    OBJECT_REPRESENTATION,
)
from pure_integer_ai.cognition.shared.scope_identity import session_scope
from pure_integer_ai.cognition.shared.unicode_representation import (
    UnicodeSequenceMaterializer,
    validate_unicode_scalars,
)
from pure_integer_ai.cognition.shared.types import Segment
from pure_integer_ai.experiments.data_manifest import (
    ManifestBinding,
    ManifestIntegrityError,
    RawDatasetManifest,
    RawFileManifest,
    read_manifest,
    sha256_file,
    write_manifest,
)
from pure_integer_ai.experiments.formal_train import (
    DefaultRoundRunner,
    FormalTrainConfig,
    STAGE1_SKELETON,
    formal_train,
    make_train_context,
)
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.ucd_adapter import (
    BINDING_EXTERNAL_PROPERTY_RELATION,
    BINDING_UCD_EPISTEMIC_ORIGIN,
    BINDING_UCD_PROVENANCE_KIND,
    BINDING_UCD_SCOPE_KIND,
    BINDING_UNICODE_SEQUENCE_FAMILY,
    PARSER_BINARY_RANGE,
    PARSER_ENUMERATED_RANGE,
    PARSER_UNICODE_DATA,
    UcdReadOnlyAdapter,
)
from pure_integer_ai.experiments.unicode_intake import UnicodeIntake
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.graph_object import GRAPH_OBJECT_TABLE
from pure_integer_ai.training.cursor import DUMP_TABLES, dump_run, load_run


def _backend(kind: str):
    """创建独立测试后端。"""
    return DictBackend() if kind == "dict" else SQLiteBackend()


def _unicode_data_line(codepoint: int, name: str, category: str) -> str:
    """构造字段数严格为 15 的最小 UnicodeData 行。"""
    return ";".join((
        f"{codepoint:04X}", name, category, "0", "L",
        "", "", "", "", "N", "", "", "", "", "",
    ))


def _write_minimal_ucd(root: Path) -> None:
    """写入只覆盖本测试码点的最小 UCD 外部格式 fixture。"""
    (root / "source/auxiliary").mkdir(parents=True)
    (root / "source/emoji").mkdir(parents=True)
    unicode_lines = [
        _unicode_data_line(0x002E, "FULL STOP", "Po"),
        _unicode_data_line(0x0041, "LATIN CAPITAL LETTER A", "Lu"),
        _unicode_data_line(0x0301, "COMBINING ACUTE ACCENT", "Mn"),
        _unicode_data_line(0x200D, "ZERO WIDTH JOINER", "Cf"),
        _unicode_data_line(0x2764, "HEAVY BLACK HEART", "So"),
        _unicode_data_line(0xFE0F, "VARIATION SELECTOR-16", "Mn"),
        _unicode_data_line(0x1F469, "WOMAN", "So"),
        _unicode_data_line(0x1F4BB, "PERSONAL COMPUTER", "So"),
        _unicode_data_line(0x20000, "CJK IDEOGRAPH-20000", "Lo"),
    ]
    (root / "source/UnicodeData.txt").write_text(
        "\n".join(unicode_lines) + "\n", encoding="utf-8")
    (root / "source/Scripts.txt").write_text(
        "# @missing: 0000..10FFFF; Unknown\n"
        "002E ; Common\n0041 ; Latin\n0301 ; Inherited\n"
        "200D ; Inherited\n2764 ; Common\nFE0F ; Inherited\n"
        "1F469 ; Common\n1F4BB ; Common\n20000 ; Han\n",
        encoding="utf-8")
    (root / "source/auxiliary/GraphemeBreakProperty.txt").write_text(
        "# @missing: 0000..10FFFF; Other\n"
        "0301 ; Extend\n200D ; ZWJ\nFE0F ; Extend\n",
        encoding="utf-8")
    (root / "source/auxiliary/SentenceBreakProperty.txt").write_text(
        "# @missing: 0000..10FFFF; Other\n"
        "002E ; ATerm\n0301 ; Extend\n200D ; Extend\nFE0F ; Extend\n"
        "20000 ; OLetter\n",
        encoding="utf-8")
    (root / "source/PropList.txt").write_text(
        "FE0F ; Variation_Selector\n", encoding="utf-8")
    (root / "source/emoji/emoji-data.txt").write_text(
        "2764 ; Extended_Pictographic\n"
        "1F469 ; Extended_Pictographic\n"
        "1F4BB ; Extended_Pictographic\n",
        encoding="utf-8")


def _file_manifest(root: Path, relative_path: str, parser_kind: str,
                   namespace: str, property_name: str,
                   record_count: int) -> RawFileManifest:
    """为测试原始文件生成真实大小和哈希。"""
    path = root / relative_path
    return RawFileManifest(
        relative_path,
        sha256_file(path),
        path.stat().st_size,
        "utf-8",
        "UCD",
        parser_kind,
        namespace,
        property_name,
        record_count,
        0,
    )


def _manifest(root: Path, *, version: str = "17.0.0") -> RawDatasetManifest:
    """构造与最小 fixture 逐文件匹配的 UCD manifest。"""
    files = (
        _file_manifest(
            root, "source/UnicodeData.txt", PARSER_UNICODE_DATA,
            "UCD", "General_Category", 9),
        _file_manifest(
            root, "source/Scripts.txt", PARSER_ENUMERATED_RANGE,
            "UCD", "Script", 9),
        _file_manifest(
            root, "source/auxiliary/GraphemeBreakProperty.txt",
            PARSER_ENUMERATED_RANGE, "UAX29", "Grapheme_Cluster_Break", 3),
        _file_manifest(
            root, "source/auxiliary/SentenceBreakProperty.txt",
            PARSER_ENUMERATED_RANGE, "UAX29", "Sentence_Break", 5),
        _file_manifest(
            root, "source/PropList.txt", PARSER_BINARY_RANGE,
            "UCD-PropList", "", 1),
        _file_manifest(
            root, "source/emoji/emoji-data.txt", PARSER_BINARY_RANGE,
            "UTS51", "", 3),
    )
    return RawDatasetManifest(
        "unicode-ucd-uax29", version, 1, 7, "Unicode-3.0", files,
        (
            ManifestBinding(BINDING_UNICODE_SEQUENCE_FAMILY, (1001,)),
            ManifestBinding(BINDING_EXTERNAL_PROPERTY_RELATION, (1002,)),
            ManifestBinding(BINDING_UCD_PROVENANCE_KIND, (1003,)),
            ManifestBinding(BINDING_UCD_EPISTEMIC_ORIGIN, (1004,)),
            ManifestBinding(BINDING_UCD_SCOPE_KIND, (1005,)),
        ),
    )


def _adapter(tmp_path: Path, *, version: str = "17.0.0"):
    """创建最小 UCD 原始目录和已核验 adapter。"""
    root = tmp_path / f"raw-{version}"
    _write_minimal_ucd(root)
    return root, UcdReadOnlyAdapter(root, _manifest(root, version=version))


def test_ucd_adapter_reads_versioned_properties_without_role_inference(tmp_path):
    root, adapter = _adapter(tmp_path)
    assert root.is_dir()

    dot = {(item.property_name, item.value)
           for item in adapter.properties_for(0x002E)}
    combining = {(item.property_name, item.value)
                 for item in adapter.properties_for(0x0301)}
    supplementary = {(item.property_name, item.value)
                     for item in adapter.properties_for(0x20000)}
    assert ("Sentence_Break", "ATerm") in dot
    assert ("Grapheme_Cluster_Break", "Extend") in combining
    assert ("Script", "Han") in supplementary
    assert all(item.property_name != "sentence_final_role"
               for item in adapter.properties_for(0x002E))


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_single_multi_combining_vs_zwj_and_supplementary_sequences_materialize(
        kind: str, tmp_path):
    _, adapter = _adapter(tmp_path)
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        intake = UnicodeIntake(ctx.graph_ontology, adapter)
        segments = [Segment(1, tokens=[
            "A",
            "A\u0301",
            "\u2764\ufe0f",
            "\U0001F469\u200D\U0001F4BB",
            "\U00020000",
        ])]
        result = intake.observe_segments(segments)

        assert result.token_count == 5
        assert result.unique_sequence_count == 5
        assert len(set(result.sequence_refs)) == 5
        assert result.property_link_count > 20
        assert all(ref.object_kind == OBJECT_REPRESENTATION
                   for ref in result.sequence_refs)
        object_kinds = {
            row["object_kind"] for row in backend.select(GRAPH_OBJECT_TABLE)}
        assert OBJECT_LANGUAGE_ATOM not in object_kinds
    finally:
        backend.close()


def test_unicode_version_changes_properties_not_sequence_identity(tmp_path):
    _, adapter17 = _adapter(tmp_path, version="17.0.0")
    _, adapter18 = _adapter(tmp_path, version="18.0.0")
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        intake17 = UnicodeIntake(ctx.graph_ontology, adapter17)
        intake18 = UnicodeIntake(ctx.graph_ontology, adapter18)
        first = intake17.observe_segments([Segment(1, tokens=["A"])])
        second = intake18.observe_segments([Segment(1, tokens=["A"])])

        assert first.sequence_refs == second.sequence_refs
        statements = ctx.graph_ontology.statements(subject=first.sequence_refs[0])
        property_identities = {
            ctx.graph_ontology.identity_of(item.object).components
            for item in statements}
        assert any(components[:3] == (17, 0, 0)
                   for components in property_identities)
        assert any(components[:3] == (18, 0, 0)
                   for components in property_identities)
    finally:
        backend.close()


def test_invalid_scalars_and_mismatched_evidence_fail_closed(tmp_path):
    _, adapter = _adapter(tmp_path)
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        materializer = UnicodeSequenceMaterializer(
            ctx.graph_ontology,
            family_key=(1001,),
            external_property_relation_key=(1002,),
        )
        with pytest.raises(ValueError, match="Unicode scalar"):
            validate_unicode_scalars((0xD800,))
        with pytest.raises(ValueError, match="Unicode scalar"):
            materializer.materialize((0x110000,))
        evidence = adapter.properties_for(0x0041)[0].integer_key(
            parser_version=7, sequence_index=0)
        with pytest.raises(ValueError, match="不一致"):
            materializer.materialize_with_properties(
                (0x002E,), (evidence,),
                scope=session_scope(1), provenance_kind=1)
    finally:
        backend.close()


def test_manifest_missing_or_changed_file_fails_closed(tmp_path):
    root, _ = _adapter(tmp_path)
    manifest = _manifest(root)
    target = root / "source/Scripts.txt"
    target.write_text(target.read_text(encoding="utf-8") + "0042 ; Latin\n",
                      encoding="utf-8")
    with pytest.raises(ManifestIntegrityError, match="大小变化|哈希变化"):
        UcdReadOnlyAdapter(root, manifest)
    target.unlink()
    with pytest.raises(ManifestIntegrityError, match="缺失"):
        UcdReadOnlyAdapter(root, manifest)


def test_manifest_output_cannot_overwrite_raw_or_changed_version(tmp_path):
    root, _ = _adapter(tmp_path)
    manifest = _manifest(root)
    with pytest.raises(ManifestIntegrityError, match="不得位于"):
        write_manifest(manifest, root / "manifest.json", raw_root=root)
    output = tmp_path / "manifests/v1/manifest.json"
    write_manifest(manifest, output, raw_root=root)
    assert read_manifest(output) == manifest
    with pytest.raises(ManifestIntegrityError, match="新版本目录"):
        write_manifest(_manifest(root, version="18.0.0"), output, raw_root=root)


def test_default_round_runner_consumes_configured_unicode_intake(tmp_path):
    _, adapter = _adapter(tmp_path)
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        ctx.unicode_intake = UnicodeIntake(ctx.graph_ontology, adapter)
        runner = DefaultRoundRunner()
        item = CollectedItem(tokens=["A", "A\u0301"], role_seq=[1, 1])
        runner.run_round(ctx, item, STAGE1_SKELETON, 0)

        representations = backend.select(
            GRAPH_OBJECT_TABLE, where={"object_kind": OBJECT_REPRESENTATION})
        assert len(representations) == 2
    finally:
        backend.close()


def test_formal_train_loads_only_paired_unicode_configuration(tmp_path):
    root, _ = _adapter(tmp_path)
    manifest_path = tmp_path / "manifests/v1/manifest.json"
    write_manifest(_manifest(root), manifest_path, raw_root=root)
    backend = DictBackend()
    try:
        config = FormalTrainConfig(
            run_dir=str(tmp_path / "runs"),
            run_id="u02",
            rounds_per_stage=1,
            active_training_stages=(STAGE1_SKELETON,),
            persist_graph_dump=False,
            unicode_raw_root=str(root),
            unicode_manifest_path=str(manifest_path),
        )
        formal_train(
            config,
            [CollectedItem(tokens=["A"], role_seq=[1])],
            backend=backend,
            runner=DefaultRoundRunner(),
        )
        assert backend.count(
            GRAPH_OBJECT_TABLE,
            where={"object_kind": OBJECT_REPRESENTATION}) == 1
    finally:
        backend.close()

    with pytest.raises(ValueError, match="必须同时配置"):
        formal_train(
            FormalTrainConfig(
                run_dir=str(tmp_path / "bad"),
                run_id="bad",
                unicode_raw_root=str(root),
            ),
            [],
            backend=DictBackend(),
        )


def test_unicode_intake_rebinds_to_evaluation_sandbox(tmp_path):
    _, adapter = _adapter(tmp_path)
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        ctx.unicode_intake = UnicodeIntake(ctx.graph_ontology, adapter)
        before = backend.snapshot()
        with isolated_evaluation(ctx, label="u02") as evaluation:
            result = evaluation.unicode_intake.observe_segments(
                [Segment(1, tokens=["A"])])
            assert result.unique_sequence_count == 1
            assert evaluation.backend.count(
                GRAPH_OBJECT_TABLE,
                where={"object_kind": OBJECT_REPRESENTATION}) == 1
        assert backend.snapshot() == before
        assert ctx.unicode_intake._sequence_cache == {}
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_unicode_sequence_and_property_qualifiers_roundtrip_dump(
        kind: str, tmp_path):
    _, adapter = _adapter(tmp_path)
    first_backend = _backend(kind)
    try:
        first = make_train_context(first_backend)
        intake = UnicodeIntake(first.graph_ontology, adapter)
        result = intake.observe_segments([Segment(1, tokens=["A\u0301"])])
        dump_run(
            first_backend,
            str(tmp_path / "runs"),
            "u02_dump",
            spaces=[first.space_id],
            tables=DUMP_TABLES,
        )
    finally:
        first_backend.close()

    second_backend = _backend(kind)
    try:
        second = make_train_context(second_backend)
        assert load_run(
            second_backend,
            str(tmp_path / "runs"),
            "u02_dump") == [1]
        materializer = UnicodeSequenceMaterializer(
            second.graph_ontology,
            family_key=adapter.manifest.binding(
                BINDING_UNICODE_SEQUENCE_FAMILY),
            external_property_relation_key=adapter.manifest.binding(
                BINDING_EXTERNAL_PROPERTY_RELATION),
        )
        restored = second.graph_ontology.resolve(
            materializer.identity((0x0041, 0x0301)))
        assert restored == result.sequence_refs[0]
        links = materializer.property_links(restored)
        assert links
        assert {link.evidence.sequence_index for link in links} == {0, 1}
        assert {link.evidence.codepoint for link in links} == {0x0041, 0x0301}
        assert {link.evidence.parser_version for link in links} == {7}
    finally:
        second_backend.close()
