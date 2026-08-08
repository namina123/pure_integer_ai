from __future__ import annotations

from dataclasses import replace
import gzip
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from pure_integer_ai.experiments import ph2_d03_v2_w02_compiler as compiler
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import (
    V2PhysicalRoots,
)
from pure_integer_ai.experiments.ph2_d03_v2_schema import validate_v2_record
from pure_integer_ai.experiments.ph2_mediawiki_multistream_adapter import (
    MediaWikiScanBudget,
    parse_mediawiki_page,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import (
    W02_FIRST_RUN_GUARD_AVAILABLE,
    W02_FIRST_RUN_GUARD_CONSUMED,
    W02_LAYOUTS,
    W02_TARGET_TRAIN_OBSERVATIONS,
    W02CompileFreezeError,
    build_w02_code_freeze,
    consume_w02_first_run_guard,
    formal_w02_compile_plan,
    publish_w02_first_run_guard,
    read_w02_compile_freeze,
    w02_first_run_guard_value,
)
from pure_integer_ai.experiments.ph2_ud_gsdsimp_adapter import (
    iter_ud_conllu_sentences,
)


def _roots(root: Path) -> V2PhysicalRoots:
    """建立测试专用的六个 sibling owner root。"""
    paths = tuple(root / name for name in ("c", "t", "d", "s", "p", "l"))
    for path in paths:
        path.mkdir()
    return V2PhysicalRoots(*paths)


def _bundle(split: str, ordinal: int, carrier: str = "plain_text"):
    """形成一个不依赖正式 private family 的公开合成小闭环。"""
    units = ("龘甲", f"q{ordinal}")
    surface = "".join(units)
    source = compiler._source_record(
        "AUTHORED_CC0", split, ordinal,
        snapshot_id=compiler.W02_AUTHORED_GENERATOR_VERSION,
        revision_id="1",
        official_url="https://creativecommons.org/publicdomain/zero/1.0/",
        source_identity=f"fixture:{split}:{ordinal}",
        upstream_checksum="sha256:" + f"{ordinal:064x}",
        local_sha256=f"{ordinal:064x}",
        license_id="CC0-1.0", attribution="fixture",
        locator_kind="record", locator_value=str(ordinal),
        span_end=len(surface),
    )
    observation = compiler._observation_record(
        "AUTHORED_CC0", split, ordinal, source,
        carrier_kind=carrier, surface=surface, family_ordinal=ordinal,
        sample_role="support" if split == "train" else "read_only_probe",
        perturbation_kind="NONE" if split == "train" else "FIXTURE",
    )
    raw, start, end, _, _ = compiler._carrier_serialization(carrier, surface)
    owner = compiler._owner_record(
        "AUTHORED_CC0", split, ordinal, source, observation,
        compiler._authored_expected(units, carrier, raw, start, end),
        dimension_name=compiler._dimension("AUTHORED_CC0", ordinal),
    )
    return source, observation, owner


def test_w02_formal_plan_counts_candidate_observations_only() -> None:
    plan = formal_w02_compile_plan()

    assert plan.split_total("train") == W02_TARGET_TRAIN_OBSERVATIONS
    assert plan.total_observations() == 73_014
    assert plan.to_dict()["split_totals"] == {
        "adversarial": 3_403,
        "dev": 7_306,
        "held_out": 10_700,
        "train": 51_200,
        "wall": 405,
    }
    assert all(
        plan.source_counts[0].count(split) % 9 == 0
        for split in ("train", "dev", "held_out", "adversarial", "wall")
    )


def test_w02_all_nine_carriers_preserve_structure_without_expected_label() -> None:
    for ordinal, carrier in enumerate(compiler.W02_CARRIER_KINDS, start=1):
        source, observation, owner = _bundle("train", ordinal, carrier)
        assert validate_v2_record(source.to_dict()) == source
        assert validate_v2_record(observation.to_dict()) == observation
        assert validate_v2_record(owner.to_dict()) == owner
        payload = observation.typed_payload.to_value()
        assert payload["carrier"]["carrier_kind"] == carrier
        assert payload["language_payload"]["surface"]
        assert "expected" not in payload["language_payload"]
        assert "oov_units" not in payload["language_payload"]


def test_w02_real_ud_first_sentence_builds_valid_source_bundle() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    ud_path = (
        repository_root.parent / "ph2_dataset_raw" / "UD_ZH_GSDSIMP_R2_18"
        / "zh_gsdsimp-ud-train.conllu")
    sentence = next(iter_ud_conllu_sentences(ud_path))
    source = compiler._source_record(
        "UD_ZH_GSDSIMP_R2_18", "train", 1,
        snapshot_id="ud-zh-gsdsimp-r2.18", revision_id="revision",
        official_url="https://github.com/UniversalDependencies/UD_Chinese-GSDSimp",
        source_identity=f"train:{sentence.sent_id}",
        upstream_checksum="sha1:" + "1" * 40,
        local_sha256="2" * 64, license_id="CC-BY-SA-4.0",
        attribution="fixture", locator_kind="sentence",
        locator_value=sentence.sent_id, span_end=len(sentence.text),
    )
    observation = compiler._observation_record(
        "UD_ZH_GSDSIMP_R2_18", "train", 1, source,
        carrier_kind="plain_text", surface=sentence.text,
        family_ordinal=1, sample_role="support", perturbation_kind="NONE")
    owner = compiler._owner_record(
        "UD_ZH_GSDSIMP_R2_18", "train", 1, source, observation,
        compiler._ud_expected(sentence, "plain_text"),
        dimension_name=compiler._dimension("UD_ZH_GSDSIMP_R2_18", 1))

    assert validate_v2_record(source.to_dict()) == source
    assert validate_v2_record(observation.to_dict()) == observation
    assert validate_v2_record(owner.to_dict()) == owner


def test_w02_mediawiki_upstream_sha1_builds_valid_source() -> None:
    page = ET.fromstring(
        "<page><title>龘词</title><ns>0</ns><id>7</id><revision>"
        "<id>9</id><timestamp>2026-07-01T00:00:00Z</timestamp>"
        "<contributor><username>fixture</username><id>1</id></contributor>"
        "<model>wikitext</model><format>text/x-wiki</format>"
        "<text bytes='2'>定义</text><sha1>0zbase36revision</sha1>"
        "</revision></page>")
    record = parse_mediawiki_page(
        page, source_key="ZHWIKTIONARY_20260701", extract_templates=False,
        budget=MediaWikiScanBudget(1, 64, 1024, 1, 1))
    source = compiler._source_record(
        "ZHWIKTIONARY_20260701", "train", 1,
        snapshot_id="fixture", revision_id=str(record.revision_id),
        official_url="https://zh.wiktionary.org/?curid=7&oldid=9",
        source_identity="page:7:revision:9",
        upstream_checksum=compiler._mediawiki_upstream_checksum(record.upstream_sha1),
        local_sha256="b" * 64, license_id="CC-BY-SA-4.0",
        attribution="fixture", locator_kind="page", locator_value="7",
        span_end=len(record.title))

    assert validate_v2_record(source.to_dict()) == source
    assert source.upstream_checksum.startswith("sha256:")


def test_w02_authored_family_surfaces_are_unique_within_split() -> None:
    for split in ("train", "dev", "held_out", "adversarial", "wall"):
        nonce = b"public-train" * 4 if split == "train" else b"private-family" * 3
        surfaces = {
            "".join(compiler._authored_units(nonce, split, ordinal))
            for ordinal in range(1, 1_001)
        }
        assert len(surfaces) == 1_000


def test_w02_spool_writes_all_owner_layouts_and_private_commitments(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    spool = compiler._W02Spool(tmp_path / "spool.sqlite3")
    for split_index, split in enumerate(
            ("train", "dev", "held_out", "adversarial", "wall"), start=1):
        for carrier_index, carrier in enumerate(compiler.W02_CARRIER_KINDS, start=1):
            ordinal = split_index * 100 + carrier_index
            spool.add(*_bundle(split, ordinal, carrier))
    commitments = spool.private_commitments("1" * 64)
    files = spool.write_files(roots)
    spool.close()

    assert len(files) == len(W02_LAYOUTS) == 17
    assert len(set(commitments)) == 3
    assert all(len(item) == 64 for item in commitments)
    for item in files:
        root = roots.by_root_key(item.root_key)
        target = root / Path(*item.storage_relative_path.split("/"))
        assert target.stat().st_size == item.transport_size_bytes
        with gzip.open(target, "rb") as stream:
            assert sum(1 for _ in stream) == item.record_count


def test_w02_spool_rejects_cluster_cross_split(tmp_path: Path) -> None:
    spool = compiler._W02Spool(tmp_path / "spool.sqlite3")
    train = _bundle("train", 1)
    dev = list(_bundle("dev", 2))
    dev[1] = replace(dev[1], content_group_key=train[1].content_group_key)
    spool.add(*train)
    with pytest.raises(compiler.W02CompilerError, match="cluster 跨 split"):
        spool.add(*dev)
    spool.close()


def test_w02_code_freeze_and_first_run_guard_are_single_use(
        tmp_path: Path,
        ) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    code_files, code_sha = build_w02_code_freeze(repository_root)
    assert len(code_files) == 9
    assert len(code_sha) == 64
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    value = w02_first_run_guard_value(
        candidate_contract_sha256="2" * 64,
        code_freeze_sha256=code_sha,
        pack_commitment="3" * 64,
    )
    guard_sha = publish_w02_first_run_guard(candidate_root, value)
    consume_w02_first_run_guard(
        candidate_root, expected_guard_sha256=guard_sha,
        run_id=1, run_identity_sha256="4" * 64)

    assert not (candidate_root / W02_FIRST_RUN_GUARD_AVAILABLE).exists()
    assert (candidate_root / W02_FIRST_RUN_GUARD_CONSUMED).is_file()
    with pytest.raises(W02CompileFreezeError, match="已经消费"):
        consume_w02_first_run_guard(
            candidate_root, expected_guard_sha256=guard_sha,
            run_id=2, run_identity_sha256="5" * 64)


def test_w02_formal_compile_freeze_is_canonical_and_code_bound() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    freeze = read_w02_compile_freeze(repository_root)
    _, current_code_sha = build_w02_code_freeze(repository_root)

    assert freeze.code_freeze_sha256 == current_code_sha
    assert freeze.plan.split_total("train") == 51_200
    assert freeze.plan.total_observations() == 73_014
    assert len(freeze.files) == 17
    assert freeze.to_dict()["status"] == "W02_COMPILE_FREEZE_COMPLETE"
    assert freeze.to_dict()["formal_training_runs"] == 0
    assert freeze.to_dict()["formal_private_evaluation_runs"] == 0
    assert freeze.to_dict()["private_payload_reads"] == 0
