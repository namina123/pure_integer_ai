"""Public Kyoto train/dev probe for the W-02 language morphology overlay."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterator

from pure_integer_ai.experiments.ph2_d03_contract_core import canonical_json_bytes
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v4_language_overlay import (
    W02_MORPH_SUCCESSOR_V4_VERSION,
    W02_MORPH_V4_BACKOFF_CANDIDATE_LIMIT,
    W02_MORPH_V4_EXACT_CANDIDATE_LIMIT,
    W02MorphologySuccessorV4Index,
    build_w02_morphology_successor_v4_from_counts,
    rank_w02_morphology_successor_v4,
)


W02_MORPH_V4_PUBLIC_PROBE_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V4-PUBLIC-PROBE-V1")
W02_MORPH_V4_PUBLIC_LANGUAGE = "lzh"
W02_MORPH_V4_PUBLIC_SOURCE_KEY = "UD_LZH_KYOTO_R2_18_PUBLIC_TRAIN_DEV"
W02_MORPH_V4_PUBLIC_COMMIT = "2f5ff2e1ac5df5315cbe547283cca80fb69224e0"
W02_MORPH_V4_PUBLIC_REPOSITORY = (
    "https://github.com/UniversalDependencies/UD_Classical_Chinese-Kyoto")
W02_MORPH_V4_PUBLIC_LICENSE = "CC-BY-SA-4.0"
W02_MORPH_V4_MIN_COVERAGE_BASIS_POINTS = 9_400

W02_MORPH_V4_PUBLIC_FILES = {
    "LICENSE.txt": (202,
                    "899b1804a12ebc090b96339614eede1b64b686721b650a71430b55b5235f7f79"),
    "README.md": (4_742,
                  "2067b50bc8f23189d60bfed740e3438c8186b9a0297533362a1c6a0504d415e6"),
    "lzh_kyoto-ud-dev.conllu": (
        3_032_014,
        "b67614202e30006f9cada1abccbd8f04371f50bd9821791db4c3932a3fd6b3a7"),
    "lzh_kyoto-ud-train.conllu": (
        36_466_278,
        "ce1202b74d176440a8a94d959e7ccd22496de49c80ca70bcda283bbc49191b68"),
    "stats.xml": (7_800,
                  "685411136aaa804cf574e3cf37109dbdc769352df7bfdb84d0cc205d3d69e7c7"),
}


# object-model: exception
class W02MorphologySuccessorV4PublicProbeError(RuntimeError):
    """The fixed public source, CoNLL-U payload, or probe result drifted."""


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _verify_public_files(root: Path) -> tuple[dict[str, object], ...]:
    rows = []
    for name, (expected_size, expected_sha) in sorted(
            W02_MORPH_V4_PUBLIC_FILES.items()):
        path = root / name
        if not path.is_file():
            raise W02MorphologySuccessorV4PublicProbeError(
                f"public Kyoto file is missing: {name}")
        size, digest = _sha256_file(path)
        if size != expected_size or digest != expected_sha:
            raise W02MorphologySuccessorV4PublicProbeError(
                f"public Kyoto file identity drifted: {name}")
        rows.append({"relative_path": name, "sha256": digest,
                     "size_bytes": size})
    return tuple(rows)


def _canonical_feats(raw: str) -> str:
    if raw == "_":
        return "{}"
    values: dict[str, str] = {}
    for item in raw.split("|"):
        if "=" not in item:
            raise W02MorphologySuccessorV4PublicProbeError(
                "public Kyoto FEATS item is malformed")
        key, value = item.split("=", 1)
        if not key or not value or key in values:
            raise W02MorphologySuccessorV4PublicProbeError(
                "public Kyoto FEATS key/value is invalid")
        values[key] = value
    return canonical_json_bytes(values).decode("utf-8")


def _sentences(path: Path) -> Iterator[tuple[tuple[str, str, str, str], ...]]:
    sentence: list[tuple[str, str, str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if not line:
                if sentence:
                    yield tuple(sentence)
                    sentence.clear()
                continue
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 10:
                raise W02MorphologySuccessorV4PublicProbeError(
                    "public Kyoto CoNLL-U column count drifted")
            if "-" in fields[0] or "." in fields[0]:
                continue
            if not fields[0].isdigit():
                raise W02MorphologySuccessorV4PublicProbeError(
                    "public Kyoto token id is invalid")
            form, lemma, upos = fields[1], fields[2], fields[3]
            if not form or not lemma or lemma == "_" or not upos or upos == "_":
                raise W02MorphologySuccessorV4PublicProbeError(
                    "public Kyoto morphology annotation is incomplete")
            sentence.append((form, lemma, upos, _canonical_feats(fields[5])))
    if sentence:
        yield tuple(sentence)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorV4PublicTraining:
    """Frozen in-memory index and payload-free public training statistics."""

    index: W02MorphologySuccessorV4Index
    sentence_count: int
    token_count: int
    unique_form_count: int
    unique_tuple_count: int
    source_files: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.index, W02MorphologySuccessorV4Index):
            raise W02MorphologySuccessorV4PublicProbeError(
                "public V4 training index type drifted")
        if any(type(value) is not int or value <= 0 for value in (
                self.sentence_count, self.token_count,
                self.unique_form_count, self.unique_tuple_count)):
            raise W02MorphologySuccessorV4PublicProbeError(
                "public V4 training counts are invalid")
        if not self.source_files:
            raise W02MorphologySuccessorV4PublicProbeError(
                "public V4 source identities are empty")


def build_w02_morphology_successor_v4_public_training(
        raw_root: str | Path,
        ) -> W02MorphologySuccessorV4PublicTraining:
    """Verify fixed Kyoto files and learn only from the public train split."""
    root = Path(raw_root).resolve()
    source_files = _verify_public_files(root)
    counts: Counter[tuple[str, str, str, str]] = Counter()
    sentence_count = 0
    token_count = 0
    forms: set[str] = set()
    for sentence in _sentences(root / "lzh_kyoto-ud-train.conllu"):
        sentence_count += 1
        token_count += len(sentence)
        for form, lemma, upos, feats_json in sentence:
            counts[(form, lemma, upos, feats_json)] += 1
            forms.add(form)
    index = build_w02_morphology_successor_v4_from_counts(tuple(
        (W02_MORPH_V4_PUBLIC_LANGUAGE, *key, support)
        for key, support in sorted(counts.items())))
    if index.training_token_count != token_count:
        raise W02MorphologySuccessorV4PublicProbeError(
            "public V4 training token count drifted")
    return W02MorphologySuccessorV4PublicTraining(
        index, sentence_count, token_count, len(forms), len(counts),
        source_files)


def _partition_report(state: dict[str, int]) -> dict[str, int | str]:
    token_count = state["token_count"]
    if token_count <= 0:
        raise W02MorphologySuccessorV4PublicProbeError(
            "public V4 probe partition is empty")
    passed = state["passed_tuple_count"]
    return {
        **state,
        "candidate_count_milli_per_token":
            state["candidate_count"] * 1_000 // token_count,
        "coverage_basis_points": passed * 10_000 // token_count,
        "status": ("PASS" if passed * 10_000
                   >= W02_MORPH_V4_MIN_COVERAGE_BASIS_POINTS * token_count
                   else "FAIL"),
    }


def evaluate_w02_morphology_successor_v4_public_probe(
        training: W02MorphologySuccessorV4PublicTraining,
        raw_root: str | Path,
        ) -> dict[str, object]:
    """Use disjoint dev sentence partitions for calibration and shadow evidence."""
    if not isinstance(training, W02MorphologySuccessorV4PublicTraining):
        raise TypeError("public V4 training type drifted")
    root = Path(raw_root).resolve()
    if _verify_public_files(root) != training.source_files:
        raise W02MorphologySuccessorV4PublicProbeError(
            "public V4 source identity changed after training")
    states = {
        "dev": Counter(),
        "shadow": Counter(),
    }
    first_seen_form = ""
    for sentence_ordinal, sentence in enumerate(
            _sentences(root / "lzh_kyoto-ud-dev.conllu"), start=1):
        partition = "dev" if sentence_ordinal % 2 else "shadow"
        state = states[partition]
        state["sentence_count"] += 1
        for form, lemma, upos, feats_json in sentence:
            if not first_seen_form:
                first_seen_form = form
            ranking = rank_w02_morphology_successor_v4(
                training.index, W02_MORPH_V4_PUBLIC_LANGUAGE, form)
            predicted = {(row[0], row[1], row[2])
                         for row in ranking.candidates}
            state["token_count"] += 1
            state["candidate_count"] += len(ranking.candidates)
            state["max_candidates_per_token"] = max(
                state["max_candidates_per_token"], len(ranking.candidates))
            state["logic_operations"] += ranking.logic_operations
            state["passed_tuple_count"] += int(
                (lemma, upos, feats_json) in predicted)
            state["failed_tuple_count"] += int(
                (lemma, upos, feats_json) not in predicted)
            state["exact_token_count"] += int(
                ranking.evidence_mode == "EXACT_LEXEME")
            state["backoff_token_count"] += int(
                ranking.evidence_mode == "LANGUAGE_BACKOFF")
            state["no_candidate_token_count"] += int(
                ranking.evidence_mode == "NONE")
    dev = _partition_report(dict(states["dev"]))
    shadow = _partition_report(dict(states["shadow"]))
    novel_form = "\U00020000"
    if (W02_MORPH_V4_PUBLIC_LANGUAGE, novel_form) in training.index.exact_counts:
        raise W02MorphologySuccessorV4PublicProbeError(
            "public V4 metamorphic novel form is not novel")
    novel = rank_w02_morphology_successor_v4(
        training.index, W02_MORPH_V4_PUBLIC_LANGUAGE, novel_form)
    isolation = rank_w02_morphology_successor_v4(
        training.index, "zh", novel_form)
    repeated_a = rank_w02_morphology_successor_v4(
        training.index, W02_MORPH_V4_PUBLIC_LANGUAGE, first_seen_form)
    repeated_b = rank_w02_morphology_successor_v4(
        training.index, W02_MORPH_V4_PUBLIC_LANGUAGE, first_seen_form)
    metamorphic = {
        "deterministic_repeat_equal": int(repeated_a == repeated_b),
        "language_isolation_candidate_count": len(isolation.candidates),
        "novel_backoff_candidate_count": len(novel.candidates),
        "novel_backoff_identity_lemma_count": sum(
            row[0] == novel_form for row in novel.candidates),
        "novel_backoff_mode": novel.evidence_mode,
        "status": "PASS" if (
            repeated_a == repeated_b
            and not isolation.candidates
            and novel.evidence_mode == "LANGUAGE_BACKOFF"
            and 0 < len(novel.candidates)
            <= W02_MORPH_V4_BACKOFF_CANDIDATE_LIMIT
            and all(row[0] == novel_form for row in novel.candidates)
        ) else "FAIL",
    }
    hard = (
        dev["status"] == "PASS"
        and shadow["status"] == "PASS"
        and metamorphic["status"] == "PASS"
        and int(dev["max_candidates_per_token"])
        <= W02_MORPH_V4_EXACT_CANDIDATE_LIMIT
        and int(shadow["max_candidates_per_token"])
        <= W02_MORPH_V4_EXACT_CANDIDATE_LIMIT
    )
    return {
        "artifact_kind": "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V4_PUBLIC_PROBE",
        "artifact_version": W02_MORPH_V4_PUBLIC_PROBE_VERSION,
        "candidate_v1_v2_v3_mutations": 0,
        "dev": dev,
        "language": W02_MORPH_V4_PUBLIC_LANGUAGE,
        "license_id": W02_MORPH_V4_PUBLIC_LICENSE,
        "metamorphic": metamorphic,
        "public_commit_sha1": W02_MORPH_V4_PUBLIC_COMMIT,
        "repository_url": W02_MORPH_V4_PUBLIC_REPOSITORY,
        "shadow": shadow,
        "source_files": list(training.source_files),
        "source_key": W02_MORPH_V4_PUBLIC_SOURCE_KEY,
        "status": "PASS" if hard else "FAIL",
        "successor_version": W02_MORPH_SUCCESSOR_V4_VERSION,
        "test_split_content_reads": 0,
        "training": {
            "backoff_lexeme_row_count":
                training.index.backoff_lexeme_row_count,
            "exact_lexeme_row_count": training.index.exact_lexeme_row_count,
            "logic_operations": training.index.logic_operations,
            "row_count": training.index.row_count,
            "semantic_sha256": training.index.semantic_sha256,
            "sentence_count": training.sentence_count,
            "token_count": training.token_count,
            "unique_form_count": training.unique_form_count,
            "unique_tuple_count": training.unique_tuple_count,
        },
    }


__all__ = [
    "W02_MORPH_V4_PUBLIC_COMMIT",
    "W02_MORPH_V4_PUBLIC_FILES",
    "W02_MORPH_V4_PUBLIC_LANGUAGE",
    "W02_MORPH_V4_PUBLIC_LICENSE",
    "W02_MORPH_V4_PUBLIC_PROBE_VERSION",
    "W02_MORPH_V4_PUBLIC_REPOSITORY",
    "W02_MORPH_V4_PUBLIC_SOURCE_KEY",
    "W02MorphologySuccessorV4PublicProbeError",
    "W02MorphologySuccessorV4PublicTraining",
    "build_w02_morphology_successor_v4_public_training",
    "evaluate_w02_morphology_successor_v4_public_probe",
]
