"""为 recovery-v7 atom identifiability 提供来源化离散事实。

本模块只解析 sealed OpenCC 与固定 UniMorph English 文件，不承担 proposal、
授权、评分或文件发布。raw surface 只保留在调用方内存中。
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_conceptnet_alias_records import (
    neutral_source_units,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_source_pack import (
    read_normalization_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


UNIMORPH_ENGLISH_REPOSITORY = "https://github.com/unimorph/eng"
UNIMORPH_ENGLISH_COMMIT = "66e0e9e8e2dcd196da081a25a48e5c1fe3d8b49b"
UNIMORPH_ENGLISH_TREE = "1bce2d510492917dd3b37b875751ceed49276796"
UNIMORPH_ENGLISH_COMMIT_DATE = "2023-02-16T21:53:39+11:00"
UNIMORPH_ENGLISH_LICENSE_ID = "CC-BY-SA-3.0"
UNIMORPH_ENGLISH_README_BYTES = 107
UNIMORPH_ENGLISH_README_SHA256 = (
    "204b46bfdd6c41f909c8a0ba9a559989e25097bc9c5e940ce608d94e709eabc0")
UNIMORPH_ENGLISH_DATA_BYTES = 18_675_382
UNIMORPH_ENGLISH_DATA_SHA256 = (
    "0c7bef5064e3ae1a0a3e7dc7eb3c7912dbada4b210abef0a4cc06cc733fef43f")
UNIMORPH_ENGLISH_LICENSE_BYTES = 22_240
UNIMORPH_ENGLISH_LICENSE_SHA256 = (
    "3f941b3b89cf7b8370ceb83cc76d2120d471b58735d8ca60238a751a48d7f72f")

_MARKED_MORPHOLOGY_FEATURES = (
    "V.PTCP", "PST", ";PL", "PRS;3;SG", "NOM(3,SG)")


def _sha256(payload: bytes) -> str:
    """返回来源文件或 analysis-set 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _verified_payload(
        path: str | Path,
        *,
        expected_bytes: int,
        expected_sha256: str,
        label: str,
        ) -> bytes:
    """读取并核对一份固定来源文件。"""
    payload = Path(path).resolve().read_bytes()
    if len(payload) != expected_bytes or _sha256(payload) != expected_sha256:
        raise BroadQaExternalDataError(f"{label} identity 漂移")
    return payload


def parse_unimorph_english(
        *,
        data_path: str | Path,
        readme_path: str | Path,
        license_path: str | Path,
        ) -> tuple[dict[str, tuple[tuple[str, str], ...]], dict[str, int]]:
    """解析固定 UniMorph English 三列 inflection 表。"""
    readme = _verified_payload(
        readme_path, expected_bytes=UNIMORPH_ENGLISH_README_BYTES,
        expected_sha256=UNIMORPH_ENGLISH_README_SHA256,
        label="UniMorph English README")
    if (b"Source: Wikipedia" not in readme
            or b"creativecommons.org/licenses/by-sa/3.0" not in readme):
        raise BroadQaExternalDataError("UniMorph English README license 漂移")
    _verified_payload(
        license_path, expected_bytes=UNIMORPH_ENGLISH_LICENSE_BYTES,
        expected_sha256=UNIMORPH_ENGLISH_LICENSE_SHA256,
        label="UniMorph English legalcode")
    payload = _verified_payload(
        data_path, expected_bytes=UNIMORPH_ENGLISH_DATA_BYTES,
        expected_sha256=UNIMORPH_ENGLISH_DATA_SHA256,
        label="UniMorph English data")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise BroadQaExternalDataError(
            "UniMorph English data 非 UTF-8") from error
    values: dict[str, set[tuple[str, str]]] = defaultdict(set)
    lemma_casefold_count = 0
    form_casefold_count = 0
    line_count = 0
    for raw in text.splitlines():
        line_count += 1
        fields = raw.split("\t")
        if len(fields) != 3 or any(not item for item in fields):
            raise BroadQaExternalDataError("UniMorph English row 漂移")
        lemma, form, features = fields
        folded_lemma = lemma.casefold()
        folded_form = form.casefold()
        lemma_casefold_count += int(folded_lemma != lemma)
        form_casefold_count += int(folded_form != form)
        values[folded_form].add((folded_lemma, features))
    frozen = {
        form: tuple(sorted(analyses)) for form, analyses in values.items()}
    return frozen, {
        "ambiguous_form_count": sum(
            len(analyses) > 1 for analyses in frozen.values()),
        "form_casefold_count": form_casefold_count,
        "lemma_casefold_count": lemma_casefold_count,
        "line_count": line_count,
        "unique_form_count": len(frozen),
    }


def unimorph_segment_facts(
        segment: str,
        morphology_by_form: dict[str, tuple[tuple[str, str], ...]],
        ) -> tuple[str, ...]:
    """把 source segment 投影为逐 unit analysis-set commitment。"""
    if not isinstance(segment, str):
        raise BroadQaExternalDataError("UniMorph source segment 非字符串")
    values = []
    for unit in neutral_source_units(segment) if segment else ():
        analyses = morphology_by_form.get(unit, ())
        if not analyses:
            continue
        marked = any(
            any(marker in features for marker in _MARKED_MORPHOLOGY_FEATURES)
            for _lemma, features in analyses)
        prefix = "UNIMORPH_MARKED:" if marked else "UNIMORPH:"
        values.append(prefix + _sha256(canonical_json_bytes(analyses)))
    return tuple(sorted(set(values)))


def read_opencc_unique_t2s_routes(
        source_pack_dir: str | Path,
        ) -> tuple[dict[str, str], dict[str, int]]:
    """回读 sealed OpenCC pack，只保留唯一 first-value t2s route。"""
    root = Path(source_pack_dir).resolve()
    manifest = read_normalization_source_pack(root)
    expected = {
        str(item["relative_path"]): str(item["sha256"])
        for item in manifest["files"]}
    routes = {}
    ambiguous = 0
    line_count = 0
    for relative in (
            "dictionary/TSPhrases.txt", "dictionary/TSCharacters.txt"):
        payload = (root / relative).read_bytes()
        if _sha256(payload) != expected[relative]:
            raise BroadQaExternalDataError("OpenCC sealed dictionary SHA 漂移")
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise BroadQaExternalDataError(
                "OpenCC sealed dictionary 非 UTF-8") from error
        for raw in text.splitlines():
            line_count += 1
            fields = raw.split("\t")
            if len(fields) != 2 or not fields[0] or not fields[1]:
                raise BroadQaExternalDataError("OpenCC dictionary row 漂移")
            outputs = tuple(item for item in fields[1].split(" ") if item)
            if len(set(outputs)) != 1:
                ambiguous += 1
                continue
            routes[fields[0]] = outputs[0]
    return routes, {
        "ambiguous_route_count": ambiguous,
        "line_count": line_count,
        "unique_route_count": len(routes),
    }


__all__ = [
    "UNIMORPH_ENGLISH_COMMIT",
    "UNIMORPH_ENGLISH_COMMIT_DATE",
    "UNIMORPH_ENGLISH_DATA_BYTES",
    "UNIMORPH_ENGLISH_DATA_SHA256",
    "UNIMORPH_ENGLISH_LICENSE_ID",
    "UNIMORPH_ENGLISH_REPOSITORY",
    "UNIMORPH_ENGLISH_TREE",
    "parse_unimorph_english",
    "read_opencc_unique_t2s_routes",
    "unimorph_segment_facts",
]
