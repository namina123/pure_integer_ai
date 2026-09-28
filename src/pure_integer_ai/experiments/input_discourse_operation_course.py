"""Load a licensed integer course of trained discourse-operation surfaces."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.cognition.understanding.query_input_structure import (
    TrainedInputDiscourseOperator,
)


COURSE_FORMAT = 91592
COURSE_SCHEMA_VERSION = 1
MANIFEST_FORMAT = "PURE_INTEGER_INPUT_DISCOURSE_OPERATION_MANIFEST_V1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_input_discourse_operator_course(
        course_path: str | Path,
        manifest_path: str | Path,
        ) -> tuple[TrainedInputDiscourseOperator, ...]:
    """Validate and restore train-split graph relations, rejecting text fields."""
    course_file = Path(course_path).resolve(strict=True)
    manifest_file = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_file.read_text(encoding="ascii"))
    if (not isinstance(manifest, dict)
            or manifest.get("format") != MANIFEST_FORMAT
            or manifest.get("split") != "train"
            or manifest.get("license_ids") != ["MIT"]
            or manifest.get("source_kind") != "project_owned_integer_annotation"
            or manifest.get("free_dialogue_claim") != 0
            or manifest.get("course_sha256") != sha256_file(course_file)):
        raise ValueError("input discourse operation course manifest mismatch")
    payload = json.loads(course_file.read_text(encoding="ascii"))
    if (not isinstance(payload, list) or len(payload) != 3
            or payload[0] != COURSE_FORMAT
            or payload[1] != COURSE_SCHEMA_VERSION
            or not isinstance(payload[2], list)):
        raise ValueError("input discourse operation course schema mismatch")
    records = []
    for row in payload[2]:
        if (not isinstance(row, list) or len(row) != 2
                or any(type(value) is not int for value in row[0])
                or any(type(value) is not int for value in row[1])):
            raise ValueError("input discourse operation row is not integer-only")
        records.append(TrainedInputDiscourseOperator(
            tuple(row[0]), tuple(row[1])))
    operators = tuple(sorted(set(records), key=lambda item: item.stable_key()))
    if (len(operators) != len(records)
            or len(operators) != manifest.get("operator_count")):
        raise ValueError("input discourse operation identities are not unique")
    return operators


__all__ = [
    "COURSE_FORMAT", "COURSE_SCHEMA_VERSION", "MANIFEST_FORMAT",
    "load_input_discourse_operator_course", "sha256_file",
]
