"""Load a licensed integer course for compositional input dialogue acts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.input_dialogue_act_graph import (
    TrainedInputDialogueActFrame,
    TrainedInputDialogueActLexeme,
)


COURSE_FORMAT = 91692
COURSE_SCHEMA_VERSION = 1
MANIFEST_FORMAT = "PURE_INTEGER_INPUT_DIALOGUE_ACT_COURSE_V1"
SOURCE_IDENTITY = (91692, 1, 20261108)
LICENSE_IDENTITY = (91696, 1)


@dataclass(frozen=True, slots=True)
class InputDialogueActCourse:
    lexemes: tuple[TrainedInputDialogueActLexeme, ...]
    frames: tuple[TrainedInputDialogueActFrame, ...]
    source_identity: tuple[int, ...]
    license_identity: tuple[int, ...]
    course_sha256: tuple[int, ...]


def load_input_dialogue_act_course(
        course_path: str | Path,
        manifest_path: str | Path,
        ) -> InputDialogueActCourse:
    """Verify a train-split integer course and restore its typed graph rows."""
    course_file = Path(course_path).resolve(strict=True)
    manifest_file = Path(manifest_path).resolve(strict=True)
    payload = course_file.read_bytes()
    digest = hashlib.sha256(payload).digest()
    manifest = json.loads(manifest_file.read_text(encoding="ascii"))
    if not isinstance(manifest, dict):
        raise ValueError("input dialogue-act course manifest mismatch")
    source_identity = manifest.get("source_identity")
    if (manifest.get("format") != MANIFEST_FORMAT
            or manifest.get("schema_version") != COURSE_SCHEMA_VERSION
            or manifest.get("split") != "train"
            or manifest.get("source_key") != "AUTHORED_CC0_V1"
            or manifest.get("source_kind") != "project_owned_integer_annotation"
            or manifest.get("license_ids") != ["CC0-1.0"]
            or manifest.get("free_dialogue_claim") != 0
            or type(source_identity) is not list or not source_identity
            or any(type(item) is not int or item < 0
                   for item in source_identity)
            or manifest.get("license_identity") != list(LICENSE_IDENTITY)
            or manifest.get("course_sha256") != digest.hex()):
        raise ValueError("input dialogue-act course manifest mismatch")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("input dialogue-act course must be ASCII integer JSON") from error
    if (type(value) is not list or len(value) != 4
            or value[0] != COURSE_FORMAT
            or value[1] != COURSE_SCHEMA_VERSION
            or type(value[2]) is not list or type(value[3]) is not list):
        raise ValueError("input dialogue-act course schema mismatch")

    lexemes = []
    for row in value[2]:
        if (type(row) is not list or len(row) != 3
                or type(row[0]) is not int or row[0] <= 0
                or type(row[1]) is not list or type(row[2]) is not list
                or any(type(item) is not int for item in (*row[1], *row[2]))):
            raise ValueError("input dialogue-act lexeme row is not integer-only")
        lexemes.append(TrainedInputDialogueActLexeme(
            tuple(row[1]), tuple(row[2])))

    frames = []
    for row in value[3]:
        if (type(row) is not list or len(row) != 4
                or type(row[0]) is not int or row[0] <= 0
                or type(row[1]) is not list
                or any(type(item) is not int for item in row[1])
                or type(row[2]) is not int or row[2] <= 0
                or type(row[3]) is not list):
            raise ValueError("input dialogue-act frame row is not integer-only")
        symbols = []
        for symbol in row[3]:
            if (type(symbol) is not list or not symbol
                    or any(type(item) is not int for item in symbol)):
                raise ValueError("input dialogue-act frame symbol is invalid")
            symbols.append(tuple(symbol))
        frames.append(TrainedInputDialogueActFrame(
            tuple(row[1]), row[2], tuple(symbols)))

    if (not lexemes or not frames
            or len(lexemes) != manifest.get("lexeme_count")
            or len(frames) != manifest.get("frame_count")
            or len(set(item.stable_key() for item in lexemes)) != len(lexemes)
            or len(set(item.stable_key() for item in frames)) != len(frames)):
        raise ValueError("input dialogue-act course rows are incomplete or duplicate")
    ordinals = tuple(row[0] for row in value[2])
    frame_ordinals = tuple(row[0] for row in value[3])
    if (ordinals != tuple(range(1, len(ordinals) + 1))
            or frame_ordinals != tuple(range(1, len(frame_ordinals) + 1))):
        raise ValueError("input dialogue-act course ordinals are not contiguous")
    known_symbols = {item.symbol_key for item in lexemes}
    if any(symbol not in known_symbols
           for frame in frames for symbol in frame.symbol_sequence):
        raise ValueError("input dialogue-act frame references an untrained symbol")
    return InputDialogueActCourse(
        tuple(lexemes), tuple(frames), tuple(source_identity), LICENSE_IDENTITY,
        tuple(digest),
    )


__all__ = [
    "COURSE_FORMAT", "COURSE_SCHEMA_VERSION", "LICENSE_IDENTITY",
    "MANIFEST_FORMAT", "SOURCE_IDENTITY", "InputDialogueActCourse",
    "load_input_dialogue_act_course",
]
