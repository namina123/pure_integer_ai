"""Learned response-organization carrier for long human dialogue answers.

The model intentionally contains no answer text, token identity, fact, or fixed
question/answer mapping.  It retains only integer block kinds, their order,
length buckets, transition counts, and source commitments.  This keeps the
artifact deterministic and portable while giving generation a real learned
layout consumer for paragraphs and structured text.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


RESPONSE_ORGANIZATION_SCHEMA = 1
RESPONSE_ORGANIZATION_MAGIC = (21402, 260826, 71)

BLOCK_PARAGRAPH = 1
BLOCK_HEADING = 2
BLOCK_BULLET_LIST = 3
BLOCK_NUMBERED_LIST = 4
BLOCK_QUOTE = 5
BLOCK_CODE = 6
BLOCK_TABLE = 7
BLOCK_HTML = 8
BLOCK_KIND_COUNT = 8

_TRANSITION_NODE_COUNT = BLOCK_KIND_COUNT + 2
_TRANSITION_START = 0
_TRANSITION_END = BLOCK_KIND_COUNT + 1
_LENGTH_LIMITS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
_LENGTH_BUCKET_COUNT = len(_LENGTH_LIMITS) + 1

_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+\S")
_BULLET_RE = re.compile(r"^[ \t]{0,3}[-+*][ \t]+\S")
_NUMBERED_RE = re.compile(r"^[ \t]{0,3}\d{1,6}[.)][ \t]+\S")
_QUOTE_RE = re.compile(r"^[ \t]{0,3}>[ \t]?\S")
_TABLE_DELIMITER_RE = re.compile(
    r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*"
    r"(?:\|[ \t]*:?-{3,}:?[ \t]*)+\|?[ \t]*$")
_HTML_RE = re.compile(
    r"^[ \t]*</?[A-Za-z][A-Za-z0-9:-]*(?:[ \t][^<>]*)?>")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？!?])\s*")


# object-model: exception; interop=response-organization-v1
class ResponseOrganizationError(ValueError):
    """Response structure or integer artifact violates the frozen contract."""


# object-model: value; representation=struct; interop=response-organization-v1
@dataclass(frozen=True, slots=True)
class ResponseBlock:
    """One content-free structural block measurement."""

    kind: int
    item_count: int
    line_count: int
    char_count: int

    def __post_init__(self) -> None:
        if (type(self.kind) is not int
                or self.kind < 1 or self.kind > BLOCK_KIND_COUNT):
            raise ResponseOrganizationError("response block kind is invalid")
        for label, value in (
                ("item_count", self.item_count),
                ("line_count", self.line_count),
                ("char_count", self.char_count)):
            if type(value) is not int or value <= 0:
                raise ResponseOrganizationError(
                    f"response block {label} must be a positive integer")


# object-model: value; representation=struct; interop=response-organization-v1
@dataclass(frozen=True, slots=True)
class ResponseOrganizationProfile:
    """Content-free structure projected from one assistant response."""

    blocks: tuple[ResponseBlock, ...]
    total_chars: int

    def __post_init__(self) -> None:
        if (not isinstance(self.blocks, tuple) or not self.blocks
                or any(not isinstance(item, ResponseBlock)
                       for item in self.blocks)):
            raise ResponseOrganizationError(
                "response profile must contain structural blocks")
        if type(self.total_chars) is not int or self.total_chars <= 0:
            raise ResponseOrganizationError(
                "response profile total_chars must be positive")

    @property
    def signature(self) -> tuple[int, ...]:
        return tuple(item.kind for item in self.blocks)


# object-model: value; representation=struct; interop=response-organization-v1
@dataclass(frozen=True, slots=True)
class ResponseOrganizationModel:
    """Portable aggregate model; no source response surface is retained."""

    train_count: int
    heldout_count: int
    preferred_paragraph_chars: int
    course_sha256: tuple[int, ...]
    pack_sha256: tuple[int, ...]
    cursor_sha256: tuple[int, ...]
    summary_sha256: tuple[int, ...]
    block_counts: tuple[int, ...]
    transition_counts: tuple[int, ...]
    length_counts: tuple[int, ...]
    sequence_counts: tuple[tuple[tuple[int, ...], int], ...]

    def __post_init__(self) -> None:
        if (type(self.train_count) is not int or self.train_count <= 0
                or type(self.heldout_count) is not int
                or self.heldout_count <= 0):
            raise ResponseOrganizationError(
                "organization model requires non-empty train and heldout sets")
        if (type(self.preferred_paragraph_chars) is not int
                or self.preferred_paragraph_chars <= 0):
            raise ResponseOrganizationError(
                "preferred paragraph size must be positive")
        for label, value in (
                ("course_sha256", self.course_sha256),
                ("pack_sha256", self.pack_sha256),
                ("cursor_sha256", self.cursor_sha256),
                ("summary_sha256", self.summary_sha256)):
            if (not isinstance(value, tuple) or len(value) != 32
                    or any(type(item) is not int or item < 0 or item > 255
                           for item in value)):
                raise ResponseOrganizationError(f"{label} must be 32 bytes")
        if (len(self.block_counts) != BLOCK_KIND_COUNT
                or any(type(item) is not int or item < 0
                       for item in self.block_counts)
                or not any(self.block_counts)):
            raise ResponseOrganizationError("block count vector is invalid")
        if (len(self.transition_counts)
                != _TRANSITION_NODE_COUNT * _TRANSITION_NODE_COUNT
                or any(type(item) is not int or item < 0
                       for item in self.transition_counts)):
            raise ResponseOrganizationError("transition matrix is invalid")
        if (len(self.length_counts) != _LENGTH_BUCKET_COUNT
                or any(type(item) is not int or item < 0
                       for item in self.length_counts)
                or sum(self.length_counts) != self.train_count):
            raise ResponseOrganizationError("length histogram is invalid")
        if not isinstance(self.sequence_counts, tuple) or not self.sequence_counts:
            raise ResponseOrganizationError("sequence histogram is empty")
        previous: tuple[int, ...] | None = None
        total = 0
        for signature, count in self.sequence_counts:
            if (not isinstance(signature, tuple) or not signature
                    or any(type(item) is not int or item < 1
                           or item > BLOCK_KIND_COUNT for item in signature)
                    or type(count) is not int or count <= 0):
                raise ResponseOrganizationError(
                    "sequence histogram entry is invalid")
            if previous is not None and signature <= previous:
                raise ResponseOrganizationError(
                    "sequence histogram must be uniquely sorted")
            previous = signature
            total += count
        if total != self.train_count:
            raise ResponseOrganizationError(
                "sequence histogram does not cover all training responses")

    @property
    def supported_kinds(self) -> tuple[int, ...]:
        return tuple(index + 1 for index, count in enumerate(self.block_counts)
                     if count > 0)

    def integer_stream(self) -> tuple[int, ...]:
        """Return the complete normative integer representation."""
        result = [
            *RESPONSE_ORGANIZATION_MAGIC,
            RESPONSE_ORGANIZATION_SCHEMA,
            self.train_count,
            self.heldout_count,
            self.preferred_paragraph_chars,
            *self.course_sha256,
            *self.pack_sha256,
            *self.cursor_sha256,
            *self.summary_sha256,
            BLOCK_KIND_COUNT,
            *self.block_counts,
            _TRANSITION_NODE_COUNT,
            *self.transition_counts,
            _LENGTH_BUCKET_COUNT,
            *self.length_counts,
            len(self.sequence_counts),
        ]
        for signature, count in self.sequence_counts:
            result.extend((len(signature), *signature, count))
        return tuple(result)

    @classmethod
    def from_integer_stream(
            cls, values: tuple[int, ...],
            ) -> "ResponseOrganizationModel":
        """Strictly restore a model and reject extension or truncation."""
        if (not isinstance(values, tuple) or not values
                or any(type(item) is not int for item in values)):
            raise ResponseOrganizationError("model stream must be integer tuple")
        cursor = 0

        def take(count: int) -> tuple[int, ...]:
            nonlocal cursor
            if type(count) is not int or count < 0 or cursor + count > len(values):
                raise ResponseOrganizationError("model integer stream is truncated")
            result = values[cursor:cursor + count]
            cursor += count
            return result

        if take(len(RESPONSE_ORGANIZATION_MAGIC)) != RESPONSE_ORGANIZATION_MAGIC:
            raise ResponseOrganizationError("model magic is incompatible")
        if take(1) != (RESPONSE_ORGANIZATION_SCHEMA,):
            raise ResponseOrganizationError("model schema is incompatible")
        train_count, heldout_count, preferred = take(3)
        course_sha = take(32)
        pack_sha = take(32)
        cursor_sha = take(32)
        summary_sha = take(32)
        if take(1) != (BLOCK_KIND_COUNT,):
            raise ResponseOrganizationError("model block vector width drifted")
        block_counts = take(BLOCK_KIND_COUNT)
        if take(1) != (_TRANSITION_NODE_COUNT,):
            raise ResponseOrganizationError("model transition width drifted")
        transitions = take(_TRANSITION_NODE_COUNT * _TRANSITION_NODE_COUNT)
        if take(1) != (_LENGTH_BUCKET_COUNT,):
            raise ResponseOrganizationError("model length width drifted")
        lengths = take(_LENGTH_BUCKET_COUNT)
        sequence_count = take(1)[0]
        if sequence_count <= 0:
            raise ResponseOrganizationError("model sequence count is invalid")
        sequences = []
        for _ in range(sequence_count):
            size = take(1)[0]
            if size <= 0:
                raise ResponseOrganizationError("model signature is empty")
            signature = take(size)
            count = take(1)[0]
            sequences.append((signature, count))
        if cursor != len(values):
            raise ResponseOrganizationError("model stream has trailing integers")
        model = cls(
            train_count, heldout_count, preferred,
            course_sha, pack_sha, cursor_sha, summary_sha,
            block_counts, transitions, lengths, tuple(sequences),
        )
        if model.integer_stream() != values:
            raise ResponseOrganizationError("model stream is not canonical")
        return model


# object-model: value; representation=struct; interop=response-organization-v1
@dataclass(frozen=True, slots=True)
class ResponseOrganizationResult:
    """One model-backed organization decision over caller-owned content."""

    surface: str
    used: bool
    reason: str
    pattern_id: int
    trace: tuple[int, ...]


def _char_count(lines: list[str]) -> int:
    return sum(len(line.strip()) for line in lines)


def _block(kind: int, lines: list[str], *, item_count: int | None = None) -> ResponseBlock:
    count = len(lines) if item_count is None else item_count
    return ResponseBlock(kind, max(1, count), len(lines), max(1, _char_count(lines)))


def _is_table(lines: list[str], index: int) -> bool:
    return (index + 1 < len(lines) and "|" in lines[index]
            and _TABLE_DELIMITER_RE.fullmatch(lines[index + 1]) is not None)


def _starts_structured(lines: list[str], index: int) -> bool:
    line = lines[index]
    return bool(
        _FENCE_RE.match(line) or _HEADING_RE.match(line)
        or _BULLET_RE.match(line) or _NUMBERED_RE.match(line)
        or _QUOTE_RE.match(line) or _HTML_RE.match(line)
        or _is_table(lines, index)
    )


def profile_response_surface(surface: str) -> ResponseOrganizationProfile:
    """Project Markdown/HTML/plain response layout without retaining content."""
    if not isinstance(surface, str) or not surface.strip():
        raise ResponseOrganizationError("response surface must be non-empty")
    normalized = surface.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = normalized.split("\n")
    blocks: list[ResponseBlock] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        fence = _FENCE_RE.match(lines[index])
        if fence is not None:
            marker = fence.group(1)
            selected = [lines[index]]
            index += 1
            while index < len(lines):
                selected.append(lines[index])
                closing = lines[index].lstrip()
                index += 1
                if closing.startswith(marker[0] * len(marker)):
                    break
            blocks.append(_block(BLOCK_CODE, selected, item_count=1))
            continue
        if _is_table(lines, index):
            selected = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines) and lines[index].strip() and "|" in lines[index]:
                selected.append(lines[index])
                index += 1
            blocks.append(_block(
                BLOCK_TABLE, selected, item_count=max(1, len(selected) - 2)))
            continue
        single_kinds = (
            (_HEADING_RE, BLOCK_HEADING),
            (_BULLET_RE, BLOCK_BULLET_LIST),
            (_NUMBERED_RE, BLOCK_NUMBERED_LIST),
            (_QUOTE_RE, BLOCK_QUOTE),
            (_HTML_RE, BLOCK_HTML),
        )
        matched = next(((regex, kind) for regex, kind in single_kinds
                        if regex.match(lines[index]) is not None), None)
        if matched is not None:
            regex, kind = matched
            selected = []
            while (index < len(lines) and lines[index].strip()
                   and regex.match(lines[index]) is not None):
                selected.append(lines[index])
                index += 1
            blocks.append(_block(kind, selected))
            continue
        selected = [lines[index]]
        index += 1
        while (index < len(lines) and lines[index].strip()
               and not _starts_structured(lines, index)):
            selected.append(lines[index])
            index += 1
        blocks.append(_block(BLOCK_PARAGRAPH, selected, item_count=1))
    return ResponseOrganizationProfile(tuple(blocks), len(normalized))


def _length_bucket(value: int) -> int:
    for index, limit in enumerate(_LENGTH_LIMITS):
        if value <= limit:
            return index
    return len(_LENGTH_LIMITS)


def _weighted_median(values: list[int]) -> int:
    if not values:
        return 160
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def learn_response_organization_model(
        rows: Iterable[tuple[str, str]], *,
        course_sha256: tuple[int, ...],
        pack_sha256: tuple[int, ...],
        cursor_sha256: tuple[int, ...],
        summary_sha256: tuple[int, ...],
        ) -> ResponseOrganizationModel:
    """Learn aggregate response layout from explicit train/heldout rows."""
    train_profiles: list[ResponseOrganizationProfile] = []
    heldout_count = 0
    for split, surface in rows:
        if split not in {"train", "heldout"}:
            raise ResponseOrganizationError("response split must be train or heldout")
        profile = profile_response_surface(surface)
        if split == "train":
            train_profiles.append(profile)
        else:
            heldout_count += 1
    if not train_profiles or heldout_count <= 0:
        raise ResponseOrganizationError(
            "response organization needs train and heldout rows")
    block_counts = [0] * BLOCK_KIND_COUNT
    transitions = [0] * (_TRANSITION_NODE_COUNT * _TRANSITION_NODE_COUNT)
    length_counts = [0] * _LENGTH_BUCKET_COUNT
    sequence_counts: dict[tuple[int, ...], int] = {}
    paragraph_lengths: list[int] = []
    for profile in train_profiles:
        signature = profile.signature
        sequence_counts[signature] = sequence_counts.get(signature, 0) + 1
        length_counts[_length_bucket(profile.total_chars)] += 1
        previous = _TRANSITION_START
        for block in profile.blocks:
            block_counts[block.kind - 1] += 1
            transitions[previous * _TRANSITION_NODE_COUNT + block.kind] += 1
            previous = block.kind
            if block.kind == BLOCK_PARAGRAPH:
                paragraph_lengths.append(block.char_count)
        transitions[previous * _TRANSITION_NODE_COUNT + _TRANSITION_END] += 1
    return ResponseOrganizationModel(
        len(train_profiles), heldout_count,
        _weighted_median(paragraph_lengths),
        course_sha256, pack_sha256, cursor_sha256, summary_sha256,
        tuple(block_counts), tuple(transitions), tuple(length_counts),
        tuple(sorted(sequence_counts.items())),
    )


def response_feature_counts(
        profiles: Iterable[ResponseOrganizationProfile],
        ) -> tuple[tuple[str, int], ...]:
    """Return stable aggregate feature counts for heldout reporting only."""
    counts = {
        "paragraph": 0,
        "list": 0,
        "heading": 0,
        "code": 0,
        "table": 0,
        "html": 0,
        "mixed": 0,
    }
    for profile in profiles:
        kinds = set(profile.signature)
        counts["paragraph"] += int(BLOCK_PARAGRAPH in kinds)
        counts["list"] += int(bool(
            kinds.intersection({BLOCK_BULLET_LIST, BLOCK_NUMBERED_LIST})))
        counts["heading"] += int(BLOCK_HEADING in kinds)
        counts["code"] += int(BLOCK_CODE in kinds)
        counts["table"] += int(BLOCK_TABLE in kinds)
        counts["html"] += int(BLOCK_HTML in kinds)
        counts["mixed"] += int(len(kinds) > 1)
    return tuple(sorted(counts.items()))


def _pattern_id(signature: tuple[int, ...]) -> int:
    result = 170001
    for kind in signature[:16]:
        result = result * 11 + kind
    return result


def _paragraph_groups(surface: str, target: int) -> tuple[str, ...]:
    parts = tuple(item for item in _SENTENCE_BOUNDARY_RE.split(surface)
                  if item)
    if len(parts) < 2:
        return (surface,)
    groups: list[str] = []
    current = ""
    for part in parts:
        if current and len(current) >= target:
            groups.append(current.strip())
            current = part
        else:
            current += part
    if current.strip():
        groups.append(current.strip())
    return tuple(groups)


def organize_response_surface(
        model: ResponseOrganizationModel, surface: str,
        ) -> ResponseOrganizationResult:
    """Consume learned layout without changing caller-owned lexical content."""
    if not isinstance(model, ResponseOrganizationModel):
        raise TypeError("response organization requires a learned model")
    profile = profile_response_surface(surface)
    supported = set(model.supported_kinds)
    if not set(profile.signature).issubset(supported):
        return ResponseOrganizationResult(
            surface.strip(), False, "unsupported_response_block", 0,
            (0, *profile.signature))
    pattern_id = _pattern_id(profile.signature)
    trace = (
        1, model.preferred_paragraph_chars, len(profile.blocks),
        *profile.signature,
    )
    if (len(profile.blocks) > 1
            or profile.blocks[0].kind != BLOCK_PARAGRAPH):
        return ResponseOrganizationResult(
            surface.strip(), True, "organization_validated",
            pattern_id, trace)
    original = surface.strip()
    target = max(48, min(512, model.preferred_paragraph_chars))
    if len(original) < max(96, target * 2):
        return ResponseOrganizationResult(
            original, False, "short_plain_response", 0, trace)
    groups = _paragraph_groups(original, target)
    if len(groups) < 2:
        return ResponseOrganizationResult(
            original, False, "no_safe_sentence_boundary", 0, trace)
    organized = "\n\n".join(groups)
    if organized.replace("\n", "") != original.replace("\n", ""):
        return ResponseOrganizationResult(
            original, False, "lexical_content_drift", 0, trace)
    return ResponseOrganizationResult(
        organized, True, "learned_paragraph_spacing", pattern_id, trace)


__all__ = [
    "BLOCK_BULLET_LIST", "BLOCK_CODE", "BLOCK_HEADING", "BLOCK_HTML",
    "BLOCK_KIND_COUNT", "BLOCK_NUMBERED_LIST", "BLOCK_PARAGRAPH",
    "BLOCK_QUOTE", "BLOCK_TABLE", "ResponseBlock",
    "ResponseOrganizationError", "ResponseOrganizationModel",
    "ResponseOrganizationProfile", "ResponseOrganizationResult",
    "learn_response_organization_model", "organize_response_surface",
    "profile_response_surface", "response_feature_counts",
]
