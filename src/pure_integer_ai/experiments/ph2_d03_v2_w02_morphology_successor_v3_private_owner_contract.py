"""Pure identities for the successor V3 PUD-news private owner receipt."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


W02_MORPH_V3_PRIVATE_OWNER_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v3_private_owner_receipt_v1.json"
)
W02_MORPH_V3_PRIVATE_OWNER_RECEIPT_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V3-PRIVATE-OWNER-RECEIPT-V1"
)
W02_MORPH_V3_PRIVATE_OWNER_METADATA_SHA256 = (
    "15b173af001f01843f597ca077f75e4368c9f3e0f13d02ae0124a9400ac82873"
)
W02_MORPH_V3_PRIVATE_OWNER_METADATA_SIZE_BYTES = 8_889
W02_MORPH_V3_PRIVATE_OWNER_ID = "cf491c57bc5e9868"
W02_MORPH_V3_PRIVATE_OWNER_FAMILY_KEY = (
    "PH2-D03-V2-W02-SUCCESSOR-V3-PUD-NEWS-BLIND-R2-cf491c57bc5e9868"
)
W02_MORPH_V3_PRIVATE_PUBLIC_BASE_COMMIT = (
    "14cac016545bade1b8f359e5e43f2175d3227ea7"
)
W02_MORPH_V3_PRIVATE_SOURCE_EXTENSION_V3_SHA256 = (
    "54962c192f0d49b135646badbbb2ac81bea1b245bbacf87f27560eb3f6ebd1c2"
)
W02_MORPH_V3_PRIVATE_SOURCE_SNAPSHOT_COMMITMENT = (
    "66eb15478baa20bbf6399ef39c45cdbc5b4a675179d0ba0bb688e2176d616c3a"
)
W02_MORPH_V3_PRIVATE_SOURCE_KEY = "UD_ZH_PUD_R2_18_NEWS_BLIND_PRIVATE"
W02_MORPH_V3_PRIVATE_SOURCE_COUNT = 500
W02_MORPH_V3_PRIVATE_PAIR_COUNT = 500
W02_MORPH_V3_PRIVATE_SPLIT_COUNTS = {
    "adversarial": 100,
    "held_out": 350,
    "wall": 50,
}
W02_MORPH_V3_PRIVATE_DIMENSION_COUNTS = {
    "W-02-V2-BOUNDARY-WITHDRAWAL": 100,
    "W-02-V2-GENERATION-HARD-CONJUNCT": 100,
    "W-02-V2-MULTI-CANDIDATE": 100,
    "W-02-V2-NEW-CONTENT-MORPHOLOGY": 100,
    "W-02-V2-OOV": 100,
}
W02_MORPH_V3_PRIVATE_LAYOUTS = (
    "PRIVATE_SOURCE",
    "PRIVATE_HELD_OUT_OBSERVATION",
    "PRIVATE_ADVERSARIAL_OBSERVATION",
    "PRIVATE_WALL_OBSERVATION",
    "PRIVATE_HELD_OUT_LABEL",
    "PRIVATE_ADVERSARIAL_LABEL",
    "PRIVATE_WALL_LABEL",
)
W02_MORPH_V3_PRIVATE_PATHS = {
    "PRIVATE_SOURCE": "source/source_refs.jsonl.gz",
    "PRIVATE_HELD_OUT_OBSERVATION": "observations/held_out.jsonl.gz",
    "PRIVATE_ADVERSARIAL_OBSERVATION": "observations/adversarial.jsonl.gz",
    "PRIVATE_WALL_OBSERVATION": "observations/wall.jsonl.gz",
    "PRIVATE_HELD_OUT_LABEL": "evaluator/held_out.labels.jsonl.gz",
    "PRIVATE_ADVERSARIAL_LABEL": "evaluator/adversarial.labels.jsonl.gz",
    "PRIVATE_WALL_LABEL": "evaluator/wall.labels.jsonl.gz",
}
W02_MORPH_V3_PRIVATE_SPLITS = ("held_out", "adversarial", "wall")


# object-model: exception
class W02MorphologySuccessorV3PrivateOwnerError(RuntimeError):
    """The payload-free V3 owner receipt or a public dependency drifted."""


def require_sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W02MorphologySuccessorV3PrivateOwnerError(
            f"{where} is not lowercase SHA-256")
    return value


def require_positive(value: object, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise W02MorphologySuccessorV3PrivateOwnerError(
            f"{where} is not a positive integer")
    return value


def require_exact_dict(
        value: object,
        fields: set[str],
        *,
        where: str,
        ) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise W02MorphologySuccessorV3PrivateOwnerError(
            f"{where} fields drifted")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorV3PrivateFileIdentity:
    """Public file identity for the CC-BY-SA-3.0 PUD owner payload."""

    layout_key: str
    root_key: str
    record_kind: str
    split: str
    record_count: int
    content_size_bytes: int
    content_sha256: str
    transport_size_bytes: int
    transport_sha256: str
    first_record_key: tuple[int, ...]
    last_record_key: tuple[int, ...]
    license_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (self.layout_key not in W02_MORPH_V3_PRIVATE_PATHS
                or self.root_key != "PRIVATE_EVALUATOR_ROOT"):
            raise W02MorphologySuccessorV3PrivateOwnerError(
                "V3 owner file layout drifted")
        if self.layout_key == "PRIVATE_SOURCE":
            expected = ("source_ref", "")
        else:
            kind = "observation" if self.layout_key.endswith(
                "_OBSERVATION") else "evaluator_label"
            split = self.layout_key.removeprefix("PRIVATE_").removesuffix(
                "_OBSERVATION").removesuffix("_LABEL").lower()
            expected = (kind, split)
        if (self.record_kind, self.split) != expected:
            raise W02MorphologySuccessorV3PrivateOwnerError(
                "V3 owner file kind or split drifted")
        for name in (
                "record_count", "content_size_bytes", "transport_size_bytes"):
            require_positive(getattr(self, name), where=f"V3 owner file {name}")
        require_sha256(self.content_sha256, where="V3 owner content SHA")
        require_sha256(self.transport_sha256, where="V3 owner transport SHA")
        for name in ("first_record_key", "last_record_key"):
            key = getattr(self, name)
            if (not isinstance(key, tuple) or not key
                    or any(type(item) is not int or item <= 0 for item in key)):
                raise W02MorphologySuccessorV3PrivateOwnerError(
                    f"V3 owner {name} drifted")
        if self.first_record_key > self.last_record_key:
            raise W02MorphologySuccessorV3PrivateOwnerError(
                "V3 owner file key range reversed")
        if self.license_ids != ("CC-BY-SA-3.0",):
            raise W02MorphologySuccessorV3PrivateOwnerError(
                "V3 owner license identity drifted")

    @classmethod
    def from_dict(
            cls,
            value: object,
            ) -> "W02MorphologySuccessorV3PrivateFileIdentity":
        raw = require_exact_dict(value, {
            "content_sha256", "content_size_bytes", "first_record_key",
            "last_record_key", "layout_key", "license_ids", "record_count",
            "record_kind", "root_key", "split", "transport_sha256",
            "transport_size_bytes",
        }, where="V3 owner file identity")
        if (not isinstance(raw["first_record_key"], list)
                or not isinstance(raw["last_record_key"], list)
                or not isinstance(raw["license_ids"], list)):
            raise W02MorphologySuccessorV3PrivateOwnerError(
                "V3 owner file array fields drifted")
        return cls(
            str(raw["layout_key"]), str(raw["root_key"]),
            str(raw["record_kind"]), str(raw["split"]), raw["record_count"],
            raw["content_size_bytes"], str(raw["content_sha256"]),
            raw["transport_size_bytes"], str(raw["transport_sha256"]),
            tuple(raw["first_record_key"]), tuple(raw["last_record_key"]),
            tuple(str(item) for item in raw["license_ids"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "content_sha256": self.content_sha256,
            "content_size_bytes": self.content_size_bytes,
            "first_record_key": list(self.first_record_key),
            "last_record_key": list(self.last_record_key),
            "layout_key": self.layout_key,
            "license_ids": list(self.license_ids),
            "record_count": self.record_count,
            "record_kind": self.record_kind,
            "root_key": self.root_key,
            "split": self.split,
            "transport_sha256": self.transport_sha256,
            "transport_size_bytes": self.transport_size_bytes,
        }


def validate_v3_private_file_inventory(
        value: object,
        ) -> tuple[W02MorphologySuccessorV3PrivateFileIdentity, ...]:
    """Validate the seven-file metadata inventory without opening payload."""
    if not isinstance(value, list) or len(value) != len(
            W02_MORPH_V3_PRIVATE_LAYOUTS):
        raise W02MorphologySuccessorV3PrivateOwnerError(
            "V3 owner file inventory is incomplete")
    identities = []
    for row in value:
        if (not isinstance(row, dict)
                or row.get("relative_path")
                != W02_MORPH_V3_PRIVATE_PATHS.get(row.get("layout_key"))):
            raise W02MorphologySuccessorV3PrivateOwnerError(
                "V3 owner receipt relative path drifted")
        identities.append(
            W02MorphologySuccessorV3PrivateFileIdentity.from_dict({
                key: item for key, item in row.items()
                if key != "relative_path"
            }))
    files = tuple(identities)
    if tuple(row.layout_key for row in files) != W02_MORPH_V3_PRIVATE_LAYOUTS:
        raise W02MorphologySuccessorV3PrivateOwnerError(
            "V3 owner layout order drifted")
    by_layout = {row.layout_key: row for row in files}
    if (by_layout["PRIVATE_SOURCE"].record_count
            != W02_MORPH_V3_PRIVATE_SOURCE_COUNT):
        raise W02MorphologySuccessorV3PrivateOwnerError(
            "V3 owner source count drifted")
    for split in W02_MORPH_V3_PRIVATE_SPLITS:
        name = split.upper()
        observation = by_layout[f"PRIVATE_{name}_OBSERVATION"]
        label = by_layout[f"PRIVATE_{name}_LABEL"]
        if (observation.record_count != label.record_count
                or observation.record_count
                != W02_MORPH_V3_PRIVATE_SPLIT_COUNTS[split]):
            raise W02MorphologySuccessorV3PrivateOwnerError(
                "V3 owner observation/label inventory drifted")
    return files


__all__ = [
    "W02_MORPH_V3_PRIVATE_DIMENSION_COUNTS",
    "W02_MORPH_V3_PRIVATE_LAYOUTS",
    "W02_MORPH_V3_PRIVATE_OWNER_FAMILY_KEY",
    "W02_MORPH_V3_PRIVATE_OWNER_ID",
    "W02_MORPH_V3_PRIVATE_OWNER_METADATA_SHA256",
    "W02_MORPH_V3_PRIVATE_OWNER_METADATA_SIZE_BYTES",
    "W02_MORPH_V3_PRIVATE_OWNER_RECEIPT_PATH",
    "W02_MORPH_V3_PRIVATE_OWNER_RECEIPT_VERSION",
    "W02_MORPH_V3_PRIVATE_PAIR_COUNT",
    "W02_MORPH_V3_PRIVATE_PATHS",
    "W02_MORPH_V3_PRIVATE_PUBLIC_BASE_COMMIT",
    "W02_MORPH_V3_PRIVATE_SOURCE_COUNT",
    "W02_MORPH_V3_PRIVATE_SOURCE_EXTENSION_V3_SHA256",
    "W02_MORPH_V3_PRIVATE_SOURCE_KEY",
    "W02_MORPH_V3_PRIVATE_SOURCE_SNAPSHOT_COMMITMENT",
    "W02_MORPH_V3_PRIVATE_SPLIT_COUNTS",
    "W02MorphologySuccessorV3PrivateFileIdentity",
    "W02MorphologySuccessorV3PrivateOwnerError",
    "require_exact_dict",
    "require_positive",
    "require_sha256",
    "validate_v3_private_file_inventory",
]
