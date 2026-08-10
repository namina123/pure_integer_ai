"""Pure identities for the successor V3 R4 private owner receipt."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_contract import (
    W02_MORPH_V3_PRIVATE_DIMENSION_COUNTS,
    W02_MORPH_V3_PRIVATE_LAYOUTS,
    W02_MORPH_V3_PRIVATE_PAIR_COUNT,
    W02_MORPH_V3_PRIVATE_PATHS,
    W02_MORPH_V3_PRIVATE_SOURCE_COUNT,
    W02_MORPH_V3_PRIVATE_SPLIT_COUNTS,
    W02_MORPH_V3_PRIVATE_SPLITS,
)


W02_MORPH_V3_PRIVATE_R4_OWNER_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v3_private_owner_r4_receipt_v1.json"
)
W02_MORPH_V3_PRIVATE_R4_OWNER_RECEIPT_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V3-PRIVATE-OWNER-R4-RECEIPT-V1"
)
W02_MORPH_V3_PRIVATE_R4_METADATA_SHA256 = (
    "8541a0b6bbb331d10c5bbf829326b90bcb2ea8cd50286514d88f01c6f6ebb228"
)
W02_MORPH_V3_PRIVATE_R4_METADATA_SIZE_BYTES = 8_663
W02_MORPH_V3_PRIVATE_R4_OWNER_ID = "98f7b12080d29f44"
W02_MORPH_V3_PRIVATE_R4_OWNER_FAMILY_KEY = (
    "PH2-D03-V2-W02-SUCCESSOR-V3-R4-"
    "UD_LZH_KYOTO_R2_18_TEST_BLIND_PRIVATE-"
    "2f5ff2e1-226c48f1-98f7b12080d29f44"
)
W02_MORPH_V3_PRIVATE_R4_PUBLIC_BASE_COMMIT = (
    "04746fc0f9ffc189549424ec81b765eccaa1cf24"
)
W02_MORPH_V3_PRIVATE_R4_SOURCE_KEY = (
    "UD_LZH_KYOTO_R2_18_TEST_BLIND_PRIVATE"
)
W02_MORPH_V3_PRIVATE_R4_SOURCE_EXTENSION_CODE_SHA256 = (
    "da799df8a505e01dcc74c969c350dfb33b99b85e683d3ca5c1088133ac43169e"
)
W02_MORPH_V3_PRIVATE_R4_SOURCE_EXTENSION_MANIFEST_SHA256 = (
    "76003a6531df489ee1ec91772ca8b4be449d628418de387bcc0f67c9b3c2be11"
)
W02_MORPH_V3_PRIVATE_R4_SOURCE_SNAPSHOT_COMMITMENT = (
    "a2b7ed5df9b4ec8c94db18e3e4a05bbf611df94d1aaa1ebd0e0dee264c1fcae4"
)
W02_MORPH_V3_PRIVATE_R4_LABEL_BINDING_SHA256 = (
    "3e3f9f9116772f7153d4d2467e26757e9775ee30f8bc1b6ad426b00bbf83cd50"
)
W02_MORPH_V3_PRIVATE_R4_DOUBLE_PASS_SHA256 = (
    "674156b886c24c47dd0556ef813dec6d6d27aac2f0294932ac811f18c3f0b447"
)
W02_MORPH_V3_PRIVATE_R4_DIMENSION_BINDINGS = (
    ("W-02-V2-BOUNDARY-WITHDRAWAL", (1, 100_494_257_788_431_926), 100),
    ("W-02-V2-MULTI-CANDIDATE", (1, 7_231_649_387_489_044_918), 100),
    ("W-02-V2-NEW-CONTENT-MORPHOLOGY", (1, 1_480_950_902_982_636_330), 100),
    ("W-02-V2-OOV", (1, 1_424_821_230_297_126_309), 100),
    ("W-02-V2-GENERATION-HARD-CONJUNCT", (1, 5_646_924_068_434_684_793), 100),
)


# object-model: exception
class W02MorphologySuccessorV3PrivateOwnerR4Error(RuntimeError):
    """The payload-free R4 owner receipt or a public dependency drifted."""


def require_exact_dict(
        value: object, fields: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            f"{where} fields drifted")
    return value


def require_sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            f"{where} is not lowercase SHA-256")
    return value


def require_positive(value: object, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            f"{where} is not a positive integer")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorV3PrivateR4FileIdentity:
    """Public file identity for the CC-BY-SA-4.0 Kyoto owner payload."""

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
            raise W02MorphologySuccessorV3PrivateOwnerR4Error(
                "R4 owner file layout drifted")
        if self.layout_key == "PRIVATE_SOURCE":
            expected = ("source_ref", "")
        else:
            kind = ("observation" if self.layout_key.endswith("_OBSERVATION")
                    else "evaluator_label")
            split = self.layout_key.removeprefix("PRIVATE_").removesuffix(
                "_OBSERVATION").removesuffix("_LABEL").lower()
            expected = (kind, split)
        if (self.record_kind, self.split) != expected:
            raise W02MorphologySuccessorV3PrivateOwnerR4Error(
                "R4 owner file kind or split drifted")
        for name in (
                "record_count", "content_size_bytes", "transport_size_bytes"):
            require_positive(getattr(self, name), where=f"R4 owner {name}")
        require_sha256(self.content_sha256, where="R4 owner content SHA")
        require_sha256(self.transport_sha256, where="R4 owner transport SHA")
        for name in ("first_record_key", "last_record_key"):
            key = getattr(self, name)
            if (not isinstance(key, tuple) or not key
                    or any(type(item) is not int or item <= 0 for item in key)):
                raise W02MorphologySuccessorV3PrivateOwnerR4Error(
                    f"R4 owner {name} drifted")
        if self.first_record_key > self.last_record_key:
            raise W02MorphologySuccessorV3PrivateOwnerR4Error(
                "R4 owner file key range reversed")
        if self.license_ids != ("CC-BY-SA-4.0",):
            raise W02MorphologySuccessorV3PrivateOwnerR4Error(
                "R4 owner license identity drifted")

    @classmethod
    def from_dict(
            cls, value: object,
            ) -> "W02MorphologySuccessorV3PrivateR4FileIdentity":
        raw = require_exact_dict(value, {
            "content_sha256", "content_size_bytes", "first_record_key",
            "last_record_key", "layout_key", "license_ids", "record_count",
            "record_kind", "root_key", "split", "transport_sha256",
            "transport_size_bytes",
        }, where="R4 owner file identity")
        if (not isinstance(raw["first_record_key"], list)
                or not isinstance(raw["last_record_key"], list)
                or not isinstance(raw["license_ids"], list)):
            raise W02MorphologySuccessorV3PrivateOwnerR4Error(
                "R4 owner file array fields drifted")
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


def validate_r4_private_file_inventory(
        value: object,
        ) -> tuple[W02MorphologySuccessorV3PrivateR4FileIdentity, ...]:
    if not isinstance(value, list) or len(value) != len(
            W02_MORPH_V3_PRIVATE_LAYOUTS):
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 owner file inventory is incomplete")
    identities = []
    for row in value:
        if (not isinstance(row, dict)
                or row.get("relative_path")
                != W02_MORPH_V3_PRIVATE_PATHS.get(row.get("layout_key"))):
            raise W02MorphologySuccessorV3PrivateOwnerR4Error(
                "R4 owner receipt relative path drifted")
        identities.append(W02MorphologySuccessorV3PrivateR4FileIdentity.from_dict({
            key: item for key, item in row.items() if key != "relative_path"
        }))
    files = tuple(identities)
    if tuple(row.layout_key for row in files) != W02_MORPH_V3_PRIVATE_LAYOUTS:
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 owner layout order drifted")
    by_layout = {row.layout_key: row for row in files}
    if by_layout["PRIVATE_SOURCE"].record_count != W02_MORPH_V3_PRIVATE_SOURCE_COUNT:
        raise W02MorphologySuccessorV3PrivateOwnerR4Error(
            "R4 owner source count drifted")
    for split in W02_MORPH_V3_PRIVATE_SPLITS:
        name = split.upper()
        observation = by_layout[f"PRIVATE_{name}_OBSERVATION"]
        label = by_layout[f"PRIVATE_{name}_LABEL"]
        if (observation.record_count != label.record_count
                or observation.record_count
                != W02_MORPH_V3_PRIVATE_SPLIT_COUNTS[split]):
            raise W02MorphologySuccessorV3PrivateOwnerR4Error(
                "R4 owner observation/label inventory drifted")
    return files


__all__ = [
    "W02_MORPH_V3_PRIVATE_DIMENSION_COUNTS",
    "W02_MORPH_V3_PRIVATE_LAYOUTS",
    "W02_MORPH_V3_PRIVATE_PAIR_COUNT",
    "W02_MORPH_V3_PRIVATE_PATHS",
    "W02_MORPH_V3_PRIVATE_R4_DIMENSION_BINDINGS",
    "W02_MORPH_V3_PRIVATE_R4_DOUBLE_PASS_SHA256",
    "W02_MORPH_V3_PRIVATE_R4_LABEL_BINDING_SHA256",
    "W02_MORPH_V3_PRIVATE_R4_METADATA_SHA256",
    "W02_MORPH_V3_PRIVATE_R4_METADATA_SIZE_BYTES",
    "W02_MORPH_V3_PRIVATE_R4_OWNER_FAMILY_KEY",
    "W02_MORPH_V3_PRIVATE_R4_OWNER_ID",
    "W02_MORPH_V3_PRIVATE_R4_OWNER_RECEIPT_PATH",
    "W02_MORPH_V3_PRIVATE_R4_OWNER_RECEIPT_VERSION",
    "W02_MORPH_V3_PRIVATE_R4_PUBLIC_BASE_COMMIT",
    "W02_MORPH_V3_PRIVATE_R4_SOURCE_EXTENSION_CODE_SHA256",
    "W02_MORPH_V3_PRIVATE_R4_SOURCE_EXTENSION_MANIFEST_SHA256",
    "W02_MORPH_V3_PRIVATE_R4_SOURCE_KEY",
    "W02_MORPH_V3_PRIVATE_R4_SOURCE_SNAPSHOT_COMMITMENT",
    "W02_MORPH_V3_PRIVATE_SOURCE_COUNT",
    "W02_MORPH_V3_PRIVATE_SPLIT_COUNTS",
    "W02_MORPH_V3_PRIVATE_SPLITS",
    "W02MorphologySuccessorV3PrivateOwnerR4Error",
    "W02MorphologySuccessorV3PrivateR4FileIdentity",
    "require_exact_dict",
    "require_positive",
    "require_sha256",
    "validate_r4_private_file_inventory",
]
