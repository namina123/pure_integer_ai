"""successor V3 R5 私有 owner receipt 的纯身份合同。"""
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


W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v3_private_owner_r5_receipt_v1.json"
)
W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V3-PRIVATE-OWNER-R5-RECEIPT-V1"
)
W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_SHA256 = (
    "bb645a0a94f3aa6eb2973a293181d5f4be0e3cfa54a3d2948d117db1b25d05a3"
)
W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_SIZE_BYTES = 14_385
W02_MORPH_V3_PRIVATE_R5_METADATA_SHA256 = (
    "20f063002659114f23345630e6f39335f8a8371f56f0e101444fa20cfc247472"
)
W02_MORPH_V3_PRIVATE_R5_METADATA_SIZE_BYTES = 12_742
W02_MORPH_V3_PRIVATE_R5_OWNER_ID = "77a594e8813f77876b79c356e9161eb9"
W02_MORPH_V3_PRIVATE_R5_OWNER_FAMILY_KEY = (
    "PH2-D03-V2-W02-SUCCESSOR-V3-R5-UD-LZH-KYOTO-R2-18-TEST-REMAINDER-"
    "77a594e8813f77876b79c356e9161eb9"
)
W02_MORPH_V3_PRIVATE_R5_PUBLIC_BASE_COMMIT = (
    "a62dd65ab706e08cd1ab02809c73745e8f659a2a"
)
W02_MORPH_V3_PRIVATE_R5_SOURCE_KEY = (
    "UD_LZH_KYOTO_R2_18_TEST_REMAINDER_BLIND_PRIVATE"
)
W02_MORPH_V3_PRIVATE_R5_SOURCE_EXTENSION_CODE_SHA256 = (
    "c931a57db70d60456290d1e0457b4b1aa215cecefa89e6c0616987afc1e96876"
)
W02_MORPH_V3_PRIVATE_R5_SOURCE_EXTENSION_MANIFEST_SHA256 = (
    "55853ce32be99ca955209957b86eab569a58ec5dc0f18c9c38e7d9a8357b6535"
)
W02_MORPH_V3_PRIVATE_R5_ADAPTER_CODE_SHA256 = (
    "a71dd56a68e94eea8bb7d87e569d96a98a029c488a4ea02d4de11d3c4cf58a69"
)
W02_MORPH_V3_PRIVATE_R5_PROBE_CODE_SHA256 = (
    "e997947a1e72fe0b9ac9afd84cb2738ccf3a2c2c15d5ee179db9d6e4d24ee867"
)
W02_MORPH_V3_PRIVATE_R5_PROBE_REPORT_SHA256 = (
    "41fd0bd004c2c6fe5c1f8b6f6d6c8251af8e3fc7be0778975edbaabaaf979a31"
)
W02_MORPH_V3_PRIVATE_R5_ORDINAL_SLICE_COMMITMENT = (
    "1b44c1421dc7dad35f80220a2caac8941828022b46da34f3eba8f02f0e1ba817"
)
W02_MORPH_V3_PRIVATE_R5_LABEL_BINDING_SHA256 = (
    "adb9136747577d81126fb8225080b7cd909944584c7160526ac86b9f7c6be233"
)
W02_MORPH_V3_PRIVATE_R5_DIMENSION_BINDINGS = (
    ("W-02-V2-BOUNDARY-WITHDRAWAL", (1, 100_494_257_788_431_926), 100),
    ("W-02-V2-MULTI-CANDIDATE", (1, 7_231_649_387_489_044_918), 100),
    ("W-02-V2-NEW-CONTENT-MORPHOLOGY", (1, 1_480_950_902_982_636_330), 100),
    ("W-02-V2-OOV", (1, 1_424_821_230_297_126_309), 100),
    ("W-02-V2-GENERATION-HARD-CONJUNCT", (1, 5_646_924_068_434_684_793), 100),
)


# object-model: exception
class W02MorphologySuccessorV3PrivateOwnerR5Error(RuntimeError):
    """无 payload 的 R5 owner receipt 或公开依赖发生漂移。"""


def require_exact_dict(
        value: object, fields: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            f"{where} fields drifted")
    return value


def require_sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            f"{where} is not lowercase SHA-256")
    return value


def require_positive(value: object, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            f"{where} is not a positive integer")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorV3PrivateR5FileIdentity:
    """R5 CC-BY-SA-4.0 payload 的公开文件身份。"""

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
            raise W02MorphologySuccessorV3PrivateOwnerR5Error(
                "R5 owner file layout drifted")
        if self.layout_key == "PRIVATE_SOURCE":
            expected = ("source_ref", "")
        else:
            kind = ("observation" if self.layout_key.endswith("_OBSERVATION")
                    else "evaluator_label")
            split = self.layout_key.removeprefix("PRIVATE_").removesuffix(
                "_OBSERVATION").removesuffix("_LABEL").lower()
            expected = (kind, split)
        if (self.record_kind, self.split) != expected:
            raise W02MorphologySuccessorV3PrivateOwnerR5Error(
                "R5 owner file kind or split drifted")
        for name in (
                "record_count", "content_size_bytes", "transport_size_bytes"):
            require_positive(getattr(self, name), where=f"R5 owner {name}")
        require_sha256(self.content_sha256, where="R5 owner content SHA")
        require_sha256(self.transport_sha256, where="R5 owner transport SHA")
        for name in ("first_record_key", "last_record_key"):
            key = getattr(self, name)
            if (not isinstance(key, tuple) or not key
                    or any(type(item) is not int or item <= 0 for item in key)):
                raise W02MorphologySuccessorV3PrivateOwnerR5Error(
                    f"R5 owner {name} drifted")
        if self.first_record_key > self.last_record_key:
            raise W02MorphologySuccessorV3PrivateOwnerR5Error(
                "R5 owner file key range reversed")
        if self.license_ids != ("CC-BY-SA-4.0",):
            raise W02MorphologySuccessorV3PrivateOwnerR5Error(
                "R5 owner license identity drifted")

    @classmethod
    def from_dict(
            cls, value: object,
            ) -> "W02MorphologySuccessorV3PrivateR5FileIdentity":
        raw = require_exact_dict(value, {
            "content_sha256", "content_size_bytes", "first_record_key",
            "last_record_key", "layout_key", "license_ids", "record_count",
            "record_kind", "root_key", "split", "transport_sha256",
            "transport_size_bytes",
        }, where="R5 owner file identity")
        if (not isinstance(raw["first_record_key"], list)
                or not isinstance(raw["last_record_key"], list)
                or not isinstance(raw["license_ids"], list)):
            raise W02MorphologySuccessorV3PrivateOwnerR5Error(
                "R5 owner file array fields drifted")
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


def validate_r5_private_file_inventory(
        value: object,
        ) -> tuple[W02MorphologySuccessorV3PrivateR5FileIdentity, ...]:
    if not isinstance(value, list) or len(value) != len(
            W02_MORPH_V3_PRIVATE_LAYOUTS):
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 owner file inventory is incomplete")
    identities = []
    for row in value:
        if (not isinstance(row, dict)
                or row.get("relative_path")
                != W02_MORPH_V3_PRIVATE_PATHS.get(row.get("layout_key"))):
            raise W02MorphologySuccessorV3PrivateOwnerR5Error(
                "R5 owner receipt relative path drifted")
        identities.append(W02MorphologySuccessorV3PrivateR5FileIdentity.from_dict({
            key: item for key, item in row.items() if key != "relative_path"
        }))
    files = tuple(identities)
    if tuple(row.layout_key for row in files) != W02_MORPH_V3_PRIVATE_LAYOUTS:
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 owner layout order drifted")
    by_layout = {row.layout_key: row for row in files}
    if by_layout["PRIVATE_SOURCE"].record_count != W02_MORPH_V3_PRIVATE_SOURCE_COUNT:
        raise W02MorphologySuccessorV3PrivateOwnerR5Error(
            "R5 owner source count drifted")
    for split in W02_MORPH_V3_PRIVATE_SPLITS:
        name = split.upper()
        observation = by_layout[f"PRIVATE_{name}_OBSERVATION"]
        label = by_layout[f"PRIVATE_{name}_LABEL"]
        if (observation.record_count != label.record_count
                or observation.record_count
                != W02_MORPH_V3_PRIVATE_SPLIT_COUNTS[split]):
            raise W02MorphologySuccessorV3PrivateOwnerR5Error(
                "R5 owner observation/label inventory drifted")
    return files


__all__ = [
    "W02_MORPH_V3_PRIVATE_DIMENSION_COUNTS",
    "W02_MORPH_V3_PRIVATE_LAYOUTS",
    "W02_MORPH_V3_PRIVATE_PAIR_COUNT",
    "W02_MORPH_V3_PRIVATE_PATHS",
    "W02_MORPH_V3_PRIVATE_R5_ADAPTER_CODE_SHA256",
    "W02_MORPH_V3_PRIVATE_R5_DIMENSION_BINDINGS",
    "W02_MORPH_V3_PRIVATE_R5_LABEL_BINDING_SHA256",
    "W02_MORPH_V3_PRIVATE_R5_METADATA_SHA256",
    "W02_MORPH_V3_PRIVATE_R5_METADATA_SIZE_BYTES",
    "W02_MORPH_V3_PRIVATE_R5_ORDINAL_SLICE_COMMITMENT",
    "W02_MORPH_V3_PRIVATE_R5_OWNER_FAMILY_KEY",
    "W02_MORPH_V3_PRIVATE_R5_OWNER_ID",
    "W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_PATH",
    "W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_SHA256",
    "W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_SIZE_BYTES",
    "W02_MORPH_V3_PRIVATE_R5_OWNER_RECEIPT_VERSION",
    "W02_MORPH_V3_PRIVATE_R5_PROBE_CODE_SHA256",
    "W02_MORPH_V3_PRIVATE_R5_PROBE_REPORT_SHA256",
    "W02_MORPH_V3_PRIVATE_R5_PUBLIC_BASE_COMMIT",
    "W02_MORPH_V3_PRIVATE_R5_SOURCE_EXTENSION_CODE_SHA256",
    "W02_MORPH_V3_PRIVATE_R5_SOURCE_EXTENSION_MANIFEST_SHA256",
    "W02_MORPH_V3_PRIVATE_R5_SOURCE_KEY",
    "W02_MORPH_V3_PRIVATE_SOURCE_COUNT",
    "W02_MORPH_V3_PRIVATE_SPLIT_COUNTS",
    "W02_MORPH_V3_PRIVATE_SPLITS",
    "W02MorphologySuccessorV3PrivateOwnerR5Error",
    "W02MorphologySuccessorV3PrivateR5FileIdentity",
    "require_exact_dict",
    "require_positive",
    "require_sha256",
    "validate_r5_private_file_inventory",
]
