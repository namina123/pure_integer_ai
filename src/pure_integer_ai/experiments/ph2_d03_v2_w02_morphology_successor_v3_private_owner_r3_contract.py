"""Pure identities for the successor V3 R3 private owner receipt."""
from __future__ import annotations

from typing import Any

from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_contract import (
    W02_MORPH_V3_PRIVATE_DIMENSION_COUNTS,
    W02_MORPH_V3_PRIVATE_LAYOUTS,
    W02_MORPH_V3_PRIVATE_PAIR_COUNT,
    W02_MORPH_V3_PRIVATE_PATHS,
    W02_MORPH_V3_PRIVATE_SOURCE_COUNT,
    W02_MORPH_V3_PRIVATE_SPLIT_COUNTS,
    W02MorphologySuccessorV3PrivateFileIdentity,
    W02MorphologySuccessorV3PrivateOwnerError as _V3OwnerError,
    require_exact_dict as _require_exact_dict,
    require_sha256 as _require_sha256,
    validate_v3_private_file_inventory as _validate_v3_private_file_inventory,
)


W02_MORPH_V3_PRIVATE_R3_OWNER_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v3_private_owner_r3_receipt_v1.json"
)
W02_MORPH_V3_PRIVATE_R3_OWNER_RECEIPT_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V3-PRIVATE-OWNER-R3-RECEIPT-V1"
)
W02_MORPH_V3_PRIVATE_R3_METADATA_SHA256 = (
    "e6661e0f73ce5716e94df48d753eeee5ffae9ebaa6197beb70b269dc0d783b70"
)
W02_MORPH_V3_PRIVATE_R3_METADATA_SIZE_BYTES = 10_151
W02_MORPH_V3_PRIVATE_R3_OWNER_ID = "dd2d3c7b4a2d3cd4"
W02_MORPH_V3_PRIVATE_R3_OWNER_FAMILY_KEY = (
    "PH2-D03-V2-W02-SUCCESSOR-V3-R3-UD-ZH-PUD-R2-18-"
    "WIKIPEDIA-BLIND-PRIVATE-dd2d3c7b4a2d3cd4"
)
W02_MORPH_V3_PRIVATE_R3_PUBLIC_BASE_COMMIT = (
    "de76d6c9a941aeefadd191ecf519f54bdcb9b842"
)
W02_MORPH_V3_PRIVATE_R3_SOURCE_KEY = (
    "UD_ZH_PUD_R2_18_WIKIPEDIA_BLIND_PRIVATE"
)
W02_MORPH_V3_PRIVATE_R3_SOURCE_EXTENSION_CODE_SHA256 = (
    "c2e2d4e4542d26a42e61d465c7b4b350e2873ae3121a12decaa7181e541befbe"
)
W02_MORPH_V3_PRIVATE_R3_SOURCE_EXTENSION_MANIFEST_SHA256 = (
    "ea3d48e518e6ca9e3c7cc90a4d391ad4304ae19a66d28c28ccab87bc613165a8"
)
W02_MORPH_V3_PRIVATE_R3_SOURCE_SNAPSHOT_COMMITMENT = (
    "9e03c718001d0c34014e9f54550e033f94c7b969e7f5609a38cee3399a7a1b91"
)
W02_MORPH_V3_PRIVATE_R3_LABEL_BINDING_VERSION = (
    "PH2-D03-V2-W02-LABEL-SEMANTIC-BINDING-V1"
)
W02_MORPH_V3_PRIVATE_R3_LABEL_BINDING_SHA256 = (
    "e5751c145cb22da3cd5a9e978e2c1573b6dab68d69e148038e463b006216b0bb"
)
W02_MORPH_V3_PRIVATE_R3_LABEL_PASS_SHA256 = (
    "88c3160061120450aa6cd839a2c3b48e7d86a10cd4150146bb08b3bf1d3fa578"
)
W02_MORPH_V3_PRIVATE_R3_DIMENSION_BINDINGS = (
    (
        "W-02-V2-BOUNDARY-WITHDRAWAL",
        (1, 100_494_257_788_431_926),
        100,
    ),
    (
        "W-02-V2-MULTI-CANDIDATE",
        (1, 7_231_649_387_489_044_918),
        100,
    ),
    (
        "W-02-V2-NEW-CONTENT-MORPHOLOGY",
        (1, 1_480_950_902_982_636_330),
        100,
    ),
    (
        "W-02-V2-OOV",
        (1, 1_424_821_230_297_126_309),
        100,
    ),
    (
        "W-02-V2-GENERATION-HARD-CONJUNCT",
        (1, 5_646_924_068_434_684_793),
        100,
    ),
)


# object-model: exception
class W02MorphologySuccessorV3PrivateOwnerR3Error(RuntimeError):
    """The payload-free R3 owner receipt or label binding drifted."""


def require_exact_dict(
        value: object,
        fields: set[str],
        *,
        where: str,
        ) -> dict[str, Any]:
    """Retain the shared strict shape check under the R3 error boundary."""
    try:
        return _require_exact_dict(value, fields, where=where)
    except _V3OwnerError as exc:
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(str(exc)) from exc


def require_sha256(value: object, *, where: str) -> str:
    """Retain the shared SHA check under the R3 error boundary."""
    try:
        return _require_sha256(value, where=where)
    except _V3OwnerError as exc:
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(str(exc)) from exc


def validate_v3_private_file_inventory(
        value: object,
        ) -> tuple[W02MorphologySuccessorV3PrivateFileIdentity, ...]:
    """Retain the shared seven-file check under the R3 error boundary."""
    try:
        return _validate_v3_private_file_inventory(value)
    except _V3OwnerError as exc:
        raise W02MorphologySuccessorV3PrivateOwnerR3Error(str(exc)) from exc


__all__ = [
    "W02_MORPH_V3_PRIVATE_DIMENSION_COUNTS",
    "W02_MORPH_V3_PRIVATE_LAYOUTS",
    "W02_MORPH_V3_PRIVATE_PAIR_COUNT",
    "W02_MORPH_V3_PRIVATE_PATHS",
    "W02_MORPH_V3_PRIVATE_R3_DIMENSION_BINDINGS",
    "W02_MORPH_V3_PRIVATE_R3_LABEL_BINDING_SHA256",
    "W02_MORPH_V3_PRIVATE_R3_LABEL_BINDING_VERSION",
    "W02_MORPH_V3_PRIVATE_R3_LABEL_PASS_SHA256",
    "W02_MORPH_V3_PRIVATE_R3_METADATA_SHA256",
    "W02_MORPH_V3_PRIVATE_R3_METADATA_SIZE_BYTES",
    "W02_MORPH_V3_PRIVATE_R3_OWNER_FAMILY_KEY",
    "W02_MORPH_V3_PRIVATE_R3_OWNER_ID",
    "W02_MORPH_V3_PRIVATE_R3_OWNER_RECEIPT_PATH",
    "W02_MORPH_V3_PRIVATE_R3_OWNER_RECEIPT_VERSION",
    "W02_MORPH_V3_PRIVATE_R3_PUBLIC_BASE_COMMIT",
    "W02_MORPH_V3_PRIVATE_R3_SOURCE_EXTENSION_CODE_SHA256",
    "W02_MORPH_V3_PRIVATE_R3_SOURCE_EXTENSION_MANIFEST_SHA256",
    "W02_MORPH_V3_PRIVATE_R3_SOURCE_KEY",
    "W02_MORPH_V3_PRIVATE_R3_SOURCE_SNAPSHOT_COMMITMENT",
    "W02_MORPH_V3_PRIVATE_SOURCE_COUNT",
    "W02_MORPH_V3_PRIVATE_SPLIT_COUNTS",
    "W02MorphologySuccessorV3PrivateFileIdentity",
    "W02MorphologySuccessorV3PrivateOwnerR3Error",
    "require_exact_dict",
    "require_sha256",
    "validate_v3_private_file_inventory",
]
