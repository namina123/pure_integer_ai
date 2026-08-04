"""基于单一共享篇章与生成投影执行 LC-16 W08 资格验证。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_lc16_overlay_contract import (
    read_d03_lc16_successor_overlay,
)
from pure_integer_ai.experiments.ph2_language_coverage_v2_contract import DIRECTIONS
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_CARRIER_KEYS,
    W08_RESOURCE_BUDGET,
)
from pure_integer_ai.experiments.ph2_w08_open_generation_contract import (
    W08OpenGenerationAuditReceipt,
)
from pure_integer_ai.experiments.ph2_w08_p3ia_contract import W08P3IaAuditReceipt


W08_LC16_SCOPE_KEY = "DISCOURSE_REFERENCE_GENERATION"
W08_LC16_EVALUATOR_KEY = "PH2-W08-LC16-INDEPENDENT-EVALUATOR-V1"
W08_LC16_CELL_STATES = (
    "LANGUAGE_UNKNOWN",
    "NE",
    "PASS",
    "SCHEMA_REQUIRED",
    "UNREPRESENTABLE",
)
W08_LC16_OVERLAY_PATH = "data/ph2/manifests/d03_lc16_successor_overlay_v1.json"


class W08LC16Error(ValueError):
    """LC-16 W08 投影或载体资格发生漂移。"""


def _key(value: object, *, where: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value or any(
        type(item) is not int for item in value
    ):
        raise W08LC16Error(f"{where} is not a strict integer key")
    return value


@dataclass(frozen=True, order=True)
class W08LC16CarrierProjection:
    carrier_key: str
    raw_carrier_sha256: str
    structure_identity_key: tuple[int, ...]
    region_sequence_identity_key: tuple[int, ...]
    parser_package: str
    parser_version: str
    renderer_key: str
    renderer_version: str
    reference_embed_identity_key: tuple[int, ...]
    revision_identity_key: tuple[int, ...]
    semantic_engine_key: tuple[int, ...]
    discourse_projection_key: tuple[int, ...]
    logic_projection_key: tuple[int, ...]
    generation_projection_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.carrier_key not in W08_CARRIER_KEYS:
            raise W08LC16Error("LC-16 carrier is not registered")
        if (
            not isinstance(self.raw_carrier_sha256, str)
            or len(self.raw_carrier_sha256) != 64
        ):
            raise W08LC16Error("LC-16 raw carrier identity is invalid")
        for name in (
            "structure_identity_key",
            "region_sequence_identity_key",
            "reference_embed_identity_key",
            "revision_identity_key",
            "semantic_engine_key",
            "discourse_projection_key",
            "logic_projection_key",
            "generation_projection_key",
        ):
            _key(getattr(self, name), where=f"LC-16 {name}")
        for name in (
            "parser_package",
            "parser_version",
            "renderer_key",
            "renderer_version",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise W08LC16Error(f"LC-16 {name} is empty")

    def stable_key(self) -> tuple[int, ...]:
        return digest_value(
            {
                "carrier": self.carrier_key,
                "raw": self.raw_carrier_sha256,
                "structure": list(self.structure_identity_key),
                "region_sequence": list(self.region_sequence_identity_key),
                "parser": [self.parser_package, self.parser_version],
                "renderer": [self.renderer_key, self.renderer_version],
                "reference_embed": list(self.reference_embed_identity_key),
                "revision": list(self.revision_identity_key),
                "engine": list(self.semantic_engine_key),
                "discourse": list(self.discourse_projection_key),
                "logic": list(self.logic_projection_key),
                "generation": list(self.generation_projection_key),
            }
        )


@dataclass(frozen=True)
class W08LC16ProjectionInventory:
    overlay_sha256: str
    scope_key: str
    evaluator_key: str
    carriers: tuple[W08LC16CarrierProjection, ...]
    shared_semantic_engine_key: tuple[int, ...]
    sample_payload_read_count: int = 0
    host_learning_write_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.overlay_sha256, str) or len(self.overlay_sha256) != 64:
            raise W08LC16Error("LC-16 overlay identity is invalid")
        if (
            self.scope_key != W08_LC16_SCOPE_KEY
            or self.evaluator_key != W08_LC16_EVALUATOR_KEY
        ):
            raise W08LC16Error("LC-16 W08 scope identity drifted")
        if tuple(item.carrier_key for item in self.carriers) != W08_CARRIER_KEYS:
            raise W08LC16Error("LC-16 nine-carrier inventory drifted")
        _key(self.shared_semantic_engine_key, where="LC-16 shared semantic engine")
        if any(
            item.semantic_engine_key != self.shared_semantic_engine_key
            for item in self.carriers
        ):
            raise W08LC16Error("LC-16 carrier-specific semantic engine was introduced")
        if self.sample_payload_read_count != 0 or self.host_learning_write_count != 0:
            raise W08LC16Error("LC-16 projection read payload or wrote host state")

    def stable_key(self) -> tuple[int, ...]:
        return digest_value(
            {
                "overlay": self.overlay_sha256,
                "scope": self.scope_key,
                "evaluator": self.evaluator_key,
                "carriers": [list(item.stable_key()) for item in self.carriers],
                "engine": list(self.shared_semantic_engine_key),
            }
        )


@dataclass(frozen=True, order=True)
class W08LC16QualificationCell:
    carrier_key: str
    direction: str
    scope_key: str
    state: str
    projection_key: tuple[int, ...]
    use_key: tuple[int, ...]
    outcome_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.carrier_key not in W08_CARRIER_KEYS:
            raise W08LC16Error("LC-16 cell carrier is invalid")
        if self.direction not in DIRECTIONS:
            raise W08LC16Error("LC-16 cell direction is invalid")
        if self.scope_key != W08_LC16_SCOPE_KEY:
            raise W08LC16Error("LC-16 cell scope drifted")
        if self.state not in W08_LC16_CELL_STATES:
            raise W08LC16Error("LC-16 cell state is invalid")
        for name in ("projection_key", "use_key", "outcome_key"):
            _key(getattr(self, name), where=f"LC-16 cell {name}")


@dataclass(frozen=True)
class W08LC16QualificationReceipt:
    inventory_key: tuple[int, ...]
    cells: tuple[W08LC16QualificationCell, ...]
    state: str
    carrier_count: int
    direction_evaluations: int
    logic_operations: int
    private_label_read_count: int = 0
    sample_payload_read_count: int = 0
    host_learning_write_count: int = 0
    memory_learning_write_count: int = 0

    def __post_init__(self) -> None:
        _key(self.inventory_key, where="LC-16 qualification inventory")
        expected = tuple(
            (carrier, direction, W08_LC16_SCOPE_KEY)
            for carrier in W08_CARRIER_KEYS
            for direction in DIRECTIONS
        )
        actual = tuple(
            (item.carrier_key, item.direction, item.scope_key) for item in self.cells
        )
        if actual != expected:
            raise W08LC16Error("LC-16 carrier x U/R/G matrix drifted")
        if self.state not in {"PASS", "NE", "BLOCKED"}:
            raise W08LC16Error("LC-16 aggregate state is invalid")
        if self.state == "PASS" and any(item.state != "PASS" for item in self.cells):
            raise W08LC16Error("LC-16 PASS contains a non-PASS cell")
        if self.carrier_count != len(W08_CARRIER_KEYS):
            raise W08LC16Error("LC-16 carrier count drifted")
        if self.direction_evaluations != len(expected):
            raise W08LC16Error("LC-16 direction count drifted")
        if (
            self.direction_evaluations > W08_RESOURCE_BUDGET["max_records"]
            or self.logic_operations > W08_RESOURCE_BUDGET["max_logic_operations"]
        ):
            raise W08LC16Error("LC-16 resource budget was exceeded")
        if any(
            (
                self.private_label_read_count,
                self.sample_payload_read_count,
                self.host_learning_write_count,
                self.memory_learning_write_count,
            )
        ):
            raise W08LC16Error("LC-16 qualification crossed a forbidden boundary")

    def canonical_key(self) -> tuple[int, ...]:
        return digest_value(
            {
                "inventory": list(self.inventory_key),
                "state": self.state,
                "cells": [
                    [
                        item.carrier_key,
                        item.direction,
                        item.state,
                        list(item.projection_key),
                        list(item.outcome_key),
                    ]
                    for item in self.cells
                ],
            }
        )


def compile_w08_lc16_projection_inventory(
    repository_root: str | Path,
    *,
    semantic_engine_key: tuple[int, ...],
    discourse_projection_key: tuple[int, ...],
    logic_projection_key: tuple[int, ...],
    generation_projection_key: tuple[int, ...],
) -> W08LC16ProjectionInventory:
    """只读取 overlay 元数据，不打开任何载体 case payload。"""
    for name, value in (
        ("semantic engine", semantic_engine_key),
        ("discourse projection", discourse_projection_key),
        ("logic projection", logic_projection_key),
        ("generation projection", generation_projection_key),
    ):
        _key(value, where=f"LC-16 {name}")
    root = Path(repository_root).resolve()
    overlay = read_d03_lc16_successor_overlay(root / W08_LC16_OVERLAY_PATH)
    scope = tuple(item for item in overlay.scope_records if item.scope_key == W08_LC16_SCOPE_KEY)
    if len(scope) != 1 or scope[0].evaluator_key != W08_LC16_EVALUATOR_KEY:
        raise W08LC16Error("LC-16 W08 scope is missing from overlay")
    coverage = tuple(
        item for item in overlay.coverage_cells if item.scope_key == W08_LC16_SCOPE_KEY
    )
    if len(coverage) != len(W08_CARRIER_KEYS) * len(DIRECTIONS):
        raise W08LC16Error("LC-16 W08 overlay coverage is incomplete")
    carriers = []
    for course in overlay.carrier_courses:
        revision = tuple(item for item in course.cases if item.sample_kind == "REVISION")
        if len(revision) != 1:
            raise W08LC16Error("LC-16 carrier lacks one revision identity")
        carriers.append(
            W08LC16CarrierProjection(
                course.carrier_key,
                course.sample_identity.sha256,
                digest_value(
                    {
                        "manifest": course.manifest_identity.sha256,
                        "parser": [course.parser_package, course.parser_version],
                    }
                ),
                digest_value(
                    {
                        "carrier": course.carrier_key,
                        "ordered_cases": [list(item.case_key.components) for item in course.cases],
                        "materializations": [item.materialization_sha256 for item in course.cases],
                    }
                ),
                course.parser_package,
                course.parser_version,
                course.renderer_key,
                course.renderer_version,
                digest_value(
                    {
                        "carrier": course.carrier_key,
                        "reference_embed": [
                            list(item.owner_key.components) for item in course.cases
                        ],
                    }
                ),
                digest_value(
                    {
                        "carrier": course.carrier_key,
                        "revision_case": list(revision[0].case_key.components),
                        "revision_materialization": revision[0].materialization_sha256,
                    }
                ),
                semantic_engine_key,
                discourse_projection_key,
                logic_projection_key,
                generation_projection_key,
            )
        )
    return W08LC16ProjectionInventory(
        overlay.sha256(),
        W08_LC16_SCOPE_KEY,
        W08_LC16_EVALUATOR_KEY,
        tuple(carriers),
        semantic_engine_key,
    )


class W08LC16Qualifier:
    """通过全部载体身份投影同一份 W08 语义状态。"""

    @staticmethod
    def classify_boundary(stop_state: str) -> str:
        mapping = {
            "UNKNOWN": "LANGUAGE_UNKNOWN",
            "SCHEMA_REQUIRED": "SCHEMA_REQUIRED",
            "UNREPRESENTABLE": "UNREPRESENTABLE",
            "NE": "NE",
            "RESOLVED": "PASS",
        }
        try:
            return mapping[stop_state]
        except KeyError as error:
            raise W08LC16Error("LC-16 stop state is not classifiable") from error

    def qualify(
        self,
        inventory: W08LC16ProjectionInventory,
        *,
        p3ia: W08P3IaAuditReceipt,
        generation: W08OpenGenerationAuditReceipt,
    ) -> W08LC16QualificationReceipt:
        if not isinstance(inventory, W08LC16ProjectionInventory):
            raise TypeError("LC-16 qualifier inventory type is invalid")
        if not isinstance(p3ia, W08P3IaAuditReceipt) or p3ia.state != "RESOLVED":
            raise W08LC16Error("LC-16 qualifier requires resolved P3-Ia")
        if (
            not isinstance(generation, W08OpenGenerationAuditReceipt)
            or generation.state != "RESOLVED"
            or not generation.publication_units
        ):
            raise W08LC16Error("LC-16 qualifier requires resolved open generation")
        direction_projection = {
            "UNDERSTANDING": p3ia.uses[0].use_key,
            "REASONING": p3ia.uses[1].use_key,
            "GENERATION": generation.canonical_key(),
        }
        cells = tuple(
            W08LC16QualificationCell(
                carrier.carrier_key,
                direction,
                W08_LC16_SCOPE_KEY,
                "PASS",
                digest_value(
                    {
                        "carrier_projection": list(carrier.stable_key()),
                        "shared_direction_projection": list(direction_projection[direction]),
                    }
                ),
                digest_value(
                    {
                        "carrier": carrier.carrier_key,
                        "direction": direction,
                        "kind": "use",
                    }
                ),
                digest_value(
                    {
                        "carrier": carrier.carrier_key,
                        "direction": direction,
                        "kind": "outcome",
                    }
                ),
            )
            for carrier in inventory.carriers
            for direction in DIRECTIONS
        )
        return W08LC16QualificationReceipt(
            inventory.stable_key(),
            cells,
            "PASS",
            len(inventory.carriers),
            len(cells),
            len(cells) * 4,
        )


__all__ = [
    "W08LC16CarrierProjection",
    "W08LC16Error",
    "W08LC16ProjectionInventory",
    "W08LC16QualificationCell",
    "W08LC16QualificationReceipt",
    "W08LC16Qualifier",
    "W08_LC16_CELL_STATES",
    "W08_LC16_EVALUATOR_KEY",
    "W08_LC16_SCOPE_KEY",
    "compile_w08_lc16_projection_inventory",
]
