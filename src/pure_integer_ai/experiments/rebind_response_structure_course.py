"""Rebind a frozen integer response-structure course to a new parent graph.

The observation payload remains byte-for-byte unchanged.  Only the parent
database and parent-ledger commitments are replaced after the target graph
has been checked by the existing response materializer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    TrainedRelationGraphRuntime,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("ascii")


def _role_mapping(dynamic: tuple, candidate) -> tuple[int, ...] | None:
    """按完整整数角色载体求双射，不解释表层词义。"""
    if len(candidate.bindings) != len(dynamic):
        return None
    available = list(range(len(candidate.bindings)))
    mapping = []
    for part in dynamic:
        matches = [
            index for index in available
            if tuple(map(ord, candidate.bindings[index].surface))
            == tuple(part[-1])
        ]
        if len(matches) != 1:
            return None
        mapping.append(matches[0])
        available.remove(matches[0])
    return tuple(mapping) if not available else None


def _inherited_frame_targets(
        original_parent: Path, current_parent: Path, payload: list,
        ) -> tuple[str, ...]:
    """经原父图唯一命题身份把旧 frame 迁到当前 active 图。"""
    with TrainedRelationGraphRuntime(original_parent) as original_runtime:
        original_facts = original_runtime.active_surface_facts()
    with TrainedRelationGraphRuntime(current_parent) as current_runtime:
        current_facts = current_runtime.active_surface_facts()
    current_by_key = {
        fact.proposition.stable_key(): index
        for index, fact in enumerate(current_facts, 1)
    }
    targets = []
    frame_ids = sorted({int(item[1]) for item in payload[2]})
    for frame_id in frame_ids:
        observations = tuple(
            item for item in payload[2] if int(item[1]) == frame_id)
        if frame_id <= 0 or frame_id > len(original_facts):
            raise ValueError("旧 frame ordinal 越出已锁定原父图")
        # ordinal 只在 ledger 锁定 SHA 的原父图内解释；迁移到新图时只传递
        # 完整 proposition stable key，绝不假定两个图的物理顺序相同。
        original_fact = original_facts[frame_id - 1]
        if not all(_role_mapping(tuple(
                part for part in observation[3]
                if isinstance(part, list) and part and part[0] == 2),
                original_fact) is not None for observation in observations):
            raise ValueError("旧 frame ordinal 与冻结角色载体不一致")
        proposition_key = original_fact.proposition.stable_key()
        current_index = current_by_key.get(proposition_key)
        if current_index is None:
            raise ValueError("原父图命题身份未进入当前 active 图")
        targets.append(f"{frame_id}:{current_index}")
    return tuple(targets)


def _explicit_frame_map(parent_database: Path, payload: list,
                        targets: tuple[str, ...]) -> list[list[object]]:
    """Freeze audited frame choices as old ordinal -> proposition integer key."""
    if not targets:
        return []
    requested: dict[int, int] = {}
    for value in targets:
        try:
            old_text, target_text = value.split(":", 1)
            old_frame, target_fact = int(old_text), int(target_text)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("frame target must use OLD_FRAME:TARGET_FACT") from exc
        if old_frame <= 0 or target_fact <= 0 or old_frame in requested:
            raise ValueError("frame target coordinates must be unique positive integers")
        requested[old_frame] = target_fact
    observed = {item[1] for item in payload[2]}
    if set(requested) != observed:
        raise ValueError("explicit frame targets must cover every observed parent frame")
    with TrainedRelationGraphRuntime(parent_database) as runtime:
        facts = runtime.active_surface_facts()
    result = []
    for old_frame, target_fact in sorted(requested.items()):
        if target_fact > len(facts):
            raise ValueError("explicit target fact is outside the active parent graph")
        fact = facts[target_fact - 1]
        expected: dict[int, tuple[int, ...]] = {}
        for observation in payload[2]:
            if observation[1] != old_frame:
                continue
            for part in observation[3]:
                if part[0] != 2:
                    continue
                units = tuple(part[-1])
                previous = expected.setdefault(part[1], units)
                if previous != units:
                    raise ValueError("one old role coordinate carries conflicting integers")
        actual = [tuple(map(ord, binding.surface)) for binding in fact.bindings]
        if (set(expected) != set(range(len(actual)))
                or sorted(expected.values()) != sorted(actual)):
            raise ValueError("explicit target fact does not preserve the role-carrier bijection")
        result.append([old_frame, list(fact.proposition.stable_key())])
    return result


def rebind(*, observations: Path, original_source_manifest: Path,
           parent_database: Path, output_root: Path,
           frame_targets: tuple[str, ...] = (),
           original_parent_database: Path | None = None) -> dict[str, object]:
    """Create a new licensed ledger pair without changing integer observations."""
    observations = observations.resolve(strict=True)
    original_source_manifest = original_source_manifest.resolve(strict=True)
    parent_database = parent_database.resolve(strict=True)
    output_root = output_root.resolve()
    if parent_database.drive.upper() != "K:" or output_root.drive.upper() != "K:":
        raise ValueError("rebinding must use an explicit K drive target")
    if output_root.exists():
        raise ValueError("rebind output already exists")
    source_payload = observations.read_bytes()
    source = json.loads(original_source_manifest.read_text(encoding="utf-8-sig"))
    if source.get("format") != "RESPONSE_STRUCTURE_SOURCE_LEDGER_V1":
        raise ValueError("unexpected response source ledger format")
    source_sha = hashlib.sha256(source_payload).hexdigest()
    if source.get("source_sha256") != source_sha:
        raise ValueError("observation payload does not match its ledger")
    payload = json.loads(source_payload.decode("utf-8"))
    if (not isinstance(payload, list) or len(payload) != 3
            or payload[:2] != [91525, 1]
            or not isinstance(payload[2], list) or not payload[2]):
        raise ValueError("response observations are not the frozen integer protocol")
    if original_parent_database is not None:
        if frame_targets:
            raise ValueError(
                "original parent database 与显式 frame target 不能并用")
        original_parent_database = original_parent_database.resolve(strict=True)
        if original_parent_database.drive.upper() != "K:":
            raise ValueError("原父图必须位于 K 盘")
        if _sha256(original_parent_database) != source.get(
                "parent_database_sha256"):
            raise ValueError("原父图 SHA 未被冻结课程 ledger 锁定")
        frame_targets = _inherited_frame_targets(
            original_parent_database, parent_database, payload)
    database_sha = _sha256(parent_database)
    frame_map = _explicit_frame_map(
        parent_database, payload, tuple(frame_targets))
    output_root.mkdir(parents=True)
    parent_ledger = {
        "format": "PURE_INTEGER_TRAINED_GRAPH_PARENT_LEDGER_V1",
        "schema_version": 1,
        "source_kind": "broad-structure-response-parent",
        "source_id": parent_database.parent.name,
        "source_version": 1,
        "database_sha256": database_sha,
        "license_ids": ["MIT"],
        "attribution": "Project-owned pure integer graph materialization metadata.",
        "official_url": "https://opensource.org/license/mit",
        "rights_basis": "Project-owned graph metadata; no source prose copied.",
        "selection_rule": "Current broad-structure active relation graph only.",
    }
    parent_bytes = _canonical(parent_ledger)
    parent_path = output_root / "parent_source_manifest.json"
    parent_path.write_bytes(parent_bytes)
    ledger = dict(source)
    ledger.update({
        "source_id": f"{source.get('source_id', 'response-structure')}-rebound-{parent_database.parent.name}",
        "parent_database_sha256": database_sha,
        "parent_source_manifest_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "selection_rule": "Frozen integer observations rebound only when every dynamic role carrier has a unique exact integer-codepoint bijection to one current active relation frame; parent ordinals are not reused, and no heldout or private data is included.",
        "parent_frame_map": frame_map,
        "free_dialogue_claim": 0,
    })
    source_path = output_root / "source_manifest.json"
    source_path.write_bytes(_canonical(ledger))
    (output_root / "observations.int.json").write_bytes(source_payload)
    receipt = {
        "schema_version": 1,
        "observation_count": len(payload[2]),
        "observation_sha256": source_sha,
        "parent_database_sha256": database_sha,
        "parent_source_manifest_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "parent_frame_map_count": len(frame_map),
        "stable_parent_identity_rebind": int(
            original_parent_database is not None),
        "free_dialogue_complete": 0,
    }
    (output_root / "rebind_receipt.json").write_bytes(_canonical(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--original-source-manifest", required=True, type=Path)
    parser.add_argument("--parent-database", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--original-parent-database", type=Path,
        help="从原父图恢复每个旧 frame 的唯一 proposition stable key")
    parser.add_argument(
        "--frame-target", dest="frame_targets", action="append", default=[],
        help="audited OLD_FRAME:TARGET_ACTIVE_FACT mapping; repeat for each old frame",
    )
    result = rebind(**vars(parser.parse_args()))
    print(json.dumps(result, ensure_ascii=True, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["rebind"]
