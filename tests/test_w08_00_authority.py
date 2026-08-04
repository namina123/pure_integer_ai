from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w08_authority import (
    W08_ABLATION_KEYS,
    W08_AUTHORITY_RELATIVE_PATH,
    W08_DIMENSION_KEYS,
    W08_FUTURE_PACK_KEYS,
    W08_SUBTASK_ORDER,
    W08AuthorityError,
    build_w08_authority,
    canonical_w08_authority_bytes,
    read_w08_authority,
    validate_w08_authority,
)


ROOT = Path(__file__).resolve().parents[1]


def test_w08_authority_recomputes_frozen_public_boundary() -> None:
    value = build_w08_authority(ROOT)
    assert tuple(value["dimension_keys"]) == W08_DIMENSION_KEYS
    assert tuple(value["ablation_keys"]) == W08_ABLATION_KEYS
    assert tuple(value["subtask_order"]) == W08_SUBTASK_ORDER
    assert tuple(value["stage_inventory"]["future_pack_keys"]) == W08_FUTURE_PACK_KEYS
    assert value["p3ia_boundary"]["future_payload_reads"] == 0
    assert value["generation_account"]["aggregate_with_source_replay"] == 0
    assert value["execution_state"]["W08_STARTED"] == 0
    assert value["execution_state"]["formal_w08_training_runs"] == 0
    assert len(value["retention_identities"]) == 6
    assert len(value["visible_pack_identities"]) == 7
    assert value["baseline_public_head_commit_sha1"] == (
        "bbf610b7e05c66f5d2930cdeb3d66bc26e822010"
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["dimension_keys"].reverse(),
        lambda value: value["ablation_keys"].reverse(),
        lambda value: value["subtask_order"].reverse(),
        lambda value: value["stage_inventory"]["train_pack_keys"].append(
            W08_FUTURE_PACK_KEYS[0]
        ),
        lambda value: value["generation_account"].__setitem__(
            "aggregate_with_source_replay", 1
        ),
        lambda value: value["p3ia_boundary"].__setitem__(
            "independent_course_stage", "W-08"
        ),
        lambda value: value["parent_identities"][0].__setitem__(
            "relative_path", "/outside/manifest.json"
        ),
        lambda value: value.__setitem__("unexpected", 1),
    ],
)
def test_w08_authority_fails_closed_on_boundary_drift(mutator) -> None:
    value = deepcopy(build_w08_authority(ROOT))
    mutator(value)
    with pytest.raises(W08AuthorityError):
        validate_w08_authority(value)


def test_w08_authority_declares_no_label_path_for_candidate() -> None:
    value = build_w08_authority(ROOT)
    encoded = canonical_w08_authority_bytes(value).decode("utf-8")
    assert "evaluator_label_paths" not in encoded
    assert "held_out.labels" not in encoded
    assert "ph2_p3ia_dataset_artifacts" not in encoded


def test_w08_authority_public_artifact_is_canonical_and_live() -> None:
    value = read_w08_authority(ROOT)
    path = ROOT / W08_AUTHORITY_RELATIVE_PATH
    payload = path.read_bytes()
    assert payload == canonical_w08_authority_bytes(value)
    assert hashlib.sha256(payload).hexdigest() == (
        "1236c34b3076bee29d16508361d8405f80b0382a1403bc54acac0b2c4a15688a"
    )
    assert json.loads(payload) == value
