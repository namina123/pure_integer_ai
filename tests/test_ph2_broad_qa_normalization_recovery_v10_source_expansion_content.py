"""覆盖 recovery-v10 新 TRAIN 来源内容与重叠可行性。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_content import (
    publish_normalization_recovery_v10_source_expansion_content,
    read_normalization_recovery_v10_source_expansion_content,
    read_normalization_recovery_v10_source_expansion_content_aggregate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_content_records import (
    derive_normalization_recovery_v10_source_expansion_content,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def _pair(source: str, hant: str, hans: str) -> dict[str, object]:
    """构造一条只供 aggregate 纯函数使用的 synthetic pair。"""
    return {
        "official_source_text": source,
        "v8_training_eligible": 1,
        "zh_hans": {"translation": hans},
        "zh_hant": {"translation": hant},
    }


def _state():
    """构造两家新来源和三家 predecessor 的最小可行状态。"""
    predecessor = {
        "family_counts": {
            "KEEPASSXC_PROJECT": 1,
            "QBITTORRENT_PROJECT": 1,
            "STELLARIUM_PROJECT": 1,
        },
        "mapping_families": {
            ("Open", "開啟", "打开"): {"QBITTORRENT_PROJECT"},
        },
        "observation_count": 3,
        "source_families": {
            "Open": {"QBITTORRENT_PROJECT"},
        },
        "source_input_families": {
            ("Open", "開啟"): {"QBITTORRENT_PROJECT"},
        },
    }
    summary = {
        "content_outcome": "PASS_NONZERO_ACTIVE_COMMON_PAIR",
        "plain_pair_count": 2,
        "v8_training_eligible_pair_count": 2,
    }
    parsed = {
        "MIXXX_PROJECT": {
            "pairs": (
                _pair("Open", "開啟", "打开"),
                _pair("Close", "關閉", "关闭"),
            ),
            "roster": {
                "license": {"expression": "GPL-2.0-or-later"},
                "locale_files": [{}, {}],
            },
            "summary": summary,
        },
        "MUMBLE_PROJECT": {
            "pairs": (
                _pair("Open", "開啟", "打开"),
                _pair("Save", "儲存", "保存"),
            ),
            "roster": {
                "license": {"expression": "BSD-3-Clause"},
                "locale_files": [{}, {}],
            },
            "summary": summary,
        },
    }
    return parsed, predecessor


def test_v10_source_expansion_content_counts_real_overlap_dimensions() -> None:
    """source、source+input 与完整 mapping 分账且两家均通过硬门。"""
    outputs, census = derive_normalization_recovery_v10_source_expansion_content(
        *_state())
    records = {
        str(item["source_family"]): item
        for item in outputs["source-content.jsonl"]
    }
    assert records["MIXXX_PROJECT"][
        "predecessor_source_input_overlap_pair_count"] == 1
    assert records["MUMBLE_PROJECT"][
        "predecessor_exact_mapping_overlap_pair_count"] == 1
    assert all(item["selection_outcome"]
               == "PASS_CONTENT_AND_PREDECESSOR_SOURCE_INPUT_OVERLAP"
               for item in records.values())
    cross = outputs["source-cross-overlap.jsonl"][0]
    assert cross["source_intersection_count"] == 1
    assert cross["source_input_intersection_count"] == 1
    assert cross["exact_mapping_intersection_count"] == 1
    assert census["selected_content_pass_count"] == 2
    assert census["pair_surface_published"] == 0


def test_v10_source_expansion_content_round_trip_and_tamper(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """aggregate publisher 不覆盖，strict reader 拒绝额外文件与篡改。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_content as module

    roots = [tmp_path / name for name in (
        "roster", "predecessor", "mixxx", "mumble")]
    for root in roots:
        root.mkdir()
    target = tmp_path / "artifact"
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(module, "_state", lambda **kwargs: _state())
    published = publish_normalization_recovery_v10_source_expansion_content(
        run_root=tmp_path,
        roster_dir=roots[0],
        predecessor_observation_dir=roots[1],
        mixxx_source_root=roots[2],
        mumble_source_root=roots[3],
        target_dir=target,
    )
    reread, outputs = read_normalization_recovery_v10_source_expansion_content(
        target,
        roster_dir=roots[0],
        predecessor_observation_dir=roots[1],
        mixxx_source_root=roots[2],
        mumble_source_root=roots[3],
        expected_manifest_sha256=published["manifest_sha256"],
    )
    assert reread == published
    assert len(outputs["source-content.jsonl"]) == 2
    aggregate, aggregate_outputs = (
        read_normalization_recovery_v10_source_expansion_content_aggregate(
            target,
            expected_manifest_sha256=published["manifest_sha256"],
        ))
    assert aggregate == published
    assert aggregate_outputs == outputs
    with pytest.raises(BroadQaExternalDataError, match="input/target"):
        publish_normalization_recovery_v10_source_expansion_content(
            run_root=tmp_path,
            roster_dir=roots[0],
            predecessor_observation_dir=roots[1],
            mixxx_source_root=roots[2],
            mumble_source_root=roots[3],
            target_dir=target,
        )

    content_path = target / "source-content.jsonl"
    lines = content_path.read_bytes().splitlines(keepends=True)
    changed = json.loads(lines[0])
    changed["transient_pair_count"] = 99
    lines[0] = canonical_json_line(changed)
    content_path.write_bytes(b"".join(lines))
    manifest_path = target / "manifest.json"
    stored = json.loads(manifest_path.read_bytes())
    payload = content_path.read_bytes()
    stored["files"][0]["bytes"] = len(payload)
    stored["files"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_bytes(canonical_json_line(stored))
    forged_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(BroadQaExternalDataError, match="records"):
        read_normalization_recovery_v10_source_expansion_content(
            target,
            roster_dir=roots[0],
            predecessor_observation_dir=roots[1],
            mixxx_source_root=roots[2],
            mumble_source_root=roots[3],
            expected_manifest_sha256=forged_sha,
        )


def test_v10_source_expansion_content_rejects_non_k_run_root(
        tmp_path: Path) -> None:
    """正式 publisher 不得把内容 artifact 写回 D 或临时盘。"""
    with pytest.raises(BroadQaExternalDataError, match="K 盘"):
        publish_normalization_recovery_v10_source_expansion_content(
            run_root=tmp_path,
            roster_dir=tmp_path,
            predecessor_observation_dir=tmp_path,
            mixxx_source_root=tmp_path,
            mumble_source_root=tmp_path,
            target_dir=tmp_path / "artifact",
        )
