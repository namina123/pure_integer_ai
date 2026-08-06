"""固定 seed Hasher 热循环复用的位级与确定性工作量回归。"""
from __future__ import annotations

from scripts.object_model_lint import (
    DEFAULT_SOURCE_ROOT,
    scan_hot_loop_fixed_hasher_calls,
)
import pure_integer_ai.cognition.understanding.observe as observe_module
import pure_integer_ai.crosscut.determinism.hasher as hasher_module
import pure_integer_ai.experiments.arithmetic_structure_runtime as arithmetic_module
import pure_integer_ai.experiments.collection as collection_module
import pure_integer_ai.experiments.evaluation_runtime as evaluation_module
import pure_integer_ai.experiments.language_structure_runtime as language_module
import pure_integer_ai.experiments.round_runtime as round_module


def _hoisted_cases():
    return (
        (observe_module._OBSERVE_PROGRAM_HASHER, "observe.prog.v1", "x = 1"),
        (observe_module._OBSERVE_SEGMENT_HASHER, "observe.seg.v1", ("甲", "乙")),
        (arithmetic_module._DISC_ROOT_HASHER, "formal_train.disc_src", "1 + 2"),
        (collection_module._LOCAL_DIR_SOURCE_HASHER,
         "local_dir.source_file.v1", ("corpus", "part.txt")),
        (evaluation_module._XVER_B_HASHER, "xver.b.v1", "3 * 4"),
        (evaluation_module._DISC_LANG_HASHER,
         "formal_train.disc_lang", "甲\x1f乙"),
        (evaluation_module._DISC_LANG_ALIGN_HASHER,
         "formal_train.disc_lang_align", "1:2\x1f3:4"),
        (language_module._DISC_LANG_HASHER,
         "formal_train.disc_lang", "丙\x1f丁"),
        (language_module._DISC_LANG_ALIGN_HASHER,
         "formal_train.disc_lang_align", "5:6\x1f7:8"),
        (language_module._DISC_LANG_SENSE_HASHER,
         "formal_train.disc_lang_sense", "1:2:3:4"),
        (round_module._VERIFICATION_EPISODE_HASHER,
         "verification.episode.v1", (7, 8, 9)),
    )


def test_all_hoisted_hashers_are_bit_identical_and_do_not_mutate():
    for reused, seed, value in _hoisted_cases():
        before = reused._iv
        assert reused.h(value) == hasher_module.Hasher(seed).h(value)
        assert reused.h63(value) == hasher_module.Hasher(seed).h63(value)
        assert reused._iv == before


def test_reuse_removes_per_item_seed_encoding_work(monkeypatch):
    original_encode = hasher_module._encode
    encoded_values = []

    def tracked_encode(value):
        encoded_values.append(value)
        return original_encode(value)

    monkeypatch.setattr(hasher_module, "_encode", tracked_encode)
    seed = "hot-loop.performance.v1"
    values = tuple(range(32))
    fresh = tuple(hasher_module.Hasher(seed).h63(value) for value in values)
    fresh_seed_encodes = encoded_values.count(seed)

    encoded_values.clear()
    reused = hasher_module.Hasher(seed)
    reused_values = tuple(reused.h63(value) for value in values)
    reused_seed_encodes = encoded_values.count(seed)

    assert reused_values == fresh
    assert fresh_seed_encodes == len(values)
    assert reused_seed_encodes == 1


def test_production_has_no_fixed_seed_hasher_construction_inside_loops():
    assert scan_hot_loop_fixed_hasher_calls(DEFAULT_SOURCE_ROOT) == []
