"""对象模型 legacy baseline 与新增声明门的专项回归。"""
from __future__ import annotations

import json
from pathlib import Path

from scripts import object_model_lint


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _empty_baseline(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "src"
    baseline_path = tmp_path / "baseline.json"
    _write(source_root / "empty.py", "")
    object_model_lint.create_baseline(source_root, baseline_path)
    return source_root, baseline_path


def test_repository_object_model_baseline_is_clean_and_substantial():
    violations, class_count = object_model_lint.check_object_model()
    assert violations == []
    assert class_count >= 3000


def test_new_class_requires_an_object_model_declaration(tmp_path):
    source_root, baseline_path = _empty_baseline(tmp_path)
    _write(source_root / "new.py", "class Undeclared:\n    pass\n")

    violations, class_count = object_model_lint.check_object_model(
        source_root, baseline_path)

    assert class_count == 1
    assert len(violations) == 1
    assert "缺少 object-model 声明" in violations[0]


def test_all_four_declared_object_model_families_pass(tmp_path):
    source_root, baseline_path = _empty_baseline(tmp_path)
    _write(
        source_root / "declared.py",
        """from dataclasses import dataclass
from typing import Protocol

# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class Value:
    number: int

# object-model: lifecycle; owner=request; cleanup=scope-end
class Runtime:
    pass

# object-model: protocol
class ReaderProtocol(Protocol):
    pass

# object-model: exception
class ReadError(Exception):
    pass
""",
    )

    violations, class_count = object_model_lint.check_object_model(
        source_root, baseline_path)

    assert class_count == 4
    assert violations == []


def test_value_requires_frozen_slots_dataclass(tmp_path):
    source_root, baseline_path = _empty_baseline(tmp_path)
    _write(
        source_root / "bad_value.py",
        """from dataclasses import dataclass

# object-model: value; representation=struct; interop=pending
@dataclass
class MutableValue:
    number: int
""",
    )

    violations, _ = object_model_lint.check_object_model(
        source_root, baseline_path)

    assert any("frozen=True" in item for item in violations)
    assert any("slots=True" in item for item in violations)


def test_value_requires_registered_struct_and_interop_metadata(tmp_path):
    source_root, baseline_path = _empty_baseline(tmp_path)
    _write(
        source_root / "bad_metadata.py",
        """from dataclasses import dataclass

# object-model: value; representation=object; interop=unknown
@dataclass(frozen=True, slots=True)
class AmbiguousValue:
    number: int
""",
    )

    violations, _ = object_model_lint.check_object_model(
        source_root, baseline_path)

    assert any("representation 未注册" in item for item in violations)
    assert any("interop 未注册" in item for item in violations)


def test_value_rejects_missing_struct_migration_metadata(tmp_path):
    source_root, baseline_path = _empty_baseline(tmp_path)
    _write(
        source_root / "missing_metadata.py",
        """from dataclasses import dataclass

# object-model: value
@dataclass(frozen=True, slots=True)
class UnclassifiedValue:
    number: int
""",
    )

    violations, _ = object_model_lint.check_object_model(
        source_root, baseline_path)

    assert any("interop" in item for item in violations)
    assert any("representation" in item for item in violations)


def test_lifecycle_requires_owner_and_cleanup_boundaries(tmp_path):
    source_root, baseline_path = _empty_baseline(tmp_path)
    _write(
        source_root / "bad_runtime.py",
        """# object-model: lifecycle; owner=request
class Runtime:
    pass
""",
    )

    violations, _ = object_model_lint.check_object_model(
        source_root, baseline_path)

    assert len(violations) == 1
    assert "cleanup" in violations[0]


def test_legacy_debt_is_allowed_but_shape_regression_is_not(tmp_path):
    source_root = tmp_path / "src"
    baseline_path = tmp_path / "baseline.json"
    module = source_root / "legacy.py"
    _write(
        module,
        """from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ExistingValue:
    number: int

class ExistingRuntime:
    pass
""",
    )
    object_model_lint.create_baseline(source_root, baseline_path)
    assert object_model_lint.check_object_model(
        source_root, baseline_path)[0] == []

    _write(
        module,
        """from dataclasses import dataclass

@dataclass(frozen=True)
class ExistingValue:
    number: int

class ExistingRuntime:
    pass
""",
    )
    violations, _ = object_model_lint.check_object_model(
        source_root, baseline_path)

    assert len(violations) == 1
    assert "移除 slots" in violations[0]


def test_baseline_digest_and_overwrite_are_fail_closed(tmp_path):
    source_root, baseline_path = _empty_baseline(tmp_path)
    try:
        object_model_lint.create_baseline(source_root, baseline_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("baseline overwrite must be rejected")

    value = json.loads(baseline_path.read_text(encoding="utf-8"))
    value["inventory_sha256"] = "0" * 64
    baseline_path.write_text(json.dumps(value), encoding="utf-8")

    try:
        object_model_lint.load_baseline(baseline_path)
    except ValueError as error:
        assert "inventory_sha256" in str(error)
    else:
        raise AssertionError("corrupt baseline digest must be rejected")


def test_fixed_seed_hasher_construction_inside_loop_is_rejected(tmp_path):
    source_root, baseline_path = _empty_baseline(tmp_path)
    _write(
        source_root / "hot_loop.py",
        """from pure_integer_ai.crosscut.determinism.hasher import Hasher

_SEED = "fixed"

def values(items):
    result = []
    for item in items:
        result.append(Hasher(_SEED).h63(item))
    return result
""",
    )

    violations, class_count = object_model_lint.check_object_model(
        source_root, baseline_path)

    assert class_count == 0
    assert len(violations) == 1
    assert "循环内重复构造固定 seed Hasher" in violations[0]
