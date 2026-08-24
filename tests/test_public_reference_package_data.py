"""DLG-RAW 公开对话入口的完整 payload closure 与 wheel 资源回归。"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import shutil
import tomllib

import pytest

from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
    build_public_dialogue_runtime_v1,
)
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PUBLIC_FRAME_CONTEXT_NONE,
)
from pure_integer_ai.experiments.conversation_public_source_payload_host import (
    load_public_source_payload_closure_from_root,
)
from pure_integer_ai.experiments import run_public_frame_dialogue


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DATA_FILES_KEY = "share/pure_integer_ai/data/ph2"
_PUBLIC_DIALOGUE_RUNTIME_FILES = (
    run_public_frame_dialogue._PUBLIC_DIALOGUE_REQUIRED_RESOURCES)


def _runtime_at(resource_root: Path) -> PublicDialogueRuntimeV1:
    """只在 host 边界读取 public resources，再建立纯逻辑对话 runtime。"""
    return build_public_dialogue_runtime_v1(
        load_public_source_payload_closure_from_root(resource_root))


def test_public_dialogue_runtime_closure_is_declared_for_wheel() -> None:
    """terminal 所需的完整 68 项安装资源必须随 distribution 发布。"""
    pyproject = tomllib.loads(
        (_REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    configured = tuple(
        pyproject["tool"]["setuptools"]["data-files"][_DATA_FILES_KEY])
    runtime = _runtime_at(_REPOSITORY_ROOT)

    assert all(configured.count(relative_path) == 1
               for relative_path in _PUBLIC_DIALOGUE_RUNTIME_FILES)
    assert all((_REPOSITORY_ROOT / relative_path).is_file()
               for relative_path in _PUBLIC_DIALOGUE_RUNTIME_FILES)
    assert tuple(sorted(_PUBLIC_DIALOGUE_RUNTIME_FILES)) == (
        run_public_frame_dialogue._PUBLIC_DIALOGUE_REQUIRED_RESOURCES)
    assert runtime.source_payload_closure.records
    assert len(runtime.active_catalog.frames) == 9


def test_public_dialogue_main_reads_the_complete_installed_data_layout(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """安装式 share 布局替代源码根后，入口仍须通过 closure/runtime 完成首轮。"""
    installed_root = tmp_path / "share" / "pure_integer_ai"
    for relative_path in _PUBLIC_DIALOGUE_RUNTIME_FILES:
        destination = installed_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_REPOSITORY_ROOT / relative_path, destination)

    runtime = _runtime_at(installed_root)
    first_surface = next(
        frame.surface_bytes for frame in runtime.base_catalog.frames
        if frame.context_requirement == PUBLIC_FRAME_CONTEXT_NONE)
    monkeypatch.setattr(
        run_public_frame_dialogue,
        "_SOURCE_REPOSITORY",
        tmp_path / "missing-source-root",
    )
    monkeypatch.setattr(
        run_public_frame_dialogue,
        "_installed_public_dialogue_resource_roots",
        lambda: (installed_root,),
    )

    target = BytesIO()
    assert run_public_frame_dialogue._default_public_dialogue_resource_root() == (
        installed_root)
    assert run_public_frame_dialogue.main(
        [],
        stdin=BytesIO(bytes(first_surface) + b"\n:quit\n"),
        stdout=target,
    ) == 0
    assert b"[REJECT:" not in target.getvalue()
