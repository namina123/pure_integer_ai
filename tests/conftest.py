"""测试临时目录兼容层，避开 Windows 受限令牌的 0700 ACL 问题。"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAFE_TMP_ROOT = _REPO_ROOT.parent / ".pure_integer_ai_pytest_tmp_safe"


def _safe_name(value: str) -> str:
    """把 pytest node 名称压成可读且稳定的目录名。"""
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    prefix = normalized[:64] or "tmp"
    return f"{prefix}-{digest}"


def _fresh_session_root() -> Path:
    """为本次 pytest 进程创建普通 ACL 的 session 根目录。"""
    _SAFE_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    base_name = f"session-{os.getpid()}"
    for ordinal in range(4096):
        suffix = "" if ordinal == 0 else f"-{ordinal:04d}"
        target = _SAFE_TMP_ROOT / f"{base_name}{suffix}"
        try:
            target.mkdir()
        except FileExistsError:
            continue
        return target
    raise RuntimeError("pytest 安全临时目录槽位耗尽")


class SafeTempPathFactory:
    """提供 tests 当前用到的 mktemp/getbasetemp 子集。"""

    def __init__(self, base: Path) -> None:
        """绑定普通 ACL 的 session 根目录。"""
        self._base = base
        self._next_by_name: dict[str, int] = {}

    def getbasetemp(self) -> Path:
        """返回本 session 的临时根目录。"""
        return self._base

    def mktemp(self, basename: str, numbered: bool = True) -> Path:
        """创建一个普通 ACL 的唯一子目录，语义上等价于 pytest mktemp。"""
        safe = _safe_name(str(basename))
        if not numbered:
            target = self._base / safe
            target.mkdir(parents=True, exist_ok=False)
            return target
        start = self._next_by_name.get(safe, 0)
        for ordinal in range(start, start + 4096):
            target = self._base / f"{safe}-{ordinal:04d}"
            try:
                target.mkdir(parents=True)
            except FileExistsError:
                continue
            self._next_by_name[safe] = ordinal + 1
            return target
        raise RuntimeError("pytest 安全临时子目录槽位耗尽")


@pytest.fixture(scope="session")
def tmp_path_factory():
    """替换 pytest 内建工厂，避免 tempfile/0700 在 Windows 沙箱下不可遍历。"""
    factory = SafeTempPathFactory(_fresh_session_root())
    try:
        yield factory
    finally:
        shutil.rmtree(factory.getbasetemp(), ignore_errors=True)
        try:
            _SAFE_TMP_ROOT.rmdir()
        except OSError:
            pass


@pytest.fixture
def tmp_path(tmp_path_factory, request):
    """为单个测试提供 pathlib.Path 临时目录。"""
    return tmp_path_factory.mktemp(request.node.name, numbered=True)
