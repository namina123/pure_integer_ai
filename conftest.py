"""根 conftest.py — 钉死 PYTHONHASHSEED=0（断奶可复现性·bit-identical / CI=生产 铁律）。

#1004 根因：CPython **str / bytes / datetime** 哈希随机化（PYTHONHASHSEED·解释器启动时读一次·运行中
改 os.environ 对当前进程**无效**）。**int 与元组哈希不受 seed 影响**（故 Rational.__hash__=
hash((num,den)) 跨进程确定·rational.py:112 安全）——但 str 哈希随机化致 frozenset(str) 迭代序 +
任何 str 的 built-in hash() 跨进程非确定 → 后端状态 parity digest 跨进程变。

实测影响：测试 pass/fail 跨 seed **稳定**（输出走有序结构/deterministic Hasher·非 hash 序·PYTHONHASHSEED=1/2
均 65/65 过）——**仅 parity hash digest 跨 seed 变**。但 bit-identical 铁律要求 digest 亦复现 → 钉死 seed=0。

机制（须 spawn 子进程·非 execve / 非设 os.environ）：conftest 在解释器启动**后**导入·此时 seed 已定型·改
os.environ 对当前进程无效（CPython 启动时一次性读）。**os.execve 在 win32 段错误**（进程模型差异·实测 exit 139）→
改用 **subprocess.run spawn 子进程**（env 注入 PYTHONHASHSEED=0·子进程启动时读到确定 seed·父进程 relay
子进程 exit code）。_PURE_INTEGER_AI_HASH_REEXEC 标记防无限递归。镜像社区 pytest hash-seed pinning 范式。

铁律：纯机制（os/sys/subprocess 标准库）/ 确定性（seed=0 全确定·parity digest 跨进程复现）/ 不影响 lint 入口
（python -m pure_integer_ai.crosscut.guards.lint 非 pytest·不加载 conftest）。
诚实边界：本守卫只钉 pytest 测试进程·生产入口须各自钉（CI env / 启动脚本）·见 doc。spawn 父进程仅 relay
exit code·不做实工作·开销≈一次进程启动（~1s）·CI/本地均接受。
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

_HASH_SEED = "0"
_REEXEC_MARKER = "_PURE_INTEGER_AI_HASH_REEXEC"

_PUBLIC_EVIDENCE_DIRECTORIES = (
    "ph2_dataset_artifacts",
    "ph2_p3ia_dataset_artifacts",
    "r02_storage_profile_artifacts",
)
_PUBLIC_EVIDENCE_FILES = (
    "ph2_dataset_raw/ZHWIKTIONARY_20260701/"
    "zhwiktionary-20260701.final-adapter-v1.pass-1.report.json",
    "ph2_dataset_raw/ZHWIKTIONARY_20260701/"
    "zhwiktionary-20260701.final-adapter-v1.pass-2.report.json",
)


def _ensure_hash_seed() -> None:
    """PYTHONHASHSEED≠'0' 时 spawn 子进程钉死 seed=0·父进程 relay exit code（启动期·幂等）。

    已 seed=0（显式或本守卫已 spawn 的子进程）→ 立即返回（子进程正常跑 pytest）。
    已 spawn 过仍非 0（理论不发生·marker 在子进程 env）→ 标记守不再递归。
    父进程 spawn 子进程后 sys.exit(子 exit code)·故父进程 pytest 不继续（不双跑）。
    """
    if os.environ.get("PYTHONHASHSEED", "") == _HASH_SEED:
        return   # 已钉死 → 无需再动（含显式设 0 / 本守卫已 spawn 的子进程）
    if os.environ.get(_REEXEC_MARKER) == "1":
        # fail-closed（审 LOW-1 修）：marker 已设却 seed≠0 = 异常（spawn 原子设两者·正常子进程必走
        # 上一行 PATH A 早返·到不了此）。仅外部 env 篡改（_PURE_INTEGER_AI_HASH_REEXEC=1 + 未设 seed）可达。
        # 旧版 silent return → 静默退化随机 seed（违 bit-identical）→ 改 raise 让 tampering 显形。
        raise RuntimeError(
            f"{_REEXEC_MARKER}=1 但 PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED', '<unset>')!r}"
            f"（须 '0'）—— 内部 marker 被外部置位致 seed 钉死失效·清掉该 env 变量后重跑"
        )
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = _HASH_SEED   # 子进程启动时读 → str 哈希确定化
    env[_REEXEC_MARKER] = "1"            # 防子进程再次 spawn（无限递归）
    # spawn 子进程·保 argv + cwd + 继承 stdio·父进程 relay exit code（非 execve·win32 安全）
    proc = subprocess.run([sys.executable, *sys.argv], env=env)
    sys.exit(proc.returncode)


_ensure_hash_seed()


def _materialize_public_evidence_view() -> None:
    """把仓库内公开证据映射到不可改名的历史 WORKSPACE 相对路径。"""
    repository = Path(__file__).resolve().parent
    workspace = repository.parent
    for relative_path in _PUBLIC_EVIDENCE_DIRECTORIES:
        source = repository / relative_path
        target = workspace / relative_path
        if target.exists() or not source.is_dir():
            continue
        shutil.copytree(source, target)
    for relative_path in _PUBLIC_EVIDENCE_FILES:
        source = repository / Path(*relative_path.split("/"))
        target = workspace / Path(*relative_path.split("/"))
        if target.exists() or not source.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def pytest_sessionstart(session: object) -> None:
    """在收集测试前生成公开 clone 可复现的历史证据布局。"""
    del session
    _materialize_public_evidence_view()
