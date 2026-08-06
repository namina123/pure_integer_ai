"""为 W05 历史行为回归装配已经发布但不再可重建的冻结上下文。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pure_integer_ai.experiments.ph2_w05_contract as contract_owner
from pure_integer_ai.experiments.ph2_d03_release_catalog import (
    FORMAL_GLOBAL_MANIFEST_PATH,
)
from pure_integer_ai.experiments.ph2_j_lc_pre_w04_contract import (
    read_j_lc_pre_w04_gate,
)


def open_historical_w05_context(
    repository_root: str | Path,
    global_manifest_path: str = FORMAL_GLOBAL_MANIFEST_PATH,
    **arguments: Any,
):
    """只消费冻结 parent gate，不改变生产 opener 的拒绝语义。"""
    root = Path(repository_root).resolve()
    gate_path = root / Path(*contract_owner.PRE_W04_GATE_PATH.split("/"))
    gate = read_j_lc_pre_w04_gate(gate_path)
    original_builder = contract_owner.build_j_lc_pre_w04_gate
    original_verifier = contract_owner.verify_j_lc_pre_w04_files

    def frozen_builder(_repository_root: str | Path):
        """返回已经发布的 gate，供历史 W05 行为装配使用。"""
        return gate

    def frozen_verifier(_gate: object, *, repository_root: str | Path) -> None:
        """跳过已由 authority 测试确认的历史 evidence 漂移。"""
        del _gate, repository_root

    contract_owner.build_j_lc_pre_w04_gate = frozen_builder
    contract_owner.verify_j_lc_pre_w04_files = frozen_verifier
    try:
        return contract_owner.open_w05_frozen_context(
            root,
            global_manifest_path,
            **arguments,
        )
    finally:
        contract_owner.build_j_lc_pre_w04_gate = original_builder
        contract_owner.verify_j_lc_pre_w04_files = original_verifier
