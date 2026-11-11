"""从授权 train facts 派生有界的多关系路径自然输入批次。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_dataset_contract import parse_canonical_json_bytes


FACT_FIELDS = frozenset({
    "anchor", "case_id", "relation", "sample_role", "seed_id", "split", "surface", "terms",
})
TARGET_FAMILIES = ("SIMILAR", "ANTONYM", "PURE_ALIAS", "PART_OF")
TARGET_SEED_PREFIX = {
    "SIMILAR": "similar", "ANTONYM": "antonym", "PURE_ALIAS": "alias", "PART_OF": "part",
}


def read_authored_path_rows(path: str | Path) -> tuple[dict, ...]:
    """读取并校验本课程的规范事实行，不接受非 train 或非 support 样本。"""
    raw = Path(path).read_bytes()
    if not raw or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("path facts require one canonical final newline")
    rows = []
    for ordinal, line in enumerate(raw.splitlines(), 1):
        row = parse_canonical_json_bytes(line, require_object=True)
        if (set(row) != FACT_FIELDS or row.get("split") != "train"
                or row.get("sample_role") != "support"):
            raise ValueError(f"path fact {ordinal} is not a licensed train support row")
        rows.append(row)
    result = tuple(rows)
    if (len(result) != 28 or len({row["seed_id"] for row in result}) != len(result)
            or {row["relation"] for row in result}
            != {"REFERS", *TARGET_FAMILIES}):
        raise ValueError("path facts do not close the frozen 28-row, five-family source")
    return result


def build_train_path_input_batch(rows: tuple[dict, ...]) -> tuple[dict, ...]:
    """生成 20 条 train-only 普通问句，四个末端关系族各五条。"""
    if type(rows) is not tuple or len(rows) != 28:
        raise ValueError("path input batch requires the complete frozen train facts")
    by_seed = {row["seed_id"]: row for row in rows}
    cases = []
    for family in TARGET_FAMILIES:
        for source_ordinal in range(1, 6):
            target_ordinal = 1 + (source_ordinal - 1) % 2
            source_seed = f"concept-reference-{source_ordinal:02d}"
            target_seed = f"{TARGET_SEED_PREFIX[family]}-{target_ordinal:02d}"
            source = by_seed[source_seed]
            target = by_seed[target_seed]
            if target["relation"] != family or source["relation"] != "REFERS":
                raise ValueError("path input endpoints do not match their train relation family")
            source_text = source["terms"][0][0]
            target_index = 0 if family == "PART_OF" else 1
            target_text = target["terms"][target_index][0]
            question = f"{source_text}到{target_text}之间有哪些来源关系？"
            cases.append({
                "ordinal": len(cases) + 1,
                "case_id": f"{family.casefold()}-{source_ordinal:02d}",
                "split": "train",
                "target_family": family,
                "source_seed_id": source_seed,
                "target_seed_id": target_seed,
                "request_instruction": f"instructions/{family.casefold()}.request.train.json",
                "input_values": list(map(ord, question)),
            })
    result = tuple(cases)
    if (len(result) != 20 or len({tuple(row["input_values"]) for row in result}) != 20
            or any(sum(row["target_family"] == family for row in result) != 5
                   for family in TARGET_FAMILIES)):
        raise ValueError("train path batch must contain 20 unique inputs across four target families")
    return result


__all__ = ["FACT_FIELDS", "TARGET_FAMILIES", "build_train_path_input_batch",
           "read_authored_path_rows"]
