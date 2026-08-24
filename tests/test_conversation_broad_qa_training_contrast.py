import pytest

from pure_integer_ai.experiments.conversation_broad_qa_training_contrast import (
    ConversationBroadQaTrainingContrastError,
    _select_questions,
    _selection_sha256,
)


_QUESTIONS = (
    {"item_id": "a", "question": "甲"},
    {"item_id": "b", "question": "乙"},
    {"item_id": "c", "question": "丙"},
)


def test_contrast_selection_preserves_pack_order_and_is_deterministic() -> None:
    selected = _select_questions(_QUESTIONS, ("c", "a"))
    assert tuple(item["item_id"] for item in selected) == ("a", "c")
    assert _selection_sha256(selected) == _selection_sha256(selected)
    assert _select_questions(_QUESTIONS, ()) == _QUESTIONS


@pytest.mark.parametrize("item_ids", [("a", "a"), ("missing",)])
def test_contrast_selection_rejects_invalid_item_ids(item_ids: tuple[str, ...]) -> None:
    with pytest.raises(ConversationBroadQaTrainingContrastError):
        _select_questions(_QUESTIONS, item_ids)
