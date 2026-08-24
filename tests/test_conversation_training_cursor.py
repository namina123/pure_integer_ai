from pathlib import Path

import pytest

from pure_integer_ai.experiments.conversation_training_cursor import (
    DialogueTrainingCursor,
    DialogueTrainingCursorError,
    recover_training_cursor,
    write_training_cursor,
)
from pure_integer_ai.storage.k_run_boundary import open_existing_run_root


def _cursor() -> DialogueTrainingCursor:
    return DialogueTrainingCursor(
        tuple(range(32)),
        tuple("dialogue-run".encode("utf-8")),
        (1, 2, 3, 4),
        (1, 2),
        445,
        184,
        1409,
        False,
    )


def test_training_cursor_round_trips_as_integer_stream(tmp_path: Path) -> None:
    root = open_existing_run_root(tmp_path, require_k_drive=False)
    cursor = _cursor()
    path = write_training_cursor(root, cursor)
    assert path.name == "training_cursor.int"
    assert recover_training_cursor(tmp_path, require_k_drive=False) == cursor
    assert recover_training_cursor(tmp_path, require_k_drive=False).identity() == cursor.identity()


def test_training_cursor_is_exclusive(tmp_path: Path) -> None:
    root = open_existing_run_root(tmp_path, require_k_drive=False)
    cursor = _cursor()
    write_training_cursor(root, cursor)
    with pytest.raises(DialogueTrainingCursorError):
        write_training_cursor(root, cursor)


def test_training_cursor_rejects_stage_and_sha_drift() -> None:
    with pytest.raises(DialogueTrainingCursorError):
        DialogueTrainingCursor(tuple(range(31)), (1,), (1,), (), 0, 0, 0, False)
    with pytest.raises(DialogueTrainingCursorError):
        DialogueTrainingCursor(tuple(range(32)), (1,), (1,), (2,), 0, 0, 0, False)
