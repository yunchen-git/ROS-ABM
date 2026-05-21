import pytest

from CountDirectory import newDirect


def test_new_direct_returns_one_for_empty_directory(tmp_path):
    assert newDirect(str(tmp_path)) == 1.0


def test_new_direct_uses_max_numeric_entry_and_ignores_non_numeric(tmp_path):
    (tmp_path / "1.0").mkdir()
    (tmp_path / "5.5").mkdir()
    (tmp_path / "notes").mkdir()
    (tmp_path / "latest").write_text("ignore me", encoding="utf-8")

    assert newDirect(str(tmp_path)) == 6.5


def test_new_direct_raises_for_missing_directory(tmp_path):
    missing = tmp_path / "missing_dir"
    with pytest.raises(FileNotFoundError):
        newDirect(str(missing))
