"""Unit tests for ocr.config._env_bool (boolean env-var parsing)."""

import pytest

from ocr import config


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("x", True),
        ("true", True),  # only the exact strings 0 / false / False / "" are falsey
        ("0", False),
        ("false", False),
        ("False", False),
        ("", False),
    ],
)
def test_env_bool_explicit_values(monkeypatch, value, expected):
    monkeypatch.setenv("OCR_TEST_FLAG", value)
    assert config._env_bool("OCR_TEST_FLAG") is expected


def test_env_bool_default_true_when_unset(monkeypatch):
    monkeypatch.delenv("OCR_TEST_FLAG", raising=False)
    assert config._env_bool("OCR_TEST_FLAG") is True
    assert config._env_bool("OCR_TEST_FLAG", default=True) is True


def test_env_bool_default_false_when_unset(monkeypatch):
    monkeypatch.delenv("OCR_TEST_FLAG", raising=False)
    assert config._env_bool("OCR_TEST_FLAG", default=False) is False
