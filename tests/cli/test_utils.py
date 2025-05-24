"""Tests for the Oboyu CLI utilities."""

import pytest

from oboyu.cli.utils import contains_japanese, format_snippet


def test_contains_japanese() -> None:
    """Test Japanese character detection."""
    # Test with Japanese text
    japanese_text = "これは日本語のテキストです"
    assert contains_japanese(japanese_text) is True

    # Test with English text
    english_text = "This is English text"
    assert contains_japanese(english_text) is False

    # Test with mixed text
    mixed_text = "This contains 日本語"
    assert contains_japanese(mixed_text) is True

    # Test with empty text
    empty_text = ""
    assert contains_japanese(empty_text) is False


def test_format_snippet() -> None:
    """Test snippet formatting."""
    # Test basic snippet formatting
    text = "This is a test document that contains some words for testing the snippet formatting function"
    query = "test"
    snippet = format_snippet(text, query, length=30, highlight=False)
    assert len(snippet) <= 40  # Account for ellipsis
    assert "test" in snippet

    # Test highlighting (now disabled)
    highlighted = format_snippet(text, query, length=30, highlight=True)
    assert "test" in highlighted  # Highlighting is disabled, so just check for plain text

    # Test with Japanese text
    japanese_text = "これは日本語のテキストです。テストのためのサンプルテキストです。"
    japanese_query = "テスト"
    japanese_snippet = format_snippet(japanese_text, japanese_query, length=20, highlight=True)
    assert "テスト" in japanese_snippet

    # Test with no match
    no_match_snippet = format_snippet(text, "xyz", length=30, highlight=True)
    assert len(no_match_snippet) <= 40  # Account for ellipsis
    assert no_match_snippet.startswith(text[:30]) or no_match_snippet.startswith("...")

    # Test with very short query term (should be skipped for highlighting)
    short_query_snippet = format_snippet(text, "is", length=30, highlight=True)
    assert "[bold][yellow]is[/yellow][/bold]" not in short_query_snippet