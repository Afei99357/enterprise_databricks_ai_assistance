"""Tests for deterministic template-page detection."""

from __future__ import annotations

from wnv_assistant.documents.template_flag import is_template_page


class TestIsTemplatePage:
    def test_flags_bracketed_insert_marker(self) -> None:
        text = "Contact us at [INSERT PHONE NUMBER] for more information."
        assert is_template_page(text) is True

    def test_flags_multiple_markers_case_insensitively(self) -> None:
        text = "[insert county name] reported cases in [Insert Month]."
        assert is_template_page(text) is True

    def test_does_not_flag_normal_narrative_text(self) -> None:
        text = "Cook County reported 12 positive mosquito pools in July."
        assert is_template_page(text) is False

    def test_does_not_flag_unrelated_bracket_usage(self) -> None:
        text = "See the appendix [1] for full case counts."
        assert is_template_page(text) is False
