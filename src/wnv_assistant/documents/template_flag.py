"""Deterministic template-page detection.

Regexes the extracted text for `[INSERT ...]`-style bracket placeholders
rather than trusting any upstream source's self-reported flag -- decouples
correctness from whatever produced the text.
"""

from __future__ import annotations

import re

_TEMPLATE_PATTERN = re.compile(r"\[INSERT[^\]]*\]", re.IGNORECASE)


def is_template_page(text: str) -> bool:
    """True if the text contains a genuine fill-in-the-blank template marker."""
    return bool(_TEMPLATE_PATTERN.search(text))
