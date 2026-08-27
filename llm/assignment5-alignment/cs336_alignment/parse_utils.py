"""Utilities for extracting answers from model evaluation outputs."""

from __future__ import annotations

import re


# Accept signed integers/decimals, optionally formatted with thousands separators.
_NUMBER_RE = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)")


def parse_gsm8k_response(model_output: str) -> str | None:
    """Return the last numeric answer in a GSM8K model response.

    GSM8K's reference answers are numbers.  Taking the final numeric token
    handles common chain-of-thought outputs while returning ``None`` for an
    output that contains no numeric answer.
    """
    matches = _NUMBER_RE.findall(model_output)
    if not matches:
        return None
    return matches[-1].replace(",", "")


# Keep the misspelled name used by older local evaluation scripts working.
parese_gsm8k_response = parse_gsm8k_response
