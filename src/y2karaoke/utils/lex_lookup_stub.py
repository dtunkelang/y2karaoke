"""
Simple CMU pronunciation-backed `lex_lookup` proxy so Epitran can keep using
phonetic transcriptions even when Flite isn't installed.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Iterable, Sequence

import pronouncing

logger = logging.getLogger(__name__)


def lex_lookup_main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint that mimics the lex_lookup CLI."""
    if argv is None:
        argv = sys.argv[1:]
    text = " ".join(argv).strip()
    if not text:
        return 0

    tokens = _tokenize(text)
    phonemes: list[str] = []
    for token in tokens:
        phones = pronouncing.phones_for_word(token.lower())
        if phones:
            phonemes.extend(phones[0].split())
        else:
            logger.debug("No pronunciation for %s; emitting text literal", token)
            phonemes.extend([token.upper()])

    if not phonemes:
        return 0

    print(f"[{' '.join(phonemes)}]")
    return 0


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text)
