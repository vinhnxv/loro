"""Split a long Vietnamese passage into TTS-sized chunks.

Autoregressive TTS (Higgs Audio v3) degrades, loops, or silently truncates when
one request carries a whole paragraph: in a real run a ~3500-char segment came
back as 81s of audio for ~186s of text — most of the content was dropped and
the QA gate's wide window let it through. Splitting on sentence boundaries and
packing greedily up to a syllable budget keeps every request inside the model's
reliable zone. Packing (rather than one-call-per-sentence) minimizes call count:
each Higgs call pads silence, so fewer-but-fuller chunks concatenate more
cleanly and avoid choppy single-word clips.

`count` is injected (loro.harness.qa.syllable_count) so chunk sizing matches the
QA duration model exactly and this module stays a leaf with no harness import.
"""

import re
from typing import Callable

# Sentence enders followed by whitespace. The lookbehind keys on the punctuation
# only, so "3.5 giây" (digit after the dot) and "ADK." mid-clause split on the
# trailing space as intended; a decimal like "3.5" never splits because no space
# follows the dot.
_SENTENCE = re.compile(r"(?<=[.!?…])\s+")
# Clause separators, used only to break a single sentence that is over budget.
_CLAUSE = re.compile(r"(?<=[,;:])\s+")

CountFn = Callable[[str], int]


def _split_keep(pattern: re.Pattern, text: str) -> list[str]:
    return [part for part in pattern.split(text.strip()) if part.strip()]


def _hard_wrap(piece: str, max_syllables: int, count: CountFn) -> list[str]:
    """Last resort: break one over-budget piece on word boundaries so no atom
    exceeds the budget when it can be avoided."""
    out: list[str] = []
    cur: list[str] = []
    for word in piece.split():
        cur.append(word)
        if count(" ".join(cur)) >= max_syllables:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


# Provenance of the break that PRECEDES an atom. "boundary" = a real
# sentence/clause break (a natural pause belongs there); "hardwrap" = an
# over-budget clause cut mid-phrase on a word boundary (no pause belongs there).
BOUNDARY = "boundary"
HARDWRAP = "hardwrap"


def _atoms(text: str, max_syllables: int, count: CountFn) -> list[tuple[str, str]]:
    """Sentences, but any over-budget sentence is broken into clauses and, if a
    clause is still over budget, hard-wrapped on words. Each atom is tagged with
    the break that precedes it: the first fragment of a hard-wrapped clause is a
    real clause/sentence boundary, the rest are mid-clause hard-wrap cuts (U4)."""
    atoms: list[tuple[str, str]] = []
    for sentence in _split_keep(_SENTENCE, text):
        if count(sentence) <= max_syllables:
            atoms.append((sentence, BOUNDARY))
            continue
        for clause in _split_keep(_CLAUSE, sentence):
            if count(clause) <= max_syllables:
                atoms.append((clause, BOUNDARY))
            else:
                for i, frag in enumerate(_hard_wrap(clause, max_syllables, count)):
                    atoms.append((frag, BOUNDARY if i == 0 else HARDWRAP))
    return atoms


def chunk_for_tts_typed(text: str, max_syllables: int,
                        count: CountFn) -> tuple[list[str], list[str]]:
    """Greedily pack sentences into chunks of at most `max_syllables` syllables,
    also returning the break type at each chunk join: `break_types[i]` is the
    boundary kind between `chunks[i]` and `chunks[i+1]` (BOUNDARY = a natural
    sentence/clause pause, HARDWRAP = a mid-clause cut). `break_types` has length
    `len(chunks) - 1`. A short single-sentence text returns one chunk and no
    joins, so callers can treat one chunk as the unchanged single-call path."""
    if max_syllables < 1:
        raise ValueError("max_syllables must be >= 1")
    atoms = _atoms(text, max_syllables, count)
    if not atoms:
        stripped = text.strip()
        return ([stripped], []) if stripped else ([], [])

    chunks: list[str] = []
    break_types: list[str] = []
    cur: list[str] = []
    cur_syllables = 0
    for atom, brk in atoms:
        atom_syllables = count(atom)
        if cur and cur_syllables + atom_syllables > max_syllables:
            chunks.append(" ".join(cur))
            break_types.append(brk)  # the join inherits this atom's break-before
            cur, cur_syllables = [], 0
        cur.append(atom)
        cur_syllables += atom_syllables
    if cur:
        chunks.append(" ".join(cur))
    return chunks, break_types


def chunk_for_tts(text: str, max_syllables: int, count: CountFn) -> list[str]:
    """Greedily pack sentences into chunks of at most `max_syllables` syllables.

    Returns a list of non-empty chunk strings. A short single-sentence text
    returns a one-element list, so callers can treat one chunk as the unchanged
    single-call path. (Thin wrapper over chunk_for_tts_typed for callers that
    don't need the per-join break types.)
    """
    return chunk_for_tts_typed(text, max_syllables, count)[0]
