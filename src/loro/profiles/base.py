"""LanguageProfile contract + the length counters it dispatches over.

A `LanguageProfile` is the single source of every per-language decision the
pipeline used to hardcode for Vietnamese: the length/rate model, the translation
framing, the default TTS engine + voice strategy, the burn-in font + glyph probe,
and the script/segmentation metadata. It mirrors the `providers/` registry
pattern (a contract + a registry + per-engine entries) so that adding a language
is adding a registry entry, not editing nodes (R2).

This module is intentionally PURE: it imports nothing from `loro.nodes` and
nothing heavy (no numpy/soundfile), so the registry can be resolved from
`config.py` through a lazy accessor without dragging the node graph or pulling an
import cycle. The Vietnamese counter `vi_syllable_count` lives here as the SINGLE
source (U5 unified it): the VI profile's `counter` and `harness.qa.syllable_count`
both reference this one implementation, so the QA gate and the translate length
budget can never drift and the VI budget fingerprint stays frozen (KTD2, R19, #9).
"""

import re
from dataclasses import dataclass
from typing import Callable, NamedTuple

_DIGITS = re.compile(r"\d+")


class ContextLabels(NamedTuple):
    """The language-specific labels the translate node wraps its assembled layered
    context with (R10): the structural phrases plus the per-line directive. The
    node owns the assembly; the profile owns only these fixed strings, so adding a
    language adds a label set, not a node branch. VI carries the exact legacy
    Vietnamese phrases; every other profile shares an English set (the LLM reads
    English labels around any-language content fine). `instruction` is the
    literal user-message directive that names the JSON output array (its key
    matches the profile's `output_key`)."""
    video_context: str
    shot_visuals: str
    summary: str
    neighbors_before: str
    neighbors_after: str
    prev_translations: str
    instruction: str


def vi_syllable_count(text: str) -> int:
    """Vietnamese is monosyllabic: one written word ~ one spoken syllable. Digit
    groups are read out ("22" -> "hai mươi hai"), so count them at ~1.5 spoken
    syllables per digit. The single VI-counter implementation — the VI profile's
    `counter` AND what `qa.syllable_count` re-exports — so the VI translation
    budget and the QA expected-length floor never drift (R19, #9)."""
    count = 0
    for token in text.split():
        digits = sum(len(m) for m in _DIGITS.findall(token))
        count += max(1, round(1.5 * digits)) if digits else 1
    return max(1, count)


def char_count(text: str) -> int:
    """CPS counter for Latin/whitespace scripts: visible characters (including
    internal spaces) of the target text — the unit subtitle reading speed and
    automatic-dubbing length control budget in (KTD1). Demotes syllables to a
    VI-only heuristic for every other language."""
    return max(1, len(text.strip()))


@dataclass(frozen=True)
class LanguageProfile:
    """Everything language-specific for one target (and EN as source/target).

    Resolved once from `Config.target_lang` through the registry; every node reads
    fields here instead of branching on language. `counter` + `rate` form the
    length budget (`length_model` records which family it is); `system_prompt` +
    the labels + `output_key` form the translation framing the `translate` node
    relabels its assembled context with; `tts_engine`/`voice_strategy`/
    `preset_pool`/`tts_language_code` drive voice selection; `font`/`glyph_sample`
    drive the burn-in + glyph preflight; `script`/`text_direction`/
    `segmentation_rule` are carried for the deferred CJK/RTL work (KTD8).
    """

    locale: str               # BCP-47 primary tag, the registry key (e.g. "fr")
    iso639_2: str             # ISO 639-2/B tag for the muxed subtitle language= (e.g. "fra")
    script: str               # ISO 15924 (e.g. "Latn"); future segmentation/bidi dispatch
    text_direction: str       # "ltr" | "rtl"

    # Length & timing budget.
    length_model: str         # "syllable" (VI only) | "cps" (everything else)
    rate: float               # spoken rate in the counter's unit per second (syl/s or char/s)
    cps_max: float            # subtitle reading-speed ceiling (chars/sec) for cue tiling / QA floor
    expansion_factor: float   # rough source->target text length growth (seed/documentation)
    counter: Callable[[str], int]  # target text -> length units (syllables for VI, chars for CPS)
    chunk_budget: int         # TTS sub-chunk size in the SAME unit as `counter`

    # Translation framing (the language-specific strings; assembly stays in the node).
    system_prompt: str        # translation system message
    src_lang_label: str       # source-language name as written in the prompt (tier-1 assumes EN source)
    tgt_lang_label: str       # target-language name as written in the prompt
    english_name: str         # English exonym ("Vietnamese"/"French") for the vision grounding prompt
    output_key: str           # JSON key the model fills per line ("vi" for the VI byte-identity)
    context_labels: ContextLabels  # labels the translate node wraps the layered context with

    # Voice & TTS.
    tts_engine: str           # default TTS engine for this language
    voice_strategy: str       # "clone" | "preset"
    preset_pool: tuple        # per-language preset voice rotation pool (preset engines)
    tts_language_code: str    # spoken-language param for preset cloud engines (e.g. Soniox `language`)

    # Subtitles / burn-in.
    font: str                 # burn-in font family (fontconfig name or bundled file stem)
    glyph_sample: str         # representative glyphs for the preflight coverage probe

    segmentation_rule: str    # "whitespace" | "icu" | ... (future word segmentation dispatch)
