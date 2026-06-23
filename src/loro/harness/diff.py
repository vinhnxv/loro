"""Mechanical transcript comparison for the cross-check stage (R8).

Pure functions, fully offline-testable.

`compare` is the original two-way comparison (Nemotron x one verify
reading); it remains the degradation path when Granite is unavailable (R32).

`needs_arbiter` + `vote3` implement the weighted three-way ensemble
(R28/R29/R30): Nemotron stays the pivot (it owns timing), Granite is the
primary verify reading, Gemma is a lazy arbiter consulted only when Nemotron
and Granite diverge on a content word. Votes are tallied per divergence
region against the pivot; replacing text requires the replacement side to
weigh strictly more than the keep side — ties keep Nemotron.
"""

import difflib
import re

# Small inline stopword list: function words whose substitution alone should
# not trigger a replacement.
STOPWORDS = frozenset(
    "a an the this that these those is are was were be been being am "
    "i you he she it we they me him her us them my your his its our their "
    "and or but so nor yet of in on at to for from by with as if than then "
    "do does did done have has had having will would shall should can could "
    "may might must not no yes uh um oh".split()
)

_PUNCT = re.compile(r"[^\w\s']", flags=re.UNICODE)


def normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into tokens."""
    return _PUNCT.sub(" ", text.lower()).split()


def align_ratio(text_a: str, text_b: str) -> float:
    """Token-level alignment ratio in [0, 1] (shared by the cross-check vote
    and the subtitle guard, R35). 0 when either side is empty."""
    a, b = normalize(text_a), normalize(text_b)
    if not a or not b:
        return 0.0
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / max(len(a), len(b))


def compare(
    text_nemotron: str,
    text_gemma: str,
    *,
    wer_threshold: float = 0.2,
    align_floor: float = 0.25,
    min_length_ratio: float = 0.3,
) -> dict:
    """Compare the two readings; returns decision + metrics.

    decision: "keep" | "replace" | "keep_low_confidence"
    """
    ref = normalize(text_nemotron)
    hyp = normalize(text_gemma)

    if not ref:
        # Nothing to verify against; trust Nemotron's (empty) reading
        return {"decision": "keep", "wer": 0.0, "align_ratio": 1.0,
                "content_substitution": False}

    if not hyp or len(hyp) < min_length_ratio * len(ref):
        return {"decision": "keep_low_confidence", "wer": 1.0, "align_ratio": 0.0,
                "content_substitution": False}

    matcher = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    align_ratio = matched / max(len(ref), len(hyp))

    edits = 0
    content_substitution = False
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == "equal":
            continue
        changed = set(ref[a0:a1]) | set(hyp[b0:b1])
        if not changed - STOPWORDS:
            continue  # stopword-only noise never counts toward divergence
        edits += max(a1 - a0, b1 - b0)
        if op == "replace":
            content_substitution = True
    wer = edits / len(ref)

    if align_ratio < align_floor:
        decision = "keep_low_confidence"
    elif content_substitution or wer > wer_threshold:
        decision = "replace"
    else:
        decision = "keep"
    return {"decision": decision, "wer": round(wer, 4),
            "align_ratio": round(align_ratio, 4),
            "content_substitution": content_substitution}


# --- Weighted three-way ensemble (R28/R29/R30) ---

# Reference decision-table weights (plan R28): Granite leads so it can replace
# Nemotron alone. This is the regime documented by test_diff_vote. The
# *deployed* default is calibrated separately in Config.crosscheck_weights
# (U6 raised Nemotron to parity to require verify-engine corroboration); the
# node always passes that, so this constant only applies to direct vote3 calls.
DEFAULT_WEIGHTS = {"nemotron": 0.2, "granite": 0.5, "gemma": 0.3}


def _tokenize_raw(text: str) -> tuple[list[str], list[str], list[int]]:
    """Whitespace tokens plus their normalized forms. Normalized token i
    came from raw token raw[owner[i]], so a winning candidate's region can be
    spliced back with original casing and punctuation."""
    raw = text.split()
    norm: list[str] = []
    owner: list[int] = []
    for i, tok in enumerate(raw):
        for nt in _PUNCT.sub(" ", tok.lower()).split():
            norm.append(nt)
            owner.append(i)
    return raw, norm, owner


def _suspect(ref: list[str], hyp: list[str],
             align_floor: float, min_length_ratio: float) -> tuple[bool, float]:
    """R30: empty / far too short / barely aligned verify reading means the
    verify engine itself is in doubt. Returns (suspect, align_ratio)."""
    if not hyp or len(hyp) < min_length_ratio * len(ref):
        return True, 0.0
    matcher = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    align = matched / max(len(ref), len(hyp))
    return align < align_floor, align


def needs_arbiter(
    text_nemotron: str,
    text_granite: str,
    *,
    align_floor: float = 0.25,
    min_length_ratio: float = 0.3,
) -> bool:
    """Does the Gemma vote matter for this segment (R29)?

    True only when Nemotron and Granite diverge on at least one content word
    AND Granite's reading is not itself suspect (a suspect reading goes
    straight to keep_low_confidence — there is nothing to arbitrate, R30).
    """
    ref = normalize(text_nemotron)
    hyp = normalize(text_granite)
    if not ref:
        return False
    suspect, _ = _suspect(ref, hyp, align_floor, min_length_ratio)
    if suspect:
        return False
    matcher = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == "equal":
            continue
        if (set(ref[a0:a1]) | set(hyp[b0:b1])) - STOPWORDS:
            return True
    return False


def _region_tokens(opcodes: list, start: int, end: int) -> list[int]:
    """Indices of the engine's normalized tokens aligned to pivot[start:end]."""
    out: list[int] = []
    for op, a0, a1, b0, b1 in opcodes:
        if op == "equal":
            lo, hi = max(a0, start), min(a1, end)
            if lo < hi:
                out.extend(range(b0 + (lo - a0), b0 + (hi - a0)))
        elif a0 == a1:  # insertion relative to the pivot
            if start <= a0 <= end:
                out.extend(range(b0, b1))
        elif a0 < end and a1 > start:
            out.extend(range(b0, b1))
    return out


def vote3(
    text_nemotron: str,
    text_granite: str,
    text_gemma: str | None = None,
    *,
    weights: dict[str, float] | None = None,
    align_floor: float = 0.25,
    min_length_ratio: float = 0.3,
) -> dict:
    """Weighted per-region vote over up to three readings (R28).

    Nemotron is the pivot: divergence regions from Granite->Nemotron and
    Gemma->Nemotron alignments are merged on the pivot axis and voted
    independently. A region only changes when the winning candidate weighs
    strictly more than the pivot side; ties keep Nemotron (and set
    `contested`). `text_gemma=None` means Gemma was not consulted (R29) or
    failed (R32) — the vote runs over the two remaining readings.

    Returns {decision, text_effective, contested, regions, metrics}.
    """
    weights = weights or DEFAULT_WEIGHTS
    raw_n, ref, owner_n = _tokenize_raw(text_nemotron)
    base = {"decision": "keep", "text_effective": text_nemotron,
            "contested": False, "regions": [],
            "metrics": {"align_granite": None, "align_gemma": None,
                        "weights": weights}}

    if not ref:
        return base  # nothing to verify against; trust the (empty) pivot

    raw_g, hyp_g, owner_g = _tokenize_raw(text_granite)
    suspect, align_g = _suspect(ref, hyp_g, align_floor, min_length_ratio)
    base["metrics"]["align_granite"] = round(align_g, 4)
    if suspect:
        base["decision"] = "keep_low_confidence"
        return base

    readings: dict[str, tuple[list[str], list[str], list[int]]] = {
        "granite": (raw_g, hyp_g, owner_g),
    }
    if text_gemma is not None:
        raw_m, hyp_m, owner_m = _tokenize_raw(text_gemma)
        readings["gemma"] = (raw_m, hyp_m, owner_m)
        matcher = difflib.SequenceMatcher(a=ref, b=hyp_m, autojunk=False)
        matched = sum(block.size for block in matcher.get_matching_blocks())
        base["metrics"]["align_gemma"] = round(
            matched / max(len(ref), len(hyp_m) or 1), 4)

    opcodes = {
        eng: difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False).get_opcodes()
        for eng, (_, hyp, _) in readings.items()
    }

    # Merge divergence intervals from every alignment on the pivot axis;
    # touching intervals merge so overlapping disagreements vote as one region
    intervals = sorted(
        (a0, a1)
        for ops in opcodes.values()
        for op, a0, a1, _, _ in ops
        if op != "equal"
    )
    merged: list[list[int]] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    regions = []
    contested = False
    any_changed = False
    for s, e in merged:
        pivot_key = tuple(ref[s:e])
        candidates: dict[tuple, dict] = {
            pivot_key: {"weight": weights["nemotron"], "engines": ["nemotron"],
                        "source": None}
        }
        for eng, (raw, hyp, owner) in readings.items():
            idxs = _region_tokens(opcodes[eng], s, e)
            key = tuple(hyp[i] for i in idxs)
            entry = candidates.setdefault(
                key, {"weight": 0.0, "engines": [], "source": (eng, idxs)})
            entry["weight"] += weights[eng]
            entry["engines"].append(eng)

        # Stopword-only regions never change the text (same rule as compare)
        all_tokens = set().union(*candidates.keys())
        has_content = bool(all_tokens - STOPWORDS)

        keep_weight = candidates[pivot_key]["weight"]
        challengers = [(k, v) for k, v in candidates.items() if k != pivot_key]
        best = max(challengers, key=lambda kv: kv[1]["weight"], default=None)

        changed = bool(best and has_content and best[1]["weight"] > keep_weight)
        tie = bool(best and has_content and best[1]["weight"] == keep_weight)
        contested = contested or tie
        any_changed = any_changed or changed
        winner = best[1] if changed else candidates[pivot_key]

        regions.append({
            "pivot_span": [s, e],
            "pivot": " ".join(pivot_key),
            "candidates": {" ".join(k): round(v["weight"], 3)
                           for k, v in candidates.items()},
            "winner_engines": winner["engines"],
            "winner_text": " ".join(best[0]) if changed else " ".join(pivot_key),
            "changed": changed,
            "tie": tie,
            "_source": winner["source"] if changed else None,
        })

    base["contested"] = contested
    base["regions"] = regions
    if not any_changed:
        for region in regions:
            region.pop("_source")
        return base

    # Splice winning regions back onto the pivot frame using raw tokens, so
    # the effective text keeps the winners' casing and punctuation
    out: list[str] = []
    emitted_n: set[int] = set()

    def emit_pivot(lo: int, hi: int) -> None:
        for i in range(lo, hi):
            oi = owner_n[i]
            if oi not in emitted_n:
                emitted_n.add(oi)
                out.append(raw_n[oi])

    pos = 0
    for region in regions:
        s, e = region["pivot_span"]
        emit_pivot(pos, s)
        if region["changed"]:
            eng, idxs = region.pop("_source")
            raw, _, owner = readings[eng]
            seen: set[int] = set()
            for i in idxs:
                oi = owner[i]
                if oi not in seen:
                    seen.add(oi)
                    out.append(raw[oi])
            emitted_n.update(owner_n[i] for i in range(s, e))
        else:
            region.pop("_source")
            emit_pivot(s, e)
        pos = e
    emit_pivot(pos, len(ref))

    base["decision"] = "replace"
    base["text_effective"] = " ".join(out)
    return base
