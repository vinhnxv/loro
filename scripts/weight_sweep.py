"""Offline weight calibration for U6: re-vote the 136 stored (Nemotron,
Granite, Gemma) readings under candidate weight sets — no model calls — and
report replace counts plus the four known-wrong regression segments."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loro.harness import diff

DST = Path("work/ai-interview-ensemble/crosscheck")
_VERDICT_RE = re.compile(r"seg_\d+\.json$")
WRONG = ["seg_0041", "seg_0035", "seg_0053", "seg_0119"]

CANDIDATES = {
    "plan (0.2/0.5/0.3)":        {"nemotron": 0.2, "granite": 0.5, "gemma": 0.3},
    "corroborate (0.5/0.5/0.3)": {"nemotron": 0.5, "granite": 0.5, "gemma": 0.3},
    "granite-lead-soft (0.4/0.5/0.3)": {"nemotron": 0.4, "granite": 0.5, "gemma": 0.3},
}


def load():
    rows = {}
    for path in sorted(DST.glob("seg_*.json")):
        if not _VERDICT_RE.fullmatch(path.name):
            continue
        d = json.loads(path.read_text())
        rows[path.stem] = (d.get("text_nemotron", ""), d.get("text_granite", ""),
                           d.get("text_gemma", "") or None)
    return rows


def revote(rows, weights):
    out = {}
    for seg, (n, g, m) in rows.items():
        # M is only meaningful when the arbiter was (or would be) consulted
        arbiter = diff.needs_arbiter(n, g)
        out[seg] = diff.vote3(n, g, m if arbiter else None, weights=weights)
    return out


def main():
    rows = load()
    print(f"loaded {len(rows)} verdicts\n")
    for name, weights in CANDIDATES.items():
        res = revote(rows, weights)
        replaced = [s for s, v in res.items() if v["decision"] == "replace"]
        contested = sum(1 for v in res.values() if v["contested"])
        print(f"== {name} ==")
        print(f"   replaced: {len(replaced)}/{len(rows)} "
              f"({100*len(replaced)/len(rows):.0f}%)   contested(tie): {contested}")
        for seg in WRONG:
            v = res.get(seg)
            if v:
                mark = "✗ replaced" if v["decision"] == "replace" else "✓ kept"
                print(f"   {seg}: {mark} -> {v['text_effective'][:60]!r}")
        print()


if __name__ == "__main__":
    main()
