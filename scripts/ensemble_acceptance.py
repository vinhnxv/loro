"""U6 acceptance: run the ensemble cross-check on the ai-interview smoke clip
and diff the verdicts against the 39 segments the old 2-way Gemma run replaced.

Reuses the existing Nemotron ASR segments and 16k audio (read-only); writes
all ensemble artifacts into a fresh workdir so the smoke-run output is left
intact. Vision runs first to supply the keyword list (R31).
"""

import json
import re
import sys
import time
from pathlib import Path

_VERDICT_RE = re.compile(r"seg_\d+\.json$")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loro.config import Config
from loro.nodes import crosscheck as xck
from loro.nodes import vision as vision_mod
from loro.state import Segment

SRC = Path("work/ai-interview")
VIDEO = Path.home() / "Downloads/YTDown_YouTube_Ai-Interview-Questions-and-Answers-for-B_Media_uWAwqu2QPiM_001_1080p.mp4"
DST = Path("work/ai-interview-ensemble")

# Segments the smoke run flagged as wrong replacements (must NOT be replaced now)
WRONG_BEFORE = {
    "seg_0053": 'underfitting -> overfitting (reversed meaning)',
    "seg_0119": 'Transfer learning -> LSTM',
    "seg_0041": 'Key concepts -> subsets',
    "seg_0035": 'Unsupervised -> The provided',
}
# Replacements the smoke run got right (ideally still corrected by Granite)
GOOD_BEFORE = {"F1 score", "semi-structured", "23"}


def load_segments() -> list[Segment]:
    data = json.loads((SRC / "asr" / "segments.json").read_text())
    return [Segment(index=i, start=s["start"], end=s["end"], text_en=s["text"].strip())
            for i, s in enumerate(s for s in data["segments"] if s["text"].strip())]


def main() -> None:
    cfg = Config()
    DST.mkdir(parents=True, exist_ok=True)

    # Vision -> keyword list (R31)
    keywords: list[str] = []
    if VIDEO.exists():
        vstate = {"video_path": str(VIDEO), "workdir": str(DST),
                  "video_duration": 838.0}
        t0 = time.monotonic()
        vres = vision_mod.vision(vstate, cfg)
        keywords = vres.get("video_keywords", [])
        print(f"[vision] {time.monotonic()-t0:.0f}s — {len(keywords)} keywords: "
              f"{', '.join(keywords[:20])}")
    else:
        print(f"[vision] video not found at {VIDEO}; running without keywords")

    segments = load_segments()
    state = {"workdir": str(DST), "audio_16k": str(SRC / "ingest" / "audio_16k.wav"),
             "segments": segments, "video_keywords": keywords}

    t0 = time.monotonic()
    xck.crosscheck(state, cfg)
    print(f"[crosscheck] {time.monotonic()-t0:.0f}s over {len(segments)} segments")

    # Tally verdicts
    verdicts = {}
    for path in sorted((DST / "crosscheck").glob("seg_*.json")):
        if not _VERDICT_RE.fullmatch(path.name):
            continue  # skip .granite.json readings and .meta.json sidecars
        verdicts[path.stem] = json.loads(path.read_text())

    replaced = {k: v for k, v in verdicts.items() if v["decision"] == "replace"}
    subtitle = [k for k, v in verdicts.items() if v["decision"] == "subtitle"]
    contested = [k for k, v in verdicts.items() if v.get("contested")]
    low_conf = [k for k, v in verdicts.items() if v.get("low_confidence")]
    by_engine: dict[str, int] = {}
    for v in replaced.values():
        by_engine[v.get("winner", "?")] = by_engine.get(v.get("winner", "?"), 0) + 1

    print("\n==== ENSEMBLE RESULT ====")
    print(f"replaced: {len(replaced)}/{len(verdicts)} "
          f"({100*len(replaced)/max(len(verdicts),1):.0f}%) — smoke run was 39 (29%)")
    print(f"subtitle: {len(subtitle)}  contested(tie): {len(contested)}  "
          f"low_confidence: {len(low_conf)}")
    print(f"winning engine on replaces: {by_engine}")

    print("\n---- regression check: previously-WRONG replacements ----")
    for seg, desc in WRONG_BEFORE.items():
        v = verdicts.get(seg)
        if v is None:
            print(f"  {seg}: MISSING from this run ({desc})")
            continue
        flag = "STILL REPLACED ✗" if v["decision"] == "replace" else "kept ✓"
        print(f"  {seg}: {v['decision']} [{v.get('winner','')}] {flag}")
        print(f"      N: {v.get('text_nemotron','')[:80]}")
        print(f"      G: {v.get('text_granite','')[:80]}")
        if v.get("text_gemma"):
            print(f"      M: {v.get('text_gemma','')[:80]}")
        print(f"      => {v.get('text_effective','')[:80]}")

    print("\n---- all replacements this run ----")
    for seg, v in sorted(replaced.items()):
        print(f"  {seg} [{v.get('winner','')}]: "
              f"{v.get('text_nemotron','')[:55]!r} -> {v.get('text_effective','')[:55]!r}")

    (DST / "acceptance_summary.json").write_text(json.dumps({
        "total": len(verdicts), "replaced": len(replaced),
        "subtitle": len(subtitle), "contested": len(contested),
        "low_confidence": len(low_conf), "by_engine": by_engine,
        "keywords": keywords,
        "wrong_before_now": {k: verdicts.get(k, {}).get("decision", "missing")
                             for k in WRONG_BEFORE},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsummary -> {DST / 'acceptance_summary.json'}")


if __name__ == "__main__":
    main()
