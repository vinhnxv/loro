"""CLI: dub an English video into Vietnamese.

    loro input.mp4 [-o output.mp4] [--no-vision] [--no-cross-check]
                   [--no-seg-visual] [--no-summary] [--original-audio duck|replace]
    loro https://www.youtube.com/watch?v=<id> [-o output.mp4] ...

Both local file paths and URLs (http/https) are accepted as the ``video``
positional argument. URL inputs are downloaded via yt-dlp before the
pipeline runs; all yt-dlp-supported platforms work (R9).

Exit codes (R25): 0 completed clean; 2 completed with skipped/accepted-skipped
segments OR placement-layer fit_overflows (a dub clip materially overran its slot
and was trimmed, U4); 3 aborted (systemic failure signature); 1 fatal. Callers
that remediate should branch on report.json (report["fit_overflows"] vs
report["skipped"]), not the exit code alone — retrying does not heal a fit_overflow.
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from loro.config import Config
from loro.graph import build_graph
from loro.harness import report as report_mod
from loro.harness.artifacts import LockError, WorkdirLock
from loro.harness.ledger import AbortRun
from loro.harness.preflight import PreflightError, preflight
from loro.services.ytdl import download as ytdl_download, sanitize_title
from loro.utils.url import is_url, derive_workdir_stem
log = logging.getLogger("loro")


def main() -> None:
    parser = argparse.ArgumentParser(prog="loro", description=__doc__)
    parser.add_argument("video", help="input English video")
    parser.add_argument("-o", "--output", help="output path (default: <video>.vi.mp4)")
    parser.add_argument("-w", "--workdir", help="working directory (default: work/<video-stem>)")
    parser.add_argument("--no-vision", action="store_true",
                        help="skip the visual-context agent")
    parser.add_argument("--no-cross-check", action="store_true",
                        help="skip the ensemble transcript cross-check stage")
    parser.add_argument("--no-seg-visual", action="store_true",
                        help="skip per-shot visual grounding of the translation")
    parser.add_argument("--no-summary", action="store_true",
                        help="skip the running-summary layer of the translation context")
    parser.add_argument("--no-embedded-subs", action="store_true",
                        help="ignore embedded/sidecar subtitles even when present")
    parser.add_argument("--original-audio", choices=["duck", "replace"], default="duck",
                        help="keep original audio quietly under the dub, or replace it")
    parser.add_argument("--burn-subs", action="store_true",
                        help="also hard-burn the target-language subtitles into the video "
                             "picture (re-encodes video); the soft track and locale-named "
                             "sidecar SRT are still produced")
    parser.add_argument("--asr-engine", choices=["soniox", "assemblyai", "local"], default=None,
                        help="ASR backend (default: soniox, or $ASR_ENGINE)")
    parser.add_argument("--tts-engine", choices=["vieneu", "higgs", "soniox", "gemini"], default=None,
                        help="TTS backend (default: soniox, or $TTS_ENGINE)")
    parser.add_argument("--target-lang", default=None,
                        help="target dubbing language as a BCP-47 tag (default: vi, "
                             "or $TARGET_LANG); tier-1: vi, fr, es")
    parser.add_argument("--source-lang", default=None,
                        help="source language BCP-47 tag (default: en, or $SOURCE_LANG); "
                             "'auto' enables Soniox language detection")
    parser.add_argument("--allow-fallback", action="store_true",
                        help="run an unprofiled --target-lang best-effort on the generic "
                             "profile instead of failing preflight")
    parser.add_argument("--ref-audio", help="preset voice-clone reference clip "
                        "(default: auto-extracted from the original speaker)")
    parser.add_argument("--ref-text", help="transcript of --ref-audio")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Root stays at INFO so third-party libraries (urllib3, openai, httpcore, ...)
    # never emit their DEBUG records — at -v those dump base64 audio request/response
    # bodies, which bloats the log to hundreds of KB per line and makes it unusable.
    # -v deepens only loro's own loggers to DEBUG.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("loro").setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # --tts-engine/--asr-engine win over their env vars; when unset, leave each
    # field to its env-reading default_factory rather than overriding it with None.
    engine_override: dict = {}
    if args.tts_engine:
        engine_override["tts_engine"] = args.tts_engine
    if args.asr_engine:
        engine_override["asr_engine"] = args.asr_engine
    # --target-lang/--source-lang win over their env vars; when unset, leave each
    # field to its env-reading default_factory (mirrors the engine flags above).
    if args.target_lang:
        engine_override["target_lang"] = args.target_lang
    if args.source_lang:
        engine_override["source_lang"] = args.source_lang
    if args.allow_fallback:
        engine_override["allow_fallback"] = True
    cfg = Config(
        enable_vision=not args.no_vision,
        enable_cross_check=not args.no_cross_check,
        enable_seg_visual=not args.no_seg_visual,
        enable_summary=not args.no_summary,
        enable_embedded_subs=not args.no_embedded_subs,
        original_audio=args.original_audio,
        subtitle_burn=args.burn_subs,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        **engine_override,
    )
    # --- URL vs file path input (U3) ---
    # When the input is a URL, download the video before preflight so the rest
    # of the pipeline always works on a local file (KTD2). Local file paths
    # continue unchanged (R3).
    video_input = args.video
    download_meta = None
    if is_url(video_input):
        # Derive workdir stem from the URL (KTD3) unless the user overrode -w
        if args.workdir:
            workdir = Path(args.workdir)
        else:
            workdir = cfg.workdir / derive_workdir_stem(video_input)
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            download_meta = ytdl_download(video_input, workdir / "ingest", cfg=cfg)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            sys.exit(1)
        video_path = download_meta["path"]
        log.info("downloaded %s -> %s", video_input, video_path)
    else:
        workdir = Path(args.workdir) if args.workdir else cfg.workdir / Path(video_input).stem
        video_path = video_input

    try:
        preflight(cfg, video_path, workdir)
    except PreflightError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    graph_state = {"video_path": video_path, "workdir": str(workdir)}
    if args.output:
        graph_state["output_path"] = args.output
    elif download_meta is not None:
        # Output naming for URL inputs: derive from the video title or video ID
        # (R7). Falls back to video_id when title is empty.
        title = sanitize_title(download_meta["title"]) or download_meta["video_id"]
        if title:
            graph_state["output_path"] = str(workdir / f"{title}.{cfg.target_lang.lower()}.mp4")

    timings: dict[str, float] = {}
    graph = build_graph(cfg, timings=timings)
    status, abort_info, final = "completed", None, None
    try:
        with WorkdirLock(workdir):
            final = graph.invoke(graph_state, {"recursion_limit": 50})
    except LockError as exc:
        # The workdir belongs to a live run: exit without touching its
        # ledger or overwriting its report
        print(exc, file=sys.stderr)
        sys.exit(1)
    except AbortRun as exc:
        status = "aborted"
        stage, error_class, code = exc.signature
        abort_info = {"signature": {"stage": stage, "class": error_class, "code": code},
                      "count": exc.count}
        log.error("%s", exc)
    except Exception:
        status = "failed"
        log.exception("run failed")

    # The report is written for every outcome, including aborts (R22, R26)
    run_report = report_mod.build_report(workdir, timings, status, abort_info, cfg=cfg)
    report_mod.write_report(workdir, run_report)
    print()
    print(report_mod.console_summary(run_report))
    if final:
        print(f"\nDone: {final['output_path']}")
        print(f"  {cfg.source_lang} subtitles: {final['srt_src']}")
        print(f"  {cfg.target_lang} subtitles: {final['srt_target']}")
        if final.get("srt_sidecar"):
            print(f"  {cfg.target_lang} sidecar (next to video): {final['srt_sidecar']}")
        print(f"  report: {workdir / 'report.json'}")
    sys.exit(report_mod.exit_code(run_report))


def _cli() -> None:
    """Console-script / `python -m loro` entry point.

    Load `.env` into the environment *before* main() builds Config (whose
    field default_factories read os.environ at construction), so a `.env` at
    the project root works without `set -a; source .env`. Deliberately kept out
    of main(): tests and library callers invoke main() directly and must
    control the environment themselves, never having the developer's real
    `.env` pulled into os.environ behind their back. override=False keeps a
    variable already exported in the shell ahead of `.env`, and CLI flags still
    win over both (engine_override in main)."""
    load_dotenv()
    main()


if __name__ == "__main__":
    _cli()
