# Loro 🦜

A LangGraph dubbing harness for video (ASR and TTS local or cloud, with the core
running on local models). By default it dubs **English into Vietnamese**, but the
target language is now selectable per run (see [Multi-language dubbing](#multi-language-dubbing)):
ASR (Soniox cloud by default, or AssemblyAI cloud, or Nemotron local) → translation
+ vision (Gemma 4 via a model server) → TTS (Soniox cloud — preset voices,
**one voice per speaker** by diarization, the default; or VieNeu/Higgs to clone the
original voice) → video mux (ffmpeg).

## Architecture

```
ingest ─┬─► asr ──► sentence_seg ─┬─► [crosscheck*] ──► voice_ref ──► translate ──► tts ──► fit ──► mux
        └─► vision ───────────────┘
  * `local` ASR engine only; cloud engines (`soniox`/`assemblyai`) skip crosscheck, and voice_ref waits directly on sentence_seg + vision
```

| Node | Role | Backend |
|---|---|---|
| `ingest` | Extract 16 kHz mono audio (for ASR) + 44.1 kHz stereo (for the final mix) | ffmpeg |
| `asr` | Source-language transcript; **engine selected** by `ASR_ENGINE`/`--asr-engine`: `soniox` (default) runs one async `stt-async-v5` job (upload → create → poll → fetch transcript), groups sub-word tokens into words + speaker labels, and caches the result so reruns are not re-billed; `assemblyai` calls cloud `universal-3-pro` once for the whole audio (with word timestamps + speaker labels), also cached; `local` runs Nemotron over overlapping windows (~600 s / 10 s) plus a cross-check ensemble. All three emit a word-with-timestamps stream + raw segments | Soniox stt-async-v5 (cloud, default) **or** AssemblyAI universal-3-pro (cloud) **or** Nemotron 0.6B (subprocess NDJSON, `nemo` venv) |
| `sentence_seg` | Turn the word stream into **complete sentence units** (the backbone of the dub): split on existing punctuation; for long under-punctuated spans, Gemma segments the sentences and timestamps are re-mapped back; fall back to silence-based splitting when Gemma errors — never cuts mid-sentence | Gemma 4 12B via model server (offline fallback) |
| `crosscheck` | **Runs only with the `local` ASR engine** (the cloud engines `soniox`/`assemblyai` skip it — the cloud engine is the single source of truth). Gemma re-listens to each segment, mechanically compares it against Nemotron's text, replaces text on content-word divergence, and flags large divergences as low-confidence (disable: `--no-cross-check`) | Gemma 4 12B via model server |
| `voice_ref` | **Branches on the TTS engine.** Cloning engines (vieneu/higgs): cut the cleanest 3–12 s of the original speaker as a voice-clone reference. Preset engine (soniox): assign each speaker (from diarization) a Soniox preset voice in sorted order + `SONIOX_VOICE_MAP` pins, and save `voice/cast.json` (no audio read) | ffmpeg (clone) / pure casting (preset) |
| `vision` | Sample frames evenly across the video → summarize the scene (who is speaking, tone, topic) for the translator | Gemma 4 12B via model server |
| `translate` | Translate source→target in batches (preserving conversational flow), under a per-slot **length budget**, and emit `transcript.<tag>.srt` (sub-style cues, **timed to the source-language word timestamps** — anchored to when things are actually said rather than spread evenly across each sentence) | Gemma 4 12B via model server |
| `tts` | Synthesize the target speech per segment; the preset engine reads each speaker's cast voice, the cloning engine reads in the original speaker's cloned voice | **Soniox** (cloud REST, preset voices, default) **or** VieNeu-TTS (on-device worker, `vieneu` venv) **or** Higgs Audio v3 4B (sglang-omni server) — chosen via `TTS_ENGINE`/`--tts-engine` |
| `fit` | Clip shorter than its slot → centered with a bounded offset (≤ 0.2 s) so it doesn't end too early; clip overrunning its slot → sped up with `atempo` (capped at ×1.35); assemble the timeline at the correct timestamps | ffmpeg + numpy |
| `mux` | Mix the dub with the original audio (duck to 15% or replace entirely); deliver the target subtitles **three ways**: a toggleable soft-sub (default), a sidecar `.<tag>.srt` next to the video, and an **optional hard burn-in** (`--burn-subs`, re-encodes the video) | ffmpeg |

The `asr → sentence_seg` branch and the `vision` branch run **in parallel** from
`ingest` (LangGraph fan-out/fan-in). With the `local` ASR engine, `crosscheck`
waits on both before `voice_ref`; with a cloud engine (`soniox`/`assemblyai`)
there is no `crosscheck`, so `voice_ref` waits directly on `sentence_seg` + `vision`
(vision still runs to feed scene context into the translation). The dub is built
from **sentence units**, while the subtitles (`transcript.<src>.srt` /
`transcript.<tag>.srt`) are still tiled into short, readable cues.

**Target-language subtitles — timing & delivery.** Target-language cues have no
per-word timestamps, so they used to be spread evenly by word count; now they are
**anchored to the source-language word timeline** (reusing the ASR `state["words"]`),
so cue boundaries follow when things are actually said (e.g. just before a pause
mid-sentence) rather than at an even rate. Subtitles stay on the **original
timeline** (not the post-`fit` timeline); this is a monotonic approximation,
better than even spreading when word gaps are uneven, never frame-accurate. When
there are no word timestamps (local ASR without per-word timing) it falls back to
the old even-spreading behavior.

Target subtitles are delivered **three ways** at once:
- **Soft-sub** embedded in `.<tag>.mp4`, toggleable (default, unchanged behavior).
- **Sidecar `<name>.<tag>.srt`** placed **next to the output video** on every mux
  (including a cache hit) — the `<name>.<lang>.srt` convention players/YouTube
  recognize; keeps the `.<tag>` suffix for both `foo.<tag>.mp4` and `-o bar.mp4`
  (`bar.<tag>.srt`).
- **Optional hard burn-in** `--burn-subs`: writes the subtitles directly into the
  picture for players/platforms that ignore embedded sub tracks. Burning
  **re-encodes the video** (libx264, dropping the fast `-c:v copy` path; HDR/10-bit
  sources are downgraded to 8-bit SDR), so enable it only when needed; the soft-sub
  + sidecar are still produced alongside. Burning needs **an ffmpeg with libass**
  and **a font that covers the target glyphs** — preflight checks both before
  running and fails clearly if the `subtitles` filter or a glyph-covering font is
  missing. Because subtitles follow the original timeline while the dub can drift
  slightly in onset after `fit`, burning pairs best with `duck` mode (you still hear
  the original audio) or short segments.

## Installation

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Configure via environment variables: put them in a `.env` at the project root
(already gitignored) — the CLI **loads it automatically** at startup (via
`python-dotenv`), no `source .env` needed. A variable already exported in the
shell still takes precedence over `.env`, and a CLI flag (e.g. `--asr-engine`)
wins over both. See `.env.example` for the full list.

ASR has three engines, chosen via `ASR_ENGINE` (default `soniox`) or the
`--asr-engine` flag:

- **Soniox (default, cloud)** — runs one async `stt-async-v5` job: upload audio →
  create transcription → poll → fetch token transcript. Soniox returns **sub-word
  tokens** ("Beautiful" → "Beau"/"ti"/"ful"); Loro groups them back into words with
  timestamps + speaker labels and maps them into the same `{segments, words, srt_src}`
  contract as the other engines. The `soniox` ASR engine **shares `SONIOX_API_KEY`**
  with the `soniox` TTS engine (one key for both products) — **no** NeMo/Granite venv
  needed. Audio leaves your machine and is billed per minute, but the transcript is
  cached (`asr/soniox.json`), so rerunning on the same audio does not re-upload or
  re-bill; after the transcript is fetched, the uploaded file and the server-side
  transcription are **deleted best-effort** (less audio retained on Soniox).
  Diarization is on by default and speaker labels feed multi-voice TTS when TTS is
  also `soniox` (via `voice_ref`). Bias domain terms via
  `SONIOX_STT_CONTEXT_TERMS`/`_TEXT` (off by default). Other options:
  `SONIOX_STT_MODEL`, `SONIOX_STT_LANGUAGE_HINTS`, `SONIOX_STT_CLEANUP` (see `.env.example`).
  > **Key-rotation note:** `SONIOX_API_KEY` is shared by STT + TTS, so rotate the key
  > between runs (not mid-run while a run is in flight).
- **AssemblyAI (cloud)** — calls `universal-3-pro` once for the whole audio
  (`--asr-engine assemblyai`), returning text + word timestamps + speaker labels.
  Kept as a first-class alternative (in case Soniox errors or changes pricing).
  Needs only `ASSEMBLYAI_API_KEY` in `.env` (gitignored); **no** NeMo/Granite venv.
  Audio leaves your machine and is billed per minute, but the result is cached
  (`asr/assemblyai.json`), so rerunning on the same audio does not re-upload or
  re-bill. Options: `ASSEMBLYAI_SPEECH_MODELS`, `ASSEMBLYAI_SPEAKER_LABELS`,
  `ASSEMBLYAI_LANGUAGE_DETECTION`/`_CODE` (see `.env.example`).
- **Local (Nemotron + ensemble)** — runs fully offline (`--asr-engine local`),
  **no per-clip cost**. NeMo needs its own virtualenv because `nemo_toolkit` pins
  transformers 4.53.x (conflicting with the `speech` env on ≥5.4):

  ```bash
  pyenv virtualenv 3.11.15 nemo
  ~/.pyenv/versions/nemo/bin/pip install "nemo_toolkit[asr]"
  # or point NEMOTRON_PYTHON at another env that already has nemo
  ```

TTS has four engines, chosen via `TTS_ENGINE` (default `soniox`) or the
`--tts-engine` flag:

- **Soniox (default, cloud)** — calls REST `tts-rt-v1`, returning 24 kHz WAV in
  one of 28 **preset voices** (no cloning). Since it can't clone the original
  voice, each speaker diarization finds is **assigned a preset voice** (rotating
  `SONIOX_VOICE_POOL`, pin per speaker with `SONIOX_VOICE_MAP=A=Adrian,B=Maya`);
  the map is saved to `voice/cast.json`. Single-speaker / no-diarization audio →
  one `SONIOX_DEFAULT_VOICE`. Needs only `SONIOX_API_KEY` in `.env` (gitignored);
  **no** vieneu venv or Higgs server. **Trade-off (this is now the default, changed
  from `vieneu`):** the dub no longer matches the original speaker's voice, and the
  TTS text leaves your machine and is billed per use — clips are still
  content-addressed, so rerunning with unchanged text is not re-billed. **Multi-voice
  needs the `assemblyai` ASR engine** (the source of speaker labels); `soniox` +
  `--asr-engine local` reads the whole video in one default voice. To clone the
  original voice on-device, use `--tts-engine vieneu`. Listen to a real dub before
  trusting the Soniox default.

  Override: `SONIOX_MODEL`, `SONIOX_LANGUAGE`, `SONIOX_SAMPLE_RATE`,
  `SONIOX_VOICE_POOL`, `SONIOX_VOICE_MAP`, `SONIOX_DEFAULT_VOICE` (see `.env.example`;
  voice list: soniox.com/docs/tts/concepts/voices). For non-Vietnamese targets the
  spoken language sent to the engine is derived from the language profile
  (`tts_language_code`), so `--target-lang fr` synthesizes French without touching
  `SONIOX_LANGUAGE` — see [Multi-language dubbing](#multi-language-dubbing).

- **VieNeu-TTS (on-device, clones the original voice)** — runs in its own `vieneu`
  venv because it pulls in onnxruntime + sea-g2p + the MOSS codec (and PyTorch on
  CUDA), heavier than the main env:

  ```bash
  pyenv virtualenv 3.14.5 vieneu
  ~/.pyenv/versions/vieneu/bin/pip install vieneu
  # or point VIENEU_PYTHON at another env that already has vieneu
  ```

  Override: `VIENEU_PYTHON`, `VIENEU_MODEL`, `VIENEU_TEMPERATURE`, `VIENEU_EMOTION`.
  The first run downloads ~0.1B weights from Hugging Face into the cache (network
  needed). The 48 kHz output is resampled by `fit` to `timeline_sr` (default 24000);
  set `timeline_sr=48000` to keep full fidelity. The worker is local, so it reads
  the reference file directly instead of spinning up a temporary HTTP server like
  Higgs. **VieNeu is Vietnamese-only** and is rejected at preflight for any non-VI
  target.

- **Higgs Audio v3 (fallback)** — sglang-omni server, needed only when `TTS_ENGINE=higgs`.

- **Gemini (cloud, preset voices)** — `--tts-engine gemini`, calls REST
  `generateContent`, returning 24 kHz PCM (the client wraps it in WAV) in one of 30
  **preset voices** (no cloning, same preset family as Soniox: assign each speaker a
  voice via `GEMINI_VOICE_POOL` + `GEMINI_VOICE_MAP`, saved to `voice/cast.json`).
  Multi-speaker is **capped at 2 voices per call**. To conserve Gemini's daily/RPM
  quotas, the engine **batches several consecutive segments into ONE multi-speaker
  call, then slices the returned audio into per-segment clips** at the silences; if
  the slicing yields the wrong count or a clip fails QA, it **falls back to
  per-segment calls** — so quality does not depend on a clean cut
  (`GEMINI_BATCH_SEGMENTS`, `GEMINI_BATCH_MAX_SYLLABLES`, `GEMINI_SPLIT_MIN_GAP_MS`
  tune the batching). Needs only `GEMINI_API_KEY` in `.env`. **Data egress:**
  choosing `gemini` sends the dub's translated text to Google's Gemini API, which
  retains it under Google's Generative AI terms — weigh this for sensitive/proprietary
  content. Override: `GEMINI_MODEL`, `GEMINI_SAMPLE_RATE`, `GEMINI_DEFAULT_VOICE`,
  `GEMINI_STYLE_PROMPT` (see `.env.example`; voice list:
  ai.google.dev/gemini-api/docs/speech-generation).

Servers you need running:
- **An OpenAI-compatible model server** (oMLX or llama.cpp) — Loro is just a
  client. Configured under the **`LLM_` namespace**: each role resolves an
  endpoint `(host, api_key, model)`; the base `LLM_HOST`/`LLM_API_KEY`/`LLM_MODEL`
  is the default, and any empty `LLM_*_<ROLE>` inherits the base. Roles: **VISION**
  (vision + seg_visual), **TRANSLATE** (translate + context), **SEG** (sentence
  segmentation), **AUDIO** (cross-check re-listen + preflight probe).
  - **Splitting hosts by role (recommended for a remote router).** A llama.cpp
    router keeps **one model resident at a time**; if several roles hit one host but
    different models, it **load/unload-swaps** on every role change (slow, crash-prone).
    Since vision/translate/segmentation **share one model (26B)**, putting them on the
    remote host keeps the 26B resident **without swapping**; push **audio (12B, needs
    a hearing mmproj)** to an always-hot host (local oMLX). Granite stays a **local
    worker**, not behind a server.

    | Role | Host (field) | Model (field) | Modality |
    |---|---|---|---|
    | Vision/text + translate + segmentation | `LLM_HOST` (base) | `LLM_MODEL` = `unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL` | image + text |
    | Audio re-listen | `LLM_HOST_AUDIO` (e.g. oMLX `http://127.0.0.1:1234/v1`) | `LLM_MODEL_AUDIO` = `gemma-4-12B-it-8bit` | audio |
    | Primary verify (cross-check) | — | `GRANITE_MODEL_ID` | audio (local worker) |

    Cold-loading a large model pays a load cost on the first call; raise `LLM_TIMEOUT`
    (seconds, default 180) if you hit a timeout. Because the gemma-4 models are
    **thinking models**, Loro disables thinking (`enable_thinking=False`) on every call
    so content is never empty. Migrating Granite behind a server is a follow-up; for
    running it natively on a GPU box see
    `docs/plans/2026-06-14-001-feat-cross-platform-cuda-rtx3060-plan.md`.
  - **Single host (Apple Silicon oMLX).** Just set the base
    `LLM_HOST=http://127.0.0.1:1234/v1` + `LLM_MODEL=gemma-4-12B-it-8bit`, leave every
    override empty → all roles share one multimodal Gemma (unchanged behavior).
- **Higgs Audio v3** at `http://localhost:8000` (override: `HIGGS_HOST`) — **only
  when `TTS_ENGINE=higgs`**; the client serves the reference audio over a temporary
  HTTP server, so it works over Tailscale. The default `soniox` engine calls the
  cloud (just needs `SONIOX_API_KEY`); the `vieneu` engine runs on-device — neither
  needs the Higgs server (preflight only probes the selected engine: the Soniox key
  for `soniox`, the Higgs server for `higgs`).

## Running

```bash
.venv/bin/loro input.mp4                          # → input.vi.mp4 (Soniox cloud ASR + Soniox multi-voice preset TTS)
.venv/bin/loro input.mp4 -o out.mp4 --no-vision   # skip the vision agent
.venv/bin/loro input.mp4 --original-audio replace # replace the original audio entirely
.venv/bin/loro input.mp4 --burn-subs              # hard-burn the target subtitles into the picture (re-encode; needs ffmpeg libass)
.venv/bin/loro input.mp4 --tts-engine vieneu      # clone the original voice on-device instead of Soniox
.venv/bin/loro input.mp4 --tts-engine higgs       # clone the original voice via the Higgs server
.venv/bin/loro input.mp4 --tts-engine gemini      # Gemini preset voices, batched to save API calls
.venv/bin/loro input.mp4 --asr-engine assemblyai  # AssemblyAI cloud ASR instead of Soniox
.venv/bin/loro input.mp4 --asr-engine local       # offline ASR (Nemotron + ensemble), no cost
.venv/bin/loro input.mp4 --target-lang fr         # dub into French (see Multi-language dubbing)
.venv/bin/loro input.mp4 --ref-audio voice.wav --ref-text "transcript of the clip"  # clone from a chosen reference (vieneu/higgs)
```

### Artifact-driven harness (resume / retry / report)

Each stage writes a durable artifact into `work/<video-name>/` alongside a
`*.meta.json` sidecar holding its input hashes — **re-running the exact same
command after a crash resumes from where it stopped**, and anything with a valid
artifact is not recomputed. Changing an input (editing a translation, swapping the
video, changing a prompt) invalidates only the affected part.

- **A failed segment is skipped, the run doesn't die**: when retries are exhausted
  the slot keeps the original audio and the reason goes into `skips.json`; re-run
  the same command to retry the skips. A second content failure becomes an
  accepted-skip (no further retry until the input changes). Many same-signature
  failures within a sliding window → early abort (the server is degrading).
- **Hand-editing a translation**: write to `work/<stem>/overrides.json` as
  `{"seg_0012": "the corrected translation"}` and re-run — the override survives
  any re-translation, and only that segment's TTS clip is re-synthesized. The key
  is a segment id `seg_NNNN` (4 digits), matching the id in `report.json` and
  `skips.json`; a key outside the segment range is **skipped with a warning** (never
  misapplied to another line).
- **Upgrading to the sentence backbone (re-segmentation)**: because the dub is now
  built from **sentences** (`sentence_seg`) instead of Nemotron's acoustic units,
  the first run after upgrading will **re-segment and rebuild the cache once** (LLM +
  TTS) — exactly per the fingerprint mechanism (changed input → changed output).
  Segment indices change, so **re-check `overrides.json`**: the same `seg_NNNN` may
  point at a different sentence after re-segmentation (this case isn't auto-detected;
  only out-of-range keys are warned).
- **Reporting**: at the end of every run (including an abort), `report.json` + a
  console summary list the skips, cross-check text replacements, low-confidence
  segments, and per-stage durations.
- **Exit codes**: `0` clean, `2` completed with skips, `3` aborted, `1` fatal.

```bash
.venv/bin/python -m pytest tests/   # unit + integration tests (offline, mocked models)
```

## Multi-language dubbing

Loro's default is English→Vietnamese, but the pipeline is language-agnostic: a
per-language **profile** drives the rate/length model, translation framing, default
TTS engine + voice strategy, subtitle font, and script handling. Vietnamese is the
validated baseline; French and Spanish are wired end-to-end with **best-effort,
calibration-pending** constants.

### Selecting source and target

| Flag | Env var | Default | Notes |
|---|---|---|---|
| `--target-lang` | `TARGET_LANG` | `vi` | BCP-47 tag for the dub target |
| `--source-lang` | `SOURCE_LANG` | `en` | BCP-47 tag for the input language; `auto` enables ASR language detection |
| `--allow-fallback` | `ALLOW_FALLBACK` | off | run an unprofiled target best-effort on the generic profile |

A CLI flag wins over its env var, which wins over the default.

```bash
.venv/bin/loro input.mp4 --target-lang fr                    # English → French
.venv/bin/loro input.mp4 --target-lang es                    # English → Spanish
.venv/bin/loro input.mp4 --source-lang auto --target-lang fr # detect source, dub into French
.venv/bin/loro input.mp4 --target-lang de --allow-fallback   # unprofiled, generic best-effort
```

`--source-lang auto` opts into the ASR engine's language identification (Soniox or
AssemblyAI). This is a **deliberate ASR re-bill**: it flips the detection flag (and
widens hints), changing the ASR fingerprint, so the cached transcript is recomputed.
The `local` (Nemotron) engine cannot detect language, so `--source-lang auto` with
`--asr-engine local` fails preflight — pass an explicit `--source-lang`.
Low-confidence or mixed detection is surfaced as a warning rather than silently
mis-targeting.

### Language profiles

A `LanguageProfile` (in `src/loro/profiles/`) is resolved per target tag (BCP-47)
and supplies, in one place, everything the pipeline used to hardcode for Vietnamese:

- the **rate / length model** (`syllable` for VI, characters-per-second for everything else),
- the **translation framing** (system prompt, source/target labels, context labels),
- the **default TTS engine + voice strategy** (`clone` or `preset`) and preset voice pool,
- the **TTS spoken-language code** (`tts_language_code`) sent to preset cloud engines,
- the **subtitle burn-in font + a representative glyph sample** for the coverage probe,
- the **ISO 639-2 tag** used in the muxed subtitle `language=` metadata, the **script**, and the **segmentation rule**.

Resolution walks a fallback chain (`es-MX` → `es` → generic), so a region/script
variant inherits its base language. Adding a language is **one registry entry in
`src/loro/profiles/data.py`** — no node edits.

**Tier-1 set and status:**

| Profile | Status | Length model | Notes |
|---|---|---|---|
| `vi` (Vietnamese) | validated baseline | syllable (4.3 syl/s) | byte-identical to the historical pipeline; default |
| `fr` (French) | wired end-to-end, **calibration pending** | CPS | CPS/rate/tolerance are seed values, not yet calibrated from real runs |
| `es` (Spanish) | wired end-to-end, **calibration pending** | CPS | CPS/rate/tolerance are seed values, not yet calibrated from real runs |
| `en` (English) | present as source/target | CPS | for EN-target dubs and as the default source |
| generic (`und`) | best-effort fallback | CPS (Latin) | resolved for any unprofiled tag behind `--allow-fallback` |

Vietnamese is the only profile validated end-to-end. **French and Spanish are wired
through every node but their CPS / rate / tolerance constants are seed values pending
empirical calibration from real cloud runs** — treat them as best-effort, not
production-validated.

### Unprofiled targets and `--allow-fallback`

A `--target-lang` with no profile **fails preflight** with a clear message listing
the profiled languages. Pass `--allow-fallback` to run it best-effort on the generic
profile instead — preflight then emits a loud warning that wrong-CPS or
missing-glyph output is possible. This guards against a typo'd target silently
producing bad output after you've already paid for cloud TTS.

### Length model

- **Vietnamese** uses a **syllable budget** (~4.3 syllables/second) — Vietnamese is
  monosyllabic, so this fits the subtitle slot well and minimizes time-stretching.
- **Every other language** uses a **characters-per-second (CPS) budget** at the
  translation step (the pre-synthesis proxy), plus a **post-synthesis measured-duration
  gate**: after a clip is synthesized, its rendered duration is checked against the
  slot, and a clip that can't fit is flagged `length_overflow` in the ledger and
  surfaced in the report (the timeline still muxes; the abort window is not tripped).

The **re-translation escalation** — when `atempo` alone can't fit, shrink the text
and re-synthesize, enabled by the `ENABLE_BUDGET_RETRY` env var (bounded by
`budget_retry_max`) — is **OFF by default** pending tier-1 calibration. Until the
tolerance band is calibrated, an uncalibrated loop could re-bill several TTS calls
per hard segment, so the baseline measured gate (just recording `length_overflow`)
is what runs by default for non-VI targets.

### Voice strategy

Voice strategy is resolved at **preflight**, not at runtime, and cloning is
language-aware:

- **VieNeu is Vietnamese-only** and is rejected at preflight for any non-VI target.
- The **preset cloud engines (Soniox / Gemini)** synthesize directly in the target
  language via the profile's `tts_language_code`, so a single `--target-lang fr`
  makes the engine speak French (no separate `SONIOX_LANGUAGE=fr` needed).
- Unprofiled / best-effort languages default to `preset` so an unvalidated
  cross-lingual clone is never the silent default.

### Subtitles, fonts, and output naming

Output artifacts are **locale-named**: a French run writes `<base>.fr.srt`,
`<base>.fr.mp4`, and muxes the subtitle track with `language=fra` (the profile's ISO
639-2 tag). The Vietnamese default stays `.vi.srt` / `.vi.mp4` / `language=vie`,
byte-identical.

Burn-in (`--burn-subs`) needs a font that covers the target glyphs. The font
**resolution order** is: the profile's font → a font bundled under `assets/fonts/`
(exposed to libass via `fontsdir`) → preflight failure. Preflight trial-renders the
profile's representative glyph sample and fails clearly if neither the profile font
nor a bundled fallback can draw it (presence ≠ coverage — a Latin substitute that
resolves but burns tofu is caught).

A work dir is **single-target by convention**. If a work dir already holds target
subtitles for a different language, switching its target is warned as a stale-override
case (`overrides.json` and cached artifacts may be stale for the new target) — use a
fresh work dir per target language.

### Per-target billing implications

Cloud ASR and TTS bill per minute. Switching the target language, or enabling
`--source-lang auto`, **re-bills**: the artifact cache is content-fingerprinted, so a
changed target text or a flipped detection flag invalidates the affected clips and
they are re-synthesized. The **Vietnamese default run stays byte-identical** at the
fingerprint level and reuses existing caches — so the EN→VI pipeline is never
re-billed by the multi-language refactor.

### Adding a language

Add one entry to the registry in `src/loro/profiles/data.py` (and register its tag
in `src/loro/profiles/__init__.py`): set the rate/length model, CPS, translation
prompt + labels, default TTS engine + voice strategy + preset pool,
`tts_language_code`, font + glyph sample, ISO 639-2 tag, script, and segmentation
rule. No node edits are required.

## Design decisions

- **Soniox preset voices by default, multi-voice by diarization** (changed from
  cloning the original voice): the default dub uses Soniox preset voices rather than
  cloning the original speaker; each speaker AssemblyAI diarizes is assigned a preset
  voice (rotating pool + manual pins), so a multi-speaker video isn't collapsed to one
  voice. `--tts-engine vieneu`/`higgs` restores on-device/server cloning of the
  original voice (just needs a few seconds of reference + transcript, reusing the
  speaker's own longest ASR span).
- **Translate under a length budget**: Vietnamese uses a syllable budget (~4.3 syl/s)
  so the line fits its subtitle slot with less time-stretching; other languages use a
  CPS budget plus a measured-duration gate (see [Length model](#length-model)).
- **Two-tier slot-overflow handling**: allow overflow into the silence before the next
  segment; only when it would overlap the following sentence apply `atempo`, capped at
  ×1.35 to avoid a chipmunk voice.
- **ASR via a subprocess worker** (`src/loro/workers/nemotron_worker.py`): isolates
  NeMo's dependency conflict, communicating over NDJSON on stdout (many files per
  invocation, model loaded once).
- **Runtime-switchable ASR engine** (`ASR_ENGINE=soniox|assemblyai|local`, default
  `soniox`): each engine is a self-contained **provider** in `src/loro/providers/asr/`
  (client/runner + fingerprint contributions + preflight + a `wants_crosscheck` flag),
  resolved through the `providers.asr(name)` registry; the `asr` node keeps only the
  shared parts (writing SRT + returning the `{segments, words, srt_src}` contract), so
  `sentence_seg`/`translate`/`tts`/`fit`/`mux` are unchanged, and **adding an ASR
  engine = one provider module + one registry line**. The cloud engines
  `soniox`/`assemblyai` (audio off-machine + per-minute billing) trade one HTTP call
  for the Nemotron+Granite+Gemma-audio trio; the raw transcript is cached by
  fingerprint (audio sha + engine + the engine's recognition params), so reruns are
  not re-billed. `--asr-engine local` keeps the full offline path.
- **Runtime-switchable TTS engine** (`TTS_ENGINE=soniox|vieneu|higgs|gemini`, default
  `soniox`): each engine is a self-contained **provider** in `src/loro/providers/tts/`
  (build client + fingerprint contributions + chunk budget + preflight + capability
  flags `clones`/`batches`/`native_long_text`), resolved through `providers.tts(name)`;
  `voice_ref`/`tts`/preflight **read capability flags instead of comparing engine
  names**, the graph topology is unchanged, and **adding a TTS engine = one provider
  module + one registry line**. Soniox calls cloud REST (one POST/clip, preset voices
  cast per speaker); VieNeu runs on-device via a *warm* NDJSON worker in the `vieneu`
  venv; Higgs is a cloning server; Gemini calls REST batching several segments. The
  engine + synthesis params live in each clip's fingerprint: the cloning path folds
  the reference (sha + text), the preset path folds **voice/segment** (omitting the
  reference field) — so changing the engine, changing `soniox_language`/`sample_rate`,
  or **re-pinning a speaker** re-synthesizes only the affected clips. **The first run
  on a new fingerprint re-synthesizes every clip once.**

## Reference tools (surveyed)

There's no off-the-shelf LangGraph dubbing harness — the pipelines below are
monolithic apps, but their techniques informed this design:

- [pyvideotrans](https://github.com/jianchang512/pyvideotrans) — modular ASR →
  translate → TTS → sync pipeline; referenced for stage structure and sync handling.
- [VideoLingo](https://aisharenet.com/en/videolingo/) — a 3-step
  "Translate-Reflect-Adaptation" translation flow; a direction for upgrading the
  `translate` node (adding a self-review step).
- [Linly-Dubbing](https://braintitan.medium.com/linly-dubbing-an-open-source-multi-language-ai-dubbing-and-video-translation-tool-2cda94cbf45e)
  — vocal/background separation via Demucs/UVR5; a direction for upgrading `mux`
  (keep clean background music instead of ducking everything).
- [youtube-auto-dub](https://github.com/mangodxd/youtube-auto-dub),
  [Subdub](https://github.com/lukaszliniewicz/Subdub) — audio-video sync strategies
  and boundary correction.
- [VideoAgent (HKUDS)](https://github.com/HKUDS/VideoAgent) — a general
  video-understanding/editing agent, heavier than needed; direct ffmpeg is enough for
  mux/dubbing.

## Roadmap

- [x] Speaker diarization → **multi-voice preset voices** (`soniox` engine): each
  speaker is assigned a Soniox preset voice from `assemblyai` diarization
- [x] Language-agnostic multi-language dubbing (per-language profiles; tier-1 VI +
  best-effort FR/ES); validated for VI, FR/ES calibration pending
- [ ] Speaker diarization → multi-voice **cloning** (per-speaker timbre cloning for
  the `vieneu`/`higgs` engines) — Soniox does multi-voice *preset* already, but
  per-speaker timbre cloning is still open
- [ ] Assign preset voices by **gender** (infer speaker gender to pick a same-gender
  voice — the current pool rotation can mis-cast gender)
- [ ] Demucs background separation, mixing the dub over clean background music
- [ ] A translation reflect/QA step (VideoLingo-style) with Gemma
- [ ] LangGraph checkpointer (SQLite) to resume long pipelines
- [ ] Tier-1 calibration of the FR/ES CPS/rate/tolerance constants from real runs;
  CJK/RTL target support (the profile carries `script`/`segmentation_rule` for it)

## License

[MIT](LICENSE) © 2026 Vinh Nguyen
