# Bundled burn-in fonts

Fonts placed here are exposed to libass via the ffmpeg `subtitles` filter's
`fontsdir=` option (wired in `src/loro/nodes/mux.py` and the preflight glyph
probe in `src/loro/harness/preflight.py`). This is the **bundled fallback** in the
burn-in font resolution order (R18):

1. the language profile's `font` (resolved by name through the host's fontconfig);
2. a font bundled in this directory (libass scans it via `fontsdir`);
3. otherwise preflight fails with a clear message.

libass resolves `FontName` through fontconfig only, so a bundled `.ttf`/`.otf`
here is ignored unless `fontsdir` points at it — which the code does whenever this
directory exists.

## What to add

For the Latin tier-1 set (VI/FR/ES), drop a Latin-covering font with full
Vietnamese diacritic coverage here — e.g. **DejaVu Sans** (`DejaVuSans.ttf`) or
**Noto Sans** (`NotoSans-Regular.ttf`), both freely redistributable. The profile
`font` field can then name that family (the tier-1 profiles currently name
`Arial`, which most hosts resolve via fontconfig; the bundled font backs hosts
that lack it).

This directory is intentionally checked in (with this README) so the `fontsdir`
wiring has a stable path; the actual font binary is a packaging asset chosen per
distribution and is not committed here.
