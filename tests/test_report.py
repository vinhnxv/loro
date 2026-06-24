import json

from loro.harness import report as rp
from loro.harness.ledger import SkipLedger


def _seed_previous_run(workdir):
    """Durable markers as a previous invocation would have left them."""
    ledger = SkipLedger(workdir)
    ledger.record_failure("seg_0005", "h5", ("tts", "qa", "too_short"), reason="too_short")
    ledger.record_failure("seg_0009", "h9", ("translate", "content", "ValueError"),
                          reason="translate_failed")
    ledger.record_failure("seg_0009", "h9", ("translate", "content", "ValueError"),
                          reason="translate_failed")  # -> accepted_skip

    xdir = workdir / "crosscheck"
    xdir.mkdir(parents=True)
    (xdir / "seg_0002.json").write_text(json.dumps({
        "decision": "replace",
        "source": "vote",
        "winner": "granite+gemma",
        "text_nemotron": "cooper netties cluster",
        "text_granite": "Kubernetes cluster",
        "text_gemma": "Kubernetes cluster",
        "text_effective": "Kubernetes cluster",
        "contested": False,
        "keywords": ["Kubernetes", "cluster"],
        "low_confidence": False,
    }), encoding="utf-8")
    # A tier-1 Granite reading must be ignored by the report
    (xdir / "seg_0002.granite.json").write_text(
        json.dumps({"text": "Kubernetes cluster"}), encoding="utf-8")
    (xdir / "seg_0007.json").write_text(json.dumps({
        "decision": "keep",
        "low_confidence": True,
        "error": "infra/connection",
        "retryable": True,
    }), encoding="utf-8")
    (xdir / "seg_0011.json").write_text(json.dumps({
        "decision": "subtitle",
        "source": "subtitle",
        "winner": "subtitle",
        "text_nemotron": "loose nemotron text",
        "text_effective": "exact subtitle text",
        "sub": {"coverage": 1.0, "align": 0.9},
    }), encoding="utf-8")
    (xdir / "seg_0013.json").write_text(json.dumps({
        "decision": "keep",
        "source": "vote",
        "winner": "nemotron",
        "text_nemotron": "kept text",
        "text_effective": "kept text",
        "sub_rejected": {"coverage": 0.4, "align": 0.2, "reason": "below_align"},
    }), encoding="utf-8")
    # Decoys that the seg_*.json glob also matches but must NOT be counted as
    # verdicts: a tier-1 Granite reading and a fingerprint sidecar
    (xdir / "seg_0050.granite.json").write_text(
        json.dumps({"text": "a granite reading"}), encoding="utf-8")
    (xdir / "seg_0002.json.meta.json").write_text(
        json.dumps({"input_fingerprint": "x", "output_sha256": "y"}), encoding="utf-8")

    vdir = workdir / "vision"
    vdir.mkdir(parents=True)
    (vdir / "context.json").write_text(json.dumps(
        {"context": "", "degraded": True, "reason": "infra/timeout"}), encoding="utf-8")


class TestBuildReport:
    def test_report_built_from_durable_markers_only(self, tmp_path):
        # AE3 (report side) + R26: this "invocation" computed nothing —
        # everything cached — yet the report lists prior skips and changes.
        _seed_previous_run(tmp_path)
        report = rp.build_report(tmp_path, stage_timings={"tts": 0.1})

        assert list(report["skipped"]) == ["seg_0005"]
        assert list(report["accepted_skips"]) == ["seg_0009"]
        repl = report["crosscheck_replacements"][0]
        assert repl["segment"] == "seg_0002"
        assert repl["text_gemma"] == "Kubernetes cluster"
        # R33: all three readings + winning engine are surfaced
        assert repl["winner"] == "granite+gemma"
        assert repl["text_nemotron"] == "cooper netties cluster"
        assert repl["text_granite"] == "Kubernetes cluster"
        assert repl["text_effective"] == "Kubernetes cluster"
        assert report["crosscheck_low_confidence"][0]["segment"] == "seg_0007"
        assert report["vision_degraded"]["reason"] == "infra/timeout"

        # Tier-1 granite reading is not mistaken for a verdict
        assert all(r["segment"] != "seg_0002.granite"
                   for r in report["crosscheck_replacements"])
        # Ensemble tallies + the run's keyword list (R33)
        summary = report["crosscheck_summary"]
        assert summary["tally"]["replace"] == 1
        assert summary["tally"]["subtitle"] == 1
        # seg_0007 + seg_0013 are the only keeps — the .meta.json sidecar and
        # .granite.json reading must not inflate the tally
        assert summary["tally"]["keep"] == 2
        assert summary["by_engine"]["granite+gemma"] == 1
        assert summary["keywords"] == ["Kubernetes", "cluster"]
        # Rejected subtitle is surfaced with its reason
        assert report["crosscheck_sub_rejected"][0]["segment"] == "seg_0013"
        assert report["crosscheck_sub_rejected"][0]["reason"] == "below_align"

    def test_abort_report_carries_window_info(self, tmp_path):
        report = rp.build_report(
            tmp_path, status="aborted",
            abort_info={"signature": ["tts", "infra", "http_503"], "count": 5},
        )
        assert report["status"] == "aborted"
        assert report["abort"]["count"] == 5
        path = rp.write_report(tmp_path, report)
        assert json.loads(path.read_text())["status"] == "aborted"

    def test_empty_workdir_clean_report(self, tmp_path):
        report = rp.build_report(tmp_path)
        assert report["skipped"] == {}
        assert report["crosscheck_replacements"] == []
        assert report["vision_degraded"] is None
        assert report["asr_lid_degraded"] is None

    def test_asr_lid_degraded_surfaced_from_marker(self, tmp_path):
        # U9/R9: a mixed/low-confidence auto-LID marker becomes a top-level field.
        (tmp_path / "asr").mkdir()
        (tmp_path / "asr" / "lid.json").write_text(
            json.dumps({"degraded": True, "detected": "en"}), encoding="utf-8")
        report = rp.build_report(tmp_path)
        assert report["asr_lid_degraded"]["detected"] == "en"


class TestExitCode:
    def test_clean_run_is_zero(self, tmp_path):
        assert rp.exit_code(rp.build_report(tmp_path)) == 0

    def test_skips_give_two(self, tmp_path):
        _seed_previous_run(tmp_path)
        assert rp.exit_code(rp.build_report(tmp_path)) == 2

    def test_abort_gives_three_fatal_gives_one(self, tmp_path):
        assert rp.exit_code(rp.build_report(tmp_path, status="aborted")) == 3
        assert rp.exit_code(rp.build_report(tmp_path, status="failed")) == 1

    def test_fit_overflow_gives_two(self, tmp_path):
        # U4/R3: a placement-layer overrun raises the exit code even on an
        # otherwise-clean run.
        SkipLedger(tmp_path).record_fit_overflow("seg_0019")
        report = rp.build_report(tmp_path)
        assert "seg_0019" in report["fit_overflows"]
        assert rp.exit_code(report) == 2

    def test_length_overflow_alone_stays_exit_zero(self, tmp_path):
        # KTD2: a CPS best-effort length_overflow is surfaced but does NOT change
        # the exit code, unlike fit_overflow.
        SkipLedger(tmp_path).record_length_overflow("seg_0003")
        report = rp.build_report(tmp_path)
        assert report["length_overflows"] and not report["fit_overflows"]
        assert rp.exit_code(report) == 0


class TestConsoleSummary:
    def test_summary_mentions_everything(self, tmp_path):
        _seed_previous_run(tmp_path)
        text = rp.console_summary(rp.build_report(tmp_path, {"asr": 12.0}))
        assert "seg_0005" in text
        assert "seg_0009" in text
        assert "Kubernetes" in text
        assert "Vision degraded" in text
        assert "asr 12s" in text
        assert "Rerun the same command" in text
        # Ensemble attribution: per-engine tally and the rejected subtitle
        assert "Cross-check:" in text
        assert "granite+gemma" in text
        assert "Subtitles rejected" in text

    def test_clean_summary(self, tmp_path):
        text = rp.console_summary(rp.build_report(tmp_path))
        assert "No segments were skipped" in text

    def test_summary_lists_fit_overflow(self, tmp_path):
        SkipLedger(tmp_path).record_fit_overflow("seg_0019")
        text = rp.console_summary(rp.build_report(tmp_path))
        assert "Fit overflow" in text
        assert "seg_0019" in text

    def test_fit_overflow_summary_omits_no_skips_all_clear(self, tmp_path):
        # A fit_overflow-only run exits 2, so the "No segments were skipped."
        # all-clear must not appear — it contradicted the exit code before.
        SkipLedger(tmp_path).record_fit_overflow("seg_0019")
        text = rp.console_summary(rp.build_report(tmp_path))
        assert "Fit overflow" in text
        assert "No segments were skipped" not in text

    def test_summary_mentions_asr_lid_degraded(self, tmp_path):
        (tmp_path / "asr").mkdir()
        (tmp_path / "asr" / "lid.json").write_text(
            json.dumps({"degraded": True, "detected": "en"}), encoding="utf-8")
        text = rp.console_summary(rp.build_report(tmp_path))
        assert "language detection uncertain" in text.lower()
