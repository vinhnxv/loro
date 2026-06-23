import json
import os
import signal
import subprocess
import sys
import textwrap

import pytest

from loro.harness.ledger import AbortRun, SkipLedger

SIG_TTS_TIMEOUT = ("tts", "infra", "timeout")
SIG_TTS_QA = ("tts", "qa", "too_short")
SIG_TR_PARSE = ("translate", "content", "ValueError")


def _ledger(tmp_path, **kw):
    return SkipLedger(tmp_path, **kw)


class TestLifecycle:
    def test_failure_records_skip_with_reason(self, tmp_path):
        led = _ledger(tmp_path)
        status = led.record_failure("seg_0007", "hashA", SIG_TR_PARSE, reason="translate_failed")
        assert status == "skipped"
        entry = led.entries()["seg_0007"]
        assert entry["status"] == "skipped"
        assert entry["reason"] == "translate_failed"
        assert entry["signature"] == list(SIG_TR_PARSE)

    def test_skipped_segment_is_retried_next_run(self, tmp_path):
        led = _ledger(tmp_path)
        led.record_failure("s1", "hashA", SIG_TTS_QA, reason="qa")
        # Same input on a rerun: R5b says retry it
        led2 = _ledger(tmp_path)
        assert led2.should_attempt("s1", "hashA") is True

    def test_second_content_failure_escalates_to_accepted(self, tmp_path):
        led = _ledger(tmp_path)
        led.record_failure("s1", "hashA", SIG_TTS_QA, reason="qa")
        status = led.record_failure("s1", "hashA", SIG_TTS_QA, reason="qa")
        assert status == "accepted_skip"
        assert led.should_attempt("s1", "hashA") is False

    def test_accepted_skip_resets_when_hash_changes(self, tmp_path):
        led = _ledger(tmp_path)
        led.record_failure("s1", "hashA", SIG_TTS_QA, reason="qa")
        led.record_failure("s1", "hashA", SIG_TTS_QA, reason="qa")
        assert led.should_attempt("s1", "hashB") is True
        # And a failure under the new hash is a *first* strike again
        status = led.record_failure("s1", "hashB", SIG_TTS_QA, reason="qa")
        assert status == "skipped"

    def test_infra_failure_never_escalates_to_accepted(self, tmp_path):
        led = _ledger(tmp_path, abort_threshold=99)  # keep abort out of the way
        led.record_failure("s1", "hashA", SIG_TTS_TIMEOUT, reason="timeout")
        status = led.record_failure("s1", "hashA", SIG_TTS_TIMEOUT, reason="timeout")
        assert status == "skipped"
        assert led.should_attempt("s1", "hashA") is True

    def test_infra_then_content_does_not_escalate(self, tmp_path):
        # An infra skip never contributes to giving up: the first content
        # failure after it is a first strike, not the second (R21)
        led = _ledger(tmp_path, abort_threshold=99)
        led.record_failure("s1", "hashA", SIG_TTS_TIMEOUT, reason="timeout")
        status = led.record_failure("s1", "hashA", SIG_TTS_QA, reason="qa")
        assert status == "skipped"
        # The next content failure does escalate (content after content)
        status = led.record_failure("s1", "hashA", SIG_TTS_QA, reason="qa")
        assert status == "accepted_skip"

    def test_record_ok_with_stage_only_clears_own_entries(self, tmp_path):
        # A cross-check success must not erase a TTS skip for the same segment
        led = _ledger(tmp_path, abort_threshold=99)
        led.record_failure("s1", "hashA", SIG_TTS_QA, reason="qa")
        led.record_ok("s1", stage="crosscheck")
        assert led.entries()["s1"]["status"] == "skipped"  # preserved
        led.record_ok("s1", stage="tts")
        assert "s1" not in led.entries()  # owning stage clears it

    def test_success_clears_entry(self, tmp_path):
        led = _ledger(tmp_path)
        led.record_failure("s1", "hashA", SIG_TTS_QA, reason="qa")
        led.record_ok("s1")
        assert "s1" not in led.entries()


class TestAbortWindow:
    def test_five_same_signature_among_mixed_outcomes_aborts(self, tmp_path):
        led = _ledger(tmp_path, window=20, abort_threshold=5)
        # interleave successes and a different-signature failure
        with pytest.raises(AbortRun):
            for i in range(20):
                seg = f"s{i}"
                if i % 4 == 0:
                    led.record_ok(seg)
                elif i % 7 == 0:
                    led.record_failure(seg, "h", SIG_TR_PARSE, reason="parse")
                else:
                    led.record_failure(seg, "h", SIG_TTS_TIMEOUT, reason="timeout")

    def test_four_same_plus_one_other_does_not_abort(self, tmp_path):
        led = _ledger(tmp_path, window=20, abort_threshold=5)
        for i in range(4):
            led.record_failure(f"a{i}", "h", SIG_TTS_TIMEOUT, reason="timeout")
        led.record_failure("b0", "h", SIG_TR_PARSE, reason="parse")  # different signature
        # No AbortRun raised

    def test_failures_outside_window_do_not_count(self, tmp_path):
        led = _ledger(tmp_path, window=5, abort_threshold=3)
        led.record_failure("a0", "h", SIG_TTS_TIMEOUT, reason="t")
        led.record_failure("a1", "h", SIG_TTS_TIMEOUT, reason="t")
        for i in range(5):  # push them out of the 5-attempt window
            led.record_ok(f"ok{i}")
        led.record_failure("a2", "h", SIG_TTS_TIMEOUT, reason="t")
        led.record_failure("a3", "h", SIG_TTS_TIMEOUT, reason="t")
        # only 2 in window -> no abort; the third triggers
        with pytest.raises(AbortRun):
            led.record_failure("a4", "h", SIG_TTS_TIMEOUT, reason="t")

    def test_strike_only_counts_toward_abort_without_skip(self, tmp_path):
        # Cross-check failures strike the window but never enter skip state (R8/R20)
        led = _ledger(tmp_path, window=20, abort_threshold=3)
        led.record_strike(("crosscheck", "infra", "connection"))
        led.record_strike(("crosscheck", "infra", "connection"))
        assert led.entries() == {}
        with pytest.raises(AbortRun):
            led.record_strike(("crosscheck", "infra", "connection"))

    def test_aborted_window_strikes_demoted_for_next_run(self, tmp_path):
        led = _ledger(tmp_path, window=20, abort_threshold=3)
        with pytest.raises(AbortRun):
            for i in range(3):
                led.record_failure(f"s{i}", "h", SIG_TTS_TIMEOUT, reason="t")
        # Next run reloads the ledger: old strikes must not re-trigger abort
        led2 = _ledger(tmp_path, window=20, abort_threshold=3)
        led2.record_failure("s9", "h", SIG_TTS_TIMEOUT, reason="t")  # no abort
        # And the aborted segments retry clean: a content fail after demotion
        # is a first strike, not an escalation to accepted
        assert led2.should_attempt("s0", "h") is True

    def test_segment_in_aborted_window_does_not_escalate_on_next_fail(self, tmp_path):
        led = _ledger(tmp_path, window=20, abort_threshold=2)
        with pytest.raises(AbortRun):
            led.record_failure("s0", "h", SIG_TTS_QA, reason="qa")
            led.record_failure("s1", "h", SIG_TTS_QA, reason="qa")
        led2 = _ledger(tmp_path, window=20, abort_threshold=2)
        status = led2.record_failure("s0", "h", SIG_TTS_QA, reason="qa")
        assert status == "skipped"  # not accepted_skip: old strike was demoted


class TestDurability:
    def test_ledger_survives_reload(self, tmp_path):
        led = _ledger(tmp_path)
        led.record_failure("s1", "hashA", SIG_TTS_QA, reason="qa")
        led2 = _ledger(tmp_path)
        assert led2.entries()["s1"]["status"] == "skipped"

    def test_file_always_parseable_under_kill(self, tmp_path):
        """Kill a writer loop hard at a random moment; skips.json must parse."""
        script = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/src")!r})
            from loro.harness.ledger import SkipLedger
            led = SkipLedger({str(tmp_path)!r}, abort_threshold=10**9)
            i = 0
            while True:
                led.record_failure(f"s{{i}}", "h", ("tts", "qa", "x"), reason="qa" * 200)
                i += 1
        """)
        proc = subprocess.Popen([sys.executable, "-c", script])
        import time
        time.sleep(1.0)
        proc.send_signal(signal.SIGKILL)
        proc.wait()
        ledger_file = tmp_path / "skips.json"
        assert ledger_file.exists()
        json.loads(ledger_file.read_text())  # must not raise
