"""Weighted three-way vote (R28/R29/R30): every row of the plan's decision
table has a test here, plus multi-region composition and weight overrides.

Engines: N = Nemotron (pivot, holds timing), G = Granite (primary verify),
M = Gemma (arbiter, only consulted when N and G diverge on a content word).
"""

import pytest

from loro.harness.diff import needs_arbiter, vote3


class TestNeedsArbiter:
    def test_agreement_skips_gemma(self):
        # seg_0053 lesson: when N and G agree, Gemma's 0.3 cannot flip 0.7
        assert needs_arbiter(
            "the model is underfitting the data",
            "the model is underfitting the data",
        ) is False

    def test_agreement_modulo_normalization(self):
        assert needs_arbiter("Hello, World!", "hello world") is False

    def test_stopword_only_divergence_skips_gemma(self):
        assert needs_arbiter("this is the plan", "this is a plan") is False

    def test_content_substitution_needs_gemma(self):
        assert needs_arbiter(
            "we use cooper netties here", "we use Kubernetes here"
        ) is True

    def test_content_insertion_needs_gemma(self):
        assert needs_arbiter(
            "we deploy the model", "we deploy the new model"
        ) is True

    def test_empty_granite_is_suspect_not_arbitrated(self):
        # R30: a suspect Granite reading goes to keep_low_confidence,
        # there is nothing for Gemma to arbitrate
        assert needs_arbiter("a full sentence here", "") is False

    def test_garbage_granite_is_suspect_not_arbitrated(self):
        assert needs_arbiter(
            "the deployment pipeline builds the artifacts and ships them",
            "lorem ipsum dolor sit amet consectetur adipiscing elit quad",
        ) is False

    def test_empty_nemotron_never_arbitrates(self):
        assert needs_arbiter("", "anything at all") is False


class TestVote3DecisionTable:
    def test_n_equals_g_keeps(self):
        # Row: N = G (M not called) -> 0.7 by default -> keep
        r = vote3("plain matching text", "plain matching text")
        assert r["decision"] == "keep"
        assert r["text_effective"] == "plain matching text"

    def test_n_equals_g_keeps_even_with_dissenting_m(self):
        # Same row exercised with all three votes: N=G 0.7 beats M 0.3
        r = vote3(
            "the model is underfitting the data",
            "the model is underfitting the data",
            "the model is overfitting the data",
        )
        assert r["decision"] == "keep"
        assert "underfitting" in r["text_effective"]

    def test_g_and_m_agree_against_n_replaces(self):
        # Row: G = M != N -> 0.8 vs 0.2 -> replace (AE4)
        r = vote3(
            "we use cooper netties to orchestrate the containers",
            "we use Kubernetes to orchestrate the containers",
            "we use Kubernetes to orchestrate the containers",
        )
        assert r["decision"] == "replace"
        assert "Kubernetes" in r["text_effective"]
        assert "cooper" not in r["text_effective"]

    def test_n_and_m_agree_against_g_is_tie_keeps(self):
        # Row: N = M != G -> 0.5 vs 0.5 -> tie keeps Nemotron (R28)
        r = vote3(
            "we apply transfer learning here",
            "we apply transformer learning here",
            "we apply transfer learning here",
        )
        assert r["decision"] == "keep"
        assert "transfer" in r["text_effective"]
        assert r["contested"] is True

    def test_all_three_differ_granite_wins(self):
        # Row: N, G, M all different -> 0.5 strict max -> Granite's reading
        r = vote3(
            "we use cooper netties here",
            "we use Kubernetes here",
            "we use cooper nettles here",
        )
        assert r["decision"] == "replace"
        assert "Kubernetes" in r["text_effective"]

    def test_empty_granite_low_confidence(self):
        # Row: Granite empty/garbage -> keep + low confidence, no vote (R30)
        r = vote3("a full sentence here", "")
        assert r["decision"] == "keep_low_confidence"
        assert r["text_effective"] == "a full sentence here"

    def test_short_granite_low_confidence(self):
        r = vote3("one two three four five six seven eight nine ten", "one")
        assert r["decision"] == "keep_low_confidence"

    def test_garbage_granite_low_confidence(self):
        r = vote3(
            "the deployment pipeline builds the artifacts and ships them",
            "lorem ipsum dolor sit amet consectetur adipiscing elit quad",
            "the deployment pipeline builds the artifacts and ships them",
        )
        assert r["decision"] == "keep_low_confidence"

    def test_stopword_only_region_never_changes(self):
        r = vote3("this is the plan", "this is a plan", "this is a plan")
        assert r["decision"] == "keep"
        assert r["text_effective"] == "this is the plan"


class TestVote3Composition:
    def test_independent_regions_vote_independently(self):
        # Region 1: G=M replace; region 2: N=M tie keeps. Effective text
        # stitches the winners back in pivot order.
        r = vote3(
            "we use cooper netties and train five models",
            "we use Kubernetes and train nine models",
            "we use Kubernetes and train five models",
        )
        assert r["decision"] == "replace"
        assert "Kubernetes" in r["text_effective"]
        assert "five" in r["text_effective"]
        assert "nine" not in r["text_effective"]
        changed = [reg for reg in r["regions"] if reg["changed"]]
        assert len(changed) == 1
        assert r["contested"] is True  # the five/nine region tied

    def test_insertion_region_replaces(self):
        r = vote3(
            "we deploy the model",
            "we deploy the trained model",
            "we deploy the trained model",
        )
        assert r["decision"] == "replace"
        assert "trained" in r["text_effective"]

    def test_gemma_absent_granite_outvotes_nemotron(self):
        # R32: Gemma failed when its vote was needed -> vote among the two
        # remaining readings, Granite 0.5 > Nemotron 0.2
        r = vote3("we use cooper netties here", "we use Kubernetes here", None)
        assert r["decision"] == "replace"
        assert "Kubernetes" in r["text_effective"]

    def test_winner_engine_attribution(self):
        r = vote3(
            "we use cooper netties here",
            "we use Kubernetes here",
            "we use Kubernetes here",
        )
        changed = [reg for reg in r["regions"] if reg["changed"]]
        assert changed and set(changed[0]["winner_engines"]) == {"granite", "gemma"}

    def test_kept_region_attribution_includes_agreeing_engines(self):
        r = vote3(
            "the model is underfitting the data",
            "the model is underfitting the data",
            "the model is overfitting the data",
        )
        region = r["regions"][0]
        assert region["changed"] is False
        assert "nemotron" in region["winner_engines"]
        assert "granite" in region["winner_engines"]

    def test_empty_nemotron_keeps(self):
        r = vote3("", "anything", "anything")
        assert r["decision"] == "keep"
        assert r["text_effective"] == ""


class TestVote3Weights:
    def test_raised_nemotron_weight_blocks_lone_granite(self):
        weights = {"nemotron": 0.6, "granite": 0.5, "gemma": 0.3}
        # Three-way split: Granite 0.5 no longer beats Nemotron 0.6
        r = vote3(
            "we use cooper netties here",
            "we use Kubernetes here",
            "we use cooper nettles here",
            weights=weights,
        )
        assert r["decision"] == "keep"
        assert "cooper netties" in r["text_effective"]

    def test_raised_nemotron_weight_still_loses_to_g_plus_m(self):
        weights = {"nemotron": 0.6, "granite": 0.5, "gemma": 0.3}
        r = vote3(
            "we use cooper netties here",
            "we use Kubernetes here",
            "we use Kubernetes here",
            weights=weights,
        )
        assert r["decision"] == "replace"  # 0.8 > 0.6

    def test_tie_requires_strictly_greater_to_replace(self):
        weights = {"nemotron": 0.25, "granite": 0.5, "gemma": 0.25}
        # N=M: keep side 0.5, G side 0.5 -> keep
        r = vote3(
            "we apply transfer learning",
            "we apply transformer learning",
            "we apply transfer learning",
            weights=weights,
        )
        assert r["decision"] == "keep"
        assert r["contested"] is True
