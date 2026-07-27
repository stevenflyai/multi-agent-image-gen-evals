"""Pricing page data tests.

This page is read for budget decisions, so the tests here guard two things that
have nothing to do with rendering: that no figure is silently invented, and that
the headline claim still follows from the numbers it quotes.
"""

import pytest

import pricing_data as pd
from config import IMAGE_GEN_MODELS

_ALL_ROWS = (
    pd.PER_IMAGE + pd.TOKEN_RATES + pd.BATCH_RATES
    + pd.REGIONAL_TIERS + pd.UNPRICED + pd.NO_PER_IMAGE
)


class TestEveryFigureIsAttributable:
    def test_all_priced_models_are_real_model_keys(self):
        """A typo'd key would render a grey 'other' bar with no label rather
        than failing, so check the join here instead."""
        keys = {row[0] for row in _ALL_ROWS}
        assert keys <= set(IMAGE_GEN_MODELS), f"unknown model keys: {keys - set(IMAGE_GEN_MODELS)}"

    def test_the_four_compared_models_all_appear_somewhere(self):
        """The page promises a comparison of these four; a model must show up
        either with a price or as an explicit gap."""
        shown = {row[0] for row in _ALL_ROWS}
        for key in ("gemini-3-pro", "gpt-image-2", "mai-image-2", "mai-image-2.5"):
            assert key in shown, f"{key} is neither priced nor listed as a gap"

    def test_no_price_is_zero_or_negative(self):
        """A zero would render as a full-width 'free' bar."""
        for key, _, tier, usd, _, _ in pd.PER_IMAGE:
            assert usd > 0, f"{key}/{tier}"
        for key, _, sku, inp, out in pd.TOKEN_RATES:
            assert inp > 0 and out > 0, f"{key}/{sku}"

    def test_unpriced_models_carry_no_number_anywhere(self):
        """The whole point of the gap rows: MAI-Image-2.5 publishes nothing, so
        it must not acquire a price by being added to a rate table."""
        unpriced = {row[0] for row in pd.UNPRICED}
        priced = {row[0] for row in pd.PER_IMAGE} | {row[0] for row in pd.TOKEN_RATES}
        assert not (unpriced & priced), "a model listed as unpriced also has a figure"

    def test_models_without_per_image_have_no_per_image_row(self):
        no_per_image = {row[0] for row in pd.NO_PER_IMAGE}
        assert not (no_per_image & {row[0] for row in pd.PER_IMAGE})

    def test_every_source_is_a_real_url(self):
        assert pd.SOURCES
        for name, url in pd.SOURCES:
            assert name and url.startswith("https://"), (name, url)

    def test_batch_undercuts_and_regional_exceeds_the_standard_rate(self):
        """Batch is the discount tier and DataZone the premium one; if either
        reads the wrong side of standard, the rows have been transposed."""
        standard = {key: rate for key, _, sku, _, rate in pd.TOKEN_RATES if "2e" not in sku}
        for key, _, sku, rate, pct in pd.BATCH_RATES:
            assert rate < standard[key], f"batch {sku} is not cheaper than standard"
            assert abs((1 - rate / standard[key]) - pct) < 0.01, f"{sku} stated discount is wrong"
        for key, _, sku, rate, pct in pd.REGIONAL_TIERS:
            assert rate > standard[key], f"regional {sku} is not dearer than standard"
            assert abs((rate / standard[key] - 1) - pct) < 0.01, f"{sku} stated uplift is wrong"


class TestTheHeadlineClaimFollowsFromTheNumbers:
    """The callout asserts token rate and per-image cost rank the models in
    opposite directions. That is a claim about the data, not a fixed string."""

    def test_inversion_holds_and_is_like_for_like(self):
        inv = pd.inversion()
        assert inv["cheap_rate"] > inv["dear_rate"], "the 'dearer per token' side is not dearer"
        assert inv["cheap_usd"] < inv["dear_usd"], "the 'cheaper per image' side is not cheaper"
        assert inv["cheap_key"] != inv["dear_key"], "the claim compares a model to itself"
        assert inv["rate_ratio"] > 1 and 0 < inv["image_saving_pct"] < 100

    def test_it_raises_rather_than_asserting_something_false(self, monkeypatch):
        """If a vendor cuts a price and the inversion stops holding, the page
        must fail loudly instead of continuing to claim it."""
        flat = [(k, plane, tier, usd if k != "gemini-3-pro" else 0.9, prov, note)
                for k, plane, tier, usd, prov, note in pd.PER_IMAGE]
        monkeypatch.setattr(pd, "PER_IMAGE", flat)
        with pytest.raises(ValueError, match="no longer holds"):
            pd.inversion()

    def test_the_quoted_rows_exist_in_the_tables(self):
        """The sentence cites specific tiers; they have to be rows, not prose."""
        inv = pd.inversion()
        tiers = {(k, tier) for k, _, tier, _, _, _ in pd.PER_IMAGE}
        assert (inv["cheap_key"], inv["cheap_tier"]) in tiers
        assert (inv["dear_key"], inv["dear_tier"]) in tiers


class TestPlanesAreNeverMixed:
    """The defect this schema exists to prevent: an earlier version of the page
    priced Gemini on Google's Developer API, GPT Image 2 on OpenAI direct, and
    MAI on Azure — three commercial agreements, presented as one comparison. The
    numbers looked authoritative and were not comparable."""

    def test_every_row_declares_a_known_plane(self):
        for row in _ALL_ROWS:
            plane = row[1]
            assert plane in pd.PLANES, f"{row[0]} carries unknown plane {plane!r}"

    def test_a_model_never_appears_on_two_planes(self):
        """One model priced on two planes in the same table is the mix, dressed
        up as extra detail."""
        planes = {}
        for row in _ALL_ROWS:
            key, plane = row[0], row[1]
            assert planes.setdefault(key, plane) == plane, (
                f"{key} is priced on both {planes[key]!r} and {plane!r}"
            )

    def test_the_headline_pair_spans_the_two_planes_deliberately(self):
        """The callout compares an Azure model against a Vertex one. That is the
        point — but it only means anything because each side is that vendor's
        own plane, so assert the pair really is cross-plane and labelled."""
        inv = pd.inversion()
        assert inv["cheap_plane"] != inv["dear_plane"]
        assert {inv["cheap_plane"], inv["dear_plane"]} <= set(pd.PLANES)

    def test_derived_figures_reproduce_from_their_own_planes_rate(self):
        """A 'derived' per-image price must be arithmetic on the rate published
        for that same model on that same plane — not carried over from another
        vendor's calculator."""
        for key, plane, tier, usd, prov, note in pd.PER_IMAGE:
            tokens = float(note.split()[0].replace(",", ""))
            rate = pd._rate(key)
            assert abs(tokens / 1e6 * rate - usd) < 0.0006, (
                f"{key}/{tier}: {note} does not reproduce ${usd}"
            )
            assert prov in ("published", "derived")

    def test_azure_side_is_sourced_from_azure(self):
        azure_models = {row[0] for row in _ALL_ROWS if row[1] == "azure"}
        assert azure_models == {"gpt-image-2", "mai-image-2", "mai-image-2.5"}
        assert any("prices.azure.com" in url for _, url in pd.SOURCES)

    def test_google_side_is_sourced_from_vertex_not_the_developer_api(self):
        vertex_models = {row[0] for row in _ALL_ROWS if row[1] == "vertex"}
        assert vertex_models == {"gemini-3-pro"}
        assert any("cloud.google.com/vertex-ai" in url for _, url in pd.SOURCES), \
            "the Google figures must cite Vertex, not ai.google.dev"
        assert not any("ai.google.dev" in url for _, url in pd.SOURCES), \
            "ai.google.dev is the Developer API plane — it must not be a source here"
