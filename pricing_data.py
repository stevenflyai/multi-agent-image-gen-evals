"""Published prices for the image generation models, with their sources.

Everything here is read off one of two commercial planes, deliberately:

    azure   Azure — Foundry Models / Azure OpenAI Media, Consumption (PAYG),
            Global deployment, commercial regions. Read from the Azure Retail
            Prices API, which returns a separate ~20% higher set of records for
            US Gov regions; those are excluded.
    vertex  Google Cloud — Vertex AI, Standard tier, global endpoint.

Mixing planes is what makes a pricing comparison lie. An earlier version of this
page put Gemini on Google's Developer API, GPT Image 2 on OpenAI direct, and MAI
on Azure — three different commercial agreements, so the numbers were not
comparable even though they looked it. Every row now carries its plane.

Nothing is estimated. Where a figure is arithmetic rather than quoted it is
marked DERIVED and the derivation is stated; where a vendor publishes nothing
the row is None and renders as a visible gap.

Verified on Vertex: Gemini 3 Pro Image's Vertex rates are identical to the
Developer API's on Standard and Flex/Batch (the Developer API additionally has a
Priority tier Vertex does not offer). So the per-image figures below are Vertex's
own published numbers, not carried over from the Developer API page.
"""

RETRIEVED = "2026-07-27"

PLANES: dict[str, str] = {
    "azure": "Azure · Consumption, Global",
    "vertex": "Google Cloud · Vertex AI, Standard",
}

# --- Image output token rate (USD per 1M tokens) -----------------------------
# The one metric both planes publish for every metered model, so this is the
# comparison that is genuinely like-for-like.
# (model_key, plane, sku label, input per 1M, image output per 1M)
TOKEN_RATES: list[tuple[str, str, str, float, float]] = [
    ("mai-image-2", "azure", "MAI-Image-2e", 5.00, 19.50),
    ("gpt-image-2", "azure", "gpt-image-2 (Image 2)", 8.00, 30.00),
    ("mai-image-2", "azure", "MAI-Image-2", 5.00, 33.00),
    ("gemini-3-pro", "vertex", "gemini-3-pro-image", 2.00, 120.00),
]

# --- Cost per generated image (USD) -----------------------------------------
# (model_key, plane, tier label, usd, provenance, note)
#   published — the vendor prints this dollar figure on its own pricing page
#   derived   — computed from that plane's own token rate and a published
#               token-per-image count; the arithmetic is in the note
PER_IMAGE: list[tuple[str, str, str, float, str, str]] = [
    ("gpt-image-2", "azure", "low · 1024²", 0.006, "derived", "200 output tokens x $30/1M"),
    ("gpt-image-2", "azure", "medium · 1024²", 0.053, "derived", "1,767 output tokens x $30/1M"),
    ("gemini-3-pro", "vertex", "1K–2K", 0.134, "published", "1,120 output tokens x $120/1M"),
    ("gpt-image-2", "azure", "high · 1024²", 0.211, "derived", "7,033 output tokens x $30/1M"),
    ("gemini-3-pro", "vertex", "4K", 0.240, "published", "2,000 output tokens x $120/1M"),
]

# --- Batch / flex tier (USD per 1M image output tokens) ----------------------
# (model_key, plane, sku label, image output per 1M, discount vs standard)
BATCH_RATES: list[tuple[str, str, str, float, float]] = [
    ("gpt-image-2", "azure", "gpt-image-2 Batch", 15.00, 0.50),
    ("gemini-3-pro", "vertex", "gemini-3-pro-image Flex/Batch", 60.00, 0.50),
]

# --- Regional tiers, where the plane charges a premium off the global rate ----
# (model_key, plane, sku label, image output per 1M, uplift vs global)
REGIONAL_TIERS: list[tuple[str, str, str, float, float]] = [
    ("gpt-image-2", "azure", "gpt-image-2 · DataZone", 33.00, 0.10),
    ("mai-image-2", "azure", "MAI-Image-2 · DataZone", 36.30, 0.10),
]

# --- Models with no published price -----------------------------------------
UNPRICED: list[tuple[str, str, str, str]] = [
    ("mai-image-2.5", "azure", "MAI-Image-2.5", "preview_unmetered"),
]

# Metered, but the plane publishes no tokens-per-image, so per-image cannot be
# computed without inventing the token count.
NO_PER_IMAGE: list[tuple[str, str, str]] = [
    ("mai-image-2", "azure", "MAI-Image-2"),
]

SOURCES: list[tuple[str, str]] = [
    ("Azure Retail Prices API — productName eq 'Azure OpenAI Media' (gpt-image-2)",
     "https://prices.azure.com/api/retail/prices"),
    ("Azure Retail Prices API — productName eq 'MAI Models'",
     "https://prices.azure.com/api/retail/prices"),
    ("Google Cloud — Vertex AI generative AI pricing",
     "https://cloud.google.com/vertex-ai/generative-ai/pricing"),
    ("Microsoft Learn — models sold directly by Azure (MAI-Image-2.5 preview status)",
     "https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-models/concepts/models-sold-directly-by-azure"),
    ("OpenAI — image generation cost calculator (token counts per image)",
     "https://developers.openai.com/api/docs/guides/image-generation"),
]


# The callout asserts that token rate and per-image cost rank the models in
# OPPOSITE directions. Demonstrating that needs a like-for-like pair — each
# vendor's top quality at a comparable resolution — not the min and max of the
# table, which would just contrast one vendor's cheapest tier with another's
# priciest and prove nothing. These point at the rows the sentence quotes, so
# the claim and its numbers cannot drift apart.
_INSIGHT_LOW = ("gemini-3-pro", "1K–2K")
_INSIGHT_HIGH = ("gpt-image-2", "high · 1024²")


def _row(model_key: str, tier: str) -> tuple[str, str, str, float]:
    for key, plane, row_tier, usd, _, _ in PER_IMAGE:
        if (key, row_tier) == (model_key, tier):
            return key, plane, row_tier, usd
    raise KeyError(f"{model_key} / {tier} not in PER_IMAGE")


def _rate(model_key: str) -> float:
    """Image-output rate per 1M tokens for a model's headline SKU."""
    for key, _, sku, _, out_rate in TOKEN_RATES:
        if key == model_key and "2e" not in sku:
            return out_rate
    raise KeyError(model_key)


def inversion() -> dict:
    """The like-for-like pair where the two rankings disagree.

    Raises if the inversion stops being true after a price update — better a
    loud failure than a page that keeps asserting something the numbers no
    longer support.
    """
    cheap_key, cheap_plane, cheap_tier, cheap_usd = _row(*_INSIGHT_LOW)
    dear_key, dear_plane, dear_tier, dear_usd = _row(*_INSIGHT_HIGH)
    cheap_rate, dear_rate = _rate(cheap_key), _rate(dear_key)
    if not (cheap_rate > dear_rate and cheap_usd < dear_usd):
        raise ValueError(
            "pricing_data.inversion(): the token-rate/per-image inversion no "
            "longer holds — update the page copy instead of shipping a false claim"
        )
    return {
        "cheap_key": cheap_key, "cheap_plane": cheap_plane,
        "cheap_tier": cheap_tier, "cheap_usd": cheap_usd, "cheap_rate": cheap_rate,
        "dear_key": dear_key, "dear_plane": dear_plane,
        "dear_tier": dear_tier, "dear_usd": dear_usd, "dear_rate": dear_rate,
        "rate_ratio": cheap_rate / dear_rate,
        "image_saving_pct": (1 - cheap_usd / dear_usd) * 100,
    }
