"""Model pricing comparison page.

Built around one rule: never mix commercial planes. Every figure is Azure
(Consumption, Global) or Google Cloud Vertex AI (Standard), and each row says
which — a comparison that quietly straddles Developer-API, OpenAI-direct and
Azure prices looks rigorous and is worthless.

Layout follows what is actually comparable. The image-output token rate is
published on both planes for every metered model, so it leads. Cost per image
comes second because only Vertex prints one; GPT Image 2's is derived from
Azure's own rate and a published token count, and says so on the row. MAI-Image-2
publishes no token-per-image count and MAI-Image-2.5 publishes nothing at all, so
both render as gaps rather than as estimates.

Bars are plain HTML/CSS to match the architecture pages. Colour comes from
ui_theme.model_color() so a model wears the same hue here as in every chart and
pill elsewhere; that palette is validated for CVD separation and lightness. In
light mode two of the hues fall under 3:1 against the surface, so every bar
carries a visible value label in text ink — identity is never colour-alone.
"""

from html import escape as esc

import streamlit as st

import pricing_data as pd
from config import IMAGE_GEN_MODELS
from i18n import t
from ui_theme import model_color


def _label(model_key: str) -> str:
    info = IMAGE_GEN_MODELS.get(model_key)
    return info["label"] if info else model_key


def _plane_chip(plane: str) -> str:
    """Trailing space is load-bearing: the chip and the model name are adjacent
    inline elements separated only by a CSS margin, so without it a screen
    reader (and any innerText scrape) reads "AzureMAI-Image 2"."""
    return f'<b class="plane {esc(plane)}">{esc(t("price_plane_" + plane))}</b> '


def _bar(model_key: str, plane: str, row_label: str, value: float, vmax: float,
         value_text: str, note: str = "", tag: str = "") -> str:
    """One horizontal bar. The value label is always rendered, never on hover
    only: the light-mode palette does not clear 3:1 against the surface, so the
    number is what carries the reading."""
    pct = max(2.0, (value / vmax) * 100) if vmax else 0
    tip = (f"{_label(model_key)} · {t('price_plane_' + plane)} · {row_label} · "
           f"{value_text}" + (f" — {note}" if note else ""))
    tag_html = f'<b class="derived">{esc(tag)}</b>' if tag else ""
    return (
        f'<div class="price-row" title="{esc(tip)}">'
        f'<span class="price-row-label">{_plane_chip(plane)}{esc(_label(model_key))}'
        f'<em>{esc(row_label)}{tag_html}</em></span>'
        f'<span class="price-track">'
        f'<i style="width:{pct:.2f}%;background:{model_color(model_key)}"></i>'
        f'</span>'
        f'<span class="price-value">{esc(value_text)}</span>'
        f'</div>'
    )


def _empty_bar(model_key: str, plane: str, row_label: str, reason: str) -> str:
    """A model with no published price. Deliberately not a zero-length bar —
    zero would read as 'free'."""
    return (
        f'<div class="price-row unpriced" title="{esc(_label(model_key))} — {esc(reason)}">'
        f'<span class="price-row-label">{_plane_chip(plane)}{esc(_label(model_key))}'
        f'<em>{esc(row_label)}</em></span>'
        f'<span class="price-track empty"><i></i></span>'
        f'<span class="price-value none">{esc(reason)}</span>'
        f'</div>'
    )


def _legend(rows: list[tuple[str, str]], hollow: set[str] | None = None) -> str:
    """Legend for the bars. A model with no published price gets a hollow swatch
    rather than its hue: a filled chip next to no bar reads as a missing bar
    rather than a missing price."""
    hollow = hollow or set()
    seen, out = set(), []
    for key, plane in rows:
        if key in seen:
            continue
        seen.add(key)
        swatch = ('<i class="none"></i>' if key in hollow
                  else f'<i style="background:{model_color(key)}"></i>')
        out.append(f'<span>{swatch}{esc(_label(key))}{_plane_chip(plane)}</span>')
    return f'<div class="price-legend">{"".join(out)}</div>'


def render_pricing_page() -> None:
    """Render the published-price comparison across the image models."""
    inv = pd.inversion()

    # 1. Token rate — the metric both planes publish for every metered model.
    tmax = max(row[4] for row in pd.TOKEN_RATES)
    token_bars = "".join(
        _bar(key, plane, sku, out_rate, tmax, f"${out_rate:,.2f}",
             t("price_input_note", usd=f"{inp:.2f}"))
        for key, plane, sku, inp, out_rate in pd.TOKEN_RATES
    )
    token_bars += "".join(
        _empty_bar(key, plane, name, t("price_preview_unmetered"))
        for key, plane, name, _ in pd.UNPRICED
    )
    token_legend = [(row[0], row[1]) for row in pd.TOKEN_RATES] + \
        [(row[0], row[1]) for row in pd.UNPRICED]

    rate_rows = "".join(
        f'<tr><td class="m"><i style="background:{model_color(key)}"></i>{esc(sku)}'
        f'{_plane_chip(plane)}</td><td>${inp:.2f}</td><td>${out_rate:,.2f}</td></tr>'
        for key, plane, sku, inp, out_rate in pd.TOKEN_RATES
    )
    rate_rows += "".join(
        f'<tr class="unpriced"><td class="m"><i class="none"></i>{esc(name)}'
        f'{_plane_chip(plane)}</td><td colspan="2">{esc(t("price_preview_unmetered"))}</td></tr>'
        for _, plane, name, _ in pd.UNPRICED
    )

    # 2. Per image — only Vertex publishes one; GPT Image 2's is derived.
    vmax = max(row[3] for row in pd.PER_IMAGE)
    per_image_bars = "".join(
        _bar(key, plane, tier, usd, vmax, f"${usd:.3f}".rstrip("0").rstrip("."),
             note, tag=t("price_derived") if prov == "derived" else "")
        for key, plane, tier, usd, prov, note in pd.PER_IMAGE
    )
    per_image_bars += "".join(
        _empty_bar(key, plane, t("price_no_per_image_tier"), t("price_no_per_image"))
        for key, plane, _ in pd.NO_PER_IMAGE
    )
    per_image_bars += "".join(
        _empty_bar(key, plane, t("price_preview_tier"), t("price_preview_unmetered"))
        for key, plane, _, _ in pd.UNPRICED
    )
    hollow = {row[0] for row in pd.NO_PER_IMAGE} | {row[0] for row in pd.UNPRICED}
    per_image_legend = [(row[0], row[1]) for row in pd.PER_IMAGE] + \
        [(row[0], row[1]) for row in pd.NO_PER_IMAGE] + \
        [(row[0], row[1]) for row in pd.UNPRICED]

    tier_rows = "".join(
        f'<tr><td class="m"><i style="background:{model_color(key)}"></i>{esc(sku)}'
        f'{_plane_chip(plane)}</td><td>${rate:,.2f}</td>'
        f'<td>{"−" if kind == "batch" else "+"}{pct * 100:.0f}%</td></tr>'
        for kind, rows in (("batch", pd.BATCH_RATES), ("regional", pd.REGIONAL_TIERS))
        for key, plane, sku, rate, pct in rows
    )

    sources = "".join(
        f'<li><a href="{esc(url)}" target="_blank" rel="noopener">{esc(name)}</a></li>'
        for name, url in pd.SOURCES
    )

    st.markdown(f"""
    <section class="arch-page price-page">
        <div class="arch-hero">
            <div>
                <div class="arch-kicker">{esc(t("nav_pricing"))}</div>
                <h2>{esc(t("price_title"))}</h2>
                <p>{esc(t("price_subtitle", date=pd.RETRIEVED))}</p>
            </div>
        </div>
        <div class="price-flow">
            <div class="price-planes" style="--step:0">
                <span><b class="plane azure">{esc(t("price_plane_azure"))}</b>
                {esc(t("price_plane_azure_detail"))}</span>
                <span><b class="plane vertex">{esc(t("price_plane_vertex"))}</b>
                {esc(t("price_plane_vertex_detail"))}</span>
            </div>
            <div class="price-block" style="--step:1">
                <div class="price-block-head">
                    <h3>{esc(t("price_token_title"))}</h3>
                    <p>{esc(t("price_token_help"))}</p>
                </div>
                {_legend(token_legend, hollow={row[0] for row in pd.UNPRICED})}
                <div class="price-chart">{token_bars}</div>
                <table class="price-table">
                    <thead><tr><th>{esc(t("price_col_sku"))}</th>
                    <th>{esc(t("price_col_input"))}</th>
                    <th>{esc(t("price_col_output"))}</th></tr></thead>
                    <tbody>{rate_rows}</tbody>
                </table>
            </div>
            <div class="price-callout" style="--step:2">
                <strong>{esc(t("price_insight_title"))}</strong>
                <p>{esc(t("price_insight_body",
                          cheap=_label(inv["cheap_key"]), cheap_tier=inv["cheap_tier"],
                          cheap_usd=f"{inv['cheap_usd']:.3f}", cheap_rate=f"{inv['cheap_rate']:.0f}",
                          dear=_label(inv["dear_key"]), dear_tier=inv["dear_tier"],
                          dear_usd=f"{inv['dear_usd']:.3f}", dear_rate=f"{inv['dear_rate']:.0f}",
                          rate_ratio=f"{inv['rate_ratio']:.0f}",
                          saving=f"{inv['image_saving_pct']:.0f}"))}</p>
            </div>
            <div class="price-block" style="--step:3">
                <div class="price-block-head">
                    <h3>{esc(t("price_per_image_title"))}</h3>
                    <p>{esc(t("price_per_image_help"))}</p>
                </div>
                {_legend(per_image_legend, hollow=hollow)}
                <div class="price-chart">{per_image_bars}</div>
            </div>
            <div class="price-block" style="--step:4">
                <div class="price-block-head">
                    <h3>{esc(t("price_tiers_title"))}</h3>
                    <p>{esc(t("price_tiers_help"))}</p>
                </div>
                <table class="price-table narrow">
                    <thead><tr><th>{esc(t("price_col_sku"))}</th>
                    <th>{esc(t("price_col_output"))}</th>
                    <th>{esc(t("price_col_delta"))}</th></tr></thead>
                    <tbody>{tier_rows}</tbody>
                </table>
            </div>
            <div class="price-block gaps" style="--step:5">
                <div class="price-block-head">
                    <h3>{esc(t("price_gaps_title"))}</h3>
                </div>
                <ul class="price-gaps">
                    <li><strong>MAI-Image-2</strong> — {esc(t("price_gap_mai2"))}</li>
                    <li><strong>MAI-Image-2.5</strong> — {esc(t("price_gap_mai25"))}</li>
                    <li><strong>GPT Image 2</strong> — {esc(t("price_gap_gpt"))}</li>
                </ul>
            </div>
            <div class="price-block sources" style="--step:6">
                <div class="price-block-head">
                    <h3>{esc(t("price_sources_title"))}</h3>
                    <p>{esc(t("price_sources_help", date=pd.RETRIEVED))}</p>
                </div>
                <ul class="price-sources">{sources}</ul>
            </div>
        </div>
    </section>
    """, unsafe_allow_html=True)
