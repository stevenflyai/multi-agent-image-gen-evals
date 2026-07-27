"""Theme asset loading and chart palette helpers for the Streamlit UI."""

from pathlib import Path

import streamlit as st

from config import IMAGE_GEN_MODELS

STATIC_DIR = Path(__file__).parent / "static"
BASE_CSS_PATH = STATIC_DIR / "style.css"
DARK_CSS_PATH = STATIC_DIR / "theme-dark.css"

# Categorical palette for per-model identity (donut segments + winner pills).
# Slots are assigned in IMAGE_GEN_MODELS order, so a model keeps its color no
# matter where it ranks or how many models are on screen — color tracks the
# entity, never its rank.
#
# Validated with the dataviz palette checker against both chart surfaces:
#   light (#ffffff): lightness/chroma/CVD pass; worst adjacent dE 24.2.
#     Aqua and yellow fall under 3:1 contrast, so the leaderboard table ships
#     alongside the donut as the required relief.
#   dark (#18181B): lightness/chroma/contrast pass; worst adjacent dE 10.3,
#     the 8-12 floor band, which is legal only with secondary encoding — hence
#     the donut's direct labels and 2px surface gaps between segments.
MODEL_PALETTE_LIGHT = (
    "#2a78d6", "#1baf7a", "#eda100", "#008300",
    "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
)
MODEL_PALETTE_DARK = (
    "#3987e5", "#199e70", "#c98500", "#008300",
    "#9085e9", "#e66767", "#d55181", "#d95926",
)
# Draws and any model past the 8 slots share a neutral — a 9th hue would be
# generated rather than validated, so it folds into "other" instead.
NEUTRAL_LIGHT = "#8a8a99"
NEUTRAL_DARK = "#A1A1AA"

_MODEL_SLOTS = {key: index for index, key in enumerate(IMAGE_GEN_MODELS)}


@st.cache_resource
def _load_css(path: str, css_mtime: float) -> str:
    return Path(path).read_text()


def _inject_css(path: Path) -> None:
    css = _load_css(str(path), path.stat().st_mtime)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def is_dark_theme() -> bool:
    return st.session_state.get("ui_theme", "light") == "dark"


def apply_theme_css() -> None:
    _inject_css(BASE_CSS_PATH)
    if is_dark_theme():
        _inject_css(DARK_CSS_PATH)


def model_color(model_key: str | None) -> str:
    """Stable chart/pill color for a model key. `None` gives the draw neutral."""
    palette = MODEL_PALETTE_DARK if is_dark_theme() else MODEL_PALETTE_LIGHT
    neutral = NEUTRAL_DARK if is_dark_theme() else NEUTRAL_LIGHT
    if model_key is None:
        return neutral
    slot = _MODEL_SLOTS.get(model_key)
    if slot is None or slot >= len(palette):
        return neutral
    return palette[slot]


def plotly_theme() -> dict[str, str]:
    if is_dark_theme():
        return {
            "paper": "#18181B",
            "plot": "#111113",
            "font": "#FAFAFA",
            "grid": "#3F3F46",
        }
    return {
        "paper": "#ffffff",
        "plot": "#f7f9fc",
        "font": "#233047",
        "grid": "#e5e9f2",
    }
