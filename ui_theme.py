"""Theme asset loading and chart palette helpers for the Streamlit UI."""

from pathlib import Path

import streamlit as st

STATIC_DIR = Path(__file__).parent / "static"
BASE_CSS_PATH = STATIC_DIR / "style.css"
DARK_CSS_PATH = STATIC_DIR / "theme-dark.css"


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
