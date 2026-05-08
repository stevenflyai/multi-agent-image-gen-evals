"""Streamlit dashboard for the Image Eval Pipeline.

Dark theme, side-by-side comparison, radar charts with before/after
critique overlay, critique transcript, and raw JSON inspection.
"""

import json
import logging
import re
import shutil
import threading
import time
import base64
from datetime import datetime
from html import escape as esc
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from i18n import dim_label, get_lang, t
from pipeline import PipelineResult, resume_pipeline_from_result, run_pipeline
from schemas import DIMENSIONS, HilAdjudication, HilAdjudicationLabel, HilArbitration, HilDimensionArbitration

logger = logging.getLogger(__name__)
RUNS_DIR = Path("runs")
HISTORY_INDEX_PATH = RUNS_DIR / "index.json"

# --- Page config ---
st.set_page_config(
    page_title="图像生成模型评估流水线" if st.session_state.get("lang", "en") == "zh" else "IMAGE GENERATION Model Evals PIPELINE",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "light"
_query_theme = st.query_params.get("theme")
if _query_theme in {"light", "dark"} and _query_theme != st.session_state.ui_theme:
    st.session_state.ui_theme = _query_theme

# --- Custom CSS (dark theme from design tokens) ---
@st.cache_resource
def _load_css(css_mtime: float) -> str:
    return (Path(__file__).parent / "static" / "style.css").read_text()

_css_path = Path(__file__).parent / "static" / "style.css"
_css = _load_css(_css_path.stat().st_mtime)
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)


def _is_dark_theme() -> bool:
    return st.session_state.get("ui_theme", "light") == "dark"


def _theme_css() -> str:
    if not _is_dark_theme():
        return ""
    return """
    <style>
    :root {
        --bg-app: #0A0A0A;
        --bg-surface: #18181B;
        --bg-subtle: #27272A;
        --border: #27272A;
        --border-strong: #3F3F46;
        --text-primary: #FAFAFA;
        --text-secondary: #D4D4D8;
        --text-tertiary: #A1A1AA;
        --brand: #6366F1;
        --brand-hover: #818CF8;
        --brand-soft: rgba(99,102,241,0.18);
        --model-a: #818CF8;
        --model-a-soft: rgba(99,102,241,0.20);
        --model-b: #F59E0B;
        --model-b-soft: rgba(245,158,11,0.18);
        --success: #22C55E;
        --warning: #F59E0B;
        --danger: #F87171;
        --info: #22D3EE;
        --surface-page: var(--bg-app);
        --surface-card: var(--bg-surface);
        --surface-muted: var(--bg-subtle);
        --surface-soft: #202024;
        --border-soft: var(--border);
        --text-main: var(--text-primary);
        --text-muted: var(--text-secondary);
        --text-soft: var(--text-tertiary);
        --accent-blue: var(--brand);
        --accent-green: var(--success);
        --accent-teal: var(--info);
        --accent-amber: var(--warning);
        --shadow-card: 0 18px 44px rgba(0,0,0,0.34);
        --shadow-soft: 0 10px 28px rgba(0,0,0,0.24);
    }
    .stApp,
    .main .block-container {
        background: var(--bg-app) !important;
        color: var(--text-primary) !important;
    }
    .eval-header,
    .dashboard-page .arch-hero,
    .dashboard-metric,
    .dashboard-cell,
    .dashboard-table,
    .dashboard-table tbody td,
    .dim-card,
    .winner-banner,
    .hil-v2-panel,
    .pipeline-panel,
    .pipeline-tracker,
    .activity-log,
    [data-testid="stExpander"],
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"],
    .score-badge,
    .hil-chip,
    .hil-gate-card,
    .gate2-dispute,
    .gate2-summary-card,
    .critique-step,
    .dim-comment-table tbody td {
        background: var(--bg-surface) !important;
        border-color: var(--border) !important;
        color: var(--text-primary) !important;
        box-shadow: var(--shadow-card);
    }
    [data-testid="stSidebar"] {
        background: var(--bg-surface) !important;
        border-right-color: var(--border) !important;
    }
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] p,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stExpander"] summary p,
    .eval-header h1,
    .dashboard-page .arch-hero h2,
    .dashboard-metric strong,
    .hil-gate-title,
    .hil-chip strong,
    .step-content strong,
    .dim-score-val,
    .score-val {
        color: var(--text-primary) !important;
    }
    .eval-header,
    .dashboard-page .arch-hero,
    .winner-banner {
        background: linear-gradient(135deg, #18181B, #111113) !important;
    }
    .eval-header {
        margin-bottom: 16px !important;
        padding: 12px 0 16px !important;
        border: 0 !important;
        border-bottom: 1px solid var(--border) !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    .pipeline-panel {
        background: var(--bg-surface) !important;
        border-color: var(--border) !important;
        box-shadow: var(--shadow-card);
    }
    .pipeline-tracker {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    .pipeline-status-text {
        background: transparent !important;
        border: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-secondary"],
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: var(--bg-subtle) !important;
        border-color: var(--border-strong) !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stBaseButton-secondary"]:hover,
    .dashboard-table tbody tr:hover td,
    .dim-comment-table tbody tr:hover td {
        background: #323238 !important;
        border-color: #52525B !important;
    }
    .dashboard-header-cell,
    .dashboard-table thead th,
    .dim-comment-table thead th {
        background: var(--bg-subtle) !important;
        border-color: var(--border) !important;
        color: var(--text-secondary) !important;
    }
    .dashboard-cell.stable,
    .eval-header p,
    .dashboard-page .arch-hero p,
    .run-label,
    .empty-state,
    .model-label,
    .dim-name,
    .score-dim,
    .activity-label,
    .activity-time,
    .hil-v2-header p,
    .hil-equivalence-row,
    .step-content,
    .dim-tooltip-text,
    .dashboard-metric span,
    .dashboard-metric em,
    .tbl-dim-cell {
        color: var(--text-secondary) !important;
    }
    .gate2-dispute-title,
    .gate2-summary-card strong {
        color: var(--text-primary) !important;
    }
    .gate2-dispute-meta span,
    .gate2-summary-card p {
        color: var(--text-secondary) !important;
    }
    .image-card,
    [data-testid="stImage"] {
        background: #09090B !important;
        border-color: #27272A !important;
    }
    .model-label,
    .dim-tooltip {
        background: #27272A !important;
        border-color: var(--border-strong) !important;
    }
    .dashboard-winner-pill.gpt,
    .dim-winner-badge.winner-a {
        color: var(--model-a) !important;
        background: var(--model-a-soft) !important;
        border-color: rgba(129,140,248,0.34) !important;
    }
    .dashboard-winner-pill.gemini,
    .dim-winner-badge.winner-b {
        color: var(--model-b) !important;
        background: var(--model-b-soft) !important;
        border-color: rgba(245,158,11,0.34) !important;
    }
    .dashboard-winner-pill.draw,
    .dim-winner-badge.winner-draw {
        color: var(--text-secondary) !important;
        background: var(--bg-subtle) !important;
        border-color: var(--border-strong) !important;
    }
    .dim-score-label.label-a,
    .dim-tooltip-model.tip-a,
    .critique-step.gpt .step-label {
        color: var(--model-a) !important;
    }
    .dim-score-label.label-b,
    .dim-tooltip-model.tip-b {
        color: var(--model-b) !important;
    }
    .critique-step.gpt { background: rgba(99,102,241,0.14) !important; }
    .critique-step.revised { background: rgba(34,197,94,0.10) !important; }
    .critique-step.opus { background: rgba(99,102,241,0.10) !important; }
    .dim-comment-table tbody td:nth-child(2) {
        border-left-color: rgba(129,140,248,0.38) !important;
        background-image: linear-gradient(90deg, rgba(99,102,241,0.12), transparent 130px) !important;
    }
    .dim-comment-table tbody td:nth-child(3) {
        border-left-color: rgba(245,158,11,0.38) !important;
        background-image: linear-gradient(90deg, rgba(245,158,11,0.12), transparent 130px) !important;
    }
    </style>
    """


def _plotly_theme() -> dict[str, str]:
    if _is_dark_theme():
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


_theme_override = _theme_css()
if _theme_override:
    st.markdown(_theme_override, unsafe_allow_html=True)


def _find_reference_image(run_dir: Path) -> Path | None:
    for filename in ("reference_image.jpg", "reference_image.jpeg", "reference_image.png", "reference_image.webp"):
        path = run_dir / filename
        if path.exists():
            return path
    return None


def _load_result_from_dir(run_dir: Path, prompt: str) -> PipelineResult | None:
    """Reconstruct a PipelineResult from saved JSON files."""
    from schemas import (
        ImageEvaluation, CritiqueResponse, RevisedEvaluation, ComparisonResult,
        GateDecision, HilAdjudication, HilArbitration, IssueEquivalenceReport,
    )
    result = PipelineResult()
    result.prompt = prompt
    result.run_dir = run_dir
    result.timestamp = run_dir.name.replace("_", " ")

    result.reference_image_path = _find_reference_image(run_dir)

    # Load images
    gpt_path = run_dir / "gpt_image_2.png"
    gemini_path = run_dir / "gemini_3_pro.png"
    if gpt_path.exists():
        result.image_paths["gpt_image_2"] = gpt_path
    if gemini_path.exists():
        result.image_paths["gemini_3_pro"] = gemini_path

    # Load evaluation
    eval_file = run_dir / "evaluation.json"
    if eval_file.exists():
        data = json.loads(eval_file.read_text())
        result.eval_a = ImageEvaluation(**data["model_a"])
        result.eval_b = ImageEvaluation(**data["model_b"])

    # Load critiques (multi-round: critique_r1.json, critique_r2.json; backward compat: critique.json)
    for r in range(1, 10):
        crit_file = run_dir / f"critique_r{r}.json"
        if crit_file.exists():
            result.critiques.append(CritiqueResponse(**json.loads(crit_file.read_text())))
        else:
            break
    if not result.critiques:
        critique_file = run_dir / "critique.json"
        if critique_file.exists():
            result.critiques.append(CritiqueResponse(**json.loads(critique_file.read_text())))

    # Load revisions (multi-round: revised_r1.json, revised_r2.json; backward compat: revised.json)
    for r in range(1, 10):
        rev_file = run_dir / f"revised_r{r}.json"
        if rev_file.exists():
            result.revisions.append(RevisedEvaluation(**json.loads(rev_file.read_text())))
        else:
            break
    if not result.revisions:
        revised_file = run_dir / "revised.json"
        if revised_file.exists():
            result.revisions.append(RevisedEvaluation(**json.loads(revised_file.read_text())))

    result.rounds_completed = len(result.revisions)

    hil_review_file = run_dir / "hil_review_r1.json"
    if hil_review_file.exists():
        result.hil_reviews.append(HilArbitration(**json.loads(hil_review_file.read_text())))

    hil_adjudication_file = run_dir / "hil_adjudication.json"
    if hil_adjudication_file.exists():
        result.hil_adjudication = HilAdjudication(**json.loads(hil_adjudication_file.read_text()))

    # Load comparison
    comp_file = run_dir / "comparison.json"
    if comp_file.exists():
        result.comparison = ComparisonResult(**json.loads(comp_file.read_text()))

    # Load V2 gate / HIL observability artifacts when present
    summary_file = run_dir / "summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        result.pipeline_status = summary.get("pipeline_status", result.pipeline_status)
        result.requires_attention = summary.get("requires_attention", result.requires_attention)
        result.prompt_difficulty = summary.get("prompt_difficulty", result.prompt_difficulty)

    for gate_file in (run_dir / "gate1_decision.json", run_dir / "gate2_decision.json"):
        if gate_file.exists():
            result.gate_decisions.append(GateDecision(**json.loads(gate_file.read_text())))

    issue_equivalence_file = run_dir / "issue_equivalence.json"
    if issue_equivalence_file.exists():
        result.issue_equivalence = IssueEquivalenceReport(**json.loads(issue_equivalence_file.read_text()))

    return result


def _history_index_entry(run_dir: Path, prompt: str, summary: dict) -> dict:
    return {
        "run_dir": str(run_dir),
        "prompt": prompt,
        "timestamp": summary.get("timestamp") or run_dir.name.replace("_", " "),
        "winner": summary.get("winner"),
        "model_a_mean": summary.get("model_a_mean"),
        "model_b_mean": summary.get("model_b_mean"),
    }


def _rebuild_history_index() -> list[dict]:
    """Build the history index from existing successful run artifacts."""
    entries: list[dict] = []
    if not RUNS_DIR.exists():
        return entries
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        summary_file = run_dir / "summary.json"
        prompt_file = run_dir / "prompt.txt"
        comparison_file = run_dir / "comparison.json"
        if not summary_file.exists() or not prompt_file.exists() or not comparison_file.exists():
            continue
        try:
            summary = json.loads(summary_file.read_text())
            if not summary.get("has_comparison"):
                continue
            entries.append(_history_index_entry(run_dir, prompt_file.read_text().strip(), summary))
        except Exception:
            continue
    entries.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    if entries:
        HISTORY_INDEX_PATH.write_text(json.dumps({"version": 1, "runs": entries}, indent=2))
    return entries


def _load_history_index() -> list[dict]:
    if not HISTORY_INDEX_PATH.exists():
        return _rebuild_history_index()
    try:
        data = json.loads(HISTORY_INDEX_PATH.read_text())
        entries = data.get("runs", []) if isinstance(data, dict) else []
    except json.JSONDecodeError:
        return _rebuild_history_index()
    if not entries:
        return _rebuild_history_index()
    return entries


def _load_indexed_history() -> list[PipelineResult]:
    """Load successful comparisons listed in the history index."""
    runs: list[PipelineResult] = []
    seen: set[str] = set()
    for entry in _load_history_index():
        run_dir_value = entry.get("run_dir")
        prompt = entry.get("prompt", "")
        if not run_dir_value or not prompt:
            continue
        run_dir = Path(run_dir_value)
        if not run_dir.exists():
            continue
        try:
            result = _load_result_from_dir(run_dir, prompt)
        except Exception:
            continue
        if not result or not result.comparison:
            continue
        run_key = _run_key(result) if "_run_key" in globals() else str(run_dir)
        if run_key in seen:
            continue
        seen.add(run_key)
        runs.append(result)
    return runs


# --- Session state init ---
if "runs" not in st.session_state:
    st.session_state.runs = []  # List of PipelineResult
if "running" not in st.session_state:
    st.session_state.running = False
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "focused_run_idx" not in st.session_state:
    st.session_state.focused_run_idx = None
if "focused_run_key" not in st.session_state:
    st.session_state.focused_run_key = None
if "loaded_prebaked_run_key" not in st.session_state:
    st.session_state.loaded_prebaked_run_key = None
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "results"
if st.session_state.view_mode == "architecture":
    st.session_state.view_mode = "architecture_v1"
if "run_progress_snapshots" not in st.session_state:
    st.session_state.run_progress_snapshots = {}
if "reviewer" not in st.session_state:
    st.session_state.reviewer = "local_reviewer"
if "show_reference_uploader" not in st.session_state:
    st.session_state.show_reference_uploader = False
if "reference_upload_bytes" not in st.session_state:
    st.session_state.reference_upload_bytes = None
if "reference_upload_name" not in st.session_state:
    st.session_state.reference_upload_name = None
if "reference_upload_type" not in st.session_state:
    st.session_state.reference_upload_type = None
if "history_loaded" not in st.session_state:
    indexed_runs = _load_indexed_history()
    merged_runs: list[PipelineResult] = []
    seen_run_keys: set[str] = set()
    for run in indexed_runs + st.session_state.runs:
        run_key = str(run.run_dir) if run.run_dir else f"{run.timestamp}|{run.prompt}"
        if run_key in seen_run_keys:
            continue
        seen_run_keys.add(run_key)
        merged_runs.append(run)
    st.session_state.runs = merged_runs
    if st.session_state.runs and st.session_state.focused_run_key is None:
        first_run = st.session_state.runs[0]
        st.session_state.focused_run_key = str(first_run.run_dir) if first_run.run_dir else f"{first_run.timestamp}|{first_run.prompt}"
    st.session_state.history_loaded = True

HISTORY_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "with", "without", "of", "for", "to", "in", "on", "at", "by", "from",
    "into", "inside", "outside", "over", "under", "about", "as", "is", "are", "be", "being", "create", "generate",
    "make", "image", "photo", "photograph", "picture", "scene", "showing", "depicting", "featuring", "style",
    "sample", "page", "editorial", "magazine", "colorized", "glossy", "smooth", "well", "laid", "distributed",
    "our", "main", "character",
}
HISTORY_ACRONYMS = {"ai": "AI", "gpt": "GPT", "ubc": "UBC", "usa": "USA", "uk": "UK", "nyc": "NYC"}


def _parse_run_datetime(timestamp: str) -> datetime | None:
    for fmt in (None, "%Y%m%d %H%M%S %f", "%Y%m%d %H%M%S", "%Y%m%d_%H%M%S_%f", "%Y%m%d_%H%M%S"):
        try:
            if fmt is None:
                return datetime.fromisoformat(timestamp)
            return datetime.strptime(timestamp, fmt)
        except ValueError:
            continue
    return None


def _history_date_label(timestamp: str) -> str:
    parsed = _parse_run_datetime(timestamp)
    if parsed:
        return parsed.strftime("%Y-%m%d")
    return timestamp[:10].replace("_", "-")


def _history_keywords(prompt: str, max_words: int = 2) -> str:
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", prompt)
    keywords: list[str] = []
    for word in words:
        normalized = word.lower().strip("-'")
        if not normalized or normalized in HISTORY_STOPWORDS or normalized.isdigit():
            continue
        if re.fullmatch(r"\d{2,4}", normalized):
            continue
        keywords.append(HISTORY_ACRONYMS.get(normalized, normalized))
        if len(keywords) >= max_words:
            break
    if keywords:
        return " ".join(keywords)
    fallback = re.sub(r"\s+", " ", prompt).strip()
    return fallback[:14] + "..." if len(fallback) > 14 else fallback


def _history_sort_key(item: tuple[int, PipelineResult]) -> datetime:
    parsed = _parse_run_datetime(item[1].timestamp)
    return parsed or datetime.min


def _run_key(run: PipelineResult) -> str:
    if run.run_dir:
        return str(run.run_dir)
    return f"{run.timestamp}|{run.prompt}"


def _remember_run(run: PipelineResult) -> None:
    run_key = _run_key(run)
    st.session_state.runs = [existing for existing in st.session_state.runs if _run_key(existing) != run_key]
    st.session_state.runs.insert(0, run)
    st.session_state.focused_run_key = run_key
    st.session_state.focused_run_idx = None
    st.session_state.view_mode = "results"


def _forget_run(run_key: str) -> None:
    st.session_state.runs = [existing for existing in st.session_state.runs if _run_key(existing) != run_key]
    if st.session_state.focused_run_key == run_key:
        st.session_state.focused_run_key = _run_key(st.session_state.runs[0]) if st.session_state.runs else None
        st.session_state.focused_run_idx = None


def _delete_run_record(run: PipelineResult) -> None:
    """Delete one selected run directory and remove only that run from local indexes."""
    run_key = _run_key(run)
    if run.run_dir:
        run_dir = run.run_dir.resolve()
        runs_root = RUNS_DIR.resolve()
        if run_dir == runs_root or runs_root not in run_dir.parents:
            raise ValueError(f"Refusing to delete path outside runs/: {run.run_dir}")
        if run_dir.exists():
            shutil.rmtree(run_dir)

    _forget_run(run_key)

    if HISTORY_INDEX_PATH.exists():
        try:
            data = json.loads(HISTORY_INDEX_PATH.read_text())
            entries = data.get("runs", []) if isinstance(data, dict) else []
            entries = [entry for entry in entries if str(Path(entry.get("run_dir", ""))) != run_key]
            HISTORY_INDEX_PATH.write_text(json.dumps({"version": 1, "runs": entries}, indent=2))
        except Exception:
            _rebuild_history_index()


def _save_progress_snapshot(result: PipelineResult, progress_html: str, activity_html: str) -> None:
    if not result.run_dir:
        return
    try:
        (result.run_dir / "activity_snapshot.json").write_text(
            json.dumps({"progress": progress_html, "activity": activity_html}, indent=2)
        )
    except Exception:
        logger.exception("Failed to save activity snapshot")


def _load_progress_snapshot(result: PipelineResult) -> dict | None:
    if not result.run_dir:
        return None
    snapshot_file = result.run_dir / "activity_snapshot.json"
    if not snapshot_file.exists():
        return None
    try:
        data = json.loads(snapshot_file.read_text())
    except Exception:
        return None
    if not isinstance(data, dict) or not data.get("progress"):
        return None
    return data


def _display_error(error: str) -> str:
    text = re.sub(r"\s+", " ", str(error)).strip()
    if not text:
        return t("error_details_unavailable")
    if re.search(r":\s*$", text):
        return f"{text} {t('error_details_unavailable')}"
    return text


def _display_errors(errors: list[str]) -> str:
    return "; ".join(_display_error(error) for error in errors)


def _attachment_preview_html(image_bytes: bytes, name: str | None, media_type: str | None) -> str:
    safe_media_type = media_type if media_type and media_type.startswith("image/") else "image/jpeg"
    encoded = base64.b64encode(image_bytes).decode()
    data_url = f"data:{safe_media_type};base64,{encoded}"
    label = esc(name or t("reference_image_preview"))
    return (
        '<div class="attachment-preview-row">'
        f'<a class="attachment-thumb-link" href="{data_url}" target="_blank" title="{esc(t("attachment_click_to_enlarge"))}">'
        f'<img src="{data_url}" alt="{label}" />'
        '</a>'
        '<div class="attachment-meta">'
        f'<strong>{label}</strong>'
        f'<span>{esc(t("attachment_click_to_enlarge"))}</span>'
        '</div>'
        '</div>'
    )


def _focused_run_entry() -> tuple[int, PipelineResult] | None:
    if not st.session_state.runs:
        return None
    focused_key = st.session_state.focused_run_key
    for idx, run in enumerate(st.session_state.runs):
        if focused_key == _run_key(run):
            return idx, run
    return sorted(enumerate(st.session_state.runs), key=_history_sort_key, reverse=True)[0]


def _time_ago(iso_ts: str) -> str:
    try:
        from datetime import datetime
        delta = datetime.now() - datetime.fromisoformat(iso_ts)
        secs = int(delta.total_seconds())
        if secs < 60:
            return t("history_just_now")
        if secs < 3600:
            return t("history_minutes_ago", n=secs // 60)
        return t("history_hours_ago", n=secs // 3600)
    except Exception:
        return iso_ts[:16]


# --- Sidebar navigation and history ---
st.sidebar.markdown(f'<div class="sidebar-product">{esc(t("sidebar_product"))}</div>', unsafe_allow_html=True)
st.sidebar.markdown(f'<div class="sidebar-section-label">{esc(t("sidebar_navigation"))}</div>', unsafe_allow_html=True)
_home_active = st.session_state.view_mode == "results"
if st.sidebar.button(
    f"{'▶ ' if _home_active else ''}{t('nav_home')}",
    key="nav_home",
    icon=":material/home:",
    use_container_width=True,
):
    st.session_state.view_mode = "results"
    st.rerun()

_dashboard_active = st.session_state.view_mode == "dashboard"
if st.sidebar.button(
    f"{'▶ ' if _dashboard_active else ''}{t('nav_dashboard')}",
    key="nav_dashboard",
    icon=":material/dashboard:",
    use_container_width=True,
):
    st.session_state.view_mode = "dashboard"
    st.rerun()

_arch_v1_active = st.session_state.view_mode == "architecture_v1"
if st.sidebar.button(
    f"{'▶ ' if _arch_v1_active else ''}{t('nav_architecture_v1')}",
    key="nav_architecture_v1",
    use_container_width=True,
):
    st.session_state.view_mode = "architecture_v1"
    st.rerun()

_arch_v2_active = st.session_state.view_mode == "architecture_v2"
if st.sidebar.button(
    f"{'▶ ' if _arch_v2_active else ''}{t('nav_architecture_v2')}",
    key="nav_architecture_v2",
    use_container_width=True,
):
    st.session_state.view_mode = "architecture_v2"
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown(f'<div class="sidebar-section-label">{esc(t("history_title"))}</div>', unsafe_allow_html=True)

_sidebar_runs = sorted(
    enumerate(st.session_state.runs),
    key=_history_sort_key,
    reverse=True,
)
if not _sidebar_runs:
    st.sidebar.caption(t("history_empty"))
else:
    for _idx, _run in _sidebar_runs:
        if _run.comparison:
            _w = _run.comparison.overall_winner
            if _w == "model_a":
                _badge = f"● {t('history_winner_a')} ✓"
                _history_winner_cls = "gpt"
            elif _w == "model_b":
                _badge = f"● {t('history_winner_b')} ✓"
                _history_winner_cls = "gemini"
            else:
                _badge = t("draw_label")
                _history_winner_cls = "draw"
        else:
            _badge = "—"
            _history_winner_cls = "none"
        _history_title = f"{_history_date_label(_run.timestamp)}: {_history_keywords(_run.prompt)}"
        _run_key_value = _run_key(_run)
        _is_focused = st.session_state.focused_run_key == _run_key_value
        _btn_label = f"{'▶ ' if _is_focused else ''}{_history_title}\n{_badge}"
        if st.sidebar.button(_btn_label, key=f"hist_{_history_winner_cls}_{_run_key_value}", use_container_width=True):
            st.session_state.focused_run_key = _run_key_value
            st.session_state.focused_run_idx = None
            st.session_state.view_mode = "results"
            st.rerun()

st.sidebar.divider()
st.sidebar.markdown(f'<div class="sidebar-section-label">{esc(t("sidebar_settings"))}</div>', unsafe_allow_html=True)
_lang_options = {"English": "en", "中文": "zh"}
_selected_lang = st.sidebar.selectbox(
    t("lang_label"),
    list(_lang_options.keys()),
    index=list(_lang_options.values()).index(get_lang()),
)
if _lang_options[_selected_lang] != st.session_state.lang:
    st.session_state.lang = _lang_options[_selected_lang]
    st.rerun()

_theme_options = {t("theme_light"): "light", t("theme_dark"): "dark"}
_current_theme = st.session_state.get("ui_theme", "light")
_selected_theme = st.sidebar.selectbox(
    t("theme_label"),
    list(_theme_options.keys()),
    index=list(_theme_options.values()).index(_current_theme),
)
if _theme_options[_selected_theme] != st.session_state.ui_theme:
    st.session_state.ui_theme = _theme_options[_selected_theme]
    st.query_params["theme"] = st.session_state.ui_theme
    st.rerun()


# --- Pre-baked results ---
@st.cache_data(ttl=60)
def load_prebaked_results() -> dict[str, PipelineResult]:
    """Load pre-baked results from runs/ directory."""
    results = {}
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return results
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        summary_file = run_dir / "summary.json"
        prompt_file = run_dir / "prompt.txt"
        if not summary_file.exists() or not prompt_file.exists():
            continue
        prompt = prompt_file.read_text().strip()
        try:
            result = _load_result_from_dir(run_dir, prompt)
            if result and result.comparison:
                results[str(run_dir)] = result
        except Exception:
            continue
    return results


def _prebaked_option_label(run_key: str, results: dict[str, PipelineResult]) -> str:
    if run_key == "":
        return t("select_example_placeholder")
    result = results.get(run_key)
    if not result:
        return run_key
    prefix = _history_date_label(result.timestamp)
    prompt = re.sub(r"\s+", " ", result.prompt).strip()
    return f"{prefix}: {prompt[:76]}" if len(prompt) > 76 else f"{prefix}: {prompt}"


CATEGORY_KEYWORDS = {
    "people": ["person", "people", "portrait", "man", "woman", "girl", "boy", "character", "model", "人物", "人像", "男人", "女人", "角色"],
    "nature": ["nature", "forest", "mountain", "river", "lake", "ocean", "garden", "landscape", "sunset", "自然", "森林", "山", "河", "湖", "海", "花园", "风景"],
    "magazine": ["magazine", "editorial", "cover", "layout", "spread", "book", "杂志", "封面", "版式", "书"],
    "animals": ["animal", "dog", "cat", "wolf", "bird", "horse", "retriever", "pet", "动物", "狗", "猫", "狼", "鸟", "马", "宠物"],
    "architecture": ["architecture", "building", "house", "interior", "city", "street", "temple", "建筑", "房子", "室内", "城市", "街道", "寺庙"],
    "food": ["food", "meal", "dish", "restaurant", "coffee", "tea", "cake", "食物", "餐", "咖啡", "茶", "蛋糕"],
    "product": ["product", "packaging", "bottle", "watch", "shoe", "device", "产品", "包装", "瓶", "手表", "鞋", "设备"],
    "ui": ["ui", "interface", "dashboard", "chart", "diagram", "app", "website", "界面", "图表", "仪表盘", "网站", "应用"],
    "fantasy": ["fantasy", "dragon", "magic", "sci-fi", "cyberpunk", "anime", "shonen", "幻想", "魔法", "科幻", "动漫"],
}


def _prompt_category(prompt: str) -> str:
    lowered = prompt.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return "other"


def _category_label(category: str) -> str:
    return t(f"category_{category}")


def _winner_label(winner: str) -> str:
    if winner == "model_a":
        return t("history_winner_a")
    if winner == "model_b":
        return t("history_winner_b")
    return t("draw_label")


def _dashboard_winner_badge(winner: str) -> str:
    if winner == "model_a":
        return f'<span class="dashboard-winner-pill gpt">● {esc(t("history_winner_a"))}</span>'
    if winner == "model_b":
        return f'<span class="dashboard-winner-pill gemini">● {esc(t("history_winner_b"))}</span>'
    return f'<span class="dashboard-winner-pill draw">{esc(t("draw_label"))}</span>'


def _dashboard_sort_button(column_key: str, label: str, *, default_ascending: bool) -> None:
    sort_by = st.session_state.setdefault("dashboard_sort_by", "date")
    sort_ascending = st.session_state.setdefault("dashboard_sort_ascending", False)
    active = sort_by == column_key
    direction = "↑" if sort_ascending else "↓"
    button_label = f"{label} {direction if active else ''}".strip()
    if st.button(button_label, key=f"dashboard_sort_{column_key}", use_container_width=True):
        if active:
            st.session_state.dashboard_sort_ascending = not sort_ascending
        else:
            st.session_state.dashboard_sort_by = column_key
            st.session_state.dashboard_sort_ascending = default_ascending
        st.rerun()


def _render_dashboard_recent_runs(runs: list[PipelineResult]) -> None:
    dashboard_columns = [1.15, 1.65, 6.85, 1.0, 0.9, 1.2]
    rows = []
    for run in runs:
        comp = run.comparison
        parsed_date = _parse_run_datetime(run.timestamp) or datetime.min
        category = _category_label(_prompt_category(run.prompt))
        rows.append({
            "date_dt": parsed_date,
            "date": _history_date_label(run.timestamp),
            "category": category,
            "prompt": run.prompt,
            "winner": comp.overall_winner,
            "margin": comp.margin or "—",
            "score": f"{comp.model_a_mean:.2f} / {comp.model_b_mean:.2f}",
        })

    sort_by = st.session_state.setdefault("dashboard_sort_by", "date")
    sort_ascending = st.session_state.setdefault("dashboard_sort_ascending", False)
    sort_key = "date_dt" if sort_by == "date" else "category"
    rows = sorted(rows, key=lambda row: row[sort_key], reverse=not sort_ascending)

    st.markdown(f'<div class="dashboard-table-title">{esc(t("dashboard_recent_runs"))}</div>', unsafe_allow_html=True)
    header_cols = st.columns(dashboard_columns, gap="small")
    with header_cols[0]:
        _dashboard_sort_button("date", t("dashboard_date"), default_ascending=False)
    with header_cols[1]:
        _dashboard_sort_button("category", t("dashboard_category"), default_ascending=True)
    with header_cols[2]:
        st.markdown(f'<div class="dashboard-header-cell">{esc(t("dashboard_prompt"))}</div>', unsafe_allow_html=True)
    with header_cols[3]:
        st.markdown(f'<div class="dashboard-header-cell">{esc(t("dashboard_winner"))}</div>', unsafe_allow_html=True)
    with header_cols[4]:
        st.markdown(f'<div class="dashboard-header-cell">{esc(t("dashboard_margin"))}</div>', unsafe_allow_html=True)
    with header_cols[5]:
        st.markdown(f'<div class="dashboard-header-cell">{esc(t("dashboard_score"))}</div>', unsafe_allow_html=True)

    st.markdown('<div class="dashboard-row-separator"></div>', unsafe_allow_html=True)
    for row in rows[:20]:
        cols = st.columns(dashboard_columns, gap="small")
        with cols[0]:
            st.markdown(f'<div class="dashboard-cell stable">{esc(row["date"])}</div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f'<div class="dashboard-cell stable category">{esc(row["category"])}</div>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f'<div class="dashboard-cell prompt">{esc(row["prompt"])}</div>', unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f'<div class="dashboard-cell">{_dashboard_winner_badge(row["winner"])}</div>', unsafe_allow_html=True)
        with cols[4]:
            st.markdown(f'<div class="dashboard-cell stable">{esc(str(row["margin"]))}</div>', unsafe_allow_html=True)
        with cols[5]:
            st.markdown(f'<div class="dashboard-cell stable score">{esc(row["score"])}</div>', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-row-separator"></div>', unsafe_allow_html=True)


def _dashboard_metric(label: str, value: str, subtext: str = "") -> str:
    return (
        f'<div class="dashboard-metric">'
        f'<span>{esc(label)}</span>'
        f'<strong>{esc(value)}</strong>'
        f'<em>{esc(subtext)}</em>'
        f'</div>'
    )


def render_dashboard_page() -> None:
    """Render aggregate analytics across completed comparison runs."""
    runs = [run for run in st.session_state.runs if run.comparison]
    st.markdown(f"""
    <section class="dashboard-page">
        <div class="arch-hero">
            <div class="arch-kicker">{esc(t("nav_dashboard"))}</div>
            <h2>{esc(t("dashboard_title"))}</h2>
            <p>{esc(t("dashboard_subtitle"))}</p>
        </div>
    </section>
    """, unsafe_allow_html=True)

    if not runs:
        st.markdown(f'<div class="empty-state">{esc(t("dashboard_no_data"))}</div>', unsafe_allow_html=True)
        return

    winner_counts = {"model_a": 0, "model_b": 0, "draw": 0}
    category_counts: dict[str, int] = {}
    total_a = 0.0
    total_b = 0.0
    for run in runs:
        comp = run.comparison
        winner_counts[comp.overall_winner] = winner_counts.get(comp.overall_winner, 0) + 1
        total_a += comp.model_a_mean
        total_b += comp.model_b_mean
        category = _prompt_category(run.prompt)
        category_counts[category] = category_counts.get(category, 0) + 1

    metric_html = "".join([
        _dashboard_metric(t("dashboard_total_runs"), str(len(runs))),
        _dashboard_metric(t("dashboard_gpt_wins"), str(winner_counts.get("model_a", 0))),
        _dashboard_metric(t("dashboard_gemini_wins"), str(winner_counts.get("model_b", 0))),
        _dashboard_metric(t("dashboard_draws"), str(winner_counts.get("draw", 0))),
        _dashboard_metric(t("dashboard_avg_score_a"), f"{total_a / len(runs):.2f}"),
        _dashboard_metric(t("dashboard_avg_score_b"), f"{total_b / len(runs):.2f}"),
    ])
    st.markdown(f'<div class="dashboard-metric-grid">{metric_html}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    plot_theme = _plotly_theme()
    with col1:
        category_labels = [_category_label(category) for category in category_counts]
        category_values = list(category_counts.values())
        fig = go.Figure(go.Bar(x=category_labels, y=category_values, marker_color="#4F46E5"))
        fig.update_layout(
            title=t("dashboard_category_distribution"),
            paper_bgcolor=plot_theme["paper"],
            plot_bgcolor=plot_theme["plot"],
            font=dict(color=plot_theme["font"]),
            margin=dict(t=50, b=40, l=40, r=20),
            height=330,
        )
        fig.update_xaxes(gridcolor=plot_theme["grid"])
        fig.update_yaxes(gridcolor=plot_theme["grid"], rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True, key="dashboard_categories")

    with col2:
        winner_labels = [_winner_label("model_a"), _winner_label("model_b"), _winner_label("draw")]
        winner_values = [winner_counts.get("model_a", 0), winner_counts.get("model_b", 0), winner_counts.get("draw", 0)]
        fig = go.Figure(go.Pie(labels=winner_labels, values=winner_values, hole=0.48, marker_colors=["#4F46E5", "#D97706", "#A1A1AA"]))
        fig.update_layout(
            title=t("dashboard_winner_distribution"),
            paper_bgcolor=plot_theme["paper"],
            plot_bgcolor=plot_theme["plot"],
            font=dict(color=plot_theme["font"]),
            margin=dict(t=50, b=40, l=20, r=20),
            height=330,
        )
        st.plotly_chart(fig, use_container_width=True, key="dashboard_winners")

    _render_dashboard_recent_runs(runs)


def render_architecture_page() -> None:
    """Render an animated architecture overview for the multi-agent pipeline."""
    st.markdown(f"""
    <section class="arch-page">
        <div class="arch-hero">
            <div>
                <div class="arch-kicker">{esc(t("nav_architecture"))}</div>
                <h2>{esc(t("arch_title"))}</h2>
                <p>{esc(t("arch_subtitle"))}</p>
            </div>
        </div>
        <div class="arch-flow" aria-label="Multi-agent pipeline architecture">
            <div class="arch-node input" style="--step:0"><span>01</span><strong>{esc(t("arch_user_prompt"))}</strong></div>
            <div class="arch-line request" data-label="{esc(t("arch_request_flow"))}" style="--step:1"><i></i><i></i><i></i></div>
            <div class="arch-node orchestrator" style="--step:2"><span>02</span><strong>{esc(t("arch_orchestrator"))}</strong></div>
            <div class="arch-branch-label" style="--step:3">{esc(t("arch_parallel_flow"))}</div>
            <div class="arch-split" style="--step:3">
                <div class="arch-node gen-a"><span>03A</span><strong>{esc(t("arch_gen_a"))}</strong></div>
                <div class="arch-node gen-b"><span>03B</span><strong>{esc(t("arch_gen_b"))}</strong></div>
            </div>
            <div class="arch-line data" data-label="{esc(t("arch_data_flow"))}" style="--step:4"><i></i><i></i><i></i></div>
            <div class="arch-node evaluator" style="--step:5"><span>04</span><strong>{esc(t("arch_evaluator"))}</strong></div>
            <div class="arch-branch-label" style="--step:6">{esc(t("arch_request_flow"))} + {esc(t("arch_data_flow"))}</div>
            <div class="arch-review-grid" style="--step:6">
                <div class="arch-node critic"><span>05</span><strong>{esc(t("arch_critic_a"))}</strong></div>
                <div class="arch-node revise"><span>06</span><strong>{esc(t("arch_reviser_a"))}</strong></div>
                <div class="arch-node critic"><span>07</span><strong>{esc(t("arch_critic_b"))}</strong></div>
                <div class="arch-node revise"><span>08</span><strong>{esc(t("arch_reviser_b"))}</strong></div>
            </div>
            <div class="arch-line data" data-label="{esc(t("arch_data_flow"))}" style="--step:7"><i></i><i></i><i></i></div>
            <div class="arch-split final" style="--step:8">
                <div class="arch-node compare"><span>09</span><strong>{esc(t("arch_compare"))}</strong></div>
                <div class="arch-node persist"><span>10</span><strong>{esc(t("arch_persist"))}</strong></div>
            </div>
        </div>
        <div class="arch-notes">
            <div class="arch-note" style="--step:0"><strong>01</strong><p>{esc(t("arch_stage_input"))}</p></div>
            <div class="arch-note" style="--step:1"><strong>02</strong><p>{esc(t("arch_stage_parallel"))}</p></div>
            <div class="arch-note" style="--step:2"><strong>03</strong><p>{esc(t("arch_stage_eval"))}</p></div>
            <div class="arch-note" style="--step:3"><strong>04</strong><p>{esc(t("arch_stage_review"))}</p></div>
            <div class="arch-note" style="--step:4"><strong>05</strong><p>{esc(t("arch_stage_output"))}</p></div>
        </div>
    </section>
    """, unsafe_allow_html=True)


def render_architecture_v2_page() -> None:
    """Render the human-gated V2 architecture overview."""
    st.markdown(f"""
    <section class="arch-page arch-v2-page">
        <div class="arch-hero">
            <div>
                <div class="arch-kicker">{esc(t("nav_architecture_v2"))}</div>
                <h2>{esc(t("arch_v2_title"))}</h2>
                <p>{esc(t("arch_v2_subtitle"))}</p>
            </div>
        </div>
        <div class="arch-flow arch-v2-flow" aria-label="Human-gated multi-agent architecture">
            <div class="arch-v2-legend" style="--step:0">
                <span><i class="agent"></i>{esc(t("arch_v2_legend_agent"))}</span>
                <span><i class="hil"></i>{esc(t("arch_v2_legend_hil"))}</span>
                <span><i class="gate"></i>{esc(t("arch_v2_legend_gate"))}</span>
                <span><i class="storage"></i>{esc(t("arch_v2_legend_storage"))}</span>
            </div>
            <div class="arch-node input" style="--step:1"><span>01</span><strong>{esc(t("arch_user_prompt"))}</strong></div>
            <div class="arch-v2-arrow" style="--step:2"></div>
            <div class="arch-node orchestrator" style="--step:3"><span>02</span><strong>{esc(t("arch_orchestrator"))}</strong></div>
            <div class="arch-v2-arrow" style="--step:4"></div>
            <div class="arch-v2-group parallel" style="--step:5">
                <div class="arch-v2-group-label">{esc(t("arch_v2_parallel"))}</div>
                <div class="arch-split">
                    <div class="arch-node gen-a"><span>03A</span><strong>GPT Image-2</strong></div>
                    <div class="arch-node gen-b"><span>03B</span><strong>Gemini 3 Pro</strong></div>
                </div>
            </div>
            <div class="arch-v2-arrow split-join" style="--step:6"></div>
            <div class="arch-node evaluator" style="--step:7"><span>04</span><strong>{esc(t("arch_evaluator"))}</strong><em>{esc(t("arch_v2_eval_detail"))}</em></div>
            <div class="arch-v2-arrow" style="--step:8"></div>
            <div class="arch-v2-router-row gate1-router" style="--step:9">
                <div class="arch-node gate"><span>G1</span><strong>{esc(t("arch_v2_gate1"))}</strong><em>{esc(t("arch_v2_gate1_detail"))}</em></div>
            </div>
            <div class="arch-v2-branch-lines gate1" style="--step:10" aria-hidden="true">
                <svg viewBox="0 0 900 72" preserveAspectRatio="none">
                    <path class="low" d="M450 0 V22 H150 V72" />
                    <path class="high" d="M450 0 V22 H610 V72" />
                </svg>
                <span class="route-label low">{esc(t("arch_v2_route_low_hil"))}</span>
                <span class="route-label high">{esc(t("arch_v2_route_high_review"))}</span>
            </div>
            <div class="arch-v2-two-col" style="--step:10">
                <div class="arch-v2-hil-branch">
                    <div class="arch-node hil"><span>HIL</span><strong>{esc(t("arch_v2_hil_early"))}</strong><em>{esc(t("arch_v2_hil_early_detail"))}</em></div>
                </div>
                <div class="arch-v2-group review-chain">
                    <div class="arch-v2-group-label">{esc(t("arch_v2_review_chain"))}</div>
                    <div class="arch-node critic"><span>05</span><strong>{esc(t("arch_critic_a"))}</strong><em>{esc(t("arch_v2_critique_a_detail"))}</em></div>
                    <div class="arch-v2-arrow slim"></div>
                    <div class="arch-node revise"><span>06</span><strong>{esc(t("arch_reviser_a"))}</strong><em>{esc(t("arch_v2_revise_a_detail"))}</em></div>
                    <div class="arch-v2-arrow slim"></div>
                    <div class="arch-node critic"><span>07</span><strong>{esc(t("arch_critic_b"))}</strong><em>{esc(t("arch_v2_critique_b_detail"))}</em></div>
                    <div class="arch-v2-arrow slim"></div>
                    <div class="arch-node revise"><span>08</span><strong>{esc(t("arch_reviser_b"))}</strong><em>{esc(t("arch_v2_revise_b_detail"))}</em></div>
                </div>
            </div>
            <div class="arch-v2-arrow" style="--step:11"></div>
            <div class="arch-v2-router-row gate2-router" style="--step:12">
                <div class="arch-node gate"><span>G2</span><strong>{esc(t("arch_v2_gate2"))}</strong><em>{esc(t("arch_v2_gate2_detail"))}</em></div>
            </div>
            <div class="arch-v2-branch-lines gate2" style="--step:13" aria-hidden="true">
                <svg viewBox="0 0 760 72" preserveAspectRatio="none">
                    <path class="disagree" d="M380 0 V22 H150 V72" />
                    <path class="agree" d="M380 0 V22 H535 V72" />
                </svg>
                <span class="route-label disagree">{esc(t("arch_v2_route_disagree_hil"))}</span>
                <span class="route-label agree">{esc(t("arch_v2_route_agree_decision"))}</span>
            </div>
            <div class="arch-v2-decision-row" style="--step:13">
                <div class="arch-node hil"><span>HIL</span><strong>{esc(t("arch_v2_hil_adjudicator"))}</strong><em>{esc(t("arch_v2_hil_adjudicator_detail"))}</em></div>
                <div class="arch-node compare"><span>09</span><strong>{esc(t("arch_compare"))}</strong><em>{esc(t("arch_v2_decision_detail"))}</em></div>
            </div>
            <div class="arch-v2-arrow" style="--step:14"></div>
            <div class="arch-node persist" style="--step:15"><span>10</span><strong>{esc(t("arch_persist"))}</strong><em>{esc(t("arch_v2_archive_detail"))}</em></div>
        </div>
    </section>
    """, unsafe_allow_html=True)


# --- Header ---
st.markdown(f"""
<div class="eval-header">
    <h1>{t("header_title")}</h1>
    <p>{t("header_subtitle")}</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.view_mode == "architecture_v1":
    render_architecture_page()
    st.stop()
if st.session_state.view_mode == "architecture_v2":
    render_architecture_v2_page()
    st.stop()
if st.session_state.view_mode == "dashboard":
    render_dashboard_page()
    st.stop()

# --- Comparison controls ---
prebaked = load_prebaked_results()
selected_prebaked = ""
selected_prebaked_result: PipelineResult | None = None
rerun_selected_clicked = False
generate_clicked = False
with st.container(border=True):
    st.markdown(f'<div class="control-card-title">{esc(t("new_comparison"))}</div>', unsafe_allow_html=True)
    if prebaked:
        example_select, example_delete, example_rerun = st.columns([5, 0.9, 0.9], vertical_alignment="bottom")
        options = [""] + list(prebaked.keys())
        if st.session_state.get("selected_prebaked_run") not in options:
            st.session_state.selected_prebaked_run = ""
            st.session_state.loaded_prebaked_run_key = None
        with example_select:
            selected_prebaked = st.selectbox(
                t("select_example"),
                options,
                format_func=lambda x: _prebaked_option_label(x, prebaked),
                key="selected_prebaked_run",
            )
        selected_prebaked_result = prebaked.get(selected_prebaked) if selected_prebaked else None
        with example_delete:
            delete_selected_clicked = st.button(
                t("btn_delete_example"),
                disabled=st.session_state.running or selected_prebaked_result is None,
                use_container_width=True,
            )
        with example_rerun:
            rerun_selected_clicked = st.button(
                t("btn_rerun_example"),
                disabled=st.session_state.running or selected_prebaked_result is None,
                use_container_width=True,
            )
        if delete_selected_clicked and selected_prebaked_result is not None:
            _delete_run_record(selected_prebaked_result)
            load_prebaked_results.clear()
            if st.session_state.loaded_prebaked_run_key == selected_prebaked:
                st.session_state.loaded_prebaked_run_key = None
            st.rerun()
        if selected_prebaked_result is not None:
            loaded = selected_prebaked_result
            loaded_key = _run_key(loaded)
            selection_changed = st.session_state.loaded_prebaked_run_key != selected_prebaked
            missing_from_session = all(_run_key(run) != loaded_key for run in st.session_state.runs)
            if selection_changed or missing_from_session:
                _remember_run(loaded)
                st.session_state.loaded_prebaked_run_key = selected_prebaked
                st.rerun()
        else:
            st.session_state.loaded_prebaked_run_key = None

    with st.container(border=True):
        if st.session_state.reference_upload_bytes:
            preview_col, clear_col = st.columns([8, 0.6], vertical_alignment="center")
            with preview_col:
                st.markdown(
                    _attachment_preview_html(
                        st.session_state.reference_upload_bytes,
                        st.session_state.reference_upload_name,
                        st.session_state.reference_upload_type,
                    ),
                    unsafe_allow_html=True,
                )
            with clear_col:
                if st.button("×", key="attachment_clear", help=t("attachment_remove"), disabled=st.session_state.running):
                    st.session_state.reference_upload_bytes = None
                    st.session_state.reference_upload_name = None
                    st.session_state.reference_upload_type = None
                    st.session_state.show_reference_uploader = False
                    st.rerun()

        prompt = st.text_area(
            t("prompt_label"),
            placeholder=t("input_placeholder"),
            height=118,
            key="prompt_composer",
            label_visibility="collapsed",
        )
        toolbar_add, toolbar_gap, toolbar_generate = st.columns([0.55, 5.0, 1.35], vertical_alignment="center")
        with toolbar_add:
            if st.button("＋", key="attachment_add", help=t("reference_image_help"), disabled=st.session_state.running, use_container_width=True):
                st.session_state.show_reference_uploader = not st.session_state.show_reference_uploader
        with toolbar_generate:
            generate_clicked = st.button(
                t("btn_generate"),
                type="primary",
                disabled=st.session_state.running,
                use_container_width=True,
                key="composer_generate",
            )
        if st.session_state.show_reference_uploader:
            reference_upload = st.file_uploader(
                t("reference_image_picker"),
                type=["png", "jpg", "jpeg", "webp"],
                help=t("reference_image_help"),
                disabled=st.session_state.running,
                key="reference_image_upload",
            )
            if reference_upload is not None:
                st.session_state.reference_upload_bytes = reference_upload.getvalue()
                st.session_state.reference_upload_name = reference_upload.name
                st.session_state.reference_upload_type = reference_upload.type
                st.session_state.show_reference_uploader = False
                st.rerun()

# --- Run pipeline ---

STAGE_LABELS = [
    "step_short_generate", "step_short_evaluate",
    "step_short_gate1", "step_short_critique", "step_short_revise",
    "step_short_critique2", "step_short_gate2", "step_short_revise2",
]
_DONE_INDEX = len(STAGE_LABELS)
STAGE_TO_INDEX = {
    "stage_generating": 0,
    "stage_evaluating": 1,
    "stage_gate1": 2,
    "stage_critique": 3,
    "stage_revising": 4,
    "stage_critique_round2": 5,
    "stage_gate2": 6,
    "stage_revising_round2": 7,
    "stage_complete": _DONE_INDEX,
}


def _activity_summary_text(
    stage_times: list[tuple[str, float, float | None]],
    gen_model_done: dict[str, float],
    pipeline_start: float,
    now: float,
) -> str:
    parts = []
    if gen_model_done:
        for model_key, label in (("gpt_image_2", "GPT"), ("gemini_3_pro", "Gemini")):
            done_at = gen_model_done.get(model_key)
            if done_at is not None:
                parts.append(f"{label} {max(done_at - pipeline_start, 0.0):.1f}s")
    elif stage_times:
        active_stage = stage_times[-1][0]
        active_label = t(STAGE_LABELS[STAGE_TO_INDEX.get(active_stage, 0)]) if active_stage in STAGE_TO_INDEX else active_stage
        parts.append(active_label)
    total = f"{now - pipeline_start:.1f}s"
    parts.append(t("activity_total_time", s=total))
    return " · ".join(parts)


def _render_progress(active_index: int, status_text: str, activity_summary: str = "") -> str:
    """Build HTML for the animated pipeline progress tracker."""
    labels = [t(k) for k in STAGE_LABELS]
    total_steps = len(labels)
    done_index = total_steps  # All done when active_index == _DONE_INDEX
    steps_html = []
    for i, label in enumerate(labels):
        if active_index > i or active_index == done_index:
            state = "done"
            icon = '<span class="check-icon">✓</span>'
        elif active_index == i:
            state = "active"
            icon = str(i + 1)
        else:
            state = "pending"
            icon = str(i + 1)
        steps_html.append(
            f'<div class="pipeline-step">'
            f'<div class="step-circle {state}">{icon}</div>'
            f'<div class="step-text {state}">{esc(label)}</div>'
            f'</div>'
        )
        if i < total_steps - 1:
            if active_index > i + 1 or active_index == done_index:
                conn = "done"
            elif active_index == i + 1 or (active_index == i and i < total_steps - 1):
                conn = "active"
            else:
                conn = "pending"
            steps_html.append(f'<div class="step-connector {conn}"></div>')

    is_complete = active_index == done_index
    status_cls = "pipeline-status-text complete" if is_complete else "pipeline-status-text"
    summary_html = f'<span>{esc(activity_summary)}</span>' if activity_summary else ""
    return (
        f'<div class="pipeline-panel">'
        f'<div class="pipeline-tracker">{"".join(steps_html)}</div>'
        f'<div class="{status_cls}"><strong>{esc(status_text)}</strong>{summary_html}</div>'
        f'</div>'
    )


def _render_activity_log(
    stage_times: list[tuple[str, float, float | None]],
    gen_model_done: dict[str, float],
    pipeline_start: float,
    now: float,
    *,
    historical: bool = False,
) -> str:
    """Build HTML for the right-side activity log with per-stage timing."""
    items_html = []
    for stage, start, end in stage_times:
        if stage == "stage_generating":
            for model_key, model_label in [
                ("gpt_image_2", "GPT Image-2"),
                ("gemini_3_pro", "Gemini 3 Pro"),
            ]:
                done_t = gen_model_done.get(model_key)
                if historical and done_t is not None:
                    elapsed = "saved"
                    cls, icon = "done", "✓"
                elif done_t is not None:
                    elapsed = f"{max(done_t - start, 0.0):.1f}s"
                    cls, icon = "done", "✓"
                elif end is not None:
                    elapsed = f"{max(end - start, 0.0):.1f}s"
                    cls, icon = "done", "✓"
                else:
                    elapsed = f"{now - start:.1f}s"
                    cls, icon = "active", "●"
                items_html.append(
                    f'<div class="activity-item {cls}">'
                    f'<span class="activity-status">{icon}</span>'
                    f'<span class="activity-label">{model_label}</span>'
                    f'<span class="activity-time">{elapsed}</span>'
                    f'</div>'
                )
        else:
            idx = STAGE_TO_INDEX.get(stage, -1)
            label = t(STAGE_LABELS[idx]) if 0 <= idx < len(STAGE_LABELS) else esc(stage)
            if end is not None:
                elapsed = f"{max(end - start, 0.0):.1f}s"
                cls, icon = "done", "✓"
            else:
                elapsed = f"{now - start:.1f}s"
                cls, icon = "active", "●"
            items_html.append(
                f'<div class="activity-item {cls}">'
                f'<span class="activity-status">{icon}</span>'
                f'<span class="activity-label">{label}</span>'
                f'<span class="activity-time">{elapsed}</span>'
                f'</div>'
            )
    total = f"{now - pipeline_start:.1f}s"
    return (
        f'<div class="activity-log">'
        f'<div class="activity-log-title">{esc(t("activity_log"))}</div>'
        f'{"".join(items_html)}'
        f'<div class="activity-total">{esc(t("activity_total_time", s=total))}</div>'
        f'</div>'
    )


def _artifact_mtime(run_dir: Path, names: tuple[str, ...]) -> float | None:
    existing = [run_dir / name for name in names if (run_dir / name).exists()]
    if not existing:
        return None
    return max(path.stat().st_mtime for path in existing)


def _build_historical_progress_snapshot(result: PipelineResult) -> dict | None:
    """Reconstruct a completed Activity Log for pre-baked runs loaded from disk."""
    if not result.run_dir or not result.run_dir.exists() or not result.comparison:
        return None

    run_dir = result.run_dir
    stage_artifacts = [
        ("stage_generating", ("gpt_image_2.png", "gemini_3_pro.png")),
        ("stage_evaluating", ("evaluation.json", "evaluation_v2.json")),
        ("stage_gate1", ("gate1_decision.json",)),
        ("stage_critique", ("critique_r1.json", "critique.json")),
        ("stage_revising", ("revised_r1.json", "revised.json")),
        ("stage_critique_round2", ("critique_r2.json",)),
        ("stage_gate2", ("gate2_decision.json",)),
        ("stage_revising_round2", ("revised_r2.json", "comparison.json")),
    ]
    observed = [(stage, mtime) for stage, names in stage_artifacts if (mtime := _artifact_mtime(run_dir, names)) is not None]
    if not observed:
        return None

    image_times = [
        mtime
        for filename in ("gpt_image_2.png", "gemini_3_pro.png")
        if (mtime := _artifact_mtime(run_dir, (filename,))) is not None
    ]
    base_time = min([mtime for _stage, mtime in observed] + image_times)
    stage_times: list[tuple[str, float, float | None]] = []
    previous_end = base_time
    for index, (stage, mtime) in enumerate(observed):
        start = previous_end if stage_times else max(base_time - 0.1, 0)
        end = max(mtime, start + 0.1 + index * 0.01)
        stage_times.append((stage, start, end))
        previous_end = end

    gen_model_done = {
        model_key: mtime
        for model_key, filename in (("gpt_image_2", "gpt_image_2.png"), ("gemini_3_pro", "gemini_3_pro.png"))
        if (mtime := _artifact_mtime(run_dir, (filename,))) is not None
    }
    now = max(end for _stage, _start, end in stage_times if end is not None)
    historical_summary = t("activity_total_time", s=f"{max(now - base_time, 0.0):.1f}s")
    return {
        "progress": _render_progress(_DONE_INDEX, t("status_done"), historical_summary),
        "activity": _render_activity_log(stage_times, gen_model_done, base_time, now, historical=True),
    }


def _progress_snapshot_for_result(result: PipelineResult) -> dict | None:
    run_key = _run_key(result)
    snapshot = st.session_state.run_progress_snapshots.get(run_key)
    if snapshot:
        return snapshot
    snapshot = _load_progress_snapshot(result) or _build_historical_progress_snapshot(result)
    if snapshot:
        st.session_state.run_progress_snapshots[run_key] = snapshot
    return snapshot


run_prompt = selected_prebaked_result.prompt if rerun_selected_clicked and selected_prebaked_result is not None else prompt.strip()
run_reference_image_bytes = None
run_reference_image_name = None
if rerun_selected_clicked and selected_prebaked_result is not None and selected_prebaked_result.reference_image_path:
    run_reference_image_bytes = selected_prebaked_result.reference_image_path.read_bytes()
    run_reference_image_name = selected_prebaked_result.reference_image_path.name
elif not rerun_selected_clicked and st.session_state.reference_upload_bytes:
    run_reference_image_bytes = st.session_state.reference_upload_bytes
    run_reference_image_name = st.session_state.reference_upload_name

if (generate_clicked or rerun_selected_clicked) and run_prompt.strip():
    st.session_state.running = True

    status_container = st.empty()
    status_container.markdown(
        _render_progress(0, t("status_running")),
        unsafe_allow_html=True,
    )

    stage_times: list[tuple[str, float, float | None]] = []
    gen_model_done: dict[str, float] = {}
    pipeline_start = time.time()
    _lock = threading.Lock()
    _done_event = threading.Event()
    _result_holder: dict = {}

    def _on_stage(stage: str) -> None:
        _now = time.time()
        with _lock:
            if stage_times and stage_times[-1][2] is None:
                last = stage_times[-1]
                stage_times[-1] = (last[0], last[1], _now)
            if stage != "stage_complete":
                stage_times.append((stage, _now, None))

    def _on_image_done(model_key: str) -> None:
        with _lock:
            gen_model_done[model_key] = time.time()

    def _run_pipeline() -> None:
        try:
            _result_holder["result"] = run_pipeline(
                run_prompt.strip(),
                on_stage=_on_stage,
                on_image_done=_on_image_done,
                reference_image=run_reference_image_bytes,
                reference_image_name=run_reference_image_name,
            )
        except Exception as e:
            _result_holder["error"] = e
        finally:
            _done_event.set()

    _thread = threading.Thread(target=_run_pipeline, daemon=True)
    _thread.start()

    while not _done_event.wait(timeout=0.5):
        _now = time.time()
        with _lock:
            _st = list(stage_times)
            _gmd = dict(gen_model_done)
        _active_stage = _st[-1][0] if _st else "status_running"
        _active_idx = STAGE_TO_INDEX.get(_active_stage, 0)
        status_container.markdown(
            _render_progress(_active_idx, t(_active_stage), _activity_summary_text(_st, _gmd, pipeline_start, _now)),
            unsafe_allow_html=True,
        )

    _thread.join()
    _now = time.time()
    with _lock:
        if stage_times and stage_times[-1][2] is None:
            last = stage_times[-1]
            stage_times[-1] = (last[0], last[1], _now)
        _st = list(stage_times)
        _gmd = dict(gen_model_done)

    if "result" in _result_holder:
        completed_result = _result_holder["result"]
        completed_run_key = _run_key(completed_result)
        if completed_result.pipeline_status == "failed" or not completed_result.comparison:
            error_text = _display_errors(completed_result.errors) if completed_result.errors else t("error_image_failed")
            progress_html = _render_progress(0, t("error_pipeline_failed", error=error_text), _activity_summary_text(_st, _gmd, pipeline_start, _now))
        else:
            progress_html = _render_progress(_DONE_INDEX, t("status_done"), _activity_summary_text(_st, _gmd, pipeline_start, _now))
            activity_html = _render_activity_log(_st, _gmd, pipeline_start, _now)
            st.session_state.run_progress_snapshots[completed_run_key] = {
                "progress": progress_html,
                "activity": activity_html,
            }
            _save_progress_snapshot(completed_result, progress_html, activity_html)
        _remember_run(completed_result)
        status_container.markdown(
            progress_html,
            unsafe_allow_html=True,
        )
    else:
        err = _result_holder.get("error", Exception("Unknown error"))
        logger.exception("Pipeline failed")
        st.error(t("error_pipeline_failed", error=_display_error(str(err))))

    st.session_state.running = False
    st.rerun()
elif generate_clicked and not run_prompt.strip():
    st.warning(t("prompt_required"))

# --- Empty state ---
_focused_entry = _focused_run_entry()
if not _focused_entry:
    st.markdown(f"""
    <div class="empty-state">
        {t("empty_state")}
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# --- Render results ---


def _resume_run_with_progress(result: PipelineResult) -> None:
    st.session_state.running = True
    status_container = st.empty()
    status_container.markdown(
        _render_progress(0, t("status_rerunning")),
        unsafe_allow_html=True,
    )

    stage_times: list[tuple[str, float, float | None]] = []
    gen_model_done: dict[str, float] = {}
    pipeline_start = time.time()
    _lock = threading.Lock()
    _done_event = threading.Event()
    _result_holder: dict = {}

    def _on_stage(stage: str) -> None:
        _now = time.time()
        with _lock:
            if stage_times and stage_times[-1][2] is None:
                last = stage_times[-1]
                stage_times[-1] = (last[0], last[1], _now)
            if stage != "stage_complete":
                stage_times.append((stage, _now, None))

    def _on_image_done(model_key: str) -> None:
        with _lock:
            gen_model_done[model_key] = time.time()

    def _run_resume() -> None:
        try:
            _result_holder["result"] = resume_pipeline_from_result(
                result,
                on_stage=_on_stage,
                on_image_done=_on_image_done,
            )
        except Exception as e:
            _result_holder["error"] = e
        finally:
            _done_event.set()

    _thread = threading.Thread(target=_run_resume, daemon=True)
    _thread.start()

    while not _done_event.wait(timeout=0.5):
        _now = time.time()
        with _lock:
            _st = list(stage_times)
            _gmd = dict(gen_model_done)
        _active_stage = _st[-1][0] if _st else "status_rerunning"
        _active_idx = STAGE_TO_INDEX.get(_active_stage, 0)
        status_container.markdown(
            _render_progress(_active_idx, t(_active_stage), _activity_summary_text(_st, _gmd, pipeline_start, _now)),
            unsafe_allow_html=True,
        )

    _thread.join()
    _now = time.time()
    with _lock:
        if stage_times and stage_times[-1][2] is None:
            last = stage_times[-1]
            stage_times[-1] = (last[0], last[1], _now)
        _st = list(stage_times)
        _gmd = dict(gen_model_done)

    if "result" in _result_holder:
        resumed_result = _result_holder["result"]
        resumed_run_key = _run_key(resumed_result)
        if resumed_result.pipeline_status == "failed" or not resumed_result.comparison:
            error_text = _display_errors(resumed_result.errors) if resumed_result.errors else t("error_image_failed")
            progress_html = _render_progress(0, t("error_pipeline_failed", error=error_text), _activity_summary_text(_st, _gmd, pipeline_start, _now))
        else:
            progress_html = _render_progress(_DONE_INDEX, t("status_done"), _activity_summary_text(_st, _gmd, pipeline_start, _now))
            activity_html = _render_activity_log(_st, _gmd, pipeline_start, _now)
            st.session_state.run_progress_snapshots[resumed_run_key] = {
                "progress": progress_html,
                "activity": activity_html,
            }
            _save_progress_snapshot(resumed_result, progress_html, activity_html)
        _remember_run(resumed_result)
        status_container.markdown(progress_html, unsafe_allow_html=True)
    else:
        err = _result_holder.get("error", Exception("Unknown error"))
        logger.exception("Pipeline resume failed")
        st.error(t("error_pipeline_failed", error=_display_error(str(err))))

    st.session_state.running = False
    st.rerun()


def render_result(result: PipelineResult, index: int) -> None:
    """Render a single pipeline result."""
    is_focused = st.session_state.focused_run_key == _run_key(result)
    if index > 0:
        st.markdown('<div class="run-divider"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="run-label">{esc(t("run_label", prompt=result.prompt[:100], timestamp=result.timestamp))}</div>', unsafe_allow_html=True)
    if is_focused:
        st.markdown('<div class="run-focused-banner">◀ ' + t("history_title") + '</div>', unsafe_allow_html=True)
        snapshot = _progress_snapshot_for_result(result)
        if snapshot:
            with st.expander(t("pipeline_completed_expander"), expanded=False):
                st.markdown(snapshot["progress"], unsafe_allow_html=True)
                if snapshot.get("activity"):
                    st.markdown(snapshot["activity"], unsafe_allow_html=True)

    # --- Errors ---
    if result.errors:
        for err in result.errors:
            is_warning = result.comparison is not None and err.startswith("Critique round 2 failed:")
            card_cls = "warning-card" if is_warning else "error-card"
            text_cls = "warning-text" if is_warning else "error-text"
            st.markdown(f'<div class="{card_cls}"><div class="{text_cls}">⚠ {esc(_display_error(err))}</div></div>', unsafe_allow_html=True)
        if is_focused and result.run_dir:
            if st.button(
                t("btn_rerun_from_error"),
                key=f"rerun_failed_{_run_key(result)}",
                disabled=st.session_state.running,
                type="secondary",
            ):
                _resume_run_with_progress(result)

    # --- Images ---
    gpt_path = result.image_paths.get("gpt_image_2")
    gemini_path = result.image_paths.get("gemini_3_pro")
    reference_path = result.reference_image_path

    if gpt_path or gemini_path:
        image_columns = st.columns([0.8, 1, 1]) if reference_path else st.columns(2)
        if reference_path:
            with image_columns[0]:
                st.markdown(f'<div class="model-label">{t("reference_image_used")}</div>', unsafe_allow_html=True)
                if reference_path.exists():
                    st.image(str(reference_path), width="stretch")
        col_a = image_columns[1] if reference_path else image_columns[0]
        col_b = image_columns[2] if reference_path else image_columns[1]
        with col_a:
            st.markdown(f'<div class="model-label">{t("model_gpt")}</div>', unsafe_allow_html=True)
            if isinstance(gpt_path, Path) and gpt_path.exists():
                st.image(str(gpt_path), use_container_width=True)
            else:
                st.markdown(f'<div class="error-card"><div class="error-text">⚠ {t("error_image_failed")}</div></div>', unsafe_allow_html=True)

        with col_b:
            st.markdown(f'<div class="model-label">{t("model_gemini")}</div>', unsafe_allow_html=True)
            if isinstance(gemini_path, Path) and gemini_path.exists():
                st.image(str(gemini_path), use_container_width=True)
            else:
                st.markdown(f'<div class="error-card"><div class="error-text">⚠ {t("error_image_failed")}</div></div>', unsafe_allow_html=True)

    render_hil_v2_status(result)
    render_gate1_hil_task(result)
    render_gate2_hil_task(result)

    if not result.comparison:
        return

    comp = result.comparison
    is_unchallenged = result.critique is None

    # --- Winner banner ---
    winner_display = {
        "model_a": esc(comp.model_a_name),
        "model_b": esc(comp.model_b_name),
        "draw": t("draw_label"),
    }
    winner_name = winner_display.get(comp.overall_winner, esc(comp.overall_winner))
    dims_won = comp.model_a_dimensions_won if comp.overall_winner == "model_a" else comp.model_b_dimensions_won
    unchallenged_note = f' <span class="unchallenged">{t("unchallenged")}</span>' if is_unchallenged else ""

    if comp.overall_winner == "draw":
        st.markdown(f"""
        <div class="winner-banner">
            <span class="winner-name">{t("draw_label")}</span> | {t("overall_label", a=comp.model_a_mean, b=comp.model_b_mean)}{unchallenged_note}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="winner-banner">
            {t("winner_label")}<span class="winner-name">{winner_name}</span>
            ({t("dimensions_won", count=dims_won)}) | {t("overall_label", a=comp.model_a_mean, b=comp.model_b_mean)}{unchallenged_note}
        </div>
        """, unsafe_allow_html=True)

    # --- Radar chart ---
    render_radar_chart(result)

    # --- Dimension score cards ---
    render_dimension_cards(result)

    # --- Dimension comments table ---
    render_dimension_comments_table(result)

    # --- Critique transcript ---
    render_critique_transcript(result)

    # --- Raw JSON ---
    render_raw_json(result)


def _gate_by_name(result: PipelineResult, gate_name: str):
    for gate in result.gate_decisions:
        if gate.gate == gate_name:
            return gate
    return None


def _chip(label: str, value: object, css_class: str = "") -> str:
    value_text = "—" if value in (None, "", []) else str(value)
    return (
        f'<div class="hil-chip {css_class}">'
        f'<span>{esc(label)}</span><strong>{esc(value_text)}</strong>'
        f'</div>'
    )


def render_gate1_hil_task(result: PipelineResult) -> None:
    """Render a minimal Gate 1 A/B/TIE arbitration task when HIL is pending."""
    gate1 = _gate_by_name(result, "gate1_uncertainty_risk_router")
    if not gate1 or gate1.status != "pending" or not result.run_dir:
        return

    st.markdown(f"### {t('hil_gate1_pending_title')}")
    st.caption(t("hil_gate1_pending_help"))
    with st.form(key=f"gate1_hil_form_{_run_key(result)}"):
        selections: dict[str, str] = {}
        review_items = gate1.review_items or []
        for item in review_items:
            label = dim_label(item.dimension)
            reason = ", ".join(item.uncertainty_reasons)
            selections[item.dimension] = st.radio(
                f"{label} ({reason})",
                options=["A", "B", "TIE"],
                format_func=lambda value: {
                    "A": t("hil_choice_a"),
                    "B": t("hil_choice_b"),
                    "TIE": t("hil_choice_tie"),
                }[value],
                horizontal=True,
            )
        submitted = st.form_submit_button(t("hil_submit_gate1"), type="primary")

    if submitted:
        now = datetime.now().isoformat()
        hil_review = HilArbitration(
            status="completed",
            route_score=gate1.route_score or 0,
            route_band=gate1.route_band or "required_hil",
            trigger_reasons=gate1.trigger_reasons,
            review_dimensions=gate1.review_dimensions,
            reviewer=st.session_state.reviewer,
            created_at=now,
            completed_at=now,
            dimension_arbitrations=[
                HilDimensionArbitration(dimension=dimension, human_winner=winner)
                for dimension, winner in selections.items()
            ],
        )
        result.hil_reviews = [hil_review]
        gate1.status = "completed"
        gate1.pipeline_status = None
        gate1.requires_attention = False
        result.pipeline_status = "partial"
        result.requires_attention = any(gate.requires_attention for gate in result.gate_decisions)
        (result.run_dir / "hil_review_r1.json").write_text(json.dumps(hil_review.model_dump(), indent=2))
        (result.run_dir / "gate1_decision.json").write_text(json.dumps(gate1.model_dump(), indent=2))
        _resume_run_with_progress(result)


def _find_dimension_critique(critique, dimension: str):
    if not critique:
        return None
    return next((item for item in critique.dimension_critiques if item.dimension == dimension), None)


def _score_delta_text(result: PipelineResult, dimension: str) -> str:
    if not result.eval_a or not result.eval_b or not result.revisions:
        return "—"
    revised = result.revisions[-1]
    initial_a = getattr(result.eval_a, dimension).score
    initial_b = getattr(result.eval_b, dimension).score
    revised_a = getattr(revised.model_a, dimension).score
    revised_b = getattr(revised.model_b, dimension).score
    delta_a = revised_a - initial_a
    delta_b = revised_b - initial_b
    return f"A {initial_a}->{revised_a} ({delta_a:+d}), B {initial_b}->{revised_b} ({delta_b:+d})"


def _revision_note_text(result: PipelineResult, dimension: str) -> str:
    if not result.revisions:
        return "—"
    revised = result.revisions[-1]
    note_a = getattr(revised.model_a, dimension).revision_note
    note_b = getattr(revised.model_b, dimension).revision_note
    accepted_a = t("accepted") if getattr(revised.model_a, dimension).critique_accepted else t("rejected")
    accepted_b = t("accepted") if getattr(revised.model_b, dimension).critique_accepted else t("rejected")
    return f"A {accepted_a}: {note_a} | B {accepted_b}: {note_b}"


def _issue_classification_text(result: PipelineResult, dimension: str) -> str:
    if not result.issue_equivalence:
        return "—"
    labels = [item.classification for item in result.issue_equivalence.equivalence_results if item.dimension == dimension]
    if not labels:
        return "—"
    label_map = {
        "equivalent": t("hil_equivalent"),
        "new": t("hil_new"),
        "contradicting": t("hil_contradicting"),
    }
    return ", ".join(label_map.get(label, label) for label in labels)


def _gate2_dimension_needs_hil(result: PipelineResult, dimension: str) -> bool:
    if not result.eval_a or not result.eval_b or not result.revisions:
        return False
    revised = result.revisions[-1]
    if getattr(result.eval_a, dimension).score != getattr(revised.model_a, dimension).score:
        return True
    if getattr(result.eval_b, dimension).score != getattr(revised.model_b, dimension).score:
        return True
    if result.issue_equivalence and any(
        item.dimension == dimension and item.classification == "contradicting"
        for item in result.issue_equivalence.equivalence_results
    ):
        return True

    critique_r1 = result.critiques[0] if result.critiques else None
    critique_r2 = result.critiques[-1] if len(result.critiques) >= 2 else None
    crit_05 = _find_dimension_critique(critique_r1, dimension)
    crit_07 = _find_dimension_critique(critique_r2, dimension)
    if crit_05 and crit_07:
        if (crit_05.suggested_score_model_a, crit_05.suggested_score_model_b) != (crit_07.suggested_score_model_a, crit_07.suggested_score_model_b):
            return True
    if crit_07:
        return (
            crit_07.original_score_model_a != crit_07.suggested_score_model_a
            or crit_07.original_score_model_b != crit_07.suggested_score_model_b
        )
    return False


def _critique_summary_html(label: str, critique_item) -> str:
    if not critique_item:
        body = "—"
        scores = "—"
    else:
        body = critique_item.critique
        scores = f"A {critique_item.original_score_model_a}->{critique_item.suggested_score_model_a}, B {critique_item.original_score_model_b}->{critique_item.suggested_score_model_b}"
    return (
        '<div class="gate2-summary-card">'
        f'<span>{esc(label)}</span>'
        f'<strong>{esc(scores)}</strong>'
        f'<p>{esc(body)}</p>'
        '</div>'
    )


def _gate2_dispute_summary_html(result: PipelineResult, dimensions: list[str]) -> str:
    if not dimensions:
        return ""
    critique_r1 = result.critiques[0] if result.critiques else None
    critique_r2 = result.critiques[-1] if len(result.critiques) >= 2 else None
    sections = []
    for dimension in dimensions:
        crit_05 = _find_dimension_critique(critique_r1, dimension)
        crit_07 = _find_dimension_critique(critique_r2, dimension)
        sections.append(
            '<section class="gate2-dispute">'
            f'<div class="gate2-dispute-title">{esc(dim_label(dimension))}</div>'
            '<div class="gate2-dispute-meta">'
            f'<span>{esc(t("hil_gate2_score_delta"))}: {esc(_score_delta_text(result, dimension))}</span>'
            f'<span>{esc(t("hil_gate2_issue_type"))}: {esc(_issue_classification_text(result, dimension))}</span>'
            '</div>'
            '<div class="gate2-summary-grid">'
            f'{_critique_summary_html(t("hil_gate2_05_summary"), crit_05)}'
            '<div class="gate2-summary-card revise">'
            f'<span>{esc(t("hil_gate2_06_summary"))}</span>'
            f'<strong>{esc(_score_delta_text(result, dimension))}</strong>'
            f'<p>{esc(_revision_note_text(result, dimension))}</p>'
            '</div>'
            f'{_critique_summary_html(t("hil_gate2_07_summary"), crit_07)}'
            '</div>'
            '</section>'
        )
    return "".join(sections)


def render_gate2_hil_task(result: PipelineResult) -> None:
    """Render a minimal Gate 2 adjudication task when major disagreement is pending."""
    gate2 = _gate_by_name(result, "gate2_disagreement_detector")
    if not gate2 or gate2.status != "pending" or not result.run_dir:
        return

    st.markdown(f"### {t('hil_gate2_pending_title')}")
    st.caption(t("hil_gate2_pending_help"))
    trigger_text = ", ".join(gate2.trigger_reasons) if gate2.trigger_reasons else "—"
    st.info(trigger_text)

    affected_dimensions = list(gate2.review_dimensions)
    if not affected_dimensions and result.issue_equivalence:
        affected_dimensions = sorted({
            item.dimension
            for item in result.issue_equivalence.equivalence_results
            if _gate2_dimension_needs_hil(result, item.dimension)
        })
    if not affected_dimensions:
        st.info(t("hil_gate2_no_actionable"))
        return

    st.markdown(_gate2_dispute_summary_html(result, affected_dimensions), unsafe_allow_html=True)

    with st.form(key=f"gate2_hil_form_{_run_key(result)}"):
        selections: dict[str, str] = {}
        for dimension in affected_dimensions:
            selections[dimension] = st.radio(
                dim_label(dimension),
                options=["agree_with_05", "agree_with_07", "both_partially_right"],
                format_func=lambda value: {
                    "agree_with_05": t("hil_agree_05"),
                    "agree_with_07": t("hil_agree_07"),
                    "both_partially_right": t("hil_both_partial"),
                }[value],
                horizontal=True,
            )
        submitted = st.form_submit_button(t("hil_submit_gate2"), type="primary")

    if submitted:
        now = datetime.now().isoformat()
        adjudication = HilAdjudication(
            status="completed",
            trigger_reasons=gate2.trigger_reasons,
            disagreement_items=[],
            reviewer=st.session_state.reviewer,
            created_at=now,
            completed_at=now,
            adjudication_labels=[
                HilAdjudicationLabel(dimension=dimension, label=label)
                for dimension, label in selections.items()
            ],
        )
        result.hil_adjudication = adjudication
        gate2.status = "completed"
        gate2.pipeline_status = None
        gate2.requires_attention = False
        result.pipeline_status = "partial"
        result.requires_attention = any(gate.requires_attention for gate in result.gate_decisions)
        (result.run_dir / "hil_adjudication.json").write_text(json.dumps(adjudication.model_dump(), indent=2))
        (result.run_dir / "gate2_decision.json").write_text(json.dumps(gate2.model_dump(), indent=2))
        _resume_run_with_progress(result)


def render_hil_v2_status(result: PipelineResult) -> None:
    """Render Gate 1 / Gate 2 observability and HIL status."""
    gate1 = _gate_by_name(result, "gate1_uncertainty_risk_router")
    gate2 = _gate_by_name(result, "gate2_disagreement_detector")
    attention_cls = "attention" if result.requires_attention else "quiet"
    attention_text = t("hil_yes") if result.requires_attention else t("hil_no")

    cards_html = []
    for title, gate in ((t("hil_gate1"), gate1), (t("hil_gate2"), gate2)):
        if not gate:
            body = f'<div class="hil-empty">{esc(t("hil_not_available"))}</div>'
        else:
            trigger_text = ", ".join(gate.trigger_reasons) if gate.trigger_reasons else "—"
            review_dims = ", ".join(dim_label(d) for d in gate.review_dimensions) if gate.review_dimensions else "—"
            body = (
                '<div class="hil-chip-grid">'
                + _chip(t("hil_status"), gate.status, "status")
                + _chip(t("hil_route_score"), gate.route_score)
                + _chip(t("hil_route_band"), gate.route_band)
                + _chip(t("hil_weight_profile"), gate.route_weight_profile)
                + _chip(t("hil_trigger_reasons"), trigger_text, "wide")
                + _chip(t("hil_review_dimensions"), review_dims, "wide")
                + '</div>'
            )
        cards_html.append(
            f'<div class="hil-gate-card"><div class="hil-gate-title">{esc(title)}</div>{body}</div>'
        )

    issue_html = ""
    if result.issue_equivalence:
        summary = result.issue_equivalence.summary
        issue_html = (
            '<div class="hil-equivalence-row">'
            f'<span>{esc(t("hil_issue_equivalence"))}</span>'
            f'<strong>{esc(t("hil_equivalent"))}: {summary.equivalent_count}</strong>'
            f'<strong>{esc(t("hil_new"))}: {summary.new_count}</strong>'
            f'<strong>{esc(t("hil_contradicting"))}: {summary.contradicting_count}</strong>'
            '</div>'
        )

    st.markdown(
        f"""
        <section class="hil-v2-panel {attention_cls}">
            <div class="hil-v2-header">
                <div>
                    <div class="hil-v2-kicker">{esc(t("hil_v2_title"))}</div>
                    <p>{esc(t("hil_v2_subtitle"))}</p>
                </div>
                <div class="hil-status-stack">
                    {_chip(t("hil_pipeline_status"), result.pipeline_status, "status")}
                    {_chip(t("hil_requires_attention"), attention_text, attention_cls)}
                    {_chip(t("hil_prompt_difficulty"), result.prompt_difficulty)}
                </div>
            </div>
            <div class="hil-gate-grid">{"".join(cards_html)}</div>
            {issue_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_radar_chart(result: PipelineResult) -> None:
    """Render plotly radar chart with before/after overlay."""
    if not result.comparison:
        return

    plot_theme = _plotly_theme()
    polar_bg = "#111113" if _is_dark_theme() else "#f8fafc"
    axis_line = "#3F3F46" if _is_dark_theme() else "#cbd5e1"
    axis_tick = "#D4D4D8" if _is_dark_theme() else "#64748b"
    legend_bg = "rgba(24,24,27,0.92)" if _is_dark_theme() else "rgba(255,255,255,0.9)"

    labels = [dim_label(d) for d in DIMENSIONS]
    labels_closed = labels + [labels[0]]  # Close the polygon

    # Post-critique scores (solid)
    post_a = [r.score_a for r in result.comparison.dimension_results]
    post_b = [r.score_b for r in result.comparison.dimension_results]
    post_a_closed = post_a + [post_a[0]]
    post_b_closed = post_b + [post_b[0]]

    # Pre-critique scores (faded)
    pre_a = [r.pre_critique_score_a for r in result.comparison.dimension_results]
    pre_b = [r.pre_critique_score_b for r in result.comparison.dimension_results]
    pre_a_closed = pre_a + [pre_a[0]]
    pre_b_closed = pre_b + [pre_b[0]]

    fig = go.Figure()

    # Pre-critique traces (faded)
    fig.add_trace(go.Scatterpolar(
        r=pre_a_closed, theta=labels_closed,
        fill="toself", name=f"{result.comparison.model_a_name} (pre-critique)",
        line=dict(color="rgba(79,70,229,0.28)", width=1),
        fillcolor="rgba(79,70,229,0.05)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=pre_b_closed, theta=labels_closed,
        fill="toself", name=f"{result.comparison.model_b_name} (pre-critique)",
        line=dict(color="rgba(217,119,6,0.28)", width=1),
        fillcolor="rgba(217,119,6,0.05)",
    ))

    # Post-critique traces (solid)
    fig.add_trace(go.Scatterpolar(
        r=post_a_closed, theta=labels_closed,
        fill="toself", name=f"{result.comparison.model_a_name} (post-critique)",
        line=dict(color="#4F46E5", width=2),
        fillcolor="rgba(79,70,229,0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=post_b_closed, theta=labels_closed,
        fill="toself", name=f"{result.comparison.model_b_name} (post-critique)",
        line=dict(color="#D97706", width=2),
        fillcolor="rgba(217,119,6,0.15)",
    ))

    fig.update_layout(
        polar=dict(
            bgcolor=polar_bg,
            radialaxis=dict(
                visible=True, range=[0, 10],
                gridcolor=plot_theme["grid"], linecolor=axis_line,
                tickfont=dict(color=axis_tick, size=10),
            ),
            angularaxis=dict(
                gridcolor=plot_theme["grid"], linecolor=axis_line,
                tickfont=dict(color=axis_tick, size=11),
            ),
        ),
        paper_bgcolor=plot_theme["paper"],
        plot_bgcolor=plot_theme["paper"],
        font=dict(color=plot_theme["font"]),
        showlegend=True,
        legend=dict(
            bgcolor=legend_bg,
            bordercolor=axis_line,
            font=dict(size=11, color=plot_theme["font"]),
        ),
        margin=dict(t=40, b=40, l=60, r=60),
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True, key=f"radar_{id(result)}")


def _get_reasoning(dim_score) -> str:
    """Return reasoning in the current UI language, falling back to English."""
    if get_lang() == "zh" and getattr(dim_score, "reasoning_zh", ""):
        return dim_score.reasoning_zh
    return dim_score.reasoning


def render_dimension_cards(result: PipelineResult) -> None:
    """Render per-dimension score cards in a 3x2 grid with score bars and hover tooltips."""
    if not result.comparison:
        return

    comp = result.comparison

    # Reasoning source: final revision if available, else initial eval
    if result.revisions:
        src_a = result.revisions[-1].model_a
        src_b = result.revisions[-1].model_b
    elif result.eval_a and result.eval_b:
        src_a = result.eval_a
        src_b = result.eval_b
    else:
        src_a = src_b = None

    cols = st.columns(3)
    for i, dr in enumerate(comp.dimension_results):
        with cols[i % 3]:
            label = esc(dim_label(dr.dimension))
            da = dr.score_a - dr.pre_critique_score_a
            db = dr.score_b - dr.pre_critique_score_b
            da_str = f"+{da}" if da > 0 else ("—" if da == 0 else str(da))
            db_str = f"+{db}" if db > 0 else ("—" if db == 0 else str(db))
            da_cls = "change-up" if da > 0 else ("change-down" if da < 0 else "change-flat")
            db_cls = "change-up" if db > 0 else ("change-down" if db < 0 else "change-flat")

            if dr.winner == "model_a":
                badge = f'<span class="dim-winner-badge winner-a">{esc(comp.model_a_name.split()[0])} ▲</span>'
            elif dr.winner == "model_b":
                badge = f'<span class="dim-winner-badge winner-b">{esc(comp.model_b_name.split()[0])} ▲</span>'
            else:
                badge = '<span class="dim-winner-badge winner-draw">Draw</span>'

            if src_a and src_b:
                ra = getattr(src_a, dr.dimension)
                rb = getattr(src_b, dr.dimension)
                tooltip_html = (
                    f'<div class="dim-tooltip">'
                    f'<div class="dim-tooltip-model tip-a">{esc(comp.model_a_name)}</div>'
                    f'<div class="dim-tooltip-text">{esc(_get_reasoning(ra))}</div>'
                    f'<hr class="dim-tooltip-divider">'
                    f'<div class="dim-tooltip-model tip-b">{esc(comp.model_b_name)}</div>'
                    f'<div class="dim-tooltip-text">{esc(_get_reasoning(rb))}</div>'
                    f'</div>'
                )
            else:
                tooltip_html = ""

            st.markdown(f"""
            <div class="dim-card">
                {tooltip_html}
                <div>{badge}<span class="dim-name">{label}</span></div>
                <div class="dim-score-row">
                    <span class="dim-score-label label-a">{esc(comp.model_a_name)}</span>
                    <div class="dim-score-right">
                        <span class="dim-score-val val-a">{dr.score_a}</span>
                        <span class="dim-score-delta {da_cls}">{da_str}</span>
                    </div>
                </div>
                <div class="dim-score-row">
                    <span class="dim-score-label label-b">{esc(comp.model_b_name)}</span>
                    <div class="dim-score-right">
                        <span class="dim-score-val val-b">{dr.score_b}</span>
                        <span class="dim-score-delta {db_cls}">{db_str}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_dimension_comments_table(result: PipelineResult) -> None:
    """Render a table of per-dimension reasoning comments for both models."""
    if not result.comparison:
        return

    comp = result.comparison
    if result.revisions:
        src_a = result.revisions[-1].model_a
        src_b = result.revisions[-1].model_b
    elif result.eval_a and result.eval_b:
        src_a = result.eval_a
        src_b = result.eval_b
    else:
        return

    rows_html = []
    for dr in comp.dimension_results:
        d = dr.dimension
        ra = getattr(src_a, d)
        rb = getattr(src_b, d)
        rows_html.append(
            f'<tr>'
            f'<td class="tbl-dim-cell">{esc(dim_label(d))}</td>'
            f'<td>{esc(_get_reasoning(ra))}</td>'
            f'<td>{esc(_get_reasoning(rb))}</td>'
            f'</tr>'
        )

    st.markdown(
        f'<table class="dim-comment-table">'
        f'<thead><tr>'
        f'<th></th>'
        f'<th class="th-a">{esc(comp.model_a_name)}</th>'
        f'<th class="th-b">{esc(comp.model_b_name)}</th>'
        f'</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table>',
        unsafe_allow_html=True,
    )


def render_critique_transcript(result: PipelineResult) -> None:
    """Render the multi-round critique transcript."""
    with st.expander(t("critique_transcript"), expanded=False):
        # Step 1: Initial evaluation
        if result.eval_a and result.eval_b:
            eval_summary_parts = []
            for d in DIMENSIONS:
                label = dim_label(d)
                sa = getattr(result.eval_a, d)
                sb = getattr(result.eval_b, d)
                eval_summary_parts.append(
                    f'<div class="step-row"><strong>{esc(label)}:</strong> '
                    f'A={sa.score} ({esc(sa.reasoning[:80])}…) &nbsp;|&nbsp; '
                    f'B={sb.score} ({esc(sb.reasoning[:80])}…)</div>'
                )

            st.markdown(f"""
            <div class="critique-step opus">
                <div class="step-label">{t("step1_label")}</div>
                <div class="step-content">{"".join(eval_summary_parts)}</div>
            </div>
            """, unsafe_allow_html=True)

        # Render each critique-revision round
        for round_idx, crit in enumerate(result.critiques):
            # Critique step
            step_label_key = "step2_label" if round_idx == 0 else "step4_label"
            css_class = "gpt" if round_idx == 0 else "revised"

            critique_parts = [
                f'<div class="step-row"><strong>{t("overall_assessment")}</strong> {esc(crit.overall_assessment)}</div>'
            ]
            for dc in crit.dimension_critiques:
                label = dim_label(dc.dimension)
                critique_parts.append(
                    f'<div class="step-row"><strong>{esc(label)}:</strong> {esc(dc.critique[:120])}… '
                    f'(A: {dc.original_score_model_a}→{dc.suggested_score_model_a}, '
                    f'B: {dc.original_score_model_b}→{dc.suggested_score_model_b})</div>'
                )
            critique_parts.append(
                f'<div class="step-row"><strong>{t("bias_detection")}</strong> {esc(crit.bias_detection)}</div>'
            )

            st.markdown(f"""
            <div class="critique-step {css_class}">
                <div class="step-label">{t(step_label_key)}</div>
                <div class="step-content">{"".join(critique_parts)}</div>
            </div>
            """, unsafe_allow_html=True)

            # Revision step (if available for this round)
            if round_idx < len(result.revisions):
                rev = result.revisions[round_idx]
                rev_label_key = "step3_label" if round_idx == 0 else "step5_label"
                revised_parts = []
                for d in DIMENSIONS:
                    label = dim_label(d)
                    ra = getattr(rev.model_a, d)
                    rb = getattr(rev.model_b, d)
                    accepted_a = t("accepted") if ra.critique_accepted else t("rejected")
                    accepted_b = t("accepted") if rb.critique_accepted else t("rejected")
                    revised_parts.append(
                        f'<div class="step-row"><strong>{esc(label)}:</strong> '
                        f'A={ra.score} ({accepted_a}: {esc(ra.revision_note[:60])}…) &nbsp;|&nbsp; '
                        f'B={rb.score} ({accepted_b}: {esc(rb.revision_note[:60])}…)</div>'
                    )

                st.markdown(f"""
                <div class="critique-step opus">
                    <div class="step-label">{t(rev_label_key)}</div>
                    <div class="step-content">{"".join(revised_parts)}</div>
                </div>
                """, unsafe_allow_html=True)

        # Show unavailable message if no critiques at all
        if not result.critiques:
            st.markdown(f"""
            <div class="critique-step gpt">
                <div class="step-label">{t("step2_label")}</div>
                <div class="step-content" style="color:#ffb347;">{t("step2_unavailable")}</div>
            </div>
            """, unsafe_allow_html=True)


def render_raw_json(result: PipelineResult) -> None:
    """Render raw JSON expanders for each evaluation step."""
    # Row 1: Initial + Round 1 critique/revision
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.expander(t("json_initial")):
            if result.eval_a and result.eval_b:
                st.json({
                    "model_a": result.eval_a.model_dump(),
                    "model_b": result.eval_b.model_dump(),
                })

    with col2:
        with st.expander(t("json_critique")):
            if result.critiques:
                st.json(result.critiques[0].model_dump())
            else:
                st.write(t("json_unavailable"))

    with col3:
        with st.expander(t("json_revised")):
            if result.revisions:
                st.json(result.revisions[0].model_dump())
            else:
                st.write(t("json_unavailable"))

    # Row 2: Round 2 critique/revision (if available)
    if len(result.critiques) > 1 or len(result.revisions) > 1:
        col4, col5, _col6 = st.columns(3)
        with col4:
            with st.expander(t("json_critique_r2")):
                if len(result.critiques) > 1:
                    st.json(result.critiques[1].model_dump())
                else:
                    st.write(t("json_unavailable"))
        with col5:
            with st.expander(t("json_revised_r2")):
                if len(result.revisions) > 1:
                    st.json(result.revisions[1].model_dump())
                else:
                    st.write(t("json_unavailable"))

    if result.gate_decisions or result.issue_equivalence:
        gate1 = _gate_by_name(result, "gate1_uncertainty_risk_router")
        gate2 = _gate_by_name(result, "gate2_disagreement_detector")
        col7, col8, col9 = st.columns(3)
        with col7:
            with st.expander(t("json_gate1")):
                if gate1:
                    st.json(gate1.model_dump())
                else:
                    st.write(t("json_unavailable"))
        with col8:
            with st.expander(t("json_gate2")):
                if gate2:
                    st.json(gate2.model_dump())
                else:
                    st.write(t("json_unavailable"))
        with col9:
            with st.expander(t("json_issue_equivalence")):
                if result.issue_equivalence:
                    st.json(result.issue_equivalence.model_dump())
                else:
                    st.write(t("json_unavailable"))

# --- Render selected run ---
focused_index, focused_run = _focused_entry
render_result(focused_run, 0)
