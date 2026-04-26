"""Streamlit dashboard for the Image Eval Pipeline.

Dark theme, side-by-side comparison, radar charts with before/after
critique overlay, critique transcript, and raw JSON inspection.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from i18n import dim_label, get_lang, t
from pipeline import PipelineResult, run_pipeline
from schemas import DIMENSIONS

# --- Page config ---
st.set_page_config(
    page_title="Image Eval Pipeline",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS (dark theme from design tokens) ---
st.markdown("""
<style>
    .stApp { background-color: #1a1a2e; }
    .main .block-container { max-width: 1200px; padding-top: 2rem; }

    /* Header */
    .eval-header { text-align: center; margin-bottom: 24px; border-bottom: 1px solid #333; padding-bottom: 16px; }
    .eval-header h1 { font-size: 20px; font-weight: 400; letter-spacing: 2px; color: #8888cc; margin: 0; }
    .eval-header p { font-size: 12px; color: #666; margin-top: 4px; }

    /* Image cards */
    .model-label { font-size: 11px; text-transform: uppercase; letter-spacing: 2px; color: #8888cc; margin-bottom: 8px; }
    .image-card { background: #16213e; border: 1px solid #333; border-radius: 8px; padding: 16px; }

    /* Uniform image containers */
    [data-testid="stImage"] {
        background: #16213e;
        border: 1px solid #333;
        border-radius: 8px;
        overflow: hidden;
        height: 480px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    [data-testid="stImage"] img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }

    /* Score badges */
    .score-badge { display: inline-block; background: #1a1a3e; border: 1px solid #444; border-radius: 4px; padding: 4px 8px; font-size: 11px; margin: 2px; }
    .score-dim { color: #999; }
    .score-val { color: #5c4dff; font-weight: 600; }

    /* Winner banner */
    .winner-banner {
        text-align: center; padding: 16px; margin: 16px 0;
        background: #16213e; border: 2px solid #5c4dff; border-radius: 8px;
        font-size: 18px; color: #e0e0e0;
    }
    .winner-banner .winner-name { color: #4ecdc4; font-weight: 700; font-size: 22px; }
    .winner-banner .unchallenged { color: #ffb347; font-size: 13px; }

    /* Dimension cards */
    .dim-card {
        background: #16213e; border: 1px solid #333; border-radius: 6px; padding: 12px;
        margin-bottom: 8px;
    }
    .dim-name { font-size: 11px; color: #999; text-transform: uppercase; letter-spacing: 1px; }
    .dim-scores { display: flex; justify-content: space-between; margin-top: 8px; }
    .dim-score { font-size: 18px; font-weight: 600; }
    .dim-score-a { color: #ff6b6b; }
    .dim-score-b { color: #4ecdc4; }
    .dim-delta { font-size: 10px; color: #666; margin-top: 4px; }
    .dim-winner { font-size: 10px; font-weight: 600; margin-top: 2px; }

    /* Critique steps */
    .critique-step { margin-bottom: 16px; padding: 12px 16px; border-left: 3px solid #333; background: #16213e; border-radius: 0 6px 6px 0; }
    .critique-step.opus { border-left-color: #5c4dff; }
    .critique-step.gpt { border-left-color: #ff6b6b; }
    .critique-step.revised { border-left-color: #4ecdc4; }
    .step-label { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #999; margin-bottom: 4px; }
    .step-content { font-size: 13px; color: #ccc; line-height: 1.6; }

    /* Error states */
    .error-card { background: rgba(255,107,107,0.1); border: 1px solid rgba(255,107,107,0.3); border-radius: 8px; padding: 16px; }
    .error-text { color: #ff6b6b; font-size: 13px; }

    /* Empty state */
    .empty-state { text-align: center; color: #666; padding: 48px 24px; font-size: 14px; }

    /* Run divider */
    .run-divider { border-top: 1px solid #333; margin: 32px 0; padding-top: 16px; }
    .run-label { font-size: 12px; color: #666; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


# --- Session state init ---
if "runs" not in st.session_state:
    st.session_state.runs = []  # List of PipelineResult
if "running" not in st.session_state:
    st.session_state.running = False
if "lang" not in st.session_state:
    st.session_state.lang = "zh"

# --- Language switcher (sidebar) ---
_lang_options = {"English": "en", "中文": "zh"}
_lang_display = {v: k for k, v in _lang_options.items()}
_selected_lang = st.sidebar.selectbox(
    "Language / 语言",
    list(_lang_options.keys()),
    index=list(_lang_options.values()).index(get_lang()),
)
if _lang_options[_selected_lang] != st.session_state.lang:
    st.session_state.lang = _lang_options[_selected_lang]
    st.rerun()


# --- Pre-baked results ---
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
                results[prompt] = result
        except Exception:
            continue
    return results


def _load_result_from_dir(run_dir: Path, prompt: str) -> PipelineResult | None:
    """Reconstruct a PipelineResult from saved JSON files."""
    from schemas import (
        ImageEvaluation, CritiqueResponse, RevisedEvaluation, ComparisonResult,
    )
    result = PipelineResult()
    result.prompt = prompt
    result.run_dir = run_dir
    result.timestamp = run_dir.name.replace("_", " ")

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

    # Load critique
    critique_file = run_dir / "critique.json"
    if critique_file.exists():
        result.critique = CritiqueResponse(**json.loads(critique_file.read_text()))

    # Load revised
    revised_file = run_dir / "revised.json"
    if revised_file.exists():
        result.revised = RevisedEvaluation(**json.loads(revised_file.read_text()))

    # Load comparison
    comp_file = run_dir / "comparison.json"
    if comp_file.exists():
        result.comparison = ComparisonResult(**json.loads(comp_file.read_text()))

    return result


# --- Header ---
st.markdown(f"""
<div class="eval-header">
    <h1>{t("header_title")}</h1>
    <p>{t("header_subtitle")}</p>
</div>
""", unsafe_allow_html=True)

# --- Pre-baked dropdown ---
prebaked = load_prebaked_results()
if prebaked:
    options = [""] + list(prebaked.keys())
    selected = st.selectbox(
        t("select_example"),
        options,
        format_func=lambda x: t("select_example_placeholder") if x == "" else x[:80],
    )
    if selected and selected in prebaked:
        loaded = prebaked[selected]
        if not st.session_state.runs or st.session_state.runs[0].prompt != loaded.prompt:
            st.session_state.runs.insert(0, loaded)
            st.rerun()

# --- Prompt input ---
col_input, col_btn = st.columns([5, 1])
with col_input:
    prompt = st.text_input(
        "Prompt",
        placeholder=t("input_placeholder"),
        label_visibility="collapsed",
    )
with col_btn:
    generate_clicked = st.button(
        t("btn_generate"),
        type="primary",
        disabled=st.session_state.running,
        use_container_width=True,
    )

# --- Run pipeline ---
if generate_clicked and prompt.strip():
    st.session_state.running = True

    with st.status(t("status_running"), expanded=True) as status:
        def on_stage(stage: str) -> None:
            status.update(label=stage)

        try:
            result = run_pipeline(prompt.strip(), on_stage=on_stage)
            st.session_state.runs.insert(0, result)
        except Exception as e:
            st.error(t("error_pipeline_failed", error=e))
        finally:
            st.session_state.running = False
            status.update(label=t("status_done"), state="complete", expanded=False)

    st.rerun()

# --- Empty state ---
if not st.session_state.runs:
    st.markdown(f"""
    <div class="empty-state">
        {t("empty_state")}
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# --- Render results ---


def render_result(result: PipelineResult, index: int) -> None:
    """Render a single pipeline result."""
    if index > 0:
        st.markdown('<div class="run-divider"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="run-label">{t("run_label", prompt=result.prompt[:100], timestamp=result.timestamp)}</div>', unsafe_allow_html=True)

    # --- Errors ---
    if result.errors:
        for err in result.errors:
            st.markdown(f'<div class="error-card"><div class="error-text">⚠ {err}</div></div>', unsafe_allow_html=True)

    # --- Images ---
    gpt_path = result.image_paths.get("gpt_image_2")
    gemini_path = result.image_paths.get("gemini_3_pro")

    if gpt_path or gemini_path:
        col_a, col_b = st.columns(2)
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

    if not result.comparison:
        return

    comp = result.comparison
    is_unchallenged = result.critique is None

    # --- Winner banner ---
    winner_display = {
        "model_a": comp.model_a_name,
        "model_b": comp.model_b_name,
        "draw": "Draw",
    }
    winner_name = winner_display.get(comp.overall_winner, comp.overall_winner)
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

    # --- Critique transcript ---
    render_critique_transcript(result)

    # --- Raw JSON ---
    render_raw_json(result)


def render_radar_chart(result: PipelineResult) -> None:
    """Render plotly radar chart with before/after overlay."""
    if not result.comparison:
        return

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
        line=dict(color="rgba(255,107,107,0.3)", width=1),
        fillcolor="rgba(255,107,107,0.05)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=pre_b_closed, theta=labels_closed,
        fill="toself", name=f"{result.comparison.model_b_name} (pre-critique)",
        line=dict(color="rgba(78,205,196,0.3)", width=1),
        fillcolor="rgba(78,205,196,0.05)",
    ))

    # Post-critique traces (solid)
    fig.add_trace(go.Scatterpolar(
        r=post_a_closed, theta=labels_closed,
        fill="toself", name=f"{result.comparison.model_a_name} (post-critique)",
        line=dict(color="#ff6b6b", width=2),
        fillcolor="rgba(255,107,107,0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=post_b_closed, theta=labels_closed,
        fill="toself", name=f"{result.comparison.model_b_name} (post-critique)",
        line=dict(color="#4ecdc4", width=2),
        fillcolor="rgba(78,205,196,0.15)",
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="#16213e",
            radialaxis=dict(
                visible=True, range=[0, 10],
                gridcolor="#333", linecolor="#333",
                tickfont=dict(color="#666", size=10),
            ),
            angularaxis=dict(
                gridcolor="#333", linecolor="#333",
                tickfont=dict(color="#999", size=11),
            ),
        ),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(22,33,62,0.8)",
            bordercolor="#333",
            font=dict(size=11, color="#e0e0e0"),
        ),
        margin=dict(t=40, b=40, l=60, r=60),
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True, key=f"radar_{id(result)}")


def render_dimension_cards(result: PipelineResult) -> None:
    """Render per-dimension score cards in a 3x2 grid."""
    if not result.comparison:
        return

    cols = st.columns(3)
    for i, dim_result in enumerate(result.comparison.dimension_results):
        with cols[i % 3]:
            label = dim_label(dim_result.dimension)
            delta_a = dim_result.score_a - dim_result.pre_critique_score_a
            delta_b = dim_result.score_b - dim_result.pre_critique_score_b
            delta_a_str = f"+{delta_a}" if delta_a > 0 else str(delta_a)
            delta_b_str = f"+{delta_b}" if delta_b > 0 else str(delta_b)

            winner_indicator = ""
            if dim_result.winner == "model_a":
                winner_indicator = '<div class="dim-winner" style="color:#ff6b6b;">▲ A</div>'
            elif dim_result.winner == "model_b":
                winner_indicator = '<div class="dim-winner" style="color:#4ecdc4;">▲ B</div>'
            else:
                winner_indicator = '<div class="dim-winner" style="color:#666;">= Draw</div>'

            st.markdown(f"""
            <div class="dim-card">
                <div class="dim-name">{label}</div>
                <div class="dim-scores">
                    <span class="dim-score dim-score-a">{dim_result.score_a}</span>
                    <span class="dim-score dim-score-b">{dim_result.score_b}</span>
                </div>
                <div class="dim-delta">{t("revised_delta", a=delta_a_str, b=delta_b_str)}</div>
                {winner_indicator}
            </div>
            """, unsafe_allow_html=True)


def render_critique_transcript(result: PipelineResult) -> None:
    """Render the 3-step critique transcript."""
    with st.expander(t("critique_transcript"), expanded=False):
        # Step 1: Initial evaluation
        if result.eval_a and result.eval_b:
            eval_summary_parts = []
            for d in DIMENSIONS:
                label = dim_label(d)
                sa = getattr(result.eval_a, d)
                sb = getattr(result.eval_b, d)
                eval_summary_parts.append(f"**{label}:** A={sa.score} ({sa.reasoning[:80]}...) | B={sb.score} ({sb.reasoning[:80]}...)")

            st.markdown(f"""
            <div class="critique-step opus">
                <div class="step-label">{t("step1_label")}</div>
                <div class="step-content">{"<br>".join(eval_summary_parts)}</div>
            </div>
            """, unsafe_allow_html=True)

        # Step 2: GPT-5.4 critique
        if result.critique:
            critique_parts = [f"**{t('overall_assessment')}** {result.critique.overall_assessment}"]
            for dc in result.critique.dimension_critiques:
                label = dim_label(dc.dimension)
                critique_parts.append(
                    f"**{label}:** {dc.critique[:120]}... "
                    f"(A: {dc.original_score_model_a}→{dc.suggested_score_model_a}, "
                    f"B: {dc.original_score_model_b}→{dc.suggested_score_model_b})"
                )
            critique_parts.append(f"**{t('bias_detection')}** {result.critique.bias_detection}")

            st.markdown(f"""
            <div class="critique-step gpt">
                <div class="step-label">{t("step2_label")}</div>
                <div class="step-content">{"<br>".join(critique_parts)}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="critique-step gpt">
                <div class="step-label">{t("step2_label")}</div>
                <div class="step-content" style="color:#ffb347;">{t("step2_unavailable")}</div>
            </div>
            """, unsafe_allow_html=True)

        # Step 3: Revised evaluation
        if result.revised:
            revised_parts = []
            for d in DIMENSIONS:
                label = dim_label(d)
                ra = getattr(result.revised.model_a, d)
                rb = getattr(result.revised.model_b, d)
                accepted_a = t("accepted") if ra.critique_accepted else t("rejected")
                accepted_b = t("accepted") if rb.critique_accepted else t("rejected")
                revised_parts.append(
                    f"**{label}:** A={ra.score} ({accepted_a}: {ra.revision_note[:60]}...) | "
                    f"B={rb.score} ({accepted_b}: {rb.revision_note[:60]}...)"
                )

            st.markdown(f"""
            <div class="critique-step revised">
                <div class="step-label">{t("step3_label")}</div>
                <div class="step-content">{"<br>".join(revised_parts)}</div>
            </div>
            """, unsafe_allow_html=True)


def render_raw_json(result: PipelineResult) -> None:
    """Render raw JSON expanders for each evaluation step."""
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
            if result.critique:
                st.json(result.critique.model_dump())
            else:
                st.write(t("json_unavailable"))

    with col3:
        with st.expander(t("json_revised")):
            if result.revised:
                st.json(result.revised.model_dump())
            else:
                st.write(t("json_unavailable"))


# --- Render all runs ---
for i, run in enumerate(st.session_state.runs):
    render_result(run, i)
