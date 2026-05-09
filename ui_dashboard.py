"""Dashboard page rendering for completed image-evaluation runs."""

from collections.abc import Callable
from datetime import datetime
from html import escape as esc

import plotly.graph_objects as go
import streamlit as st

from i18n import t
from pipeline import PipelineResult
from ui_theme import plotly_theme

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

ParseRunDatetime = Callable[[str], datetime | None]
HistoryDateLabel = Callable[[str], str]


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


def _render_dashboard_recent_runs(
    runs: list[PipelineResult],
    parse_run_datetime: ParseRunDatetime,
    history_date_label: HistoryDateLabel,
) -> None:
    dashboard_columns = [1.15, 1.65, 6.85, 1.0, 0.9, 1.2]
    rows = []
    for run in runs:
        comp = run.comparison
        parsed_date = parse_run_datetime(run.timestamp) or datetime.min
        category = _category_label(_prompt_category(run.prompt))
        rows.append({
            "date_dt": parsed_date,
            "date": history_date_label(run.timestamp),
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


def render_dashboard_page(
    all_runs: list[PipelineResult],
    parse_run_datetime: ParseRunDatetime,
    history_date_label: HistoryDateLabel,
) -> None:
    """Render aggregate analytics across completed comparison runs."""
    runs = [run for run in all_runs if run.comparison]
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
    chart_theme = plotly_theme()
    with col1:
        category_labels = [_category_label(category) for category in category_counts]
        category_values = list(category_counts.values())
        fig = go.Figure(go.Bar(x=category_labels, y=category_values, marker_color="#4F46E5"))
        fig.update_layout(
            title=t("dashboard_category_distribution"),
            paper_bgcolor=chart_theme["paper"],
            plot_bgcolor=chart_theme["plot"],
            font=dict(color=chart_theme["font"]),
            margin=dict(t=50, b=40, l=40, r=20),
            height=330,
        )
        fig.update_xaxes(gridcolor=chart_theme["grid"])
        fig.update_yaxes(gridcolor=chart_theme["grid"], rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True, key="dashboard_categories")

    with col2:
        winner_labels = [_winner_label("model_a"), _winner_label("model_b"), _winner_label("draw")]
        winner_values = [winner_counts.get("model_a", 0), winner_counts.get("model_b", 0), winner_counts.get("draw", 0)]
        fig = go.Figure(go.Pie(labels=winner_labels, values=winner_values, hole=0.48, marker_colors=["#4F46E5", "#D97706", "#A1A1AA"]))
        fig.update_layout(
            title=t("dashboard_winner_distribution"),
            paper_bgcolor=chart_theme["paper"],
            plot_bgcolor=chart_theme["plot"],
            font=dict(color=chart_theme["font"]),
            margin=dict(t=50, b=40, l=20, r=20),
            height=330,
        )
        st.plotly_chart(fig, use_container_width=True, key="dashboard_winners")

    _render_dashboard_recent_runs(runs, parse_run_datetime, history_date_label)
