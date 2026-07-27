"""Dashboard page rendering for completed image-evaluation runs."""

from collections.abc import Callable
from datetime import datetime
from html import escape as esc

import plotly.graph_objects as go
import streamlit as st

from config import model_label
from i18n import t
from pipeline import PipelineResult
from ui_theme import model_color, plotly_theme

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


def run_model_slots(run: PipelineResult) -> tuple[tuple[str, str], tuple[str, str]]:
    """((key_a, label_a), (key_b, label_b)) for a run.

    Slot A is not a fixed model — the same model appears in either slot across
    runs, so anything aggregated per model must go through this rather than
    counting "model_a" wins.
    """
    comp = run.comparison
    fallback_a = (comp.model_a_name if comp else "") or run.model_a_label
    fallback_b = (comp.model_b_name if comp else "") or run.model_b_label
    return (
        (run.model_a_key, model_label(run.model_a_key, fallback_a)),
        (run.model_b_key, model_label(run.model_b_key, fallback_b)),
    )


def winning_model(run: PipelineResult) -> tuple[str, str] | None:
    """(key, label) of the model that actually won, or None for a draw."""
    comp = run.comparison
    if not comp:
        return None
    slot_a, slot_b = run_model_slots(run)
    if comp.overall_winner == "model_a":
        return slot_a
    if comp.overall_winner == "model_b":
        return slot_b
    return None


def winner_pill(run: PipelineResult) -> str:
    """Winner badge naming the actual model, colored by model identity."""
    winner = winning_model(run)
    if winner is None:
        return f'<span class="dashboard-winner-pill draw">{esc(t("draw_label"))}</span>'
    key, label = winner
    return (
        f'<span class="dashboard-winner-pill" title="{esc(label)}" '
        f'style="--pill-color:{esc(model_color(key))}">'
        f'● {esc(label)}</span>'
    )


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
    # The winner column holds a full model name now, not a 3-letter abbreviation,
    # so it needs real width — taken from the prompt column, which wraps freely.
    dashboard_columns = [1.1, 1.4, 4.3, 3.0, 1.0, 1.3]
    rows = []
    for run in runs:
        comp = run.comparison
        parsed_date = parse_run_datetime(run.timestamp) or datetime.min
        category = _category_label(_prompt_category(run.prompt))
        (_key_a, label_a), (_key_b, label_b) = run_model_slots(run)
        rows.append({
            "date_dt": parsed_date,
            "date": history_date_label(run.timestamp),
            "category": category,
            "prompt": run.prompt,
            "winner_html": winner_pill(run),
            "margin": comp.margin or "—",
            "score": f"{comp.model_a_mean:.2f} / {comp.model_b_mean:.2f}",
            # Slot order varies per run, so name the models behind the numbers.
            "score_title": f"{label_a} {comp.model_a_mean:.2f} · {label_b} {comp.model_b_mean:.2f}",
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
            st.markdown(f'<div class="dashboard-cell">{row["winner_html"]}</div>', unsafe_allow_html=True)
        with cols[4]:
            st.markdown(f'<div class="dashboard-cell stable">{esc(str(row["margin"]))}</div>', unsafe_allow_html=True)
        with cols[5]:
            st.markdown(
                f'<div class="dashboard-cell stable score" title="{esc(row["score_title"])}">{esc(row["score"])}</div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div class="dashboard-row-separator"></div>', unsafe_allow_html=True)


def _dashboard_metric(label: str, value: str, subtext: str = "") -> str:
    return (
        f'<div class="dashboard-metric">'
        f'<span>{esc(label)}</span>'
        f'<strong>{esc(value)}</strong>'
        f'<em>{esc(subtext)}</em>'
        f'</div>'
    )


def _model_standings(runs: list[PipelineResult]) -> tuple[list[dict], int]:
    """Per-model totals across every run, plus the draw count.

    Keyed on the model key rather than the slot: the same model appears in both
    slots across history, so slot-based totals conflate different models.
    """
    standings: dict[str, dict] = {}
    draws = 0

    def _bucket(key: str, label: str) -> dict:
        return standings.setdefault(key, {"key": key, "label": label, "runs": 0, "wins": 0, "score_total": 0.0})

    for run in runs:
        comp = run.comparison
        (key_a, label_a), (key_b, label_b) = run_model_slots(run)
        bucket_a, bucket_b = _bucket(key_a, label_a), _bucket(key_b, label_b)
        bucket_a["runs"] += 1
        bucket_a["score_total"] += comp.model_a_mean
        bucket_b["runs"] += 1
        bucket_b["score_total"] += comp.model_b_mean
        winner = winning_model(run)
        if winner is None:
            draws += 1
        else:
            _bucket(*winner)["wins"] += 1

    rows = []
    for entry in standings.values():
        runs_count = entry["runs"] or 1
        rows.append({
            **entry,
            "win_rate": entry["wins"] / runs_count,
            "avg_score": entry["score_total"] / runs_count,
        })
    rows.sort(key=lambda row: (row["wins"], row["win_rate"], row["avg_score"]), reverse=True)
    return rows, draws


def _render_model_leaderboard(rows: list[dict]) -> None:
    """Per-model standings. Also the table view that gives the donut's
    low-contrast palette slots their required relief."""
    columns = [3.4, 1.0, 1.0, 1.2, 1.2]
    st.markdown(f'<div class="dashboard-table-title">{esc(t("dashboard_leaderboard"))}</div>', unsafe_allow_html=True)
    header = st.columns(columns, gap="small")
    for col, key in zip(header, ("dashboard_model", "dashboard_runs", "dashboard_wins", "dashboard_win_rate", "dashboard_avg_score")):
        with col:
            st.markdown(f'<div class="dashboard-header-cell">{esc(t(key))}</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-row-separator"></div>', unsafe_allow_html=True)

    for row in rows:
        cols = st.columns(columns, gap="small")
        with cols[0]:
            st.markdown(
                f'<div class="dashboard-cell"><span class="dashboard-model-swatch" '
                f'style="--pill-color:{esc(model_color(row["key"]))}"></span>{esc(row["label"])}</div>',
                unsafe_allow_html=True,
            )
        for col, value in zip(cols[1:], (
            str(row["runs"]), str(row["wins"]), f'{row["win_rate"] * 100:.0f}%', f'{row["avg_score"]:.2f}',
        )):
            with col:
                st.markdown(f'<div class="dashboard-cell stable">{esc(value)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="dashboard-row-separator"></div>', unsafe_allow_html=True)


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

    standings, draws = _model_standings(runs)
    category_counts: dict[str, int] = {}
    for run in runs:
        category = _prompt_category(run.prompt)
        category_counts[category] = category_counts.get(category, 0) + 1

    metric_html = "".join([
        _dashboard_metric(t("dashboard_total_runs"), str(len(runs))),
        _dashboard_metric(t("dashboard_models_compared"), str(len(standings))),
        _dashboard_metric(t("dashboard_draws"), str(draws)),
    ])
    st.markdown(f'<div class="dashboard-metric-grid">{metric_html}</div>', unsafe_allow_html=True)

    _render_model_leaderboard(standings)

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
        winner_rows = [row for row in standings if row["wins"] > 0]
        labels = [row["label"] for row in winner_rows]
        values = [row["wins"] for row in winner_rows]
        colors = [model_color(row["key"]) for row in winner_rows]
        if draws:
            labels.append(t("draw_label"))
            values.append(draws)
            colors.append(model_color(None))
        fig = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.48,
            marker=dict(colors=colors, line=dict(color=chart_theme["paper"], width=2)),
            # Direct labels + the 2px surface gap above are the secondary encoding
            # the palette's dark-mode CVD separation requires.
            textinfo="label+value",
            textposition="outside",
            sort=False,
        ))
        fig.update_layout(
            title=t("dashboard_winner_distribution"),
            paper_bgcolor=chart_theme["paper"],
            plot_bgcolor=chart_theme["plot"],
            font=dict(color=chart_theme["font"]),
            # Outside labels need horizontal room or long model names clip.
            margin=dict(t=50, b=40, l=90, r=90),
            height=330,
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True, key="dashboard_winners")

    _render_dashboard_recent_runs(runs, parse_run_datetime, history_date_label)
