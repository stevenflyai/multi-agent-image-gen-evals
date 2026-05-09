"""Architecture page rendering for the Streamlit dashboard."""

from html import escape as esc

import streamlit as st

from i18n import t


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
