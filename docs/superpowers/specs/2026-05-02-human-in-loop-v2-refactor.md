# Human-in-the-loop V2 Refactor Design

## Goal

根据 Architecture V2，引入 Human-in-the-loop (HIL) 评审/裁决能力，并重构当前 image evaluation pipeline，使评分输出更可校准、更可审核、更适合人工介入。

本阶段只做设计，不修改运行代码。讨论确认后再进入实现。

## Current State

当前程序流程是：

1. `generate.py` 并行生成两张图：GPT Image-2 和 Gemini 3 Pro。
2. `evaluate.py` 使用 Claude Opus 初评两张图，输出每个维度的 `score/reasoning/reasoning_zh`。
3. `critique.py` round 1 使用 GPT-5.4 critique 初评。
4. `revise.py` 使用 Claude Opus 根据 round 1 critique 修正。
5. `critique.py` round 2 使用 Gemini 3.1 Pro critique revised scores。
6. `revise.py` 使用 Claude Opus 根据 round 2 critique 修正。
7. `compare.py` 基于最终 revised scores 计算 winner。
8. `pipeline.py` 持久化 `evaluation.json`、`critique_r*.json`、`revised_r*.json`、`comparison.json`、`summary.json`。

当前限制：

- 评分 schema 缺少 `evidence` 和 `confidence`。
- compare 逻辑只基于均分和最大单维 lead，没有 prompt difficulty、margin、conflict notes。
- HIL 只存在于 Architecture V2 UI 图中，还没有落到数据模型、流程控制或 Streamlit 交互。
- critique/revision 是固定 AI 闭环，缺少低置信度维度仲裁、分歧裁决和人工标签 audit trail。
- prompt 仍偏“reasoning”叙述，没有强制每个分数给 concrete visual evidence 和 confidence。

## Target V2 Flow

Architecture V2 对应的目标流程：

1. User Prompt
2. Pipeline Orchestrator
3. Parallel Generation
   - Image A: GPT Image-2
   - Image B: Gemini 3 Pro
4. Evaluation Agent
   - 输出每张图 6 个维度的 `score/evidence/confidence`
   - 输出 `prompt_difficulty`
   - 输出初始 `comparison`
5. Gate 1: Uncertainty & Risk Router
  - 低风险：HIL 保持关闭，直接进入 AI review chain
  - 高风险：只对 04 标记的高不确定维度触发 HIL arbitration，再进入 AI review chain
  - 路由依据不是单一 self-reported confidence，而是综合 `route_score`
6. Sequential Critique & Revision Chain, Part 1
  - 05 GPT-5.4 independent critique of 04
  - 06 Claude revision using 04 + 05
  - 07 Gemini independent critique of 06 only; 07 prompt must never include 05 output
7. Gate 2: Disagreement Detector
  - agree：直接进入 08 final revision
  - disagree：触发 HIL adjudicator，让用户选择 `agree_with_05` / `agree_with_07` / `both_partially_right`，再进入 08 final revision
8. Final Revision Agent
  - 08 使用 05 + 06 + 07，并可使用 Gate 1 / Gate 2 HIL labels，产出 final revised evaluation
9. Decision Engine
  - weighted: 结合 final AI evaluation、critic signals、Gate 1 arbitration labels、Gate 2 adjudication labels 生成最终 winner
10. Archive Module
   - 持久化 prompts、images、scores、evidence、confidence、HIL labels 和所有 AI + HIL artifact

## Design Principles

- Preserve current pipeline behavior when no HIL input exists.
- Make every automated score auditable through concrete visual evidence.
- Treat confidence as uncertainty, not quality.
- Keep HIL off by default; only trigger when gate logic detects specific model difficulty.
- HIL should not redo model work. It should answer narrow arbitration questions where the model is uncertain, inconsistent, or in disagreement.
- Keep HIL optional and resumable: pipeline can pause at gates and resume after human input.
- Keep schemas explicit and backward-compatible where possible.
- Avoid model/provider identity bias in prompts and comparison logic.
- Store all human labels as structured artifacts, not silent score mutations.

## Proposed Data Model

### Evaluation Score

Replace or evolve `DimensionScore` from:

```json
{ "score": 8, "reasoning": "...", "reasoning_zh": "..." }
```

to:

```json
{
  "score": 8,
  "evidence": "Concrete visual evidence tied to image regions/objects/issues.",
  "confidence": 4,
  "reasoning_zh": "Optional localized display text."
}
```

Recommendation:

- Make `evidence` the canonical field.
- Keep `reasoning` temporarily as a backward-compatible alias or derived display field.
- Add `confidence: int` with range 1-5.
- Keep `reasoning_zh` only if UI still needs bilingual display. Otherwise translate at display time later.

### Evaluation Result

New evaluator output should align with the proposed prompt:

```json
{
  "prompt_difficulty": "easy|medium|hard",
  "image_a": { "scores": { "prompt_adherence": { ... } } },
  "image_b": { "scores": { "prompt_adherence": { ... } } },
  "comparison": {
    "per_dimension_winner": { "prompt_adherence": "A|B|TIE" },
    "overall_winner": "A|B|TIE",
    "margin": "decisive|clear|narrow|tie",
    "conflict_notes": "...|null"
  }
}
```

Mapping into existing domain terms:

| New prompt field | Existing app concept |
|---|---|
| `image_a` | `model_a` / GPT Image-2 |
| `image_b` | `model_b` / Gemini 3 Pro |
| `A` | `model_a` |
| `B` | `model_b` |
| `TIE` | `draw` |
| `margin` | New field in `ComparisonResult` |
| `conflict_notes` | New field in `ComparisonResult` |
| `prompt_difficulty` | New run-level/evaluation-level field |

### Gate 1 HIL Arbitration Artifact

Gate 1 produces an optional arbitration artifact only when the route score or dimension-level risk crosses the configured trigger. The human does not write evidence and does not rescore images. The human only answers, for each high-uncertainty dimension marked by step 04, whether A is better, B is better, or the dimension is a tie.

```json
{
  "gate": "gate1_uncertainty_risk_router",
  "status": "not_required|recommended|pending|completed|skipped",
  "route_score": 0.62,
  "route_band": "none|soft_hil|strong_hil|required_hil",
  "trigger_reasons": ["low_self_confidence", "high_sample_variance", "narrow_margin", "hard_prompt", "missing_evidence"],
  "route_features": {
    "normalized_self_confidence_risk": 0.4,
    "normalized_variance": 0.7,
    "normalized_margin_risk": 0.8,
    "cross_dim_conflict_score": 0.5,
    "difficulty_prior": 0.6,
    "evidence_quality_risk": 0.4
  },
  "review_dimensions": ["prompt_adherence", "composition"],
  "review_items": [
    {
      "dimension": "prompt_adherence",
      "uncertainty_reasons": ["low_self_confidence", "narrow_margin"],
      "model_marked_by_step": "04_evaluation_agent",
      "initial_winner": "A|B|TIE"
    }
  ],
  "reviewer": "stable local reviewer string from session state",
  "created_at": "ISO timestamp",
  "completed_at": "ISO timestamp|null",
  "dimension_arbitrations": [
    {
      "dimension": "prompt_adherence",
      "human_winner": "A|B|TIE"
    }
  ]
}
```

### Gate 2 HIL Adjudication Artifact

Gate 2 produces an optional final adjudication artifact only when the critique/revision chain has a concrete disagreement. The user does not re-evaluate images or override scores directly. The user chooses which critique position is more convincing.

```json
{
  "gate": "gate2_disagreement_detector",
  "status": "not_required|pending|completed|skipped",
  "trigger_reasons": ["critic_disagreement", "winner_flip", "large_score_delta", "round2_new_issue"],
  "disagreement_items": [
    {
      "dimension": "prompt_adherence",
      "issue_from_05": "GPT-5.4 critique claim or issue id",
      "change_from_06": "Claude revision response or score/evidence delta",
      "issue_from_07": "Gemini critique claim or issue id"
    }
  ],
  "reviewer": "stable local reviewer string from session state",
  "created_at": "ISO timestamp",
  "completed_at": "ISO timestamp|null",
  "adjudication_labels": [
    {
      "dimension": "prompt_adherence",
      "label": "agree_with_05|agree_with_07|both_partially_right"
    }
  ]
}
```

### Pipeline Result Additions

`PipelineResult` should eventually include:

- `prompt_difficulty`
- `initial_comparison`
- `hil_reviews: list[HilArbitration]`
- `hil_adjudication: HilAdjudication | None`
- `gate_decisions: list[GateDecision]`
- `pipeline_status: completed|pending_hil_gate1|pending_hil_gate2|auto_continued_high_risk|auto_continued_critical_risk|failed|partial`

### Reviewer Identity

Reviewer identity is a Phase 1 requirement, not a deferred open question. The app should ask for a reviewer string once per local session before any HIL artifact can be submitted. The default value should be the OS username when available, editable by the user, and persisted in `st.session_state` for the duration of the session.

This is not an authentication system. It is a lightweight audit identity for inter-reviewer agreement analysis, reviewer-quality tracking, fatigue monitoring, and artifact traceability. Every Gate 1 arbitration artifact and Gate 2 adjudication artifact must store the same session reviewer string in its `reviewer` field.

## Gate Logic

### Gate 1: Uncertainty & Risk Router

Purpose: detect cases where the initial evaluation is not reliable enough to enter the automated critique chain without narrow human arbitration.

HIL is disabled by default. Gate 1 should only create a HIL task when the router identifies specific model difficulty, and the task must be scoped to the dimensions/items the model marked as uncertain in step 04.

The UI/architecture diagram can keep the simpler label "confidence router", but implementation should treat confidence as only one feature in a broader routing-risk model. Model self-reported confidence is useful, but it is not sufficiently reliable as the sole trigger because it can be miscalibrated, overconfident, or inconsistent with the actual score/evidence behavior.

#### Route Score

Gate 1 should compute a normalized `route_score` from multiple uncertainty and risk signals:

```text
route_score =
    w1 * (1 - normalized_self_confidence)
  + w2 * normalized_variance
  + w3 * (1 - normalized_margin)
  + w4 * cross_dim_conflict_score
  + w5 * difficulty_prior
  + w6 * evidence_quality_risk
```

Recommended Phase 2+ weights, when repeated-sample variance or multi-evaluator variance is available:

| Feature | Weight | Rationale |
|---|---:|---|
| `1 - normalized_self_confidence` | 0.15 | Self-reported confidence is useful but should not dominate. |
| `normalized_variance` | 0.25 | Repeated-sample instability is a strong reliability signal. |
| `1 - normalized_margin` | 0.20 | Narrow score margins make the winner less robust. |
| `cross_dim_conflict_score` | 0.15 | Internal contradictions indicate fragile evaluation. |
| `difficulty_prior` | 0.15 | Hard prompts deserve more conservative routing. |
| `evidence_quality_risk` | 0.10 | Vague evidence reduces auditability. |

Phase 1 must not use the Phase 2+ table if `normalized_variance` is disabled or stored as `null`. Without renormalization, the maximum linear score would be 0.75 and the route thresholds would become invalid. Use this Phase 1 no-variance table until variance is implemented:

| Feature | Weight | Rationale |
|---|---:|---|
| `1 - normalized_self_confidence` | 0.20 | Self-reported confidence remains useful but should not dominate. |
| `1 - normalized_margin` | 0.27 | Narrow score margins are the strongest available reliability signal before variance exists. |
| `cross_dim_conflict_score` | 0.20 | Internal contradictions indicate fragile evaluation. |
| `difficulty_prior` | 0.20 | Hard prompts deserve more conservative routing. |
| `evidence_quality_risk` | 0.13 | Vague evidence reduces auditability. |
| **Total** | **1.00** | Keeps route thresholds stable in Phase 1. |

Migration note: Phase 1 runs should persist `route_weight_profile: "phase1_no_variance"`. After repeated sampling or multi-evaluator variance is implemented, new runs should switch to `route_weight_profile: "phase2_with_variance"`. Do not compare route-score distributions across the two profiles without stratifying by `route_weight_profile`, because the same case may receive a different score after variance is added.

Recommended route bands:

| Route score | Behavior |
|---:|---|
| `< 0.35` | HIL remains off; continue to critique/revision chain. |
| `0.35 - 0.55` | Soft HIL recommendation; default behavior is still auto-skip with audit. |
| `0.55 - 0.70` | Trigger targeted HIL arbitration unless configured to auto-skip. |
| `>= 0.70` | High-risk targeted HIL arbitration unless user explicitly skips. |

HIL feature-flag behavior must be explicit for each route band:

| Route band | HIL flag off | HIL flag on |
|---|---|---|
| `< 0.35` | Continue to AI review chain. | Continue to AI review chain. |
| `0.35 - 0.55` | Continue and log `would_recommend`. | Continue and log `would_recommend` unless soft-HIL is explicitly enabled. |
| `0.55 - 0.70` | Continue, flag the run as `auto_continued_high_risk`, and mark the summary prominently. | Pause for targeted HIL arbitration. |
| `>= 0.70` | Continue, flag the run as `auto_continued_critical_risk`, and mark the summary prominently. | Pause for targeted HIL arbitration; skip requires explicit confirmation. |

Phase 2 observability rule: when HIL is off, `auto_continued_high_risk` and `auto_continued_critical_risk` runs must not be silently mixed into downstream analysis as ordinary completed runs. `summary.json` must include the `pipeline_status`, route band, route score, trigger reasons, and a visible `requires_attention: true` marker for both statuses.

#### Feature Definitions

`normalized_self_confidence`:

- Derived from the evaluator's 1-5 confidence scores across all 12 image/dimension scores.
- Recommended normalization: average confidence divided by 5.
- Also track minimum confidence, because one very low-confidence critical dimension may matter more than the average.

`normalized_variance`:

- Derived from repeated evaluator samples, alternate temperature-zero retries, or multiple evaluator agents if available.
- Should consider score variance, per-dimension winner variance, and overall winner flip rate.
- If multi-sampling is too expensive in Phase 1, persist this as `null` and start with a conservative default or disable the term until Phase 2.

`normalized_margin`:

- Should reflect both overall margin and critical-dimension margin.
- Recommended calculation:

```text
normalized_margin = min(
  normalized_overall_mean_margin,
  normalized_prompt_adherence_margin,
  normalized_median_dimension_margin
)
```

- Smaller margins increase routing risk.

`cross_dim_conflict_score`:

- Prefer a 0-1 score over a boolean flag.
- Examples that should increase this score:
  - Prompt adherence is low but overall winner is decisive.
  - Evidence describes serious artifacts but photorealism score remains high.
  - Confidence is high while sample variance is high.
  - A and B split wins across many dimensions with narrow margins.
  - A dimension has severe negative evidence but still receives an 8+ score.

`difficulty_prior`:

- Should combine evaluator-provided `prompt_difficulty` with deterministic prompt heuristics.
- Prompt features that increase difficulty include multiple subjects, countable objects, text/logo rendering, UI/chart rendering, precise spatial relationships, occlusion, negative constraints, mixed style/photorealism requirements, and multi-step narrative scenes.

`evidence_quality_risk`:

- Should increase when evidence is generic, missing, not visually grounded, or not tied to concrete objects/regions/issues.
- This is an auditability signal: even a high score with vague evidence may increase Gate 1 arbitration risk.

#### Dimension-scoped HIL

Gate 1 should not automatically send the entire run to human review. It should produce `review_dimensions` and `review_items` scoped to the riskiest dimensions marked by step 04. For example, if only `prompt_adherence` and `composition` are high-risk, HIL should ask the human to arbitrate only those dimensions.

Gate 1 HIL question format:

```text
For this prompt and these two images, on dimension <dimension>, which image is better?
A / B / TIE
```

The user should see the prompt, Image A, Image B, and the selected high-uncertainty dimension. The UI should not ask for evidence, confidence, score, or freeform reasoning in the default flow. Optional notes can be considered later, but Phase 1 should keep the interaction as a single structured label per dimension.

Suggested trigger rules:

- `route_score >= 0.35` creates a HIL recommendation but should default to auto-skip unless the user enables soft HIL.
- `route_score >= 0.55` should normally pause for targeted HIL arbitration.
- `route_score >= 0.70` should be treated as high-risk and require explicit skip to continue without HIL.
- Any critical dimension with very low confidence, high variance, and narrow margin can trigger dimension-scoped review even if aggregate `route_score` is lower.

Phase behavior must be explicit: while the HIL feature flag is disabled, Gate 1 records route decisions for observability but does not pause the pipeline. For route bands below 0.55, it may record `recommended` or `skipped`; for route bands at or above 0.55, the run-level `pipeline_status` must become `auto_continued_high_risk` or `auto_continued_critical_risk`. Only after the resume flow is implemented and HIL is enabled should `pending` pause execution.

Output:

- `not_required`: continue directly to AI critique chain.
- `recommended`: HIL recommended, but auto-skip may be acceptable with audit logging.
- `pending`: pause pipeline and render HIL arbitration UI for selected dimensions/items.
- `completed`: store human arbitration labels and continue.
- `skipped`: continue, but record that human review was skipped.

Merge behavior:

- Human labels should never directly mutate scores.
- Gate 1 labels should be passed to step 06 and step 08 revision prompts as trusted arbitration context, for example: "human selected A over B for prompt_adherence".
- The decision engine should treat Gate 1 labels as dimension-level tie-break or confidence-adjustment signals, not as new evidence or scores.

### Gate 2: Disagreement Detector

Purpose: identify unresolved conflicts after AI critique/revision chain and ask the human to adjudicate the critique disagreement, not re-evaluate the images from scratch.

Suggested trigger rules:

- Pipeline-side comparison finds that 07 independently flags an issue equivalent to one raised by 05, meaning 06 may not have fully resolved it.
- Pipeline-side comparison finds that 07 independently introduces a new material issue after reviewing 06.
- Pipeline-side comparison finds that 05 and 07 recommend opposite winners on any key dimension.
- 06 revised score differs from 04 initial score by >= 3 in any dimension.
- Overall winner flips between 04 initial evaluation and 06 revised evaluation.
- 06 revised comparison says `narrow` or `tie` while per-dimension evidence is conflicting.
- Gate 1 arbitration label conflicts with the 06 revised dimension winner.

Important: these trigger rules are computed after 05, 06, and 07 are complete, before 08 final revision runs. They must not be encoded into 07's prompt as references to 05.

#### Issue Equivalence Detection

Gate 2 needs a pipeline-layer issue equivalence checker before it can decide whether 07 repeated, contradicted, or newly introduced critique issues relative to 05. This checker is an independent LLM call, recommended model: Claude Opus. It is not step 07 and it is not part of the 07 prompt. It runs only after 05, 06, and 07 artifacts already exist.

The checker may read the full 05 and 07 issue lists, plus minimal dimension metadata from 06 when needed for naming and score-delta context. This does not break 07 independence because 07 has already been generated without seeing 05; the checker is a later pipeline-layer tool used by Gate 2.

Checker input:

```json
{
  "prompt": "original user prompt",
  "issue_list_05": [
    {
      "issue_id": "05-prompt_adherence-1",
      "dimension": "prompt_adherence",
      "claim": "The evaluation underweights Image B's missing text requirement.",
      "suggested_direction": "increase_or_decrease_scores_if_available"
    }
  ],
  "issue_list_07": [
    {
      "issue_id": "07-prompt_adherence-1",
      "dimension": "prompt_adherence",
      "claim": "The revised evaluation still ignores that Image B failed the requested text.",
      "suggested_direction": "increase_or_decrease_scores_if_available"
    }
  ],
  "revision_context_06": {
    "changed_dimensions": ["prompt_adherence"],
    "score_deltas": {"prompt_adherence": {"image_a": -1, "image_b": 0}}
  }
}
```

Checker output:

```json
{
  "equivalence_results": [
    {
      "issue_id_07": "07-prompt_adherence-1",
      "classification": "equivalent|new|contradicting",
      "matched_issue_id_05": "05-prompt_adherence-1|null",
      "dimension": "prompt_adherence",
      "rationale": "Short explanation grounded in the issue claims, not a re-evaluation of the images.",
      "gate2_trigger": true
    }
  ],
  "summary": {
    "equivalent_count": 1,
    "new_count": 0,
    "contradicting_count": 0,
    "should_trigger_gate2": true
  }
}
```

Classification rules:

- `equivalent`: 07 raises materially the same visual/evaluation problem as 05, even if wording differs.
- `new`: 07 raises a material issue not present in 05.
- `contradicting`: 07 directly disputes or reverses a material claim from 05.

Prompt template draft:

```text
You are a pipeline-layer issue equivalence checker for an image-evaluation audit system.

You will receive issue lists from two independent critique steps:
- 05: critique of the initial evaluation.
- 07: critique of the revised evaluation.

Your job is not to re-evaluate the images and not to decide the final winner. Your job is only to classify each 07 issue relative to the 05 issue list.

For every issue in issue_list_07, return exactly one classification:
- equivalent: materially the same evaluation problem appears in 05.
- new: the issue is material and not present in 05.
- contradicting: the issue directly disputes or reverses a material 05 claim.

Use dimension, claim semantics, affected image, suggested score direction, and described evidence as matching signals. Do not require identical wording. If uncertain, choose new unless there is clear semantic overlap.

Return only valid JSON matching the requested schema. Include a concise rationale for each classification.
```

Output:

- `not_required`: continue to 08 final revision without Gate 2 HIL labels.
- `pending`: pause for HIL adjudication label before 08 final revision.
- `completed`: 08 final revision and the decision engine may incorporate human adjudication labels.
- `skipped`: continue to 08 final revision without adjudication labels but archive the skip.

Merge behavior:

- Gate 2 HIL produces one of three labels: `agree_with_05`, `agree_with_07`, or `both_partially_right`.
- `agree_with_05`: 08 and the decision engine should give more weight to the original GPT-5.4 critique issue, including the possibility that 06 did not adequately fix it.
- `agree_with_07`: 08 and the decision engine should give more weight to Gemini's second critique of the revised evaluation.
- `both_partially_right`: 08 and the decision engine should preserve both critique signals and mark the final result as contested/partially resolved.
- Raw AI comparison and all critique/revision artifacts should still be persisted for audit.

## Prompt Refactor Design

The provided evaluator prompt should become the new baseline for `EVAL_SYSTEM_PROMPT` because it improves calibration in four important ways:

- Scores require concrete `evidence` instead of broad reasoning.
- Scores include `confidence`, enabling Gate 1.
- Comparison is generated together with independent scores.
- Calibration rules clarify 7 vs 8, rare 10s, and non-default 5s.

### Evaluation Prompt Changes

Use the provided prompt structure with these project-specific adjustments:

- Keep model identities out of scoring instructions as much as possible.
- Still label images as `IMAGE_A` and `IMAGE_B`; map to model names only in app data.
- Include `prompt_difficulty` exactly as specified.
- Enforce single JSON output.
- Add optional Chinese display fields only if we keep bilingual artifacts; otherwise the UI can translate labels and show English evidence.

### Critique Prompt Changes

Critique should review:

- Whether evidence is concrete and visible.
- Whether confidence is calibrated.
- Whether score/evidence/confidence are mutually consistent.
- Whether comparison winners follow the stated tie/winner rules.
- Whether prompt difficulty is reasonable.

Suggested critique output additions:

- `evidence_quality_issues`
- `confidence_calibration_issues`
- `comparison_issues`
- `suggested_confidence_model_a/b`

### Critique Independence and Prompt Boundaries

Each critique round must be an independent audit of the evaluation artifact it receives. The purpose of 07 is not to review 05 or validate whether 06 followed 05; the purpose of 07 is to independently inspect the revised evaluation produced by 06.

Required information boundaries:

| Step | Agent | Prompt may include | Prompt must not include | Purpose |
|---|---|---|---|---|
| 05 | GPT-5.4 Critique | User prompt, Image A, Image B, 04 evaluation, rubric/schema | 06, 07, 08 outputs | Independent critique of 04. |
| 06 | Claude Revision | User prompt, Image A, Image B, 04 evaluation, 05 critique, optional Gate 1 arbitration labels | 07, 08 outputs | Revise 04 using 05 feedback. |
| 07 | Gemini Critique | User prompt, Image A, Image B, 06 revised evaluation, rubric/schema | 05 critique output, 05 issue IDs, 05 wording, 08 output | Independent critique of 06. |
| 08 | Final Revision Agent | User prompt, Image A, Image B, 05 critique, 06 revised evaluation, 07 critique, optional Gate 1 and Gate 2 HIL labels | N/A | Produce final revision after seeing both independent critique signals and any human arbitration/adjudication labels. |

05 and 07 prompt templates should be different but equivalent:

- Different: use separately written wording, section order, and field names where helpful to reduce correlated failure modes and prompt-template overfitting.
- Equivalent: apply the same rubric, dimensions, calibration rules, output contract, and evidence/confidence expectations.
- Neither prompt should reveal model/provider identity as a scoring cue.
- 07 must not say "previous critique", "round 1 critique", "05", "GPT-5.4 said", or include any text derived from 05.

Gate 2 disagreement detection is a local pipeline/decision-layer comparison after 05, 06, and 07 exist, before 08 final revision. It must not be implemented by asking 07 to compare itself with 05.

### Revision Prompt Changes

Revision should re-evaluate all dimensions and must preserve or update:

- `score`
- `evidence`
- `confidence`
- `critique_accepted`
- `revision_note`

If Gate 1 arbitration labels exist, revision prompt should distinguish:

- AI critic feedback
- Human arbitration labels
- The evaluator's final accepted/rejected decision

There should be two revision prompt modes:

- Step 06 revision: may see 04 + 05 + optional Gate 1 labels, but must not see 07.
- Step 08 final revision: may see 05 + 06 + 07 + optional Gate 1/Gate 2 HIL labels and should explicitly reconcile both critique signals.

## Decision Engine Design

The current `determine_winner()` should evolve from mean-only comparison to a richer decision engine.

Inputs:

- Initial evaluation
- Final revised evaluation
- AI-generated comparison block
- Gate 1 route score and route features
- Critique agreement/disagreement signals
- Gate 1 arbitration labels, if any
- Gate 2 adjudication labels, if any

Outputs:

- Per-dimension winner
- Overall winner
- Margin
- Conflict notes
- Whether human arbitration/adjudication affected the final result
- Audit summary

Recommended winner rules:

- Per dimension: use score difference, but allow `TIE` when difference ≤ 1 and no concrete reason separates candidates.
- Overall: use dimensional wins and mean score together.
- Margin:
  - `decisive`: winner leads most dimensions and largest gap ≥ 3
  - `clear`: winner leads most dimensions and largest gap = 2
  - `narrow`: winner leads more dimensions and no gap > 1
  - `tie`: overall winner is `TIE`
- Human arbitration/adjudication labels should be explicit and always archived.

Decision engine should be weighted, not a silent mutation of AI scores. Recommended signal layers:

- Final AI scores and evidence form the base automated judgment.
- Critique agreement increases confidence in the automated judgment.
- Gate 1 route score and evidence-quality risk reduce confidence in the automated judgment.
- Gate 1 arbitration labels can influence dimension-level winner/tie resolution and automated-confidence weighting, but must not directly change scores.
- Gate 2 adjudication labels can weight 05 vs 07 critique signals and may mark the result as human-adjudicated or contested, but must not directly change scores.

## Persistence Design

New run artifacts:

| File | Purpose |
|---|---|
| `evaluation_v2.json` | New score/evidence/confidence evaluation output |
| `initial_comparison.json` | Initial evaluator comparison block |
| `gate1_decision.json` | Uncertainty/risk router decision, route score, feature values, and trigger reasons |
| `hil_review_r1.json` | Gate 1 human arbitration labels, if triggered |
| `critique_r1.json` | 05 GPT-5.4 independent critique of 04 |
| `revised_r1.json` | 06 revised evaluation using 04 + 05 |
| `critique_r2.json` | 07 Gemini independent critique of 06 only |
| `gate2_decision.json` | Disagreement detector decision and trigger reasons |
| `hil_adjudication.json` | Gate 2 human critique-disagreement label, if triggered |
| `revised_r2.json` | 08 final revision using 05 + 06 + 07 plus optional Gate 1/Gate 2 HIL labels |
| `comparison.json` | Final decision engine output |
| `summary.json` | Run-level summary including HIL status, route band, high-risk auto-continue flags, and attention markers |

Archive contents should explicitly include:

- Original prompt text and any prompt-normalization metadata.
- Generated image files and model labels.
- Scores, evidence, confidence, prompt difficulty, and comparison outputs.
- Gate 1 route score, feature values, selected review dimensions, and route decision.
- Gate 2 disagreement reasons and adjudication state.
- HIL arbitration/adjudication labels, reviewer decisions, skip reasons, and timestamps.
- The session reviewer string used for every HIL artifact.
- For HIL-off high-risk runs, `pipeline_status`, `route_band`, `route_score`, `trigger_reasons`, and `requires_attention: true` in `summary.json`.
- Prompt-input metadata for 05, 06, 07, and 08 showing which artifacts were included/excluded, so information-boundary violations are auditable.

Backward compatibility:

- Continue reading old `evaluation.json`, `revised.json`, and `comparison.json` for historical runs.
- UI should detect artifact version and render old/new layouts gracefully.
- `runs/index.json` should add optional fields without breaking old entries.

## Streamlit UI Design

### HIL UX Principles

HIL should feel like a short, focused decision task rather than a second evaluation workflow. When HIL appears, the UI should make three things immediately clear:

- Why HIL was triggered.
- What exact question the human needs to answer.
- How many decisions remain before the pipeline can continue.

General UX rules:

- HIL is absent from the normal flow unless a gate triggers it. Do not show inactive HIL forms or empty HIL panels on ordinary runs.
- Use a single prominent pending-HIL task panel when human input is required, with the rest of the dashboard visually de-emphasized.
- Keep required actions to one click per arbitration item whenever possible.
- Use segmented controls for `A` / `B` / `TIE` and for `agree_with_05` / `agree_with_07` / `both_partially_right`.
- Show technical details behind expanders so advanced users can inspect them without increasing default cognitive load.
- Make skip explicit and auditable, but secondary to the submit action.
- Preserve work-in-progress labels in session state so accidental refreshes do not lose HIL choices.
- After submission, show a compact confirmation state before the pipeline resumes.

Visual design guidance:

- Use a calm attention style for pending HIL, such as an amber left border or badge, not a red error treatment.
- Use the existing dark dashboard style and compact operational layout; avoid modal-heavy or marketing-style layouts.
- Keep Image A and Image B at equal size with sticky labels and a synchronized zoom/open-preview affordance.
- Keep the arbitration controls close to the images and dimension title so the user does not need to scroll back and forth.
- Use concise labels; explanatory details should be available through tooltips or collapsed context.

### HIL Gate 1 UI

When Gate 1 is `pending`, show:

- A pending-HIL banner: "Human arbitration needed" plus a short reason such as "High uncertainty in prompt adherence".
- User prompt in a compact, readable prompt strip.
- Side-by-side Image A and Image B with equal sizing, stable aspect ratio, model labels, and click-to-enlarge.
- A progress indicator such as `1 of 2 dimensions` when multiple dimensions require arbitration.
- The current high-uncertainty dimension selected by step 04 / Gate 1, shown as the task title.
- For each selected dimension, a compact arbitration control:
  - `A better`
  - `B better`
  - `TIE`
- Optional supporting context can show 04's score/evidence/confidence collapsed by default, but the required user action is only the winner/tie label.
- Buttons:
  - `Submit Arbitration`
  - `Skip HIL`

The Gate 1 UI should not ask the user to write evidence, set confidence, or re-score either image.

Recommended Gate 1 layout:

1. Top: pending-HIL banner with trigger reason and remaining item count.
2. Middle: prompt strip and image comparison area.
3. Bottom: one arbitration card for the active dimension, with `A better` / `B better` / `TIE` as the only required input.
4. Footer: submit and skip actions, plus an audit note that skip will be recorded.

If multiple dimensions are pending, use a stepper or tabs by dimension. Do not render a long form with all dimensions expanded at once.

### HIL Gate 2 UI

When Gate 2 is `pending`, show:

- A pending-HIL banner: "Critique disagreement needs adjudication" plus the affected dimension count.
- The specific disagreement between 05, 06, and 07:
  - what 05 (GPT-5.4) raised
  - what 06 changed in response
  - what 07 (Gemini) says after the modification
- The affected dimension(s) and any score/evidence deltas as read-only context.
- Human adjudicator control:
  - `agree_with_05`
  - `agree_with_07`
  - `both_partially_right`
- Buttons:
  - `Submit Adjudication`
  - `Skip HIL`

The Gate 2 UI should not ask for score overrides, final-winner overrides, or new evidence in the default flow.

The Gate 2 UI may show 05, 06, and 07 together because it is rendered after all three artifacts exist. This UI visibility does not change the prompt boundary: 07 itself must have been generated without 05.

Recommended Gate 2 layout:

1. Top: pending-HIL banner with trigger reason, such as "07 still disagrees with the 06 revision".
2. Middle: a three-column dispute summary: `05 raised`, `06 changed`, `07 says`.
3. Bottom: one segmented adjudication control with the three labels.
4. Footer: submit and skip actions, plus a short audit note.

The three labels should be user-friendly in the UI while preserving machine-readable values in artifacts:

| UI label | Stored label | Meaning |
|---|---|---|
| `05 is more convincing` | `agree_with_05` | GPT-5.4's original issue should carry more weight. |
| `07 is more convincing` | `agree_with_07` | Gemini's second critique should carry more weight. |
| `Both have valid points` | `both_partially_right` | Keep the result contested or partially resolved. |

The UI should avoid forcing the user to read raw JSON. Raw artifacts remain available in expanders below the main task.

### Result UI Updates

For completed runs, show:

- Prompt difficulty badge.
- Margin badge.
- Confidence markers on dimension cards.
- Evidence snippets in dimension cards.
- HIL audit panel when arbitration/adjudication labels affected the run.
- Raw JSON expanders for all new artifacts.

The HIL audit panel should summarize human involvement in one compact line per gate, for example:

- Gate 1: `prompt_adherence -> A better`, submitted by local reviewer, timestamp.
- Gate 2: `composition -> 05 is more convincing`, submitted by local reviewer, timestamp.

Detailed route scores, feature values, and raw labels should remain available behind expanders.

## Implementation Phases

### Phase 1: Schema and Prompt Foundation

- Add score `evidence` and `confidence` models.
- Add comparison fields: `margin`, `conflict_notes`, `prompt_difficulty`.
- Add route-score support with the Phase 1 no-variance weight profile and persist `route_weight_profile: "phase1_no_variance"`.
- Implement reviewer identity prompt and session persistence: ask once per local session, default to OS username when available, store in `st.session_state`, and write the value into all HIL artifacts.
- Introduce V2 parsing while preserving V1 readers.
- Replace evaluator prompt with the new calibrated evaluator prompt.
- Update tests for schema validation and comparison mapping.

### Phase 2: Gate Decisions With HIL Off By Default

- Implement Gate 1 and Gate 2 decision functions.
- Implement Gate 2 issue equivalence checker as a pipeline-layer LLM call after 05/06/07 and before 08.
- Persist `gate1_decision.json` and `gate2_decision.json`.
- Keep HIL disabled by default and auto-continue while recording whether HIL would have been triggered.
- For `0.55 - 0.70`, set `pipeline_status: auto_continued_high_risk` and mark `summary.json` with `requires_attention: true`.
- For `>= 0.70`, set `pipeline_status: auto_continued_critical_risk` and mark `summary.json` with `requires_attention: true`.
- This gives observability before adding targeted arbitration pause/resume.

### Phase 3: HIL Arbitration Artifacts and Resume Flow

- Add `pending_hil_gate1` and `pending_hil_gate2` statuses.
- Let pipeline save partial state and stop at HIL gates.
- Add resume entry points that continue after Gate 1 arbitration labels are submitted, or continue into 08 final revision after Gate 2 adjudication labels are submitted.
- Add tests for resume behavior and partial artifact loading.

### Phase 4: Streamlit HIL UI

- Add Gate 1 arbitration form.
- Add Gate 2 adjudication form.
- Add audit panels to result view.
- Add i18n keys for all HIL labels.
- Add friendly pending-HIL task panels with clear trigger reason, remaining item count, compact controls, and collapsed technical context.
- Persist in-progress HIL choices in `st.session_state` before submission.
- Add image preview/zoom support for Gate 1 so users can inspect details without leaving the arbitration task.

### Phase 5: Decision Engine and Dashboard Polish

- Replace final winner logic with V2 decision engine.
- Render confidence/evidence/margin/prompt difficulty in result cards.
- Update Architecture V2 diagram only if flow semantics change during implementation.

## Test Plan

- Schema tests:
  - confidence must be 1-5
  - scores must be 1-10
  - all six dimensions required
  - winner values map between `A/B/TIE` and `model_a/model_b/draw`
- Prompt parsing tests:
  - evaluator output validates against V2 schema
  - missing evidence fails validation
  - missing confidence fails validation
- Prompt isolation tests:
  - 05 critique prompt includes 04 evaluation but excludes 06, 07, and 08
  - 06 revision prompt includes 04 and 05 but excludes 07 and 08
  - 07 critique prompt includes 06 revised evaluation but excludes 05 output, 05 issue IDs, and 05 wording
  - 07 prompt template does not contain phrases such as `previous critique`, `round 1 critique`, or `GPT-5.4 said`
  - 05 and 07 prompt templates validate against equivalent rubric/output requirements while remaining separately worded
  - 08 final revision prompt includes 05, 06, 07, and optional Gate 1/Gate 2 HIL labels
- Gate tests:
  - low route score bypasses Gate 1 HIL
  - high sample variance triggers Gate 1
  - narrow margin triggers Gate 1
  - cross-dimension conflict triggers Gate 1
  - hard prompt difficulty prior increases route score
  - missing or vague evidence increases route score
  - Phase 1 no-variance route weights sum to 1.00
  - Phase 1 route score uses `route_weight_profile: "phase1_no_variance"`
  - Phase 2+ route score uses `route_weight_profile: "phase2_with_variance"` when variance is available
  - HIL-off route band `0.55 - 0.70` sets `pipeline_status` to `auto_continued_high_risk`
  - HIL-off route band `>= 0.70` sets `pipeline_status` to `auto_continued_critical_risk` and writes `requires_attention: true` to `summary.json`
  - winner flip triggers Gate 2
  - large score delta between 04 and 06 triggers Gate 2
  - issue equivalence checker classifies 07 issues as `equivalent`, `new`, or `contradicting`
  - equivalent issue checker output triggers Gate 2 without violating 07 prompt isolation
  - pipeline-side equivalence between 05 and 07 issues triggers Gate 2 without 07 seeing 05
  - pipeline-side new material issue from 07 triggers Gate 2
- UI behavior tests:
  - normal completed runs do not show inactive HIL forms
  - Gate 1 pending UI shows prompt, two images, active uncertain dimension, and only A/B/TIE required controls
  - Gate 2 pending UI shows 05/06/07 dispute summary and only the three adjudication labels
  - HIL skip requires an explicit click and records an audit state
  - reviewer identity prompt appears once per session before HIL submission and persists through reruns
  - all submitted HIL artifacts include the session reviewer string
  - in-progress HIL selections survive Streamlit reruns before submission
  - technical route-score details are collapsed by default
- Decision tests:
  - margin classification
  - conflict notes behavior
  - Gate 1 A/B/TIE arbitration label influences dimension-level decision without changing scores
  - Gate 2 `agree_with_05` weights the GPT-5.4 critique issue
  - Gate 2 `agree_with_07` weights the Gemini critique issue
  - Gate 2 `both_partially_right` preserves contested status
- Backward compatibility tests:
  - old runs load without V2 fields
  - old comparison render still works

## Open Questions

1. Should Chinese evidence be generated by the model, translated by the UI, or omitted in V2 artifacts?
2. How aggressive should Gate 1 be? Lower `route_score` thresholds improve quality but increase review workload.
3. How many repeated samples should Gate 1 use before variance cost becomes too high?
4. Should Gate 1 always remain dimension-scoped, even when route risk is very high?

## Recommended Decisions

- Keep local deterministic `compare.py` / decision engine as the final source of truth.
- Treat model-generated `comparison` as evidence/input, not the sole authority.
- Keep HIL off by default and skippable with an explicit audit record when triggered.
- Store human arbitration/adjudication labels separately and compute final decision from layered inputs.
- Use the new evaluator prompt in Phase 1, because `confidence` is required before Gate 1 can work.
- Implement Gate 1 as a composite `route_score`, not a single confidence threshold.
- Treat self-reported confidence as one feature, with sample variance and score margin weighted more heavily.
- Prefer dimension-scoped A/B/TIE arbitration for Gate 1 to minimize reviewer workload.
- Use `agree_with_05` / `agree_with_07` / `both_partially_right` as the Gate 2 HIL label set.
- Keep 05 and 07 as independent critiques with different-but-equivalent prompts; 07 must never receive 05 output, while 08 may see 05 + 06 + 07.
- Keep backward compatibility for historical runs in `runs/`.
