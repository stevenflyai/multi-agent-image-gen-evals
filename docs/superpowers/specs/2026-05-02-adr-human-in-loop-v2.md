# ADR: Human-in-the-loop V2 Evaluation Architecture

## Status

Accepted for design. Runtime implementation has not started.

## Context

The current image evaluation pipeline generates two images, asks an evaluator model to score both images, runs two critique/revision rounds, and then computes a final winner from revised scores. This works as a fully automated V1 pipeline, but it has several limits:

- Scores are not sufficiently auditable because they rely on broad `reasoning` text rather than concrete visual `evidence`.
- The evaluator does not expose calibrated `confidence`, so the pipeline cannot tell when a result is uncertain enough to merit human attention.
- The final comparison is mostly mean-score driven and does not account for prompt difficulty, narrow margins, inconsistent evidence, or critique disagreement.
- Human-in-the-loop (HIL) is currently only a concept in the Architecture V2 diagram, not a data model, persistence contract, or resumable workflow.
- Humans should not redo model work. Human input should be narrow, structured, optional by default, and reserved for cases where the model pipeline is uncertain or internally conflicted.

The detailed refactor design is documented in [2026-05-02-human-in-loop-v2-refactor.md](2026-05-02-human-in-loop-v2-refactor.md).

## Decision

Adopt Architecture V2 as a gated, auditable evaluation pipeline with two targeted HIL gates, independent critique boundaries, and a local decision engine.

The target flow is:

1. User prompt
2. Pipeline orchestrator
3. Parallel generation: Image A from GPT Image-2, Image B from Gemini 3 Pro
4. Evaluation Agent produces per-image dimension scores with `score`, `evidence`, `confidence`, plus `prompt_difficulty` and initial comparison
5. Gate 1: Uncertainty & Risk Router
6. Step 05: GPT-5.4 independent critique of 04
7. Step 06: Claude revision using 04 + 05, plus optional Gate 1 labels
8. Step 07: Gemini independent critique of 06 only
9. Gate 2: Disagreement Detector
10. Step 08: Final Revision Agent using 05 + 06 + 07, plus optional Gate 1/Gate 2 HIL labels
11. Decision Engine computes final winner and audit summary
12. Archive module persists all prompts, artifacts, gate decisions, HIL labels, and final outputs

## Detailed Decisions

### 1. Evaluation Schema

The evaluator output will evolve from `score/reasoning/reasoning_zh` to a V2 score shape centered on:

```json
{
  "score": 8,
  "evidence": "Concrete visual evidence tied to image regions/objects/issues.",
  "confidence": 4
}
```

`evidence` is the canonical audit field. `reasoning` may remain temporarily as a backward-compatible alias or display field. `confidence` is an uncertainty signal, not a quality score.

### 2. Gate 1: Uncertainty & Risk Router

Gate 1 will not rely on model self-confidence alone. It will compute a composite `route_score` from:

- Self-confidence risk
- Repeated-sample or evaluator variance
- Narrow score margin risk
- Cross-dimension conflict
- Prompt difficulty prior
- Evidence quality risk

HIL is off by default. When HIL is disabled, Gate 1 records `recommended` or `skipped` decisions for observability but does not pause the pipeline. When HIL is enabled, Gate 1 may pause only for targeted dimension-scoped arbitration.

Gate 1 HIL asks only:

```text
For this prompt and these two images, on dimension <dimension>, which image is better?
A / B / TIE
```

Humans do not write evidence, set confidence, rescore images, or directly mutate model scores.

### 3. Independent Critique Boundaries

Each critique must be an independent audit of the artifact it receives.

Step 05 may see the user prompt, Image A, Image B, 04 evaluation, rubric, and schema. It must not see 06, 07, or 08 outputs.

Step 06 may see 04 + 05 and optional Gate 1 labels. It must not see 07 or 08.

Step 07 may see the user prompt, Image A, Image B, 06 revised evaluation, rubric, and schema. It must never see 05 output, 05 issue IDs, 05 wording, or any phrase implying a previous critique such as `previous critique`, `round 1 critique`, or `GPT-5.4 said`.

Step 05 and Step 07 prompt templates should be different but equivalent: separately worded to reduce correlated failure modes, but aligned on rubric, dimensions, calibration rules, output contract, and evidence/confidence expectations.

Step 08 is the first revision step allowed to see 05 + 06 + 07 together. It may also see optional Gate 1 and Gate 2 HIL labels.

### 4. Gate 2: Disagreement Detector

Gate 2 runs after 05, 06, and 07 are complete, and before 08 final revision. It is a local pipeline-side comparison, not a prompt task delegated to 07.

Gate 2 may trigger when:

- 07 independently flags an issue equivalent to one raised by 05, suggesting 06 may not have resolved it.
- 07 introduces a new material issue after reviewing 06.
- 05 and 07 recommend opposite winners on a key dimension.
- 06 score differs from 04 score by at least 3 in any dimension.
- The winner flips between 04 and 06.
- 06 says `narrow` or `tie` while evidence remains conflicting.
- Gate 1 arbitration conflicts with the 06 revised dimension winner.

Gate 2 HIL asks the user to adjudicate the critique disagreement with exactly one of:

- `agree_with_05`
- `agree_with_07`
- `both_partially_right`

This label can guide 08 and the Decision Engine, but it must not directly change scores.

### 5. Final Revision and Decision Engine

Step 08 reconciles both independent critique signals and optional HIL labels to produce the final revised evaluation.

The Decision Engine remains the final source of truth for winner selection. It should use final AI scores and evidence as the base automated judgment, then layer in critique agreement, route risk, evidence quality risk, Gate 1 arbitration labels, and Gate 2 adjudication labels.

Human labels can affect dimension-level winner/tie resolution, confidence weighting, and contested/audited status. They must not silently mutate raw model scores.

### 6. Persistence and Auditability

New V2 runs will persist explicit artifacts for each meaningful stage:

| Artifact | Purpose |
|---|---|
| `evaluation_v2.json` | Score/evidence/confidence evaluation output |
| `initial_comparison.json` | Initial evaluator comparison block |
| `gate1_decision.json` | Route score, feature values, decision, trigger reasons |
| `hil_review_r1.json` | Gate 1 arbitration labels, if triggered |
| `critique_r1.json` | 05 independent critique of 04 |
| `revised_r1.json` | 06 revision using 04 + 05 |
| `critique_r2.json` | 07 independent critique of 06 only |
| `gate2_decision.json` | Disagreement decision and trigger reasons |
| `hil_adjudication.json` | Gate 2 adjudication label, if triggered |
| `revised_r2.json` | 08 final revision using 05 + 06 + 07 plus optional HIL labels |
| `comparison.json` | Final Decision Engine output |
| `summary.json` | Run-level summary including HIL status |

Prompt-input metadata for 05, 06, 07, and 08 must record included/excluded artifacts so information-boundary violations are auditable.

### 7. User Experience

HIL UI should be absent unless a gate triggers it. When shown, it should present a short focused decision task, not a second evaluation workflow.

Gate 1 UI should show the prompt, both images, the uncertain dimension, and `A better` / `B better` / `TIE` controls.

Gate 2 UI should show the 05/06/07 dispute summary and one control with `05 is more convincing`, `07 is more convincing`, or `Both have valid points`.

Technical details remain available behind expanders. Skip actions are explicit and audited.

## Consequences

### Positive

- Automated scores become more auditable through concrete visual evidence.
- HIL workload stays small because users only answer scoped arbitration/adjudication questions.
- The system avoids over-trusting model self-confidence by using a composite uncertainty/risk score.
- Critique independence is preserved because 07 cannot see 05.
- 08 can perform a true synthesis step after both independent critiques and any HIL labels are available.
- Final winner selection remains deterministic and locally inspectable.
- Historical V1 runs can remain readable through backward-compatible artifact handling.

### Negative

- The pipeline becomes more complex: more artifacts, statuses, prompts, and resume states.
- Gate 1 variance features may increase cost if implemented with repeated samples.
- Prompt isolation requires explicit tests and metadata auditing.
- The UI must handle partial runs and pending HIL states, which Streamlit reruns can make tricky.
- Decision Engine weighting must be calibrated carefully so human labels influence outcomes without becoming hidden score overrides.

## Alternatives Considered

### Use Self-reported Confidence Only for Gate 1

Rejected. Model confidence can be miscalibrated, overconfident, or inconsistent with score/evidence behavior. A composite route score is more robust.

### Let Humans Rescore Images Directly

Rejected. This would turn HIL into a second evaluation workflow, increase user burden, and blur audit semantics. Humans should provide narrow structured labels only.

### Let 07 Review 05 and 06 Together

Rejected. This would make 07 a follow-up reviewer rather than an independent audit. Gate 2 can compare 05 and 07 locally after both exist.

### Run Gate 2 After 08

Rejected. If Gate 2 HIL labels should influence 08, Gate 2 must occur before 08 final revision.

### Let HIL Labels Directly Modify Scores

Rejected. Raw model scores must remain immutable for auditability. HIL labels influence weighting, tie resolution, confidence, and contested status.

## Implementation Notes

- Phase 1 should add V2 schema and calibrated prompts before HIL UI.
- Phase 2 should implement gate decisions with HIL still disabled by default, recording would-trigger decisions for observability.
- Phase 3 should add resumable pending HIL statuses and partial artifact loading.
- Phase 4 should add Streamlit HIL UI.
- Phase 5 should replace final winner logic with the V2 Decision Engine.

## Open Questions

- What reviewer identity should be persisted for local HIL decisions?
- Should high-risk Gate 1 cases remain skippable, or require explicit confirmation to skip?
- Should Chinese evidence be generated by the model, translated in the UI, or omitted from V2 artifacts?
- How many repeated samples are worth the cost for Gate 1 variance estimation?
- Should Gate 1 always remain dimension-scoped when route risk is very high?

## References

- [2026-05-02-human-in-loop-v2-refactor.md](2026-05-02-human-in-loop-v2-refactor.md)