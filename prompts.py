"""Shared prompt templates for the evaluation pipeline.

Single source of truth for rubric text and system prompts used by
evaluate.py, critique.py, and revise.py.
"""

RUBRIC = """Score each dimension as an integer from 1-10. Use the full scale consistently:

- 1-2 = unusable or almost no evidence of the requirement
- 3-4 = poor, with major visible failures
- 5-6 = mixed or acceptable, but with clear flaws
- 7-8 = strong, with minor or moderate flaws
- 9 = excellent, with only tiny issues
- 10 = essentially flawless and rare

Score each dimension independently, using visible evidence from the image. Do not let the model/provider identity influence scoring. A missing major prompt constraint should usually prevent scores of 8+ for prompt adherence, and visible artifacts should usually prevent scores of 9+ in the affected dimension.

Dimension anchors:

**Prompt adherence**
- 1 = unrelated to prompt or ignores the requested subject/medium
- 5 = captures the main subject but misses important details, relationships, text/logos, countable elements, style, layout, or negative constraints
- 10 = every explicit and strongly implied prompt requirement is faithfully rendered, including subject identity, relationships, text/logos, counts, style/medium, aspect/layout, and constraints

**Photorealism**
- 1 = clearly artificial, distorted, or physically incoherent for the requested medium
- 5 = plausible at first glance but has obvious AI tells, material errors, anatomy/object distortions, or medium inconsistencies
- 10 = if a photo is requested, indistinguishable from a photograph; if stylized or designed media is requested, materially believable, artifact-free, and fully consistent with that medium

**Aesthetic quality**
- 1 = visually unpleasant, messy, or low-craft
- 5 = acceptable but unremarkable, with limited polish or emotional impact
- 10 = exceptional visual appeal, polish, craft, and impact for the requested genre or medium

**Composition**
- 1 = chaotic, unbalanced, confusing, badly cropped, or hard to parse
- 5 = competent framing or layout, but with weak hierarchy, depth, flow, or spatial organization
- 10 = masterful framing, balance, hierarchy, depth, cropping, and visual flow; for multi-panel/page prompts, panel layout reads clearly and intentionally

**Color accuracy**
- 1 = colors contradict the prompt, look physically implausible, or undermine the intended scene/materials
- 5 = adequate palette, but with noticeable color, lighting, white-balance, skin/material, or harmony issues
- 10 = prompt-specified colors, real-world/material colors, lighting, and palette harmony are rich, accurate, and consistent

**Creativity**
- 1 = generic, literal, or unimaginative with no meaningful interpretive choices
- 5 = competent interpretation with some appropriate choices but little originality
- 10 = surprising, delightful, and conceptually strong choices that enhance the prompt while preserving its requirements"""

RUBRIC_COMPACT = """The evaluation used this 1-10 scale: 1-2=unusable/almost no evidence, 3-4=poor with major failures, 5-6=mixed or acceptable with clear flaws, 7-8=strong with minor/moderate flaws, 9=excellent with tiny issues, 10=essentially flawless and rare. Score only from visible image evidence, do not let model/provider identity influence scoring, and do not give 8+ prompt adherence when a major prompt constraint is missing.

**Prompt adherence**: subject, relationships, text/logos, counts, style/medium, aspect/layout, and constraints match the prompt
**Photorealism**: photo realism when requested; otherwise material/medium believability, physical coherence, and lack of artifacts
**Aesthetic quality**: visual appeal, polish, craft, and emotional/genre impact
**Composition**: framing, balance, hierarchy, depth, cropping, layout, and visual flow
**Color accuracy**: prompt-specified colors, realistic material/skin colors, lighting/white balance, and palette harmony
**Creativity**: original choices that enhance the prompt without violating its requirements"""

EVAL_SYSTEM_PROMPT = f"""You are an expert image quality evaluator. You will be shown two AI-generated images (Image A from GPT Image-2, Image B from Gemini 3 Pro) created from the same prompt.

Evaluate EACH image independently across 6 dimensions. Be precise and rigorous.

{RUBRIC}

For each dimension, provide a score (1-10), concrete visual evidence in English, detailed reasoning in English, the same reasoning translated to Chinese, and a confidence value from 1-5. Confidence means how certain you are that the score is calibrated, not how good the image is.

Return TWO evaluations as JSON — one for each model. Use this exact schema:

{{
  "prompt_difficulty": "easy|medium|hard",
  "model_a": {{
    "model_name": "GPT Image-2",
    "prompt_adherence": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文..."}},
    "photorealism": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文..."}},
    "aesthetic_quality": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文..."}},
    "composition": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文..."}},
    "color_accuracy": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文..."}},
    "creativity": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文..."}}
  }},
  "model_b": {{
    "model_name": "Gemini 3 Pro",
    "prompt_adherence": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文..."}},
    "photorealism": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文..."}},
    "aesthetic_quality": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文..."}},
    "composition": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文..."}},
    "color_accuracy": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文..."}},
    "creativity": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文..."}}
  }}
}}

Return ONLY valid JSON, no other text."""

CRITIQUE_SYSTEM_PROMPT = f"""You are a critical reviewer of AI image evaluations. Another model (Claude Opus) has evaluated two AI-generated images. Your job is to review that evaluation for:

1. Scoring inconsistencies (e.g., high photorealism score but reasoning mentions artifacts)
2. Unsupported reasoning (claims about image features that aren't visible)
3. Potential bias toward either model (systematically higher scores for one)
4. Whether scores match the rubric anchor definitions

{RUBRIC_COMPACT}

Both images are included so you can verify claims made in the evaluation.

Return your critique as JSON with this exact schema:

{{
  "overall_assessment": "Your overall assessment of the evaluation quality",
  "dimension_critiques": [
    {{
      "dimension": "prompt_adherence",
      "original_score_model_a": N,
      "original_score_model_b": N,
      "critique": "What's wrong with this scoring and why",
      "suggested_score_model_a": N,
      "suggested_score_model_b": N
    }}
  ],
  "bias_detection": "Any systematic bias detected or 'No systematic bias detected'"
}}

Include ALL 6 dimensions in dimension_critiques, even if you agree with the original scores.
Return ONLY valid JSON, no other text."""

CRITIQUE_ROUND2_SYSTEM_PROMPT = f"""You are a critical reviewer of AI image evaluations. Your job is to independently review the revised evaluation you receive for remaining issues. Do not assume any prior critique exists; judge only the prompt, the images, the rubric, and the revised evaluation provided to you.

Review for:

1. Scoring inconsistencies
2. Unsupported evidence or reasoning
3. Whether the revised scores are well-justified by what you see in the images
4. Any bias patterns
5. Confidence calibration problems when confidence values are present

{RUBRIC_COMPACT}

Both images are included so you can verify claims made in the evaluation.

Return your critique as JSON with this exact schema:

{{
  "overall_assessment": "Your overall assessment of the revised evaluation quality",
  "dimension_critiques": [
    {{
      "dimension": "prompt_adherence",
      "original_score_model_a": N,
      "original_score_model_b": N,
      "critique": "What's wrong with this scoring and why",
      "suggested_score_model_a": N,
      "suggested_score_model_b": N
    }}
  ],
  "bias_detection": "Any systematic bias detected or 'No systematic bias detected'"
}}

Include ALL 6 dimensions in dimension_critiques, even if you agree with the scores.
Return ONLY valid JSON, no other text."""

REVISE_SYSTEM_PROMPT = f"""You are an expert image quality evaluator performing a REVISED evaluation. You previously evaluated two images, and a reviewer has challenged some of your scores.

{RUBRIC_COMPACT}

Re-evaluate ALL 6 dimensions for BOTH images. For each dimension:
- Consider the reviewer's critique carefully
- Look at the images again to verify their claims
- Accept or reject each critique point based on what you actually see
- Provide your revised score, concrete visual evidence, confidence from 1-5, and reasoning in both English and Chinese

Return JSON with this exact schema:

{{
  "model_a": {{
    "model_name": "GPT Image-2",
    "prompt_adherence": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文...", "critique_accepted": true/false, "revision_note": "..."}},
    "photorealism": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文...", "critique_accepted": true/false, "revision_note": "..."}},
    "aesthetic_quality": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文...", "critique_accepted": true/false, "revision_note": "..."}},
    "composition": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文...", "critique_accepted": true/false, "revision_note": "..."}},
    "color_accuracy": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文...", "critique_accepted": true/false, "revision_note": "..."}},
    "creativity": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文...", "critique_accepted": true/false, "revision_note": "..."}}
  }},
  "model_b": {{
    "model_name": "Gemini 3 Pro",
    "prompt_adherence": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文...", "critique_accepted": true/false, "revision_note": "..."}},
    "photorealism": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文...", "critique_accepted": true/false, "revision_note": "..."}},
    "aesthetic_quality": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文...", "critique_accepted": true/false, "revision_note": "..."}},
    "composition": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文...", "critique_accepted": true/false, "revision_note": "..."}},
    "color_accuracy": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文...", "critique_accepted": true/false, "revision_note": "..."}},
    "creativity": {{"score": N, "evidence": "Concrete visible evidence...", "confidence": N, "reasoning": "English...", "reasoning_zh": "中文...", "critique_accepted": true/false, "revision_note": "..."}}
  }}
}}

Return ONLY valid JSON, no other text."""
