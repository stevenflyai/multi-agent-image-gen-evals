"""Internationalization support for the Image Eval Pipeline UI."""

import streamlit as st

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # Header
        "header_title": "IMAGE EVAL PIPELINE",
        "header_subtitle": "Cross-Model Evaluation with Adversarial Review",
        # Language switcher
        "lang_label": "Language / 语言",
        # Input/actions
        "input_placeholder": "Enter a prompt to compare image generators...",
        "btn_generate": "GENERATE & EVALUATE",
        "select_example": "Load a pre-baked example",
        "select_example_placeholder": "Select an example...",
        # Pipeline stages
        "stage_generating": "Generating images with GPT Image-2 and Gemini 3 Pro...",
        "stage_evaluating": "Claude Opus evaluating both images...",
        "stage_critique": "GPT-5.4 reviewing evaluation...",
        "stage_revising": "Claude Opus revising scores with critique...",
        "stage_complete": "Pipeline complete!",
        "status_running": "Running evaluation pipeline...",
        "status_done": "Pipeline complete!",
        # Model labels
        "model_gpt": "GPT Image-2 (OpenAI)",
        "model_gemini": "Gemini 3 Pro (Google)",
        # Winner banner
        "winner_label": "Winner: ",
        "draw_label": "Draw",
        "dimensions_won": "{count}/6 dimensions",
        "overall_label": "Overall: {a} vs {b}",
        "unchallenged": "(unchallenged evaluation)",
        # Dimension labels
        "dim_prompt_adherence": "Prompt Adherence",
        "dim_photorealism": "Photorealism",
        "dim_aesthetic_quality": "Aesthetic Quality",
        "dim_composition": "Composition",
        "dim_color_accuracy": "Color Accuracy",
        "dim_creativity": "Creativity",
        # Dimension cards
        "revised_delta": "Revised: {a} / {b}",
        "accepted": "Accepted",
        "rejected": "Rejected",
        # Critique transcript
        "critique_transcript": "Critique Transcript",
        "step1_label": "Step 1: Claude Opus Initial Evaluation",
        "step2_label": "Step 2: GPT-5.4 Review & Feedback",
        "step3_label": "Step 3: Claude Opus Revised Evaluation",
        "step2_unavailable": "Review step unavailable. Showing initial evaluation only.",
        "bias_detection": "Bias detection:",
        "overall_assessment": "Overall:",
        # JSON expanders
        "json_initial": "Initial Evaluation JSON",
        "json_critique": "GPT-5.4 Critique JSON",
        "json_revised": "Revised Evaluation JSON",
        "json_unavailable": "Unavailable",
        # Error/empty states
        "error_pipeline_failed": "Pipeline failed: {error}",
        "error_image_failed": "Image generation failed",
        "empty_state": "Enter a prompt above or load a pre-baked example to see side-by-side image comparison with cross-model adversarial evaluation.",
        "run_label": "Run: {prompt} ({timestamp})",
    },
    "zh": {
        # Header
        "header_title": "图像评估流水线",
        "header_subtitle": "跨模型对抗评审评估",
        # Language switcher
        "lang_label": "Language / 语言",
        # Input/actions
        "input_placeholder": "输入提示词以比较图像生成器...",
        "btn_generate": "生成并评估",
        "select_example": "加载预设示例",
        "select_example_placeholder": "选择一个示例...",
        # Pipeline stages
        "stage_generating": "正在使用 GPT Image-2 和 Gemini 3 Pro 生成图像...",
        "stage_evaluating": "Claude Opus 正在评估两张图像...",
        "stage_critique": "GPT-5.4 正在审查评估...",
        "stage_revising": "Claude Opus 正在根据反馈修正评分...",
        "stage_complete": "流水线完成！",
        "status_running": "正在运行评估流水线...",
        "status_done": "流水线完成！",
        # Model labels
        "model_gpt": "GPT Image-2 (OpenAI)",
        "model_gemini": "Gemini 3 Pro (Google)",
        # Winner banner
        "winner_label": "优胜: ",
        "draw_label": "平局",
        "dimensions_won": "{count}/6 个维度",
        "overall_label": "总分: {a} vs {b}",
        "unchallenged": "(未经质疑的评估)",
        # Dimension labels
        "dim_prompt_adherence": "提示词遵循度",
        "dim_photorealism": "真实感",
        "dim_aesthetic_quality": "美学质量",
        "dim_composition": "构图",
        "dim_color_accuracy": "色彩准确度",
        "dim_creativity": "创意",
        # Dimension cards
        "revised_delta": "修正: {a} / {b}",
        "accepted": "采纳",
        "rejected": "拒绝",
        # Critique transcript
        "critique_transcript": "评审记录",
        "step1_label": "步骤一: Claude Opus 初始评估",
        "step2_label": "步骤二: GPT-5.4 审查与反馈",
        "step3_label": "步骤三: Claude Opus 修正评估",
        "step2_unavailable": "审查步骤不可用，仅展示初始评估。",
        "bias_detection": "偏差检测:",
        "overall_assessment": "总体评价:",
        # JSON expanders
        "json_initial": "初始评估 JSON",
        "json_critique": "GPT-5.4 评审 JSON",
        "json_revised": "修正评估 JSON",
        "json_unavailable": "不可用",
        # Error/empty states
        "error_pipeline_failed": "流水线失败: {error}",
        "error_image_failed": "图像生成失败",
        "empty_state": "在上方输入提示词或加载预设示例，查看跨模型对抗评审的图像对比评估。",
        "run_label": "运行: {prompt} ({timestamp})",
    },
}

DIMENSION_KEY_MAP: dict[str, str] = {
    "prompt_adherence": "dim_prompt_adherence",
    "photorealism": "dim_photorealism",
    "aesthetic_quality": "dim_aesthetic_quality",
    "composition": "dim_composition",
    "color_accuracy": "dim_color_accuracy",
    "creativity": "dim_creativity",
}


def get_lang() -> str:
    return st.session_state.get("lang", "zh")


def t(key: str, **kwargs: object) -> str:
    lang = get_lang()
    text = TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text


def dim_label(dimension: str) -> str:
    key = DIMENSION_KEY_MAP.get(dimension)
    if key:
        return t(key)
    return dimension
