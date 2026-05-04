# Changelog

All notable changes to this project will be documented in this file.

The current project version is `0.5`. For every future version update, add a new entry at the top of this file so version changes remain traceable.

## [0.5] - 2026-05-04

### Added

- Added a bilingual README structure with [README.md](README.md) and [README.zh-CN.md](README.zh-CN.md), including language switch links at the top of both files.
- Added HIL V2 gate documentation covering Gate 1 uncertainty/risk routing and Gate 2 disagreement detection.
- Added documentation for historical/pre-baked run management, including loading, deleting the selected run, and rerunning a selected prompt.
- Added documentation for persisted run artifacts such as `gate1_decision.json`, `gate2_decision.json`, `issue_equivalence.json`, `activity_snapshot.json`, `critique_r2_raw.txt`, and prompt-input metadata files.
- Added documentation for current testing commands, run commands, environment variables, and contributor conventions.

### Changed

- Reworked the main README to describe the current multi-agent image evaluation pipeline instead of the earlier simpler critique loop.
- Clarified model roles across generation, evaluation, critique, and revision.
- Clarified failure semantics: generation failures now mark a run as failed, while later critique/revision failures are recorded with fallback behavior where possible.
- Clarified that HIL labels guide revision and decision context but do not directly mutate raw model scores.

### Current Pipeline Highlights

- Parallel GPT Image-2 and Gemini 3 Pro image generation.
- Claude Opus evaluation across six calibrated dimensions.
- GPT-5.4 first-round critique and Claude first-round revision.
- Gemini second-round critique with raw output archival for JSON/truncation debugging.
- Deterministic HIL Gate 1 and Gate 2 decision logic.
- Final deterministic comparison by mean score, largest dimension lead, then draw.
- Streamlit dashboard with activity logs, score visualizations, critique transcript, HIL panels, raw JSON inspectors, and bilingual UI.