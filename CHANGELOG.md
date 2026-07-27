# Changelog

All notable changes to this project will be documented in this file.

The current project version is `1.0`. For every future version update, add a new entry at the top of this file so version changes remain traceable.

## [1.0] - 2026-07-27

### Added

- Added a LangGraph `StateGraph` orchestrator in [graph_pipeline.py](graph_pipeline.py) as a drop-in alternative to [pipeline.py](pipeline.py): 11 nodes, 6 conditional edges, and the critique/revision cycle expressed as a real loop edge rather than a Python `while`. Both orchestrators write identical run artifacts, so history, dashboard, and reload are unaffected by which one ran.
- Added checkpointed crash recovery via `SqliteSaver` (`runs/checkpoints.db`, one thread per run). A suspended run survives the death of the process that created it; a cold process resumes without re-running upstream stages, so images already generated are not paid for twice. Threads are retired at `END` and pruned when a run is deleted.
- Added `interrupt()`-based human-in-the-loop gates. Each gate is split into a compute node (scoring, artifact writes) and a `*_wait` node containing only the interrupt, so resuming replays the wait node and never re-executes the gate computation.
- Added [orchestrator.py](orchestrator.py) as the single seam app.py calls through, selected by `USE_GRAPH_ORCHESTRATOR` (default off) and read per call.
- Added support for Claude Opus 5 (`EVAL_MODEL=claude-opus-5`) and GPT-5.6 Sol (`CRITIQUE_MODEL=gpt-5.6-sol`), both exercised in a full live run.
- Added an **Architecture V3** page documenting the LangGraph topology, and a **Model Pricing Compare** page comparing the image models on a single commercial plane (Azure or Google Cloud Vertex AI).
- Added [errors.py](errors.py) so `CheckpointMissing` can be caught without importing LangGraph.

### Changed

- Human decisions now affect scoring instead of only being archived. Gate 1 arbitration overrides per-dimension winners while leaving every score as the model produced it; gate 2 adjudication selects, per dimension, whether the round-1 or round-2 revision stands. Previously both gates were write-only in both orchestrators: a reviewer's verdict was persisted to JSON and never read back.
- Dimension cards and the reasoning table now read the same revision the verdict came from, so an adjudicated dimension no longer shows round 1's score above round 2's argument against it.
- Raised critique and evaluation token ceilings (`CRITIQUE_MAX_TOKENS`, `CRITIQUE_ROUND2_MAX_TOKENS`, `LLM_MAX_TOKENS`) after reasoning-heavy models exhausted the previous budget before emitting visible output.
- Dashboard aggregate statistics and history now derive model identity from each run rather than from fixed slots.

### Fixed

- Fixed `graph_pipeline` being imported at module scope in [orchestrator.py](orchestrator.py), which made LangGraph a hard requirement even with the flag off — `streamlit run app.py` failed with `ModuleNotFoundError` on any interpreter without it. The graph is now imported lazily; with the flag on and the dependency missing, the error names the fix instead of silently falling back to the legacy orchestrator.
- Fixed gate-pending runs rendering as failures; they now show an awaiting-review state.
- Fixed hardcoded model labels in progress text, prompts, and the dashboard when a non-default model pair is selected.

### Notes

- `USE_GRAPH_ORCHESTRATOR` defaults to **off**; the legacy orchestrator remains the default path. Turning it on requires the project virtualenv (`uv run streamlit run app.py`).
- The human-pause path is covered by tests (in-memory, SQLite, and cross-process) but has not yet been observed pausing in production: gate 1 has not triggered in 30 real runs (max route score 0.328 against a 0.35 threshold), and the one full live run through the graph cleared both gates.

## [0.8] - 2026-05-08

### Added

- Added support for uploading a reference image in the Streamlit generation composer, including an inline thumbnail preview and click-to-enlarge behavior before starting a run.
- Added end-to-end reference-image generation support across the pipeline: uploaded images are normalized, persisted with run artifacts, passed to GPT Image-2 and Gemini 3 Pro, and reloaded with historical runs.
- Added side-by-side comparison display that includes the uploaded reference image alongside the generated GPT Image-2 and Gemini 3 Pro outputs when a run uses an attachment.
- Added resumable generation for failed attachment runs so "Rerun from failed step" can regenerate missing images instead of requiring a fully completed generation stage.

### Changed

- Updated GPT Image-2 attachment generation to use streaming image edits, low-latency image quality settings, bounded request timeouts, and single-attempt attachment retries to avoid long hidden retry loops.
- Improved generation failure messages so empty provider exceptions and timeout cases are shown with actionable details in the dashboard.

### Notes

- GPT Image-2 and Azure OpenAI support reference-image editing, but attachment edits can be slower than text-only generation; the app now handles this path more directly and records attachment metadata in run summaries.

## [0.7] - 2026-05-05

### Added

- Added a [README.md](README.md) snapshot tour at the end of the document, embedding the files under [snapshot/](snapshot/) so readers can quickly understand the dashboard screens, HIL gate panels, scoring views, raw artifacts, analytics dashboard, and architecture diagrams.

### Notes

- This release is documentation-focused and does not change pipeline behavior, model configuration, run artifact layout, or Streamlit runtime logic.

## [0.6] - 2026-05-05

### Added

- Added [AGENTS.md](AGENTS.md) at the repo root as a cross-tool entry point (Copilot, Codex, Cursor, etc.), with a module map, command reference, conventions, and pitfalls; links to [CLAUDE.md](CLAUDE.md) and [README.md](README.md) instead of duplicating their content.

### Changed

- Refreshed [CLAUDE.md](CLAUDE.md) to reflect the current pipeline: documented the `gates.py` HIL routing stage (risk features, phase weights, `HIL_ENABLED_BY_DEFAULT`), expanded `schemas.py` to cover gate-side models (`GateDecision`, `RouteFeatures`, `RouteBand`, `Gate1ReviewItem`, `IssueEquivalenceReport`), noted the Python `>=3.12` requirement, and added a reminder that `uv.lock` should not be hand-edited.

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