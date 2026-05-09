# Snapshots

Languages: [English](SNAPSHOTS.md) | [简体中文](SNAPSHOTS.zh-CN.md)

The [snapshot/](snapshot/) directory contains screenshots that walk through the dashboard and architecture at a glance. They are useful when you want to understand the workflow before running Streamlit locally.

## Main Comparison View

![Dashboard prompt controls and side-by-side generated images](snapshot/Screenshot%202026-05-05%20at%208.22.19%E2%80%AFPM.png)

The main screen loads pre-baked examples, accepts a new prompt, shows pipeline completion status, and compares GPT Image-2 and Gemini 3 Pro outputs side by side.

## HIL Gate Status

![Human-in-the-loop gate status panel](snapshot/Screenshot%202026-05-05%20at%208.22.29%E2%80%AFPM.png)

The HIL V2 panel shows whether deterministic gates require attention, including route scores, route bands, trigger reasons, and review dimensions.

## Scores And Winner

![Winner banner, radar chart, and dimension score cards](snapshot/Screenshot%202026-05-05%20at%208.22.38%E2%80%AFPM.png)

The scoring section summarizes the final winner, overall averages, pre-critique versus post-critique score movement, and each dimension-level decision.

## Revised Rationale

![Dimension-by-dimension revised rationale table](snapshot/Screenshot%202026-05-05%20at%208.22.47%E2%80%AFPM.png)

The revised evaluation table records the reasoning for each rubric dimension after critique and revision, keeping the comparison auditable.

## Raw Artifacts

![Critique transcript and JSON artifact expanders](snapshot/Screenshot%202026-05-05%20at%208.22.53%E2%80%AFPM.png)

The raw artifact area exposes the critique transcript and JSON payloads for evaluation, critique, revisions, gates, and issue equivalence.

## Evaluation Dashboard

![Run-level analytics dashboard](snapshot/Screenshot%202026-05-05%20at%208.23.15%E2%80%AFPM.png)

The dashboard view aggregates historical runs with prompt categories, winner distribution, recent comparisons, margins, and score summaries.

## Human-Gated Architecture V2

![Human-gated architecture V2 diagram](snapshot/Screenshot%202026-05-05%20at%208.23.43%E2%80%AFPM.png)

Architecture V2 highlights confidence routing, optional human review, disagreement detection, adjudication, and archival before the final decision.

## Multi-Agent Architecture V1

![Multi-agent architecture V1 diagram](snapshot/Screenshot%202026-05-05%20at%208.23.52%E2%80%AFPM.png)

Architecture V1 shows the base flow from prompt intake through parallel image generation, evaluation, critique, revision, deterministic comparison, and artifact storage.

## Image-to-Image Workflow

![Image-to-image workflow snapshot](snapshot/Image2image.png)

The image-to-image workflow snapshot shows the extended comparison path for prompts that start from an existing visual reference.