# 截图导览

语言版本：[English](SNAPSHOTS.md) | [简体中文](SNAPSHOTS.zh-CN.md)

[snapshot/](snapshot/) 目录包含一组截图，用来快速浏览 dashboard 和架构设计。读者即使不先运行 Streamlit，也可以先了解整个 workflow 的界面和数据流。

## 主对比界面

![Prompt 控件和两张生成图并排对比](snapshot/Screenshot%202026-05-05%20at%208.22.19%E2%80%AFPM.png)

主界面可以加载 pre-baked 示例、输入新 prompt、查看 pipeline 完成状态，并并排比较 GPT Image-2 和 Gemini 3 Pro 的生成结果。

## HIL Gate 状态

![Human-in-the-loop gate 状态面板](snapshot/Screenshot%202026-05-05%20at%208.22.29%E2%80%AFPM.png)

HIL V2 面板展示确定性 gates 是否需要人工关注，包括 route score、route band、触发原因和需要 review 的维度。

## 分数和胜者

![胜者横幅、雷达图和维度分数卡片](snapshot/Screenshot%202026-05-05%20at%208.22.38%E2%80%AFPM.png)

评分区域汇总最终胜者、整体均分、critique 前后的分数变化，以及每个维度的单项结果。

## 修订后理由

![逐维度修订理由表格](snapshot/Screenshot%202026-05-05%20at%208.22.47%E2%80%AFPM.png)

修订后的 evaluation 表格记录每个 rubric 维度在 critique 和 revision 之后的判断依据，让比较过程保持可审计。

## 原始 Artifacts

![Critique transcript 和 JSON artifact 展开项](snapshot/Screenshot%202026-05-05%20at%208.22.53%E2%80%AFPM.png)

Raw artifact 区域展示 critique transcript，并提供 evaluation、critique、revision、gate 和 issue equivalence 等 JSON payload 的展开查看入口。

## Evaluation Dashboard

![Run 级 analytics dashboard](snapshot/Screenshot%202026-05-05%20at%208.23.15%E2%80%AFPM.png)

Dashboard 视图聚合历史 run，包括 prompt categories、winner distribution、recent comparisons、margin 和 score summary。

## Human-Gated Architecture V2

![Human-gated architecture V2 图](snapshot/Screenshot%202026-05-05%20at%208.23.43%E2%80%AFPM.png)

Architecture V2 展示 confidence routing、可选人工 review、disagreement detection、adjudication，以及最终决策前的 artifact 归档。

## Multi-Agent Architecture V1

![Multi-agent architecture V1 图](snapshot/Screenshot%202026-05-05%20at%208.23.52%E2%80%AFPM.png)

Architecture V1 展示基础流程：从 prompt intake 到并行图像生成、evaluation、critique、revision、确定性 comparison 和 artifact storage。

## Image-to-Image Workflow

![Image-to-image workflow 截图](snapshot/Image2image.png)

Image-to-image workflow 截图展示了从已有视觉参考开始的扩展比较路径。