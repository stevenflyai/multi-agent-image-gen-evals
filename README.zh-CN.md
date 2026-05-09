# Multi-Agent Image Generation Evals Pipeline

语言版本：[English](README.md) | [简体中文](README.zh-CN.md)

Multi-Agent Image Generation Evals Pipeline 是一个用于评估和比较 AI 图像生成模型的 multi-agent system。应用会把同一个 prompt 同时发送给两个图像生成 agent，再由独立 evaluation agent 按统一 rubric 评估两张输出图，并通过 critique agent 与 revision agent 多轮挑战和修正评分；当风险较高时，可暂停进入 human-in-the-loop (HIL) 人工裁决，并把每次运行的完整过程归档，便于后续复盘。

主入口是一个用于操作 multi-agent workflow 的中英双语 Streamlit Dashboard，支持并排看图、进度和 Activity Log、分数可视化、critique transcript、HIL 审核控件、历史 run 加载、删除和重跑。

## 截图导览

完整截图导览已移至 [SNAPSHOTS.zh-CN.md](SNAPSHOTS.zh-CN.md)。

## 功能概览

- 在一个可审计 workflow 中协调 generation、evaluation、critique、revision、gate 和 comparison agents。
- 使用同一个 prompt 并行生成两张图片。
- 基于 6 个维度的 1-10 分校准 rubric，对两张图片打分。
- 使用独立 critique agent 发现评估中的错误、偏见、证据不足和分数不一致。
- critique 后修订分数，同时保留原始 evaluation artifact。
- 通过确定性的 HIL gates 路由不确定或内部冲突的案例。
- 将图片、prompt、JSON artifact、原始模型输出、gate decision、activity snapshot 和最终 summary 持久化到 `runs/`。
- 可从历史 run 加载 pre-baked 示例，并支持只删除当前选中的 run，或用当前 run 的 prompt 重新运行。

## 模型

| 角色 | 模型 | 提供方 |
| --- | --- | --- |
| 图像生成 A | GPT Image-2 | OpenAI |
| 图像生成 B | Gemini 3 Pro | Google |
| 初始评估 | Claude Opus 4.7 | Anthropic |
| 分数修订 | Claude Opus 4.7 | Anthropic |
| Critique round 1 | GPT-5.4 | OpenAI |
| Critique round 2 | Gemini 3.1 Pro | Google |

模型名称和运行参数集中配置在 [config.py](config.py)。

## Pipeline

```text
Prompt
  |
  +--> GPT Image-2 --------+
  |                        |
  +--> Gemini 3 Pro -------+--> Claude evaluates both images
           |
           v
         Gate 1: uncertainty/risk router
           |
           v
         GPT-5.4 critique (round 1)
           |
           v
         Claude revision (round 1)
           |
           v
         Gemini critique (round 2)
           |
           v
         Gate 2: disagreement detector
           |
           v
         Claude final revision
           |
           v
         Deterministic comparison
```

Pipeline 编排逻辑在 [pipeline.py](pipeline.py)。各 provider/model adapter 分布在 [generate.py](generate.py)、[evaluate.py](evaluate.py)、[critique.py](critique.py) 和 [revise.py](revise.py)。Rubric 和 prompt contract 集中在 [prompts.py](prompts.py)，所有输出结构由 [schemas.py](schemas.py) 的 Pydantic schema 定义。

### 核心阶段

1. **Generation**：GPT Image-2 和 Gemini 3 Pro 并行生成图片，带 retry 和 timeout 处理。
2. **Evaluation**：Claude 按 6 个 rubric 维度评估两张图，并记录 evidence/confidence 字段。
3. **Gate 1**：确定性的 uncertainty/risk router，检查 margin risk、prompt difficulty、evidence quality、confidence 和跨维度冲突。
4. **Critique round 1**：GPT-5.4 独立审查初始 evaluation。
5. **Revision round 1**：Claude 基于 critique context 修订 evaluation。
6. **Critique round 2**：Gemini 独立审查 revised evaluation。原始 Gemini 输出会被归档，便于排查 malformed 或 truncated JSON。
7. **Gate 2**：确定性的 disagreement detector，对比 critique signal 和 score movement。
8. **Final revision**：Claude 综合 critique signal 和可选 HIL label 做最终修订。
9. **Comparison**：[compare.py](compare.py) 根据平均分、最大单维领先幅度、draw 规则计算最终胜者。

如果图像生成阶段失败，run 会被标记为 failed，而不是显示为成功。如果后续 critique/revision 阶段失败，pipeline 会记录错误，并尽可能 fallback 到已完成的最佳可用分数。

## Human-in-the-loop Gates

HIL 逻辑在 [gates.py](gates.py) 实现，并由 [app.py](app.py) 展示到 UI。

| Gate | 目的 | 可能影响 |
| --- | --- | --- |
| Gate 1: uncertainty/risk router | 标记 narrow、hard、low-confidence 或 evidence-poor 的 evaluation | HIL 开启时，可在 critique 前暂停 |
| Gate 2: disagreement detector | 标记 critique disagreement、winner flip、大幅分数变动或新的关键问题 | HIL 开启时，可在 final revision 前暂停 |

人工 label 是窄范围、结构化、可审计的。它们会指导后续 revision 和 decision context，但不会直接改写原始模型分数。

## 评估维度

分数使用完整 1-10 区间：

| 分数段 | 含义 |
| --- | --- |
| 1-2 | 基本不可用，或几乎没有符合要求的证据 |
| 3-4 | 较差，存在重大失败 |
| 5-6 | 混合或可接受，但有明显缺陷 |
| 7-8 | 较强，只有轻中度问题 |
| 9 | 优秀，仅有很小问题 |
| 10 | 几乎完美，且非常少见 |

| 维度 | 衡量内容 |
| --- | --- |
| Prompt adherence | 是否满足 prompt 中明确和隐含的要求 |
| Photorealism | 物理可信度、artifact 控制和媒介一致性 |
| Aesthetic quality | 视觉工艺、精致度、吸引力和类型匹配 |
| Composition | 构图、层级、纵深、裁切和视觉流动 |
| Color accuracy | prompt 指定颜色、光照、材质和调色一致性 |
| Creativity | 在不违背 prompt 的前提下，做出有价值且令人惊喜的解释 |

维度列表统一定义在 [schemas.py](schemas.py)。

## Dashboard

运行 Streamlit app：

```bash
streamlit run app.py
```

本地开发中常用命令：

```bash
/opt/anaconda3/bin/python -m streamlit run app.py --server.port 8504
```

Dashboard 包含：

- Prompt 输入和运行控制。
- 新 run 的实时 Activity Log 和模型/阶段耗时。
- 老 run 的历史 Activity 重建，不伪造生成耗时。
- 两张生成图并排对比。
- Winner banner 和维度级 score cards。
- 初始分数与修订分数的 radar chart。
- 自适应 category column 的 Dashboard 表格。
- 多轮 critique transcript。
- 触发时显示 HIL Gate 1 / Gate 2 审核面板。
- evaluation、critique、revision、gate decision 和 comparison 的 Raw JSON expanders。
- 基于 `runs/index.json` 和 `runs/*/summary.json` 的 pre-baked example dropdown。
- 删除当前选中 run，以及用当前 prompt rerun。
- 通过 [i18n.py](i18n.py) 支持英文和中文 UI。

## 安装配置

### 前置要求

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) 或等价 Python 环境

### 安装

```bash
git clone https://github.com/stevenflyai/multi-agent-image-gen-evals.git
cd multi-agent-image-gen-evals
uv sync
```

如果使用本 workspace 的 Conda Python，可以这样运行命令：

```bash
/opt/anaconda3/bin/python -m pytest -q
```

### 环境变量

复制 [.env.example](.env.example) 为 `.env` 并填写 key：

```bash
cp .env.example .env
```

| 变量 | 必填 | 用途 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 是 | GPT Image-2 图像生成和 GPT-5.4 critique |
| `ANTHROPIC_API_KEY` | 是 | Claude Opus evaluation 和 revision |
| `GOOGLE_API_KEY` | 是 | Gemini 图像生成和 critique |
| `OPENAI_BASE_URL` | 可选 | Azure OpenAI 或兼容代理 endpoint |
| `ANTHROPIC_BASE_URL` | 可选 | Anthropic 兼容代理 endpoint |

不要提交 `.env`；它已被 git ignore。

## 测试

运行完整测试：

```bash
pytest -q
```

或使用 workspace Conda Python：

```bash
/opt/anaconda3/bin/python -m pytest -q
```

常用 focused test：

```bash
pytest tests/test_utils.py -q
pytest tests/test_gates.py tests/test_pipeline.py -q
pytest tests/test_winner.py::test_tiebreak_by_largest_dimension_lead -v
```

当前测试覆盖 schema、JSON repair/truncation handling、generation failure handling、HIL gate decision、pipeline resume/fallback behavior 和 winner selection。

## Run Artifacts

每次运行都会保存在 `runs/YYYYMMDD_HHMMSS_microseconds/`。目录中可能包含：

```text
prompt.txt
gpt_image_2.png
gemini_3_pro.png
evaluation.json
evaluation_v2.json
gate1_decision.json
critique_r1.json
revised_r1.json
critique_r2_raw.txt
critique_r2.json
issue_equivalence.json
gate2_decision.json
revised_r2.json
comparison.json
summary.json
activity_snapshot.json
prompt_inputs_05_critique.json
prompt_inputs_06_revision.json
prompt_inputs_07_critique.json
prompt_inputs_08_final_revision.json
```

`runs/` 已被 git ignore，因为里面包含生成图片、原始模型输出和本地历史记录。

## 项目结构

```text
multi-agent-image-gen-evals/
|-- app.py                 # Streamlit dashboard 和 run 管理 UI
|-- pipeline.py            # Pipeline 编排、持久化、resume/fallback 逻辑
|-- generate.py            # 并行 GPT Image-2 和 Gemini 3 Pro 图像生成
|-- evaluate.py            # Claude 初始评分
|-- critique.py            # GPT round 1 和 Gemini round 2 critique adapter
|-- revise.py              # Claude revision adapter
|-- compare.py             # 最终确定性 winner selection
|-- gates.py               # HIL Gate 1 和 Gate 2 decision logic
|-- schemas.py             # Pydantic models 和共享 dimensions
|-- prompts.py             # Rubric 和 prompt contracts
|-- utils.py               # Retry、图像编码、JSON parsing/repair helpers
|-- i18n.py                # 英文/中文 UI 文案 helpers
|-- config.py              # 模型名、timeout、retry、HIL 默认配置
|-- static/style.css       # Streamlit UI 样式
|-- tests/                 # Pytest suite
|-- docs/superpowers/      # 设计说明和 Architecture V2 docs
|-- runs/                  # 本地生成 runs，git ignored
|-- pyproject.toml
|-- .env.example
|-- .gitignore
`-- CLAUDE.md
```

## 配置

关键配置位于 [config.py](config.py)：

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `MAX_CRITIQUE_ROUNDS` | `2` | 最大 critique/revision 轮数 |
| `CONVERGENCE_THRESHOLD` | `1` | 当所有分数 delta 低于该阈值时停止 |
| `MAX_RETRIES` | `3` | LLM/image 调用 retry 次数 |
| `RETRY_BACKOFF` | `2.0` | 指数退避基础值 |
| `IMAGE_MAX_SIZE` | `768` | 发送给 evaluator/critic 的最大图片尺寸 |
| `IMAGE_QUALITY` | `85` | LLM 输入图的 JPEG quality |
| `LLM_MAX_TOKENS` | `4096` | 默认文本模型输出 token 预算 |
| `CRITIQUE_ROUND2_MAX_TOKENS` | `8192` | Gemini critique 更大的 token 预算，降低截断概率 |
| `IMAGE_GEN_TIMEOUT` | `600` | 并行图像生成 timeout，单位秒 |
| `HIL_ENABLED_BY_DEFAULT` | `True` | gate 触发时是否允许暂停进入 HIL review |

## 贡献说明

- 用户可见 UI 文案放在 [i18n.py](i18n.py)，在 app 中使用 `t(...)` / `dim_label(...)`。
- Rubric 文本和 system prompts 放在 [prompts.py](prompts.py)。
- 模型名称和运行参数放在 [config.py](config.py)。
- 使用 `unsafe_allow_html=True` 渲染动态 HTML 前，需要 escape 动态值。
- 保留 prompt-input metadata 边界，使 critique independence 可审计。
- 不要提交本地 `runs/`、`.env`、截图、cache 或生成的 secret。

## 作者

Steven Lian (stevenlian2981@gmail.com)

仓库地址：https://github.com/stevenflyai/multi-agent-image-gen-evals

