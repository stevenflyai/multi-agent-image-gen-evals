# HIL V2 Refactor Design — Review Checklist

> 用于 review 会上逐项确认。按 **P0 必须修 / P1 必须加 / P2 建议改 / P3 细节** 四档分级。
> 每条都带具体动作和验收标准，可直接派活。

---

## P0 必须修（动手前阻塞项）

### ☐ 1. Gate 2 issue equivalence checker 未定义

**问题**：trigger rule 第一条要求"判断 07 提的 issue 是否等价于 05 提的"，但实现机制缺失。

**动作**：

- 在 Gate 2 章节加 subsection `Issue Equivalence Detection`
- 明确 checker 是独立 LLM 调用（推荐 Claude），输入 05 + 07 的 issue list，输出每条 07 issue 的分类（equivalent / new / contradicting）
- 文档里强调：checker 是 pipeline-layer tool,可以同时看 05 和 07 的全部内容,**不破坏 07 的独立性**
- 给出 checker 的 prompt 模板草稿

**验收**:Gate 2 章节包含一段可直接照着实现的 checker 规格。

---

### ☐ 2. HIL off 状态下不同 route band 的行为不明确

**问题**：Phase 2 写"HIL off + record would_trigger"，但 `route_score >= 0.70` 的 critical case 在 HIL off 时是直接出结论还是标记为不可信？

**动作**：在 Gate 1 章节加一个 4×2 行为矩阵：

| Route band | HIL flag off | HIL flag on |
|---|---|---|
| `< 0.35` | continue | continue |
| `0.35–0.55` | continue + log "would_recommend" | continue + log "would_recommend" |
| `0.55–0.70` | continue + flag result `auto_continued_high_risk` | pause for HIL |
| `>= 0.70` | continue + flag result `auto_continued_critical_risk` | pause; skip requires explicit confirmation |

- `pipeline_status` 增加 `auto_continued_high_risk` 和 `auto_continued_critical_risk` 两个值
- summary.json 必须显著标注这两个状态，避免 Phase 2 数据被无标识地混入下游分析

**验收**：文档矩阵存在，Phase 2 测试用例覆盖 `auto_continued_critical_risk` 的 flag 行为。

---

### ☐ 3. Phase 1 缺 variance 时 route_score 权重未重新归一化

**问题**：文档说 Phase 1 可以 disable variance term，但没说剩余 5 项权重怎么处理。线性加权下不归一化会让 route_score 上限变成 0.75，所有阈值都失效。

**动作**：在 route score 章节明确给出两套权重表：

```
Phase 1 (no variance):
  self_confidence_risk:    0.20
  margin_risk:             0.27
  cross_dim_conflict:      0.20
  difficulty_prior:        0.20
  evidence_quality_risk:   0.13
  Total:                   1.00

Phase 2+ (with variance):
  self_confidence_risk:    0.15
  variance:                0.25
  margin_risk:             0.20
  cross_dim_conflict:      0.15
  difficulty_prior:        0.15
  evidence_quality_risk:   0.10
  Total:                   1.00
```

**验收**：文档两套权重并存；Phase 1 → Phase 2 切换时有 migration note。

---

### ☐ 4. HIL reviewer identity 不能列为 open question

**问题**：identity 是 inter-reviewer agreement、个体质量追踪、疲劳监控的前提，不能 later 决定。

**动作**：

- 把 Open Question 1 移除，改写成 Phase 1 deliverable
- 决定方案：本地 session 启动时让用户输入一次 reviewer 字符串（默认 OS username），全程沿用，存入所有 HIL artifact 的 `reviewer` 字段
- 不做认证，但要可追踪

**验收**：Phase 1 task list 包含"实现 reviewer identity prompt + session persistence"。

---

## P1 必须加（缺失的关键章节）

### ☐ 5. Calibration Set 章节缺失

**问题**：整份文档没有离线人类标注集合的概念。所有 prompt/阈值改动没有验证基线，Phase 切换没有 agreement check。

**动作**：新增 section `Offline Calibration Infrastructure`，包含：

- **构建方式**：200 个 case，按任务类型 + 难度 + 维度覆盖采样
- **标注 schema**：完整 6 维度 1-10 + evidence + confidence
- **标注成本**：约 27 小时一次性投入
- **使用场景**：
  - 每次 evaluator/critique prompt 改动前后跑 agreement check
  - Phase 切换的 acceptance gate
  - route_score 权重的 grid search 数据源
  - 05/07 prompt equivalence 验证（见 #8）
- **验收指标**：human-model agreement（Spearman / quadratic-weighted Cohen's kappa）

**验收**：章节存在；Phase 1 deliverable 加入"build initial calibration set"。

---

### ☐ 6. Periodic Audit 章节缺失

**问题**：Gate 1 在 `< 0.35` 时无审计——如果 evaluator 在某类 prompt 上 over-confident，永远不被发现。

**动作**：新增 subsection `Periodic Audit of High-Confidence Cases`：

- 每周从 Archive 抽样 20-50 个 `route_score < 0.35` 的 case
- 让人重新盲评
- 追踪 high-confidence agreement 随时间的趋势
- 触发条件：连续 2 周 agreement < 0.7 → 回去调 evaluator prompt

**验收**：章节存在；Phase 5（dashboard polish）加入 audit drift 可视化。

---

### ☐ 7. Continuous Improvement Loop 章节缺失

**问题**：HIL labels 只在线用（喂给 06/08 revision），离线没用——每次人介入应该让模型变得更好的反馈循环没建立。

**动作**：新增 Phase 6 或独立 section `Continuous Improvement Loop`：

- 月度分析"模型 vs 人类标签差异最大的维度" → 反向调 evaluator prompt anchors
- Gate 2 的 "which judge was more accurate" 数据 → 动态调 05 vs 07 在 Decision Engine 的相对权重
- evidence_quality_risk 高的 case 集合 → 优化 evaluator prompt 的 evidence 要求
- HIL 标注积累到一定量后 → 可作为内部 reward model 的训练种子

**验收**：章节存在；包含至少 3 个具体的"每月做什么"的动作清单。

---

## P2 建议改（提升严谨度）

### ☐ 8. 05/07 prompt equivalence 验证标准不够

**问题**：现有测试只能验证 schema 等价，不能验证 behavioral 等价。

**动作**：把 equivalence 拆成 3 层：

1. **Schema equivalence**（自动）：JSON schema 字段一致
2. **Rubric equivalence**（自动）：维度定义、scale 描述、calibration rules 内容一致（normalize string 后比对）
3. **Behavioral equivalence**（半自动）：在 Calibration Set 上用同一模型分别跑两个 prompt，agreement ≥ 0.7

第 3 条作为 Phase 1 末期的 acceptance criteria，依赖 #5 的 Calibration Set。

**验收**：测试章节加 3 条对应 test case；Phase 1 exit criteria 包含 behavioral equivalence ≥ 0.7。

---

### ☐ 9. route_score 用纯线性加权可能漏掉单维度极端 risk

**问题**：5 个低风险信号 + 1 个极端风险（如某维度 confidence=1 且 evidence 完全空白）平均后可能不触发 HIL。

**动作**：route_score 改成线性加权和单维度极端 risk 的 max：

```
route_score = max(
    weighted_linear_score,
    max over all (image, dimension) of dim_specific_risk
)

dim_specific_risk fired when:
  confidence == 1 AND evidence is generic/empty
  OR score >= 8 AND evidence flags severe artifacts (contradiction)
  OR margin == 0 AND both confidences <= 2
```

**验收**：route_score 公式加 max-aggregation 段；测试覆盖单维度极端 risk 触发场景。

---

### ☐ 10. prompt_difficulty 不应只靠模型 self-report

**问题**：evaluator 自报 difficulty 有 motivated reasoning 风险（"评得不好就报 hard"）。

**动作**：difficulty 取两个独立来源的 max：

- evaluator self-report（现方案）
- 确定性 prompt 特征启发式（多主体计数、文字渲染、负面约束、UI 渲染等关键词检测）

```
final_difficulty = max(self_report, deterministic_estimate)
```

**验收**：Gate 1 章节加 difficulty heuristics 描述；测试覆盖"模型报 easy 但 prompt 含 5 个 countable subject"的场景。

---

## P3 细节（可在实现中调整）

### ☐ 11. summary.json 加 pipeline_health 字段

记录每次 run 的 health metrics：每个 agent 是否成功调用、evidence quality 整体水平、是否有 prompt boundary violation（静态扫描）、HIL gate 触发情况。

**验收**：summary.json schema 包含 `pipeline_health` 对象。

---

### ☐ 12. runs/index.json 按 schema_version 分桶

不要运行时检测每个 run 的 schema 版本，在 index 层就分桶（`index_v1.json` / `index_v2.json`），UI 分别渲染。

**验收**：persistence 章节明确分桶策略。

---

### ☐ 13. 补一份 ADR（Architecture Decision Record）

1 页文档列出"考虑过但没采用的方案 + 否定理由"：

- 为什么不用 Image Arena 风格的纯 ELO 对比
- 为什么 critique 不直接改像素而只输出 reasoning
- 为什么 HIL 分层（Tier 1/2）而非单一队列
- 为什么 Decision Engine 不让 HIL 直接覆盖 score
- 为什么 05 / 07 用不同家族模型而非同一模型多次采样

**验收**：`docs/adr/001-hil-v2-decisions.md` 存在；至少 5 条决策有 rationale。

---

## 附:Phase 1 实施前的 Definitive Pre-flight 清单

在写第一行代码之前，下面这些必须就位：

- [ ] P0 全部 4 项已修
- [ ] P1 第 5 项 Calibration Set 已构建（200 case 已标注）
- [ ] reviewer identity 决策已定（#4）
- [ ] route_score 权重表两版本已确定（#3）
- [ ] HIL off 状态下 4×2 行为矩阵已确定（#2）
- [ ] Gate 2 issue equivalence checker 的 prompt 模板已草拟（#1）
- [ ] ADR 起草（#13）

P2 / P3 可以在 Phase 进行中迭代，但 P0 + #5 是阻塞 Phase 1 启动的。

---

## 用法建议

**Review 会议安排**：

- 30 分钟过 P0（4 项）—— 必须当场决策，不能 parking lot
- 20 分钟过 P1（3 项）—— 决定要不要加、加在哪个 phase
- 15 分钟过 P2 + P3 —— 快速决议，能合并的合并
- 15 分钟过 ADR 列表 —— 把否决方案列清楚

**Owner 分配**：

- P0 由 design owner 在会后 48 小时内更新文档
- P1 章节可以拆给不同人写初稿
- P2/P3 在实现 PR 里顺手改

**版本控制**：

- 现在的文档标 `v2.0-draft`
- 修完 P0 + P1 后升 `v2.1-review`
- ADR 跟文档分开版本，长期维护

---

## 跟踪状态（Review 时填写）

| ID | 优先级 | 决议 | Owner | Due | 备注 |
|----|--------|------|-------|-----|------|
| 1  | P0 | | | | Gate 2 equivalence checker |
| 2  | P0 | | | | HIL off 4×2 矩阵 |
| 3  | P0 | | | | route_score 权重归一化 |
| 4  | P0 | | | | reviewer identity |
| 5  | P1 | | | | Calibration Set |
| 6  | P1 | | | | Periodic Audit |
| 7  | P1 | | | | Continuous Improvement Loop |
| 8  | P2 | | | | 05/07 equivalence 3 层验证 |
| 9  | P2 | | | | route_score max-aggregation |
| 10 | P2 | | | | difficulty deterministic heuristic |
| 11 | P3 | | | | pipeline_health |
| 12 | P3 | | | | runs/index 分桶 |
| 13 | P3 | | | | ADR |

决议字段建议值：`accept` / `defer-to-phaseN` / `reject` / `needs-discussion`