# 图像生成模型评测工作台 — 主操作页布局优化建议

> **目标界面**：Image Generation Evals 主操作页（Streamlit 工作台）
> **用户画像**：工程师 / 研究员，频繁发起对比、查看结果
> **设计定位**：内部技术工具 / 模型评测工作流
> **核心使命**：快速发起对比 + 看结果

---

## 一、当前布局问题诊断

### 1.1 顶部巨大空白卡片（最严重）

页面最顶部的白色大卡片，里面只放了 "IMAGE GENERATION Model Evals PIPELINE" 一行标题加副标题，**占用了约 280px 的纵向空间**（接近一屏的 25%）。

这是**非常昂贵的视觉地产**用在了最低价值的内容上：

- 标题文字本来一行就够
- 上下大量留白，没有任何信息
- 用户每次进入页面都要滚过这一大块才能开始干活

评测工作台的核心是"快速发起对比 + 看结果"，标题不该抢这么多注意力。

### 1.2 输入区缺少视觉分组

当前从上到下的结构：

```
[标题大卡]
[Pre-baked example 下拉]
[Prompt 输入框 + Generate 按钮]
[Pipeline 进度条 + Activity Log]
[图片对比]
```

输入控件之间没有任何分组容器，**Pre-baked dropdown** 和 **prompt input** 看起来是两个独立元素，但它们逻辑上是同一组——"决定要生成什么"。

而且 **Generate & Evaluate 按钮**只和 prompt 输入框对齐，跟 dropdown 不是一组，视觉关系混乱。

### 1.3 Pipeline 进度条 vs Activity Log 的关系

这两个组件**展示的是同一件事**（当前任务进度），但用了完全不同的可视化形式并排放置：

- 左边：8 步圆点进度条
- 右边：模型耗时列表

它们之间没有视觉连接，用户需要左右扫读才能拼出"现在在做什么、用了多久"。

而且 Activity Log 卡片宽度和 Pipeline 不成比例——Pipeline 占 75% 宽度，Activity Log 占 25%，但**信息密度恰好相反**：Pipeline 信息少（8 个圆点），Activity Log 信息多（模型 + 耗时 + 总耗时）。

### 1.4 "Generating images..." 提示条位置尴尬

那条紫色的 "Generating images with GPT Image-2 and Gemini 3 Pro..." 横条夹在 Pipeline 和图片之间，**和 Pipeline 进度条传达的是重复信息**。如果 Pipeline 已经显示当前在 "GENERATE" 步骤，这条横幅就是冗余。

### 1.5 垂直信息流过长

整个流程是纯线性堆叠，必须从上到下滚动。但实际上用户的工作流是：

- **频繁操作**：选 prompt、点 Generate
- **频繁查看**：图片对比结果

这两个高频区域之间隔着 Pipeline + Log + 提示条三个中间层，**操作和结果距离太远**。

### 1.6 侧边栏的 Theme 选择器位置

把 Theme（Light/Dark）放在 Language 下面、Home/Dashboard 之前，**层级错乱**：

- Language 是用户偏好（合理放顶部）
- Theme 也是偏好（OK）
- 但 Home/Dashboard/Architecture V1/V2 是导航
- Comparison History 是历史记录

应该是 **导航 > 偏好 > 历史**，而不是把偏好夹在中间。

---

## 二、布局重构方案

### 2.1 建议的新版面结构

```
┌──────────────┬─────────────────────────────────────────────┐
│              │  ▸ 页面标题区（紧凑，60px）                    │
│  侧边栏       │    Image Gen Evals · Cross-Model Review     │
│              ├─────────────────────────────────────────────┤
│  导航         │                                              │
│  • Home      │  ▸ 控制区（Input Card）                       │
│  • Dashboard │    ┌────────────────────────────────────┐   │
│  • Arch V1   │    │ Example: [____下拉____]            │   │
│  • Arch V2   │    │ Prompt:  [____________________]    │   │
│              │    │                                    │   │
│              │    │              [Generate & Evaluate] │   │
│  ─────       │    └────────────────────────────────────┘   │
│              │                                              │
│  历史         │  ▸ 状态区（运行时才显示）                      │
│  ▸ k-pop...  │    ●━━●━━○━━○━━○━━○━━○━━○                  │
│  ▸ cry bear  │    Generate · 48s   Total: 48s              │
│              │                                              │
│              ├─────────────────────────────────────────────┤
│              │                                              │
│  ─────       │  ▸ 结果区（核心，占满）                        │
│              │    ┌──────────────┬──────────────┐         │
│  偏好         │    │  GPT Image-2 │  Gemini 3 Pro │         │
│  Lang [EN▾]  │    │              │              │         │
│  Theme [☀]   │    │   [图片1]    │   [图片2]    │         │
│              │    │              │              │         │
│              │    └──────────────┴──────────────┘         │
└──────────────┴─────────────────────────────────────────────┘
```

### 2.2 标题区瘦身

把标题大卡片删掉，改成简洁的页面头：

```python
st.markdown("""
<div style="padding: 12px 0 24px; border-bottom: 1px solid #E4E4E7;">
  <h1 style="font-size: 20px; font-weight: 600; margin: 0; color: #18181B;">
    Image Generation Evals
  </h1>
  <p style="font-size: 13px; color: #71717A; margin: 4px 0 0;">
    Cross-Model Evaluation with Adversarial Review
  </p>
</div>
""", unsafe_allow_html=True)
```

**收益**：从 280px 压到 60px，多出来 220px 给真正重要的内容。

### 2.3 把输入控件包成"控制卡"

```python
with st.container(border=True):
    st.markdown("##### New Comparison")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        example = st.selectbox("Load example", ...)
        prompt = st.text_input("Or enter prompt", ...)
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Generate & Evaluate", type="primary", use_container_width=True)
```

按钮垂直居中、占满右侧列宽，和两个输入框形成清晰的"输入 → 触发"关系。

### 2.4 Pipeline + Activity Log 合并

它们是**同一件事的两种视图**，应该融合。两种处理方式：

#### 方案 A：横向单行（推荐，省空间）

```
●━━━●━━━○━━━○━━━○━━━○━━━○━━━○    GPT 48s · Gemini 29.8s · Total 48s
1   2   3   4   5   6   7   8
```

进度条 + 一行紧凑的耗时摘要，整个状态区只占 60px。

#### 方案 B：折叠（运行完就收起）

```python
with st.expander("Pipeline · ✓ Completed in 48s", expanded=is_running):
    # 详细进度条 + activity log
```

任务跑完后自动折叠，结果区立刻可见。

### 2.5 移除冗余的 "Generating images..." 横条

如果 Pipeline 进度条做到位（高亮当前步骤 + 显示步骤名），这条横幅就是重复信息，**直接删掉**。

如果一定要保留状态提示，把它做成 Pipeline 卡片底部的一行小字，不要单独占一条横幅。

### 2.6 侧边栏重新分组

```
┌──────────────┐
│ Comparison Lab │  ← Logo / 产品名
├──────────────┤
│ NAVIGATION   │  ← 灰色小字 section header
│ ⌂ Home       │
│ ▦ Dashboard  │
│ ⊞ Arch V1    │
│ ⊞ Arch V2    │
├──────────────┤
│ HISTORY      │
│ ▸ k-pop...   │
│ ▸ cry bear   │
│ ...          │
├──────────────┤
│              │  ← 弹性空间
├──────────────┤
│ ⚙ Settings   │  ← 折叠区
│   Lang  EN ▾ │
│   Theme ☀ ▾ │
└──────────────┘
```

把 Language 和 Theme 移到侧边栏底部 Settings 区，**导航和历史占据视觉重心**，偏好作为低频项放底部。这是 VS Code、Linear、Notion 等工具的通行模式。

### 2.7 Pipeline 进度条的视觉细节

当前进度条问题：

- 已完成步骤和未完成步骤的圆圈差异不够明显
- 紫色实心圆 + 浅紫描边圆并列，对比度偏低
- 步骤名 "GENERATE / EVALUATE / GATE 1 ..." 全大写挤在一起，扫读吃力

建议样式：

```css
.step-done    { background: #4F46E5; color: white; }
.step-done::after  { content: "✓"; }

.step-active  {
  background: #4F46E5;
  color: white;
  box-shadow: 0 0 0 4px #E0E7FF;
}

.step-pending {
  background: #FFFFFF;
  border: 1px solid #D4D4D8;
  color: #A1A1AA;
}

.step-line-done    { background: #4F46E5; }
.step-line-pending { background: #E4E4E7; }
```

步骤名建议从全大写改成 **Title Case**：`Generate / Evaluate / Gate 1 / Critique` —— 全大写在密集排列时严重影响可读性。

---

## 三、信息密度与折叠策略

Streamlit 应用容易"什么都放在一屏"。建议引入**折叠态**思维：

| 区域 | 任务运行中 | 任务完成后 |
|---|---|---|
| 标题 | 显示 | 显示（紧凑） |
| 输入区 | 显示 | 折叠成一行摘要 |
| Pipeline 状态 | 展开（用户在等） | 折叠成 "✓ Completed in 48s" |
| 结果图片对比 | Loading skeleton | 完整显示 |
| 评测详情（雷达图等） | 灰色占位 | 完整显示 |

### 折叠示例代码

```python
if is_running:
    with st.container(border=True):
        render_pipeline_full()
else:
    with st.expander(f"✓ Pipeline completed in {total_time}s", expanded=False):
        render_pipeline_full()
```

任务完成后，输入区也折叠成一行：

```
┌─────────────────────────────────────────┐
│ ✏ "专业美食摄影..."  [Edit]  [New]      │
└─────────────────────────────────────────┘
```

这样结果区可以最大化展示，符合用户跑完一次评测后"专心看结果"的需求。

---

## 四、改造前后效果对比

### 改造前

| 区域 | 占用高度 | 价值 |
|---|---|---|
| 标题大卡 | ~280px | 低（只有一行字） |
| Pre-baked 下拉 | ~50px | 中 |
| Prompt 输入 + 按钮 | ~60px | 高 |
| Pipeline 进度条 | ~120px | 中 |
| Activity Log（独立卡） | ~120px | 中（与 Pipeline 重复） |
| Generating 提示横条 | ~50px | 低（信息冗余） |
| **总计** | **~680px** | 结果区被推到首屏外 |

### 改造后

| 区域 | 占用高度 | 价值 |
|---|---|---|
| 紧凑页头 | ~60px | 低，但占用极小 |
| 输入卡（合并） | ~120px | 高 |
| Pipeline + Log（合并） | ~80px | 中 |
| **总计** | **~260px** | 结果区在首屏内可见 |

**节省空间约 420px**，相当于多出半屏给核心内容。

---

## 五、实施优先级

| 优先级 | 改动 | 收益 |
|---|---|---|
| **P0** | 删除巨型标题卡，改 60px 紧凑 header | 多出 220px 给核心内容 |
| **P0** | 删除冗余的 "Generating..." 紫色横幅 | 去重 |
| **P0** | 步骤名从全大写改 Title Case | 可读性立刻提升 |
| **P1** | 输入区包成 container，按钮和输入对齐 | 视觉分组清晰 |
| **P1** | Pipeline 进度条加当前步光晕 + 用对勾代替已完成数字 | 状态一目了然 |
| **P1** | Pipeline 完成后自动折叠 | 结果区最大化 |
| **P2** | 侧边栏 Language/Theme 移到底部 Settings | 信息层级正确 |
| **P2** | Pipeline + Activity Log 融合成单一组件 | 状态区紧凑 |

---

## 六、设计原则回顾

整个改造遵循以下原则，可作为后续设计决策的参考：

1. **视觉地产分配按价值** —— 高频高价值内容（输入、结果）占主区，低频内容（标题、偏好）压缩或下沉
2. **同源信息合并展示** —— Pipeline 和 Activity Log 是同一件事，不该并排两块
3. **去重原则** —— 同样的状态信息只在一处展示（删除冗余横幅）
4. **完成态折叠** —— 跑完的任务不该继续占据视觉重心，给当前关注点让位
5. **视觉分组传递语义** —— 用 container 把相关控件框成一组，而不是简单堆叠
6. **大写仅用于强调** —— 全大写不适合密集步骤名，改 Title Case
7. **侧边栏遵循"导航 > 内容 > 偏好"层级** —— 偏好类设置永远沉底

---

*文档生成日期：2026-05-03*
