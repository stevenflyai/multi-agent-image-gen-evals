# 图像生成模型评测工作台 — UI 配色优化建议

> **目标界面**：Cross-Model Evaluation with Adversarial Review（Streamlit 工作台）
> **用户画像**：工程师 / 研究员，长时间盯屏使用
> **设计定位**：内部技术工具 / 评测仪表盘

---

## 一、当前配色诊断

### 现状盘点

| 角色 | 当前颜色 | 问题 |
|---|---|---|
| 主背景 | 近纯白 `#FFFFFF` | 偏冷、偏亮，长时间用眼疲劳 |
| 卡片背景 | 浅灰带蓝调 `~#F0F2F6` | 与主背景对比太弱，卡片"浮"不起来 |
| 主操作按钮 | 深蓝紫 `#3B3B98` 左右 | 偏紫调，在数据界面里显得过于"product marketing" |
| 模型 A (GPT) | 蓝紫 `~#5B5BD6` | ✅ 合理 |
| 模型 B (Gemini) | 鲜绿 `~#21BA45` | ⚠️ 饱和度过高，与红绿色盲不友好 |
| 数据数字 | 蓝/绿强对比 | 视觉噪音，每一行都在"喊" |
| 状态文字（completed / not_required） | 青绿色 | 与 Gemini 的绿混淆，语义和品牌色撞车 |

### 三个核心问题

1. **语义色和品牌色混用** —— Gemini 用绿色、"completed" 状态也是绿色，"skipped" 是青色。用户无法瞬间分辨"这是模型标识"还是"这是状态信号"。
2. **层级几乎不存在** —— 主背景、卡片背景、嵌套卡片背景灰度差仅 2-3%，眼睛要费劲找边界。这在 gate status + 6 个评分卡 + 表格的页面尤其明显。
3. **数据色饱和度过高** —— 评测工具的数据展示应该让"差异"跳出来，而不是让每个数字都鲜艳。当前蓝紫 + 鲜绿同时高饱和,焦点反而消失。

---

## 二、推荐配色方案

整体策略：**冷静的中性灰底 + 克制的双品牌色 + 清晰的语义色** 三层体系。这是 Linear、Vercel、Stripe Dashboard、Hugging Face 这类技术工具的通行做法。

### 2.1 中性底层（界面骨架）

```css
--bg-app:        #FAFAFA   /* 主背景，比纯白柔和 */
--bg-surface:    #FFFFFF   /* 卡片背景，反而比主背景更亮 */
--bg-subtle:     #F4F4F5   /* 嵌套区域、表头 */
--border:        #E4E4E7   /* 分隔线、卡片描边 */
--border-strong: #D4D4D8   /* 输入框、强调边界 */

--text-primary:   #18181B  /* 主文字 */
--text-secondary: #52525B  /* 标签、说明 */
--text-tertiary:  #A1A1AA  /* 辅助、占位 */
```

> **关键改动**：把主背景做成 `#FAFAFA`、卡片做成 `#FFFFFF`。**反向层级** —— 卡片"亮"于背景，比当前"卡片暗于背景"更符合现代仪表盘审美，层级也立刻清晰。

### 2.2 品牌主色（操作 / 强调）

```css
--brand:       #4F46E5   /* Indigo-600，主按钮、链接 */
--brand-hover: #4338CA
--brand-soft:  #EEF2FF   /* "Comparison History" 那种胶囊背景 */
```

比当前的紫蓝更"工程化"，去掉了过度的紫调。

### 2.3 模型对比双色（关键改动）

把 Gemini 从鲜绿换成 **琥珀 / 橙**，避开和"成功状态"语义撞色：

```css
--model-a:      #4F46E5   /* GPT  – Indigo */
--model-a-soft: #E0E7FF
--model-b:      #D97706   /* Gemini – Amber-600 */
--model-b-soft: #FEF3C7
```

**为什么不是绿/红、绿/蓝？**

- 绿在仪表盘里语义"已通过"，会和 status 撞
- 蓝/紫太接近，对比度不够
- **Indigo + Amber 是经过 Stripe、Linear 验证的"中性双色对比"**，色盲友好（不是红绿对立），冷暖对比强，雷达图叠加也好看

雷达图填充用 `rgba(79,70,229,0.15)` 和 `rgba(217,119,6,0.15)`，描边用实色。

### 2.4 语义色（状态、分数变化）

```css
--success: #16A34A   /* completed, ✓ */
--warning: #CA8A04   /* requires attention */
--danger:  #DC2626   /* failed, contradicting */
--info:    #0891B2   /* skipped, not_required */
```

这样 "completed" 的绿和模型颜色 **完全分离**，用户一眼就懂"绿 = 状态好"。

### 2.5 评分数字的处理

当前 `9 / 3 -2` 这种地方蓝绿都是高饱和度，建议：

- 数字本身用 `--text-primary`（深灰），**不上色**
- 分数变化标记 `-2` `+1` 用语义色小字体
- 只有标签 "GPT Image-2" / "Gemini 3 Pro" 用品牌色
- 胜出方加一个细微的 `background: var(--model-a-soft)` 行底色

这样信息密度还在，但视觉噪音骤降。

---

## 三、Streamlit 落地配置

### 3.1 主题配置文件

新建或修改 `.streamlit/config.toml`：

```toml
[theme]
base = "light"
primaryColor = "#4F46E5"
backgroundColor = "#FAFAFA"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#18181B"
font = "sans serif"

# 边框（Streamlit 1.30+）
borderColor = "#E4E4E7"
```

### 3.2 自定义 CSS 注入

Streamlit 原生主题覆盖不了所有细节，**强烈建议**额外注入一段 CSS 处理卡片边距、雷达图、评分卡：

```python
st.markdown("""
<style>
  /* 卡片更柔和的边框 */
  [data-testid="stMetric"],
  [data-testid="stExpander"] {
    border: 1px solid #E4E4E7;
    border-radius: 8px;
    background: #FFFFFF;
  }
  /* 侧边栏微调 */
  [data-testid="stSidebar"] {
    background: #F4F4F5;
    border-right: 1px solid #E4E4E7;
  }
  /* 主按钮 */
  .stButton > button[kind="primary"] {
    background: #4F46E5;
    border: none;
    font-weight: 500;
    letter-spacing: 0.01em;
  }
  .stButton > button[kind="primary"]:hover {
    background: #4338CA;
  }
</style>
""", unsafe_allow_html=True)
```

### 3.3 Plotly 雷达图配色

```python
fig.update_traces(
    selector=dict(name="GPT Image-2 (post-critique)"),
    line_color="#4F46E5",
    fillcolor="rgba(79,70,229,0.15)"
)
fig.update_traces(
    selector=dict(name="Gemini 3 Pro (post-critique)"),
    line_color="#D97706",
    fillcolor="rgba(217,119,6,0.15)"
)
# pre-critique 用同色但虚线 + 更低透明度
```

---

## 四、可选：暗色模式

考虑到这是工程师长时间盯的工具，强烈建议加暗色模式。骨架色直接对应：

```css
--bg-app:        #0A0A0A
--bg-surface:    #18181B
--bg-subtle:    #27272A
--border:        #27272A
--text-primary:  #FAFAFA
```

品牌色亮度上调一档（`#6366F1` / `#F59E0B`），其他不变。

---

## 五、实施优先级

按这个顺序改，**每一步都能看到明显提升**：

| 优先级 | 工作量 | 内容 | 预期效果 |
|---|---|---|---|
| **P0** | 10 分钟 | 换 Gemini 颜色为 Amber `#D97706` | 立刻消除和 status 绿色的语义冲突 |
| **P1** | 30 分钟 | 注入 CSS，背景反向（`#FAFAFA` 主 + `#FFFFFF` 卡） | 卡片层级清晰，眼睛轻松定位边界 |
| **P2** | 1 小时 | 评分卡数字脱色，只保留标签上色 | 视觉噪音骤降，差异点突出 |
| **P3** | 半天 | 加暗色模式 | 长时间使用更舒适 |

---

## 六、完整色板速查表

### 浅色模式

| 用途 | 变量 | 色值 | 预览 |
|---|---|---|---|
| 主背景 | `--bg-app` | `#FAFAFA` | ⬜️ |
| 卡片背景 | `--bg-surface` | `#FFFFFF` | ⬜️ |
| 嵌套背景 | `--bg-subtle` | `#F4F4F5` | ⬜️ |
| 边框 | `--border` | `#E4E4E7` | ⬜️ |
| 强边框 | `--border-strong` | `#D4D4D8` | ⬜️ |
| 主文字 | `--text-primary` | `#18181B` | ⬛️ |
| 次文字 | `--text-secondary` | `#52525B` | ⬛️ |
| 辅文字 | `--text-tertiary` | `#A1A1AA` | ◼️ |
| 品牌主色 | `--brand` | `#4F46E5` | 🟦 |
| 品牌悬停 | `--brand-hover` | `#4338CA` | 🟦 |
| 品牌浅底 | `--brand-soft` | `#EEF2FF` | ⬜️ |
| GPT 标识 | `--model-a` | `#4F46E5` | 🟦 |
| GPT 软底 | `--model-a-soft` | `#E0E7FF` | ⬜️ |
| Gemini 标识 | `--model-b` | `#D97706` | 🟧 |
| Gemini 软底 | `--model-b-soft` | `#FEF3C7` | ⬜️ |
| 成功 | `--success` | `#16A34A` | 🟩 |
| 警告 | `--warning` | `#CA8A04` | 🟨 |
| 危险 | `--danger` | `#DC2626` | 🟥 |
| 信息 | `--info` | `#0891B2` | 🟦 |

---

*文档生成日期：2026-05-03*
