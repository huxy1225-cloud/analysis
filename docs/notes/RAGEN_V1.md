# 读书笔记：RAGEN —— 通过强化推理训练 LLM 智能体

**论文标题：** RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning  
**arXiv：** https://arxiv.org/abs/2504.20073  
**代码仓库：** https://github.com/RAGEN-AI/RAGEN  
**作者：** Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li 等（Northwestern University / Stanford / Microsoft Research 等）  
**笔记日期：** 2026-06-04

---

## 一、研究背景与动机

### 问题出发点

RL + 规则奖励（如 GRPO、PPO）在静态单轮任务（数学推理、代码生成）上已经验证有效，但直接迁移到**多轮 Agent 场景**存在两个本质困难：

1. **多轮交互（Multi-turn Interaction）：** Agent 需要根据环境反馈连续决策，动作之间有时序依赖，无法按单轮处理。
2. **随机性环境（Stochastic Environment）：** 相同动作可能导致不同结果（如 FrozenLake 的随机滑动），奖励信号噪声大。

已有方法（如 DeepSeek-R1、TinyZero）针对单轮静态任务设计，缺乏对多轮轨迹和随机环境的统一支持。

### 核心研究问题

> **如何将 RL 训练机制从单轮静态任务推广到多轮、随机、交互式 Agent 场景，并让模型同时学会"推理"和"行动"？**

---

## 二、核心贡献

### 贡献 1：StarPO 算法框架

**StarPO**（State-Thinking-Action-Reward Policy Optimization）是论文的核心算法，将多轮 Agent 交互统一为轨迹级优化。

### 贡献 2：RAGEN 系统

基于 StarPO 构建的完整训练与评估系统，包含 10 个内置环境、模块化架构，支持快速扩展自定义环境。

### 贡献 3：训练动态分析

系统研究了奖励归一化策略、上下文窗口管理、KL 惩罚等超参数对 Agent RL 训练的影响规律。

---

## 三、算法：StarPO

### 3.1 MDP 建模

将 Agent-环境交互形式化为**马尔可夫决策过程（MDP）**：

- **状态** $s_t$：当前环境观测（token 序列）
- **动作** $a_t$：模型输出的 token 序列
- **转移函数** $T(s_{t+1} | s_t, a_t)$：环境状态转移（可随机）
- **奖励** $r_t$：环境反馈的标量信号
- **目标**：最大化期望累积奖励 $\mathbb{E}\left[\sum_t r_t\right]$

与传统 RL 的关键区别：状态和动作都是**token 序列**，而非低维向量，LLM 充当策略网络 $\pi_\theta$。

### 3.2 推理引导的动作格式

模型在每一步输出固定格式：

```
<think> 推理过程 </think> <ans> 动作 </ans>
```

- `<think>` 部分：链式推理，引导模型显式分析当前状态
- `<ans>` 部分：最终执行动作，传入环境

这种设计将"思考"和"行动"解耦，使 RL 同时优化推理质量和动作质量。

### 3.3 StarPO 的两个阶段

**阶段一：Rollout（轨迹生成）**

给定初始状态 $s_0$，LLM 生成多条完整轨迹：

$$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T, a_T, r_T)$$

每条轨迹包含 $T$ 轮交互，每轮模型接收历史轨迹 + 当前状态，生成推理 + 动作。

**阶段二：Update（轨迹优化）**

用重要性采样对**完整轨迹**做策略梯度优化，而非逐步优化：

$$\mathcal{L}(\theta) = -\mathbb{E}_\tau \left[ \sum_t A_t \cdot \log \pi_\theta(a_t | s_{0:t}) \right]$$

其中优势 $A_t$ 的估计方式支持两种后端：

| 后端 | 优势估计 | 特点 |
|------|---------|------|
| **PPO** | GAE（广义优势估计），使用 Critic 网络 | Token 级精细估计，训练更稳定 |
| **GRPO** | 组内归一化奖励，无需 Critic | 计算更简单，但方差较大 |

两阶段交替进行，实现在线学习。

### 3.4 StarPO 与 GRPO/PPO 的关系

StarPO 是**框架**，GRPO 和 PPO 是其支持的两种**优化算法**。StarPO 的创新在于将它们统一扩展到多轮轨迹级场景，并加入推理引导的动作格式。

---

## 四、RAGEN 系统架构

RAGEN 由三个核心模块组成：

```
┌─────────────────────────────────────────────┐
│                  Agent Proxy                 │  ← 训练/评估入口
│         (agent_proxy.py)                     │
└──────────────┬──────────────────────────────┘
               │
    ┌──────────▼──────────┐    ┌────────────────────┐
    │   Context Manager   │◄──►│  Env State Manager  │
    │   (ctx_manager.py)  │    │  (es_manager.py)    │
    └─────────────────────┘    └────────────────────┘
```

| 模块 | 职责 |
|------|------|
| **Environment State Manager** | 管理多个并行环境实例；执行 `step()`/`reset()`；批量处理动作并返回观测 |
| **Context Manager** | 解析模型输出的动作；格式化环境观测为 prompt；管理历史上下文窗口（`max_context_window`）；整合轨迹为训练 token |
| **Agent Proxy** | 统一的 rollout 执行接口；协调 LLM 推理（vLLM）与环境交互 |

### 关键设计：上下文窗口管理

通过 `max_context_window` 控制模型能看到的历史轮数：
- `-1`（默认）：保留完整历史
- `1`：无历史（退化为单轮）
- `k`：保留最近 k 轮

这个参数对多轮任务的性能有显著影响（历史太长导致上下文过长；太短丢失关键状态）。

---

## 五、实验设置

### 5.1 实验环境

| 环境 | 类型 | 特点 |
|------|------|------|
| **Sokoban** | 空间推理 | 多步规划，确定性 |
| **FrozenLake** | 导航 | 随机滑动，需鲁棒策略 |
| **Bandit** | 探索 | 纯探索-利用权衡 |
| **Spatial** | 空间关系 | 语言描述的空间推理 |

### 5.2 基础模型

主要使用 **Qwen2.5-0.5B-Instruct**（轻量验证）和 **Qwen2.5-3B-Instruct**（主要实验）。

### 5.3 关键超参数

- 无 KL 惩罚（`kl_coef=0`）
- 仅保留任务成功轨迹的 top 25%（早期实验的过滤策略）
- PPO 比 GRPO 训练更稳定（后续实验结论）

---

## 六、主要实验结论

### 结论 1：StarPO 能有效训练多轮 Agent

在 Sokoban、FrozenLake、Bandit 任务上，从 Qwen2.5-0.5B 出发，StarPO 训练后 reward 持续上升，验证了框架有效性。

### 结论 2：推理引导（`<think>` 格式）有助于 Agent 学习

显式推理步骤使模型能更好地分析环境状态，尤其在需要多步规划的 Sokoban 任务上效果明显。

### 结论 3：泛化能力

在简单 Sokoban（6×6，1 box）上训练后，模型能泛化到：
- 更大地图（8×8，2 boxes）
- 不同的网格符号表示
- FrozenLake 等其他环境

说明模型学到了可迁移的推理策略，而非记忆特定状态。

### 结论 4：PPO 比 GRPO 更稳定

多个来源（Open-Reasoner-Zero、TinyZero）和自身实验均发现，PPO（GAE 优势估计）在 Agent 训练中比 GRPO 更稳定，因此后续版本默认切换为 PPO。

---

## 七、局限性与遗留问题

1. **奖励设计依赖领域知识：** 不同环境需要手动设计奖励函数，无法自动化。
2. **Template Collapse（模板崩溃）问题未被检测：** V1 用熵监测训练质量，但后续发现熵无法检测模型对不同输入输出相同推理模板的问题（这成为 V2 的核心研究对象）。
3. **WebShop 等复杂环境的扩展性：** 真实世界复杂环境（网页交互等）的集成较繁琐。
4. **多节点分布式训练：** V1 主要在单节点多 GPU 上验证，分布式扩展留待后续。

---

## 八、与后续工作（RAGEN V2）的关系

| 维度 | V1 | V2 |
|------|----|----|
| 核心算法 | StarPO 框架建立 | SNR-Adaptive Filtering |
| 诊断工具 | 熵监测 | 互信息（MI）+ 四类 reasoning regime |
| 发现的新问题 | — | Template Collapse（熵盲区）|
| 训练干预 | 轨迹过滤（top 25%） | 奖励方差过滤（Top-p）|

V2 的核心动机直接来自 V1 的遗留问题：发现熵指标不足以诊断训练质量，进而提出 MI 代理指标。

---

## 九、核心思想总结

```
单轮 RL（GRPO/PPO）
    ↓ 挑战：多轮 + 随机环境
StarPO = MDP 建模 + 推理引导动作格式 + 轨迹级策略梯度
    ↓ 实现
RAGEN = StarPO + 模块化系统（ESM + CTX + AP）+ 多环境支持
    ↓ 发现
PPO 比 GRPO 更稳定；推理引导有效；训练质量监测不足（→ V2）
```

---

## 十、引用

```bibtex
@misc{ragen,
  title={RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning},
  author={Zihan Wang and Kangrui Wang and Qineng Wang and Pingyue Zhang and Linjie Li and
          Zhengyuan Yang and Xing Jin and Kefan Yu and Minh Nhat Nguyen and Licheng Liu and
          Eli Gottlieb and Yiping Lu and Kyunghyun Cho and Jiajun Wu and Li Fei-Fei and
          Lijuan Wang and Yejin Choi and Manling Li},
  year={2025},
  eprint={2504.20073},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2504.20073},
}
```
