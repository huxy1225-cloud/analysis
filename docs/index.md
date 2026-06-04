# LLM Agent RL 论文笔记

本站收录 LLM Agent 强化学习方向的论文阅读笔记与复现指南，持续更新。

---

<div class="home-grid">
  <div class="home-card">
    <h3>📄 RAGEN V1</h3>
    <p>StarPO 框架：将多轮 Agent 交互统一建模为轨迹级 RL 优化，同时训练推理和行动。</p>
    <p>
      <a href="notes/RAGEN_V1/">论文笔记</a> ·
      <a href="notes/RAGEN_V1_reproduction/">复现指南</a>
    </p>
  </div>
</div>

---

## 关于本站

!!! note "说明"
    笔记基于论文原文、官方代码和实际复现经验整理。内容包括核心算法解读、系统架构分析、实验复现步骤和踩坑记录。

## 涉及方向

- **Agent RL 训练框架**：StarPO、GRPO、PPO 在多轮 Agent 场景的应用
- **训练诊断**：Reasoning Collapse 检测、互信息代理指标
- **SNR 自适应过滤**：基于奖励方差的 rollout 筛选
