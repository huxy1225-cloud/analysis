# RAGEN V1 复现指南

**论文：** RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning  
**代码：** https://github.com/RAGEN-AI/RAGEN  
**笔记日期：** 2026-06-04

---

## 一、硬件与软件要求

| 项目 | 要求 |
|------|------|
| GPU | A100 / H100 / H200 / B200（验证过），显存 ≥ 40GB |
| CUDA | ≥ 12.8 |
| Python | 3.12 |
| 操作系统 | Linux |

> 本文基于 **8× A100-80GB** 环境复现。

---

## 二、环境安装

### 2.1 克隆仓库

```bash
git clone https://github.com/RAGEN-AI/RAGEN.git
cd RAGEN
```

> 注意：必须用 `git clone`，不能直接下载 zip，因为 veRL 是 git submodule，zip 下载后子模块目录为空。

### 2.2 创建 conda 环境

```bash
conda create -n ragen python=3.12 -y
conda activate ragen
```

### 2.3 运行安装脚本

```bash
bash scripts/setup_ragen.sh
```

脚本自动完成以下步骤：

1. 初始化 veRL 子模块（`git submodule update --init --recursive`）
2. 安装 RAGEN（`pip install -e . --no-deps`）
3. 安装 veRL 及 vLLM（`USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh`）
4. 安装环境依赖（gym、gym_sokoban、gymnasium 等）
5. 下载训练数据

### 2.4 可能遇到的问题

#### 问题 1：`fatal: not a git repository`
**原因：** 项目不是通过 `git clone` 获取，导致 submodule 无法初始化。  
**解决：** 手动下载并安装 veRL。

```bash
cd /tmp
git clone --depth 1 --branch v0.6.1 https://github.com/volcengine/verl.git
cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

#### 问题 2：`ModuleNotFoundError: No module named 'hydra'`
```bash
pip install hydra-core omegaconf
```

#### 问题 3：`ModuleNotFoundError: No module named 'huggingface_hub'`
```bash
pip install huggingface_hub
```

#### 问题 4：`ModuleNotFoundError: No module named 'verl.utils'`
**原因：** veRL 是空 namespace package，未正确安装。  
**解决：** 参考问题 1，手动安装 veRL。

---

## 三、下载预训练模型

论文默认使用 `Qwen2.5-3B-Instruct`：

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
  --local-dir /path/to/models/Qwen2.5-3B-Instruct
```

如需其他规模：

```bash
# 0.5B（快速调试）
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir /path/to/models/Qwen2.5-0.5B-Instruct

# 7B（完整实验）
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /path/to/models/Qwen2.5-7B-Instruct
```

然后修改 `config/base.yaml`：

```yaml
model_path: /path/to/models/Qwen2.5-3B-Instruct
```

---

## 四、快速验证（5 步冒烟测试）

安装完成后，先用最少步数验证环境是否能跑通：

```bash
conda activate ragen
CUDA_VISIBLE_DEVICES=0 python train.py --config-name _2_sokoban \
  trainer.total_training_steps=5 \
  trainer.logger="['console']"
```

看到 reward 打印输出即为成功，不需要等待训练收敛。

---

## 五、核心实验复现

### 5.1 实验概览

论文核心结论对应三组实验：

| 实验 | 目的 | 对应脚本 |
|------|------|---------|
| Filter vs. NoFilter | 验证 SNR 过滤有效性（主表） | `run_main_table_diff_algo.sh` |
| 跨算法对比 | PPO/GRPO/DAPO/DrGRPO 对比 | `run_main_table_diff_algo.sh` |
| 跨模型规模对比 | 0.5B ~ 7B 效果对比 | `run_main_table_diff_size.sh` |

### 5.2 主表实验：Filter vs. NoFilter

验证 SNR-Adaptive Filtering 的核心结论（绿色 = filter 比 nofilter 好）。

```bash
# 4 GPU × 2 实验（filter + nofilter 并行）
bash scripts/runs/run_main_table_diff_algo.sh \
  --steps 400 \
  --tasks sokoban,frozenlake,metamathqa,countdown \
  --algos PPO \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus-per-exp 4 \
  --gpu-memory-utilization 0.3 \
  --filters all
```

**关键配置含义：**

| 参数 | filter 模式 | nofilter 模式 |
|------|------------|--------------|
| `rollout_filter_value` | `0.9` | `1.0` |
| `rollout_filter_strategy` | `top_p` | `top_p` |
| `rollout_filter_top_p_prob_mode` | `softmax` | `softmax` |
| `rollout_filter_include_zero` | `True` | `True` |

### 5.3 跨算法对比

```bash
bash scripts/runs/run_main_table_diff_algo.sh \
  --steps 400 \
  --tasks sokoban \
  --algos PPO,GRPO,DAPO,DrGRPO \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus-per-exp 2 \
  --filters all
```

### 5.4 跨模型规模对比

```bash
bash scripts/runs/run_main_table_diff_size.sh \
  --steps 400 \
  --tasks sokoban,frozenlake \
  --models Qwen2.5-0.5B,Qwen2.5-1.5B,Qwen2.5-3B,Qwen2.5-7B \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus-per-exp 2 \
  --filters filter
```

### 5.5 单任务手动训练（精细控制）

```bash
# PPO + filter，Sokoban，单卡
CUDA_VISIBLE_DEVICES=0 python train.py --config-name _2_sokoban \
  actor_rollout_ref.rollout.rollout_filter_strategy=top_p \
  actor_rollout_ref.rollout.rollout_filter_value=0.9 \
  actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode=softmax \
  actor_rollout_ref.rollout.rollout_filter_include_zero=True \
  trainer.total_training_steps=400 \
  trainer.experiment_name=sokoban_ppo_filter

# PPO，无过滤，对照组
CUDA_VISIBLE_DEVICES=1 python train.py --config-name _2_sokoban \
  actor_rollout_ref.rollout.rollout_filter_value=1.0 \
  trainer.total_training_steps=400 \
  trainer.experiment_name=sokoban_ppo_nofilter
```

---

## 六、关键配置说明

### 6.1 `config/base.yaml` 核心参数

```yaml
# 模型
model_path: Qwen/Qwen2.5-3B-Instruct

# 训练规模
es_manager:
  train:
    env_groups: 8      # 每步训练的环境组数
    group_size: 16     # 每组 rollout 数量（共 8×16=128 条轨迹）

# SNR 过滤
actor_rollout_ref:
  rollout:
    rollout_filter_strategy: top_p
    rollout_filter_value: 0.9        # 0.9=filter，1.0=nofilter
    rollout_filter_top_p_prob_mode: linear   # 推荐；论文主表用 softmax

# 算法选择
algorithm:
  adv_estimator: gae   # gae=PPO，grpo=GRPO/DrGRPO

# 训练步数
trainer:
  total_training_steps: 400
```

### 6.2 不同算法的配置差异

| 算法 | `adv_estimator` | `loss_agg_mode` | `norm_adv_by_std_in_grpo` |
|------|----------------|-----------------|--------------------------|
| PPO | `gae` | `token-mean` | — |
| GRPO | `grpo` | `seq-mean-token-mean` | `True` |
| DrGRPO | `grpo` | `seq-mean-token-sum` | `True` |
| DAPO | `grpo` | `token-mean` | `False` |

---

## 七、训练监控

### 7.1 WandB 日志

训练默认上传 WandB，需先登录：

```bash
wandb login
```

关键指标：

| 指标 | 含义 | 期望趋势 |
|------|------|---------|
| `train/reward_mean` | 平均奖励 | 持续上升 |
| `train/reward_std` | 奖励标准差 | 先升后稳 |
| `collapse/mi_estimate` | 互信息（推理与输入相关性） | 保持正值（filter 更好） |
| `collapse_first_turn_sample/retrieval_accuracy` | 推理溯源准确率 | 高于随机水平 |
| `actor/entropy` | 策略熵 | 不应过快下降 |

### 7.2 只用本地日志

```bash
python train.py --config-name _2_sokoban \
  trainer.logger="['console']"
```

---

## 八、评估

```bash
# 评估训练好的 checkpoint
python -m ragen.llm_agent.agent_proxy --config-name _2_sokoban \
  actor_rollout_ref.model.path=results/sokoban-main/checkpoints/step_400

# 评估时关闭采样（确定性输出）
python -m ragen.llm_agent.agent_proxy --config-name _2_sokoban \
  actor_rollout_ref.model.path=results/sokoban-main/checkpoints/step_400 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False
```

---

## 九、复现结果对照

复现时对照论文以下数值（允许 ±10% 误差）：

### Sokoban（Qwen2.5-3B，400 steps）

| 设置 | 论文 Reward | 复现目标 |
|------|------------|---------|
| PPO + filter | 论文 Fig.3 绿线终点 | reward 稳定上升，高于 nofilter |
| PPO + nofilter | 论文 Fig.3 蓝线终点 | reward 上升但低于 filter |

**核心验证点：** filter 曲线在训练后期显著高于 nofilter，且 MI 指标在 nofilter 下出现下降（template collapse 信号）。

---

## 十、常见问题排查

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| OOM（显存不足） | batch size 太大 | 减小 `env_groups` 或 `group_size` |
| reward 始终为 0 | 动作格式解析失败 | 检查 `<think>` / `<ans>` 标签是否正确输出 |
| 训练不收敛 | KL 惩罚过强 | 确认 `kl_coef=0` |
| MI 指标一直为负 | 正常，接近 0 即表示低 MI | 对比 filter vs nofilter 的相对趋势 |
| vLLM 报错 | GPU 内存碎片 | 重启进程，或降低 `gpu_memory_utilization` |
