---
title: "VL Reward Model 实验报告"
subtitle: "InternVL2.5-2B Reward Model on VLRewardBench"
author: "csh / Codex assisted"
date: "2026-05-16"
lang: zh-CN
mainfont: "Helvetica Neue"
CJKmainfont: "PingFang SC"
geometry: margin=1in
fontsize: 11pt
---

# 1. 任务目标

本次考核任务是基于视觉语言模型训练一个 reward model，并在 VLRewardBench 上评估。实现要求包括：

- 选择 Qwen2-VL 或 InternVL2.5-VL 作为 base model。
- 在 base model 上增加 scalar score head，训练成偏好 reward model。
- 不能使用 VLRewardBench 的训练数据。
- 记录 base model 分数 `Score_base`，训练后 reward model 分数 `Score_reward`，并保存权重。

本实验选择 `InternVL2.5-2B` 作为 backbone。最终提交的可保存 reward model 是 LoRA + scalar score head 模型，`Score_reward = 59.98%`，超过题目中 2B reward model 参考线 `58%`。

# 2. 代码与环境

服务器工作目录：

`/data4/ljl/csh/vl_reward_work`

主要脚本：

`/data4/ljl/csh/vl_reward_work/scripts/internvl_reward.py`

模型：

`/data2/ljl/csh/pretrained_model/InternVL2_5-2B`

运行环境：

- Python: `/data3/ljl/miniconda3/envs/internvl/bin/python`
- GPU: RTX 4090 24GB
- 本轮训练和评测均限制使用 4 张卡：`CUDA_VISIBLE_DEVICES=0,1,2,3`

本次补充修改：

- 新增 `pseudo-from-eval` 子命令，把 base judge 在训练池上的判别结果转换为偏好训练 JSONL。
- 新增 `train-lora-reward --init-checkpoint` 参数，允许从已有 LoRA reward checkpoint 继续初始化训练。

# 3. 数据

已有数据目录：

- RLHF-V: `data/processed/train_pairs.jsonl`, `data/processed/val_pairs.jsonl`
- RLAIF-H4 子集: `data/processed_h4_12k/train_pairs_h4.jsonl`, `data/processed_h4_12k/val_pairs_h4.jsonl`
- 混合训练集: `data/processed_h4_12k/train_pairs_mix.jsonl`
- 混合验证集: `data/processed_h4_12k/val_pairs_mix.jsonl`
- VLRewardBench eval: `data/processed/vlrewardbench_eval.jsonl`

样本规模：

| split | 文件 | 数量 |
|---|---:|---:|
| RLHF-V train | `train_pairs.jsonl` | 2,533 |
| RLHF-V val | `val_pairs.jsonl` | 281 |
| H4 train | `train_pairs_h4.jsonl` | 5,206 |
| H4 val | `val_pairs_h4.jsonl` | 273 |
| mix train | `train_pairs_mix.jsonl` | 7,739 |
| mix val | `val_pairs_mix.jsonl` | 554 |
| VLRewardBench eval | `vlrewardbench_eval.jsonl` | 1,247 |

训练没有使用 VLRewardBench 训练数据。VLRewardBench 只作为最终测试集。

# 4. 模型方法

Backbone 为 InternVL2.5-2B。Reward model 结构为：

- 冻结原始视觉语言模型大部分参数。
- 对语言模型高层注入 LoRA adapter。
- 在最终 hidden state 上增加 scalar score head。
- 对一对回答分别打分 `s_chosen` 和 `s_rejected`。
- 使用 pairwise logistic loss: `-log sigmoid(s_chosen - s_rejected)`。

最终最优模型配置：

| 参数 | 值 |
|---|---:|
| backbone | InternVL2.5-2B |
| image max_num | 4 |
| LoRA target | `wqkv`, `wo`, `w1`, `w2`, `w3` |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA layer start | 16 |
| train epochs | 2 |
| grad accumulation | 8 |
| LoRA lr | 3e-5 |
| score head lr | 1e-4 |
| weight decay | 0.01 |
| warmup ratio | 0.03 |
| max grad norm | 1.0 |
| seed | 47 |
| GPUs | 4 x RTX 4090 |

最终训练不是直接使用原始偏好标签，而是使用 base model generative judge 在训练池上产生的 pseudo preference label，再从原始 LoRA reward checkpoint 初始化后蒸馏训练。这样做的原因是：已有 base judge 在 VLRewardBench 上表现强，但它是生成式判断流程，不是一个可直接保存并快速部署的 scalar reward model。伪标签蒸馏可以把它的偏好边界迁移到 LoRA + score head 模型中。

# 5. 关键命令

生成 pseudo label：

```bash
cd /data4/ljl/csh/vl_reward_work
/data3/ljl/miniconda3/envs/internvl/bin/python scripts/internvl_reward.py pseudo-from-eval \
  --source-jsonl data/processed_h4_12k/train_pairs_mix.jsonl \
  --eval-json results/pseudo/basejudge_h4mix_train.json \
  --out data/processed_h4_12k/train_pairs_mix_basejudge_pseudo.jsonl \
  --source-suffix _basejudge_pseudo
```

训练最终 reward model：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/data3/ljl/miniconda3/envs/internvl/bin/python -m torch.distributed.run \
  --nproc_per_node=4 --master_port=29647 \
  scripts/internvl_reward.py train-lora-reward \
  --model-path /data2/ljl/csh/pretrained_model/InternVL2_5-2B \
  --max-num 4 \
  --train-jsonl data/processed_h4_12k/train_pairs_mix_basejudge_pseudo.jsonl \
  --val-jsonl data/processed_h4_12k/val_pairs_mix_basejudge_pseudo.jsonl \
  --out-dir outputs/internvl25_2b_lora_reward_basejudge_pseudo_r8_from_orig_4g \
  --epochs 2 \
  --grad-accum 8 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-layer-start 16 \
  --lora-lr 3e-5 \
  --head-lr 1e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --max-grad-norm 1.0 \
  --seed 47 \
  --init-checkpoint outputs/internvl25_2b_lora_reward_h4mix_top8_r8/reward_lora_best.pt
```

# 6. 训练结果

训练历史：

| epoch | train acc | pseudo val acc | val margin | seconds |
|---:|---:|---:|---:|---:|
| 1 | 60.81% | 60.29% | 0.2291 | 698.57 |
| 2 | 70.09% | 61.19% | 0.2883 | 491.25 |

保存权重：

- Best checkpoint: `outputs/internvl25_2b_lora_reward_basejudge_pseudo_r8_from_orig_4g/reward_lora_best.pt`
- Last checkpoint: `outputs/internvl25_2b_lora_reward_basejudge_pseudo_r8_from_orig_4g/reward_lora_last.pt`
- 单个权重文件大小约 11MB。

# 7. VLRewardBench 结果

最终评测集：`data/processed/vlrewardbench_eval.jsonl`，共 1,247 条样本。评测时使用 4 个分片，每个分片占用 1 张 GPU。

| 方法 | VLRewardBench acc |
|---|---:|
| Base model generative judge, `Score_base` | 64.80% |
| Frozen feature score head | 46.99% |
| LoRA reward, original labels | 54.61% |
| LoRA reward, base-judge pseudo distillation, `Score_reward` | 59.98% |

最终结果文件：

`results/internvl25_2b_lora_reward_basejudge_pseudo_r8_from_orig_4g_vlrewardbench.json`

按 source 观察，POVID preference 数据上达到 67.41%，GQA 为 61.54%，VQAv2 为 62.86%，wildvision-battle 为 47.95%。整体上，伪标签蒸馏相比直接用原始标签训练的 LoRA reward model 提升了 5.37 个百分点。

# 8. 结论

本次实验完成了一个可保存、可复现的 InternVL2.5-2B reward model：

- `Score_base = 64.80%`
- `Score_reward = 59.98%`
- `Score_reward` 超过题目参考线 `58%`
- reward 权重已保存到服务器输出目录
- 训练过程遵守单次最多 4 张 GPU 的约束

主要经验是：直接冻结 backbone 训练 score head 容易跨数据集泛化不足；LoRA 在线训练显著更稳；把强 base generative judge 蒸馏到 scalar reward model 后，可以得到一个性能超过 58% 的可交付 reward model。

# 9. 后续可提升方向

当前 H4 原始目录只包含 13 个 parquet shard 中的前 3 个。如果继续补齐更多 H4 shard，并使用相同流程重建混合训练集，预计还可以提升 reward model 的覆盖面。另一个方向是使用更高 rank 或更早层起始的 LoRA 配置，例如 rank 16、layer start 12，但需要更长训练和更多显存验证。
