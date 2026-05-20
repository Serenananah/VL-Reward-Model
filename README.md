# VL Reward Model Experiments

This repository contains the code, final evaluation results, and report for the VL Reward Model research assessment.

## Final VLRewardBench Results

| Model | Base score | Reward score |
| --- | ---: | ---: |
| Qwen2-VL-7B-Instruct | 43.14% | 66.56% |
| InternVL2.5-2B | 64.80% | 71.21% |

The base score is measured by direct generative A/B judging with the original model. The reward score is measured by a trained scalar reward head with LoRA adapters.

## Contents

- `scripts/internvl_reward.py`: InternVL2.5 reward training/evaluation utilities.
- `scripts/qwen2vl_reward.py`: Qwen2-VL reward training/evaluation utilities with hand-written LoRA.
- `results/`: final merged VLRewardBench JSON results.
- `reports/vl_reward_final_report.tex`: LaTeX report source.
- `reports/vl_reward_final_report.pdf`: compiled report.
- `requirements.txt`: minimal Python dependencies and the exact package versions used by the two experiment environments.

## Reproduction

The original experiments were run under `/data4/ljl/csh/vl_reward_work` with local base models at:

- `/data2/ljl/csh/pretrained_model/InternVL2_5-2B`
- `/data2/ljl/csh/pretrained_model/Qwen2-VL-7B-Instruct`

Create dependencies with two separate environments if possible, because the final InternVL and Qwen runs used different tested Transformers/Torch stacks:

```bash
# InternVL environment used in the final run
conda create -n internvl_reward python=3.9 -y
conda activate internvl_reward
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.37.2 accelerate==0.34.2 pandas==2.3.3 pyarrow==21.0.0 \
  Pillow==11.3.0 tqdm==4.67.3 timm==0.9.12 einops==0.6.1 sentencepiece==0.1.99

# Qwen2-VL environment used in the final run
conda create -n qwen2vl_reward python=3.10 -y
conda activate qwen2vl_reward
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==5.1.0 accelerate==1.12.0 qwen-vl-utils==0.0.8 \
  Pillow==12.1.0 tqdm==4.67.3 numpy==1.26.4 safetensors==0.7.0
```

Prepare data. The scripts expect preference-pair JSONL files with image paths, query, `response_a`, `response_b`, and label. In the original run, RLHF-V and HuggingFaceH4 `rlaif-v_formatted` were processed into:

```bash
python scripts/internvl_reward.py prepare \
  --rlhf-v-parquet data/raw/RLHF-V-Dataset.parquet \
  --vlrewardbench-jsonl data/processed/vlrewardbench_eval.jsonl \
  --out-dir data/processed

python scripts/internvl_reward.py prepare-h4 \
  --h4-dir data/raw/rlaif_h4 \
  --vlrewardbench-jsonl data/processed/vlrewardbench_eval.jsonl \
  --out-dir data/processed_h4_30k_nodedup \
  --max-train 29000 --max-val 1000

python scripts/internvl_reward.py mix-jsonl \
  --inputs data/processed_h4_30k_nodedup/train_pairs_h4.jsonl data/processed/train_pairs.jsonl \
  --out data/processed_h4_30k_nodedup/train_pairs_mix.jsonl
```

Train reward models with 4 GPUs:

```bash
# InternVL2.5-2B reward
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 scripts/internvl_reward.py train-lora-reward \
  --model-path /data2/ljl/csh/pretrained_model/InternVL2_5-2B \
  --train-jsonl data/processed_h4_30k_nodedup/train_pairs_mix.jsonl \
  --val-jsonl data/processed_h4_30k_nodedup/val_pairs_mix.jsonl \
  --out-dir outputs/internvl25_2b_lora_reward_h4mix30k_r16_l8_len1536_no_logits_4g \
  --max-num 4 --max-length 1536 --lora-rank 16 --lora-alpha 32 --lora-layer-start 8 \
  --grad-accum 8 --epochs 1

# Qwen2-VL-7B reward
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 scripts/qwen2vl_reward.py train \
  --model-path /data2/ljl/csh/pretrained_model/Qwen2-VL-7B-Instruct \
  --train-jsonl data/processed_h4_30k_nodedup/train_pairs_mix.jsonl \
  --val-jsonl data/processed_h4_30k_nodedup/val_pairs_mix.jsonl \
  --out-dir outputs/qwen2vl7b_lora_reward_h4mix30k_r8_l20_len768_4g \
  --max-length 768 --min-pixels 50176 --max-pixels 200704 \
  --lora-rank 8 --lora-alpha 16 --lora-layer-start 20 --grad-accum 8 --epochs 1
```

Evaluate by launching 4 single-GPU shards and then merging:

```bash
# Example: merge final InternVL reward shards
python scripts/internvl_reward.py merge-eval-json \
  --inputs results/internvl25_2b_lora_reward_h4mix30k_r16_l8_len1536_no_logits_4g_vlrewardbench_shard*.json \
  --out results/internvl25_2b_lora_reward_h4mix30k_r16_l8_len1536_no_logits_4g_vlrewardbench.json
```

The uploaded `results/*.json` files are the final merged VLRewardBench outputs used in the report.

## Main Artifacts

Model weights are intentionally kept in the Hugging Face model repositories rather than this code repository:

- `Hanzi7na/internvl25-2b-vlreward-lora`
- `Hanzi7na/qwen2vl-7b-vlreward-lora`

## Data Note

VLRewardBench was used only for final evaluation. Training used RLHF-V and HuggingFaceH4 `rlaif-v_formatted` preference data, with VLRewardBench queries filtered out during preprocessing.
