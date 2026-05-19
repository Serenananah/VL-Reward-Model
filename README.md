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

## Main Artifacts

Model weights are intentionally kept in the Hugging Face model repositories rather than this code repository:

- `Hanzi7na/internvl25-2b-vlreward-lora`
- `Hanzi7na/qwen2vl-7b-vlreward-lora`

## Data Note

VLRewardBench was used only for final evaluation. Training used RLHF-V and HuggingFaceH4 `rlaif-v_formatted` preference data, with VLRewardBench queries filtered out during preprocessing.
