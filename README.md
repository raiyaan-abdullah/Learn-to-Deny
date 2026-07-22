# Learning to Deny: Action Denial in Multimodal Large Language Models

[![Paper](https://img.shields.io/badge/arXiv-2606.31187-b31b1b.svg)](https://arxiv.org/abs/2606.31187)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-UCF101--AD-yellow)](https://huggingface.co/datasets/raiyaanabdullah/UCF101-AD)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://raiyaan.xyz/Learn-to-Deny-webpage/)

Official code and dataset for **Learning to Deny: Action Denial in Multimodal Large Language Models**, accepted to the ECCV 2026 main conference.

**Authors:** Raiyaan Abdullah, Shehreen Azad, and Yogesh Singh Rawat

## Overview

Multimodal large language models can recognize actions when they occur, but they often predict a plausible action from the surrounding people, objects, and scene even when its defining interaction or motion is absent. We introduce **UCF101-AD**, a benchmark of paired action-presence and action-denial videos for measuring this behavior, and **CausalAct**, a causal prompting formulation that links persons, objects, locations, spatial relations, interactions, motions, and actions.

## Multiple-choice questions

| File | Description |
| --- | --- |
| `mcq_questions/ucf101_ad_action_denial_standard.json` | Contains 3,549 standard UCF101-AD MCQs. Each denial video retains its corresponding positive action as a hard distractor, while the correct answer is the “none” option. |
| `mcq_questions/ucf101_ad_action_denial_primary_distractor_removed.json` | Contains 3,549 action-denial MCQs for the primary-distractor-removal analysis. The corresponding positive action caption is removed from the choices. |
| `mcq_questions/ucf101_ad_action_denial_explicit_denial.json` | Contains 3,549 action-denial MCQs for the explicit-denial analysis. The generic “none” answer is replaced by an explicit caption stating that the contextual cues are present but the expected action is not occurring. |
| `mcq_questions/ucf101_ad_action_presence.json` | Contains 675 MCQs for UCF101 action-presence videos in which the target action is present. |

Each question file is a JSON array. Entries contain a relative `video_name`, an 11-caption `choices` list, and zero-based indices used for evaluation:

- `none_index`: the correct denial/none choice for action-denial questions.
- `main_positive`: the corresponding UCF101 action label, when included.
- `main_positive_index`: the correct choice for action-presence questions and the primary distractor for action-denial questions, when included.

The evaluation scripts convert the stored indices to one-based values to match the numbered choices presented to the model.

## Zero-shot evaluation

Follow the [Ovis2.5-9B instructions](https://huggingface.co/ATH-MaaS/Ovis2.5-9B) to set up the environment.

| File | Description |
| --- | --- |
| `ovis2_5_baseline.py` | Baseline Ovis2.5-9B multiple-choice evaluator. It asks the model to select the caption that best describes the action in each video. |
| `ovis2_5_causalact_zero.py` | CausalAct-Zero evaluator. It contains the paper's causal graph prompt over person (P), object (O), location (L), spatial relation (S), interaction (I), motion (M), and action (A) before asking the model to select a choice number. |

To run the baseline, add the following call at the end of `ovis2_5_baseline.py`, replacing the placeholder paths, and then run `python ovis2_5_baseline.py`:

```python
run_ovis_mcq("/path/to/dataset", "/path/to/mcq_questions.json", "/path/to/output.json")
```

Run CausalAct-Zero with:

```bash
python ovis2_5_causalact_zero.py \
  --question_type data_and_graph \
  --model_path AIDC-AI/Ovis2.5-9B \
  --output_file /path/to/output.json
```

You may adapt either script to work with other MLLMs.

## Auxiliary questions for fine-tuning CausalAct

| File | Description |
| --- | --- |
| `mcq_questions/auxilary_finetuning_questions.json` | Contains 7,059 auxiliary MCQs from the UCF101-AD training set for the paper's fine-tuning experiments. The filename retains the original `auxilary` spelling. |

We provide instructions for Qwen here, but the data can be adapted for other MLLMs. Our experiments primarily trained and evaluated Qwen2.5-VL. Follow the [Qwen2.5-VL repository](https://github.com/QwenLM/Qwen2.5-VL) to set up the environment. The QwenLM maintainers may update this link when a future version of Qwen-VL is released.

In `qwen-vl-finetune/qwenvl/data/__init__.py`, add an entry named `ucf101_ad_auxilary_finetuning`. Set its annotation path to the auxiliary fine-tuning JSON file and its data path to the dataset directory. Then run the following command from the `qwen-vl-finetune` directory:

```bash
torchrun --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=12345 \
  qwenvl/train/train_qwen.py \
  --model_name_or_path pretrained_checkpoints/Qwen2.5-VL-3B-Instruct \
  --tune_mm_llm True \
  --tune_mm_vision True \
  --tune_mm_mlp True \
  --dataset_use ucf101_ad_auxilary_finetuning \
  --output_dir ./output_path \
  --cache_dir ./cache \
  --bf16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-7 \
  --mm_projector_lr 1e-5 \
  --vision_tower_lr 1e-6 \
  --optim adamw_torch \
  --model_max_length 8192 \
  --data_flatten True \
  --data_packing True \
  --max_pixels $((576*28*28)) \
  --min_pixels $((16*28*28)) \
  --base_interval 2 \
  --video_max_frames 6 \
  --video_min_frames 6 \
  --video_max_frame_pixels $((1408*28*28)) \
  --video_min_frame_pixels $((256*28*28)) \
  --num_train_epochs 2 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --logging_steps 10 \
  --save_steps 100 \
  --save_total_limit 10 \
  --deepspeed scripts/zero3.json \
  --gradient_checkpointing True
```

We used these hyperparameters in our experiments, but they may be adjusted as needed.

## Citation

```bibtex
@InProceedings{Abdullah_2026_ECCV,
    author    = {Abdullah, Raiyaan and Azad, Shehreen and Rawat, Yogesh Singh},
    title     = {Learning to Deny: Action Denial in Multimodal Large Language Models},
    booktitle = {Computer Vision -- ECCV 2026},
    publisher = {Springer Nature Switzerland},
    address   = {Cham},
    pages     = {}
}
```
