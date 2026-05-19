#!/usr/bin/env python3
import argparse
import contextlib
import json
import math
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def setup_distributed(device_arg):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, local_rank, int(os.environ["RANK"]), world_size, torch.device("cuda", local_rank)
    return False, 0, 0, 1, torch.device(device_arg)


def cleanup_distributed(enabled):
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank, alpha, dropout=0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear expects nn.Linear")
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(self.rank, 1)
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(self.rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        result = self.base_layer(x)
        lora_dtype = self.lora_A.dtype
        lora = F.linear(self.dropout(x).to(lora_dtype), self.lora_A)
        lora = F.linear(lora, self.lora_B).to(result.dtype)
        return result + lora * self.scale


class OnlineRewardModel(nn.Module):
    def __init__(self, model, head):
        super().__init__()
        self.model = model
        self.head = head

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        idx = attention_mask.sum(dim=1) - 1
        feats = hidden[torch.arange(hidden.shape[0], device=hidden.device), idx].float()
        return self.head(feats).squeeze(-1)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad_(False)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "use_cache"):
        text_config.use_cache = False


def _set_child_module(root, name, module):
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def add_lora_adapters(model, rank, alpha, dropout, layer_start, target_modules):
    target_modules = tuple(target_modules)
    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith("model.language_model.layers."):
            continue
        parts = name.split(".")
        try:
            layer_idx = int(parts[3])
        except Exception:
            continue
        if layer_idx < int(layer_start):
            continue
        if parts[-1] not in target_modules:
            continue
        _set_child_module(model, name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(name)
    if not replaced:
        raise ValueError("No LoRA target modules were replaced")
    return replaced


def trainable_state_dict(model):
    return {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}


def count_trainable_params(module):
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def unwrap(module):
    return module.module if isinstance(module, DDP) else module


def make_score_head(hidden_size):
    return nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))


def qwen_messages(image, question, answer):
    return [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]


def truncate_for_judge(text, max_chars):
    text = str(text or "")
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n[truncated]"
    return text


def qwen_judge_messages(image, question, response_a, response_b, max_response_chars=0):
    response_a = truncate_for_judge(response_a, max_response_chars)
    response_b = truncate_for_judge(response_b, max_response_chars)
    prompt = (
        "You are evaluating two candidate answers to a visual question. "
        "Use the image and the user question to decide which response is better. "
        "Prefer the response that is more visually grounded, factual, complete, and follows the question. "
        "If one response contains hallucinated visual details or contradicts the image, choose the other response.\n\n"
        f"Question:\n{question}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        "Answer with exactly one letter: A or B."
    )
    return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]


def parse_choice(text):
    text = str(text or "").strip().upper()
    match = re.search(r"\b([AB])\b", text)
    if match:
        return 0 if match.group(1) == "A" else 1
    if text.startswith("A"):
        return 0
    if text.startswith("B"):
        return 1
    return None


def pair_inputs_from_row(processor, row, args):
    image = Image.open(row["image_path"]).convert("RGB")
    texts = [
        processor.apply_chat_template(
            qwen_messages(image, row["query"], row["response_a"]),
            tokenize=False,
            add_generation_prompt=False,
        ),
        processor.apply_chat_template(
            qwen_messages(image, row["query"], row["response_b"]),
            tokenize=False,
            add_generation_prompt=False,
        ),
    ]
    old_side = processor.tokenizer.padding_side
    old_truncation_side = processor.tokenizer.truncation_side
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.truncation_side = "right"
    inputs = processor(
        text=texts,
        images=[image, image],
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    processor.tokenizer.padding_side = old_side
    processor.tokenizer.truncation_side = old_truncation_side
    input_ids = inputs["input_ids"].to(args.device)
    attention_mask = inputs["attention_mask"].to(args.device)
    pixel_values = inputs["pixel_values"].to(device=args.device, dtype=torch.bfloat16)
    image_grid_thw = inputs["image_grid_thw"].to(args.device)
    return input_ids, attention_mask, pixel_values, image_grid_thw, int(row["label"])


def build_reward_model(args, checkpoint=None, train=False):
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_fast=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    freeze_model(model)
    config = checkpoint.get("lora_config") if checkpoint else None
    rank = int(config.get("rank", args.lora_rank)) if config else args.lora_rank
    alpha = float(config.get("alpha", args.lora_alpha)) if config else args.lora_alpha
    dropout = float(config.get("dropout", args.lora_dropout)) if config else args.lora_dropout
    layer_start = int(config.get("layer_start", args.lora_layer_start)) if config else args.lora_layer_start
    target_modules = config.get("target_modules", args.lora_target_modules) if config else args.lora_target_modules
    replaced = add_lora_adapters(
        model,
        rank=rank,
        alpha=alpha,
        dropout=dropout if train else 0.0,
        layer_start=layer_start,
        target_modules=target_modules,
    )
    head = make_score_head(int(model.config.text_config.hidden_size)).to(args.device)
    reward_model = OnlineRewardModel(model, head).to(args.device)
    if checkpoint:
        missing, unexpected = model.load_state_dict(checkpoint["model_lora_state"], strict=False)
        head.load_state_dict(checkpoint["head_state"])
        if unexpected:
            print(f"unexpected checkpoint keys: {unexpected}", flush=True)
    reward_model.train(mode=train)
    return processor, reward_model, replaced


@torch.inference_mode()
def online_accuracy(reward_model, processor, rows, args):
    module = unwrap(reward_model)
    module.eval()
    correct = 0
    skipped = 0
    margins = []
    for row in tqdm(rows, desc="online-eval", disable=getattr(args, "rank", 0) != 0):
        try:
            input_ids, attention_mask, pixel_values, image_grid_thw, label = pair_inputs_from_row(processor, row, args)
            scores = module(input_ids, attention_mask, pixel_values, image_grid_thw)
            pred = 0 if float(scores[0]) >= float(scores[1]) else 1
            margin = float(scores[0] - scores[1]) if label == 0 else float(scores[1] - scores[0])
        except Exception as exc:
            print(f"skip eval {row.get('id')}: {exc}", flush=True)
            skipped += 1
            continue
        correct += int(pred == label)
        margins.append(margin)
    total = max(1, len(margins))
    module.train()
    return correct / total, (sum(margins) / total if margins else 0.0), skipped


def save_checkpoint(path, reward_module, args, best):
    module = unwrap(reward_module)
    payload = {
        "model_lora_state": trainable_state_dict(module.model),
        "head_state": module.head.state_dict(),
        "hidden_size": int(module.model.config.text_config.hidden_size),
        "best": best,
        "lora_config": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "layer_start": args.lora_layer_start,
            "target_modules": args.lora_target_modules,
        },
        "args": {k: str(v) if isinstance(v, torch.device) else v for k, v in vars(args).items()},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def cmd_train(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    distributed, local_rank, rank, world_size, device = setup_distributed(args.device)
    args.device = device
    args.rank = rank
    try:
        train_rows = list(read_jsonl(args.train_jsonl))
        val_rows = list(read_jsonl(args.val_jsonl)) if args.val_jsonl else []
        rng = random.Random(args.seed)
        rng.shuffle(train_rows)
        rng.shuffle(val_rows)
        if args.train_limit:
            train_rows = train_rows[: args.train_limit]
        if args.val_limit:
            val_rows = val_rows[: args.val_limit]
        processor, reward_model, replaced = build_reward_model(args, train=True)
        module = unwrap(reward_model)
        trainable, total = count_trainable_params(module)
        if distributed:
            reward_model = DDP(
                reward_model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
        params = [
            {"params": [p for n, p in module.model.named_parameters() if p.requires_grad], "lr": args.lora_lr},
            {"params": list(module.head.parameters()), "lr": args.head_lr},
        ]
        opt = torch.optim.AdamW(params, weight_decay=args.weight_decay)
        steps_per_epoch = max(1, math.ceil(len(train_rows) / max(1, world_size) / args.grad_accum))
        total_steps = max(1, steps_per_epoch * args.epochs)
        warmup_steps = int(total_steps * args.warmup_ratio)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        best = {"val_acc": -1.0, "epoch": -1, "step": 0}
        history = []
        global_step = 0
        if rank == 0:
            print(
                json.dumps(
                    {
                        "train_rows": len(train_rows),
                        "val_rows": len(val_rows),
                        "world_size": world_size,
                        "lora_modules": len(replaced),
                        "trainable_params": trainable,
                        "total_params": total,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        indices_all = list(range(len(train_rows)))
        sampler = None
        if distributed:
            sampler = DistributedSampler(indices_all, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        local_rng = random.Random(args.seed + rank)
        for epoch in range(1, args.epochs + 1):
            reward_model.train()
            if sampler:
                sampler.set_epoch(epoch)
                indices = list(iter(sampler))
            else:
                indices = indices_all[:]
                local_rng.shuffle(indices)
            opt.zero_grad(set_to_none=True)
            losses = []
            correct = 0
            seen = 0
            start_time = time.time()
            for local_step, idx in enumerate(indices):
                row = train_rows[int(idx) % len(train_rows)]
                sync_now = ((local_step + 1) % args.grad_accum == 0) or (local_step + 1 == len(indices))
                sync_context = contextlib.nullcontext() if (sync_now or not distributed) else reward_model.no_sync()
                try:
                    input_ids, attention_mask, pixel_values, image_grid_thw, label = pair_inputs_from_row(
                        processor, row, args
                    )
                    with sync_context:
                        scores = reward_model(input_ids, attention_mask, pixel_values, image_grid_thw)
                        margin = scores[0] - scores[1] if label == 0 else scores[1] - scores[0]
                        loss = -F.logsigmoid(margin) / args.grad_accum
                        loss.backward()
                    pred = 0 if float(scores[0].detach()) >= float(scores[1].detach()) else 1
                    correct += int(pred == label)
                    seen += 1
                    losses.append(float(loss.detach().cpu()) * args.grad_accum)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        torch.cuda.empty_cache()
                    if distributed:
                        raise
                    print(f"skip train {row.get('id')}: {exc}", flush=True)
                    continue
                if sync_now:
                    if args.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_([p for p in module.parameters() if p.requires_grad], args.max_grad_norm)
                    opt.step()
                    scheduler.step()
                    opt.zero_grad(set_to_none=True)
                    global_step += 1
                if rank == 0 and args.log_every and seen and seen % args.log_every == 0:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "seen_rank0": seen,
                                "loss": sum(losses[-args.log_every :]) / max(1, min(args.log_every, len(losses))),
                                "train_acc_rank0": correct / max(1, seen),
                                "lr_lora": scheduler.get_last_lr()[0],
                                "lr_head": scheduler.get_last_lr()[1],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            stats = torch.tensor([sum(losses), len(losses), correct, seen], dtype=torch.float64, device=args.device)
            if distributed:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            train_loss = float(stats[0].item() / max(1.0, stats[1].item()))
            train_acc = float(stats[2].item() / max(1.0, stats[3].item()))
            val_acc = val_margin = None
            val_skipped = 0
            if distributed:
                dist.barrier()
            if rank == 0 and val_rows:
                val_acc, val_margin, val_skipped = online_accuracy(reward_model, processor, val_rows, args)
                if val_acc > best["val_acc"]:
                    best = {"val_acc": val_acc, "epoch": epoch, "step": global_step}
                    save_checkpoint(out_dir / "reward_lora_best.pt", reward_model, args, best)
            if distributed:
                dist.barrier()
            row = {
                "epoch": epoch,
                "loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "val_margin": val_margin,
                "val_skipped": val_skipped,
                "seconds": time.time() - start_time,
                "step": global_step,
            }
            if rank == 0:
                history.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                (out_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if rank == 0:
            save_checkpoint(out_dir / "reward_lora_last.pt", reward_model, args, best)
            print("best", json.dumps(best, ensure_ascii=False), flush=True)
    finally:
        cleanup_distributed(distributed)


def cmd_eval(args):
    args.device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    processor, reward_model, _ = build_reward_model(args, checkpoint=checkpoint, train=False)
    reward_model.eval()
    rows = list(read_jsonl(args.jsonl))
    if args.num_shards > 1:
        rows = [row for i, row in enumerate(rows) if i % args.num_shards == args.shard_index]
    if args.limit:
        rows = rows[: args.limit]
    results = []
    correct = 0
    by_source = {}
    for row in tqdm(rows, desc="qwen-eval"):
        try:
            input_ids, attention_mask, pixel_values, image_grid_thw, label = pair_inputs_from_row(processor, row, args)
            with torch.inference_mode():
                scores = reward_model(input_ids, attention_mask, pixel_values, image_grid_thw)
            score_a = float(scores[0].cpu())
            score_b = float(scores[1].cpu())
            pred = 0 if score_a >= score_b else 1
        except Exception as exc:
            print(f"skip eval {row.get('id')}: {exc}", flush=True)
            continue
        ok = pred == label
        correct += int(ok)
        source = row.get("source", "")
        by_source.setdefault(source, [0, 0])
        by_source[source][0] += int(ok)
        by_source[source][1] += 1
        results.append(
            {
                "id": row.get("id"),
                "pred": pred,
                "label": label,
                "correct": ok,
                "score_a": score_a,
                "score_b": score_b,
                "source": source,
            }
        )
        if args.flush_every and len(results) % args.flush_every == 0:
            summary = {
                "accuracy": correct / len(results),
                "num_examples": len(results),
                "by_source": {
                    k: {"accuracy": v[0] / v[1], "num_examples": v[1]}
                    for k, v in sorted(by_source.items())
                },
                "checkpoint": args.checkpoint,
                "checkpoint_best": checkpoint.get("best"),
            }
            write_json(args.out, {"summary": summary, "results": results})
    summary = {
        "accuracy": correct / max(1, len(results)),
        "num_examples": len(results),
        "by_source": {
            k: {"accuracy": v[0] / v[1], "num_examples": v[1]}
            for k, v in sorted(by_source.items())
        },
        "checkpoint": args.checkpoint,
        "checkpoint_best": checkpoint.get("best"),
    }
    write_json(args.out, {"summary": summary, "results": results})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_base_judge_eval(args):
    args.device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_fast=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    model.eval()
    rows = list(read_jsonl(args.jsonl))
    if args.num_shards > 1:
        rows = [row for i, row in enumerate(rows) if i % args.num_shards == args.shard_index]
    if args.limit:
        rows = rows[: args.limit]

    results = []
    correct = 0
    parsed = 0
    by_source = {}
    old_side = processor.tokenizer.padding_side
    old_truncation_side = processor.tokenizer.truncation_side
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.truncation_side = "right"
    try:
        for row in tqdm(rows, desc="qwen-base-judge"):
            raw_output = ""
            try:
                image = Image.open(row["image_path"]).convert("RGB")
                text = processor.apply_chat_template(
                    qwen_judge_messages(
                        image,
                        row["query"],
                        row["response_a"],
                        row["response_b"],
                        max_response_chars=args.max_response_chars,
                    ),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = processor(
                    text=[text],
                    images=[image],
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                model_inputs = {}
                for key, value in inputs.items():
                    if not torch.is_tensor(value):
                        model_inputs[key] = value
                    elif key == "pixel_values":
                        model_inputs[key] = value.to(device=args.device, dtype=torch.bfloat16)
                    else:
                        model_inputs[key] = value.to(args.device)
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                new_ids = generated_ids[:, model_inputs["input_ids"].shape[1] :]
                raw_output = processor.batch_decode(
                    new_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                pred = parse_choice(raw_output)
            except Exception as exc:
                print(f"skip base judge {row.get('id')}: {exc}", flush=True)
                pred = None
            if pred is None:
                pred = 0
            else:
                parsed += 1
            label = int(row["label"])
            ok = pred == label
            correct += int(ok)
            source = row.get("source", "")
            by_source.setdefault(source, [0, 0])
            by_source[source][0] += int(ok)
            by_source[source][1] += 1
            results.append(
                {
                    "id": row.get("id"),
                    "label": label,
                    "prediction": pred,
                    "correct": ok,
                    "raw_output": raw_output,
                    "source": source,
                }
            )
            if args.flush_every and len(results) % args.flush_every == 0:
                summary = {
                    "accuracy": correct / len(results),
                    "num_examples": len(results),
                    "parse_rate": parsed / len(results),
                    "by_source": {
                        k: {"accuracy": v[0] / v[1], "num_examples": v[1]}
                        for k, v in sorted(by_source.items())
                    },
                    "method": "base_model_generative_judge",
                }
                write_json(args.out, {"summary": summary, "results": results})
    finally:
        processor.tokenizer.padding_side = old_side
        processor.tokenizer.truncation_side = old_truncation_side

    summary = {
        "accuracy": correct / max(1, len(results)),
        "num_examples": len(results),
        "parse_rate": parsed / max(1, len(results)),
        "by_source": {
            k: {"accuracy": v[0] / v[1], "num_examples": v[1]}
            for k, v in sorted(by_source.items())
        },
        "method": "base_model_generative_judge",
    }
    write_json(args.out, {"summary": summary, "results": results})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_merge(args):
    all_results = []
    inputs = []
    for path in args.inputs:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        all_results.extend(payload["results"])
        inputs.append(path)
    correct = sum(1 for row in all_results if row.get("correct"))
    by_source = {}
    for row in all_results:
        source = row.get("source", "")
        by_source.setdefault(source, [0, 0])
        by_source[source][0] += int(bool(row.get("correct")))
        by_source[source][1] += 1
    summary = {
        "accuracy": correct / max(1, len(all_results)),
        "num_examples": len(all_results),
        "by_source": {
            k: {"accuracy": v[0] / v[1], "num_examples": v[1]}
            for k, v in sorted(by_source.items())
        },
        "inputs": inputs,
    }
    write_json(args.out, {"summary": summary, "results": all_results})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def add_common_args(p):
    p.add_argument("--model-path", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--min-pixels", type=int, default=50176)
    p.add_argument("--max-pixels", type=int, default=200704)
    p.add_argument("--attn-implementation", default="sdpa")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("train")
    add_common_args(p)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--val-limit", type=int, default=0)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--lora-layer-start", type=int, default=24)
    p.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    p.add_argument("--lora-lr", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval")
    add_common_args(p)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--flush-every", type=int, default=20)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--lora-layer-start", type=int, default=24)
    p.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("base-judge-eval")
    add_common_args(p)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--flush-every", type=int, default=20)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--max-response-chars", type=int, default=1200)
    p.set_defaults(func=cmd_base_judge_eval)

    p = sub.add_parser("merge-eval-json")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
