from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Subset

from .config import ModelConfig, TrainConfig, write_json
from .data import MathDataset, collate_examples, dataset_provenance, make_loader
from .evaluate import evaluate_dataset
from .model import CausalTransformer
from .tokenizer import FrozenTokenizer
from .verifier import ANSWER_POLICY_VERSION


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def choose_device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


def save_checkpoint(path: Path, model, optimizer, model_config, train_config, step, completed_epochs, processed_sequence_tokens, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "model_config": model_config.to_dict(), "train_config": train_config.to_dict(), "step": step, "completed_epochs": completed_epochs, "processed_sequence_tokens": processed_sequence_tokens, "manifest": manifest}, path)


def choose_epoch_validation_examples(valid: MathDataset, seed: int, per_type: int) -> tuple[list, dict[str, list[str]]]:
    """Choose one reproducible, stratified validation subset for all epochs.

    The old ``--eval-limit`` path takes the first rows, which can overrepresent
    the ordering of the JSONL file.  The epoch curve instead uses a fixed
    random sample from the three high-volume scalar types.  Reusing the same
    IDs at every epoch makes the curve measure model change rather than sample
    change, while the seed makes comparisons across codebook runs paired.
    """
    if per_type <= 0:
        return [], {key: [] for key in ("integer", "decimal", "fraction")}
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(valid.records):
        answer_type = str(row.get("answer_type", "unknown"))
        if answer_type in ("integer", "decimal", "fraction"):
            groups[answer_type].append(index)
    rng = random.Random(seed + 104729)
    selected: list[int] = []
    selected_ids: dict[str, list[str]] = {}
    for answer_type in ("integer", "decimal", "fraction"):
        candidates = groups.get(answer_type, [])
        chosen = sorted(rng.sample(candidates, min(per_type, len(candidates))))
        selected.extend(chosen)
        selected_ids[answer_type] = [str(valid.records[index].get("id", index)) for index in chosen]
    selected.sort()
    return [valid[index] for index in selected], selected_ids


@torch.no_grad()
def validation_loss(model: CausalTransformer, examples: list, device: torch.device, pad_id: int,
                    eos_loss_weight: float, batch_size: int = 32) -> dict[str, float | int]:
    """Compute fast teacher-forced validation losses without free generation."""
    model.eval()
    answer_nll_sum = 0.0
    answer_token_count = 0
    eos_loss_sum = 0.0
    eos_example_count = 0
    for offset in range(0, len(examples), batch_size):
        batch = collate_examples(examples[offset:offset + batch_size], pad_id)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention = batch["attention_mask"].to(device)
        logits = model(input_ids, attention)
        answer_sum = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
            ignore_index=-100, reduction="sum",
        )
        token_count = int(labels.ne(-100).sum().item())
        answer_mean = answer_sum / max(token_count, 1)
        eos_mean = model.eos_loss(logits, input_ids, batch["examples"]) if eos_loss_weight else logits.new_zeros(())
        combined = answer_mean + eos_loss_weight * eos_mean
        answer_nll_sum += float(answer_sum.item())
        answer_token_count += token_count
        eos_loss_sum += float(eos_mean.item()) * len(batch["examples"])
        eos_example_count += len(batch["examples"])
    answer_nll = answer_nll_sum / max(answer_token_count, 1)
    eos_loss = eos_loss_sum / max(eos_example_count, 1)
    return {
        "examples": len(examples),
        "answer_tokens": answer_token_count,
        "answer_nll": answer_nll,
        "eos_loss": eos_loss,
        "loss": answer_nll + eos_loss_weight * eos_loss,
    }


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    if not args.resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing run directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = FrozenTokenizer(args.tokenizer_dir)
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, max_seq_len=args.max_seq_len)
    train_config = TrainConfig(seed=args.seed, batch_size=args.batch_size, learning_rate=args.learning_rate, epochs=args.epochs, max_steps=args.max_steps, eos_loss_weight=args.eos_loss_weight, device=args.device)
    device = choose_device(args.device)
    train_path = args.train_path or (args.dataset_dir / "train.jsonl")
    valid_path = args.valid_path or (args.dataset_dir / "valid.jsonl")
    train = MathDataset(train_path, tokenizer, model_config.max_seq_len, codebook_mode=args.codebook_mode, codebook_index=args.codebook_index, seed=args.seed)
    valid = MathDataset(valid_path, tokenizer, model_config.max_seq_len, codebook_mode=args.codebook_mode, codebook_index=args.codebook_index, seed=args.seed)
    loader = make_loader(train, train_config.batch_size, args.seed, True, train_config.num_workers)
    epoch_valid_examples, epoch_valid_ids = choose_epoch_validation_examples(valid, args.seed, args.epoch_valid_per_type)
    model = CausalTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    train_sequence_tokens = sum(len(example.input_ids) for example in train)
    answer_tokens = sum(example.answer_end - example.answer_start for example in train)
    manifest = {"code_version": "transformer-pilot-v3", "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "seed": args.seed, "model": model_config.to_dict(), "actual_parameter_count": model.parameter_count(), "tokenizer_model_sha256": tokenizer.hash(), "dataset": {s: dataset_provenance(args.dataset_dir, s) for s in ("train", "valid", "test_interpolate", "test_extrapolate")}, "training_path": train_path.as_posix(), "validation_path": valid_path.as_posix(), "answer_policy_version": ANSWER_POLICY_VERSION, "codebook_mode": args.codebook_mode, "codebook_index": args.codebook_index if args.codebook_mode == "fixed" else None, "eos_loss_weight": args.eos_loss_weight, "truncation_count": {"train": train.truncation_count, "valid": valid.truncation_count}, "epoch_validation": {"enabled": not args.no_epoch_validation, "sampling": "fixed_random_stratified", "seed": args.seed + 104729, "per_type": args.epoch_valid_per_type, "types": ["integer", "decimal", "fraction"], "examples": len(epoch_valid_examples), "ids_by_type": epoch_valid_ids}, "data_stats": {"train_examples": len(train), "train_sequence_tokens_per_epoch": train_sequence_tokens, "supervised_answer_tokens_per_epoch": answer_tokens, "optimizer_steps_per_epoch": len(loader), "requested_epochs": train_config.epochs, "requested_sequence_tokens": train_sequence_tokens * train_config.epochs, "requested_tokens_per_parameter": train_sequence_tokens * train_config.epochs / max(model.parameter_count(), 1)}}
    loss_history: list[dict] = []
    best_valid_answer_nll = float("inf")
    best_epoch = None
    write_json(args.output_dir / "run_manifest.json", manifest)
    step = 0
    completed_epochs = 0
    processed_sequence_tokens = 0
    if args.resume:
        checkpoint_data = torch.load(args.resume, map_location=device, weights_only=False)
        loss_history = list(checkpoint_data.get("manifest", {}).get("loss_history", []))
        best_valid_answer_nll = float(checkpoint_data.get("manifest", {}).get("best_valid_answer_nll", float("inf")))
        best_epoch = checkpoint_data.get("manifest", {}).get("best_epoch")
        model.load_state_dict(checkpoint_data["model"])
        optimizer.load_state_dict(checkpoint_data["optimizer"])
        step = int(checkpoint_data.get("step", 0))
        processed_sequence_tokens = int(checkpoint_data.get("processed_sequence_tokens", 0))
        if "completed_epochs" in checkpoint_data:
            completed_epochs = int(checkpoint_data["completed_epochs"])
        else:
            # Backward compatibility: older checkpoints only stored global step.
            # Infer fully completed epochs only when step lands exactly on an epoch boundary.
            steps_per_epoch = len(loader)
            completed_epochs = min(train_config.epochs, step // steps_per_epoch) if steps_per_epoch and step % steps_per_epoch == 0 else 0
            if completed_epochs:
                processed_sequence_tokens = train_sequence_tokens * completed_epochs
            print(json.dumps({"event": "legacy_checkpoint_epoch_inference", "step": step, "steps_per_epoch": steps_per_epoch, "inferred_completed_epochs": completed_epochs}))
        if completed_epochs >= train_config.epochs and args.max_steps is None:
            print(json.dumps({"event": "already_complete", "completed_epochs": completed_epochs, "target_epochs": train_config.epochs, "step": step}))
    max_steps_already_reached = train_config.max_steps is not None and step >= train_config.max_steps
    if max_steps_already_reached:
        completed_epochs = min(completed_epochs, train_config.epochs)
        print(json.dumps({"event": "max_steps_already_reached", "step": step, "max_steps": train_config.max_steps}, ensure_ascii=False))
    for epoch in range(completed_epochs, train_config.epochs) if not max_steps_already_reached else ():
        model.train()
        stopped_by_max_steps = False
        epoch_loss_sum = 0.0
        epoch_answer_nll_sum = 0.0
        epoch_answer_token_count = 0
        epoch_eos_loss_sum = 0.0
        epoch_example_count = 0
        epoch_batch_count = 0
        for batch in loader:
            input_ids, labels, attention = batch["input_ids"].to(device), batch["labels"].to(device), batch["attention_mask"].to(device)
            processed_sequence_tokens += int(attention.sum().item())
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention)
            loss = model.answer_loss(logits, labels)
            answer_loss_value = float(loss.item())
            eos_loss_value = 0.0
            if train_config.eos_loss_weight:
                eos_loss = model.eos_loss(logits, input_ids, batch["examples"])
                eos_loss_value = float(eos_loss.item())
                loss = loss + train_config.eos_loss_weight * eos_loss
            batch_answer_tokens = int(labels.ne(-100).sum().item())
            epoch_loss_sum += float(loss.item())
            epoch_answer_nll_sum += answer_loss_value * batch_answer_tokens
            epoch_answer_token_count += batch_answer_tokens
            epoch_eos_loss_sum += eos_loss_value * len(batch["examples"])
            epoch_example_count += len(batch["examples"])
            epoch_batch_count += 1
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm); optimizer.step(); step += 1
            if step % train_config.log_every_steps == 0: print(json.dumps({"step": step, "epoch": epoch, "loss": float(loss.item())}))
            if train_config.max_steps is not None and step >= train_config.max_steps:
                stopped_by_max_steps = True
                break
        if not stopped_by_max_steps:
            completed_epochs = epoch + 1
            print(json.dumps({"event": "epoch_complete", "epoch": completed_epochs, "target_epochs": train_config.epochs, "step": step}))
            train_answer_nll = epoch_answer_nll_sum / max(epoch_answer_token_count, 1)
            train_eos_loss = epoch_eos_loss_sum / max(epoch_example_count, 1)
            train_metrics = {"loss": train_answer_nll + train_config.eos_loss_weight * train_eos_loss, "answer_nll": train_answer_nll, "eos_loss": train_eos_loss, "answer_tokens": epoch_answer_token_count}
            valid_metrics = validation_loss(model, epoch_valid_examples, device, tokenizer.id("<PAD>"), train_config.eos_loss_weight, train_config.batch_size) if not args.no_epoch_validation else {"loss": None, "answer_nll": None, "eos_loss": None, "examples": 0, "answer_tokens": 0}
            epoch_record = {"epoch": completed_epochs, "step": step, "train": train_metrics, "valid": valid_metrics}
            loss_history.append(epoch_record)
            if valid_metrics["answer_nll"] is not None and valid_metrics["answer_nll"] < best_valid_answer_nll:
                best_valid_answer_nll = float(valid_metrics["answer_nll"])
                best_epoch = completed_epochs
            manifest["loss_history"] = loss_history
            manifest["best_valid_answer_nll"] = best_valid_answer_nll if best_epoch is not None else None
            manifest["best_epoch"] = best_epoch
            write_json(args.output_dir / "loss_history.json", {"epoch_validation": manifest["epoch_validation"], "history": loss_history})
            print(json.dumps({"event": "epoch_metrics", **epoch_record}, ensure_ascii=False))
            if best_epoch == completed_epochs:
                save_checkpoint(args.output_dir / "checkpoints" / "best.pt", model, optimizer, model_config, train_config, step, completed_epochs, processed_sequence_tokens, manifest)
        # Save after every completed epoch so a finished run is resumable without repeating work.
        save_checkpoint(args.output_dir / "checkpoints" / "last.pt", model, optimizer, model_config, train_config, step, completed_epochs, processed_sequence_tokens, manifest)
        if stopped_by_max_steps:
            break
    checkpoint = args.output_dir / "checkpoints" / "last.pt"
    save_checkpoint(checkpoint, model, optimizer, model_config, train_config, step, completed_epochs, processed_sequence_tokens, manifest)
    training_complete = completed_epochs >= train_config.epochs if train_config.max_steps is None else (step >= train_config.max_steps)
    print(json.dumps({"event": "training_complete", "completed_epochs": completed_epochs, "target_epochs": train_config.epochs, "step": step, "training_complete": training_complete}, ensure_ascii=False))
    if args.skip_eval or not args.eval_after_training:
        metrics = {"skipped": True, "reason": "--skip-eval" if args.skip_eval else "evaluation is a separate command by default"}
    else:
        eval_dataset = valid if args.eval_limit is None else Subset(valid, range(min(args.eval_limit, len(valid))))
        metrics = evaluate_dataset(model, eval_dataset, tokenizer, device, max_new_tokens=args.max_new_tokens)
        write_json(args.output_dir / "valid_metrics.json", metrics)
    manifest["loss_history"] = loss_history; manifest["best_valid_answer_nll"] = best_valid_answer_nll if best_epoch is not None else None; manifest["best_epoch"] = best_epoch; manifest["best_checkpoint_path"] = (args.output_dir / "checkpoints" / "best.pt").as_posix() if best_epoch is not None else None; manifest["checkpoint_path"] = checkpoint.as_posix(); manifest["completed_steps"] = step; manifest["completed_epochs"] = completed_epochs; manifest["training_complete"] = training_complete; manifest["evaluation_after_training"] = bool(args.eval_after_training and not args.skip_eval)
    manifest["data_stats"]["completed_sequence_tokens"] = processed_sequence_tokens
    manifest["data_stats"]["completed_tokens_per_parameter"] = processed_sequence_tokens / max(model.parameter_count(), 1)
    manifest["checkpoint_bytes"] = checkpoint.stat().st_size if checkpoint.is_file() else None
    write_json(args.output_dir / "run_manifest.json", manifest)
    parameter_count = model.parameter_count()
    weight_bytes = parameter_count * torch.tensor([], dtype=torch.float32).element_size()
    print(json.dumps({"output_dir": str(args.output_dir), "step": step, "parameter_count": parameter_count, "parameter_count_millions": round(parameter_count / 1_000_000, 3), "model_weight_bytes_estimate": weight_bytes, "checkpoint_bytes": manifest.get("checkpoint_bytes"), "valid_exact_match": metrics.get("exact_match")}, ensure_ascii=False))


def build_parser():
    p = argparse.ArgumentParser(description="Train the answer-only Transformer pilot")
    p.add_argument("--dataset-dir", type=Path, required=True); p.add_argument("--tokenizer-dir", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-path", type=Path); p.add_argument("--valid-path", type=Path)
    p.add_argument("--max-seq-len", type=int, default=128); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--learning-rate", type=float, default=3e-4); p.add_argument("--epochs", type=int, default=1); p.add_argument("--max-steps", type=int); p.add_argument("--eos-loss-weight", type=float, default=0.0, help="optional EOS protocol loss; 0 preserves the answer-only pilot contract"); p.add_argument("--seed", type=int, default=1234); p.add_argument("--device", default="auto")
    p.add_argument("--eval-limit", type=int); p.add_argument("--max-new-tokens", type=int, default=64); p.add_argument("--eval-after-training", action="store_true", help="run slow full validation in this process; otherwise use evaluate.py separately"); p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--epoch-valid-per-type", type=int, default=100, help="fixed random validation examples per integer/decimal/fraction type after each epoch (0 disables sampling)"); p.add_argument("--no-epoch-validation", action="store_true", help="disable the fast teacher-forced validation loss after each epoch")
    p.add_argument("--resume", type=Path)
    p.add_argument("--codebook-mode", choices=("none", "fixed", "random"), default="none")
    p.add_argument("--codebook-index", type=int, default=0)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
