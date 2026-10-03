from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import torch

from .data import MathDataset, collate_examples, encode_record, normalize_answer_text, serialize_example
from .config import ModelConfig
from .model import CausalTransformer, generate
from .tokenizer import FrozenTokenizer
from .verifier import ANSWER_POLICY_VERSION, verify_answer


EVAL_BATCH_SIZE = 32


@torch.no_grad()
def teacher_forced_score(model: CausalTransformer, example, device: torch.device) -> float:
    inputs = torch.tensor(example.input_ids, dtype=torch.long, device=device)[None, :]
    logits = model(inputs)[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    values = [log_probs[i - 1, example.input_ids[i]].item() for i in range(example.answer_start, example.answer_end)]
    return float(sum(values))


@torch.no_grad()
def _teacher_forced_scores_batched(
    model: CausalTransformer, examples: list[Any], device: torch.device
) -> dict[int, float]:
    """Score examples in length groups, avoiding one Transformer call per row."""
    scores: dict[int, float] = {}
    groups: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[len(example.input_ids)].append((index, example))
    for length, items in groups.items():
        for offset in range(0, len(items), EVAL_BATCH_SIZE):
            chunk = items[offset : offset + EVAL_BATCH_SIZE]
            inputs = torch.tensor([item.input_ids for _, item in chunk], dtype=torch.long, device=device)
            logits = model(inputs)
            log_probs = torch.log_softmax(logits, dim=-1)
            for row, (index, example) in enumerate(chunk):
                targets = torch.tensor(example.input_ids[example.answer_start:example.answer_end], device=device)
                positions = torch.arange(example.answer_start - 1, example.answer_end - 1, device=device)
                scores[index] = float(log_probs[row, positions, targets].sum().item())
    return scores


@torch.no_grad()
def _generate_grouped(
    model: CausalTransformer,
    examples: list[Any],
    device: torch.device,
    eos_id: int,
    max_new_tokens: int,
) -> dict[int, tuple[list[int], bool]]:
    """Generate in groups with equal prefix length for a large speedup."""
    outputs: dict[int, tuple[list[int], bool]] = {}
    groups: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[example.answer_start].append((index, example))
    for prefix_length, items in groups.items():
        for offset in range(0, len(items), EVAL_BATCH_SIZE):
            chunk = items[offset : offset + EVAL_BATCH_SIZE]
            prefix = torch.tensor(
                [item.input_ids[:prefix_length] for _, item in chunk], dtype=torch.long, device=device
            )
            sequences, finished = generate(model, prefix, eos_id, max_new_tokens)
            for row, (index, _) in enumerate(chunk):
                outputs[index] = (sequences[row].tolist(), bool(finished[row].item()))
    return outputs


def _prediction_from_sequence(tokenizer: FrozenTokenizer, sequence: list[int]) -> str:
    decoded = tokenizer.decode(sequence)
    return decoded.split("<ANSWER>", 1)[-1].split("<EOS>", 1)[0].strip() if "<ANSWER>" in decoded else ""


@torch.no_grad()
def evaluate_dataset(model: CausalTransformer, dataset: MathDataset, tokenizer: FrozenTokenizer, device: torch.device,
                     max_new_tokens: int = 64) -> dict[str, Any]:
    model.eval()
    examples = list(dataset)
    total_loss, token_count = 0.0, 0
    exact_correct, generated = 0, 0
    statuses = Counter()
    by_type = defaultdict(lambda: {"count": 0, "exact_correct": 0, "verifier": Counter()})
    records = []
    eos_id = tokenizer.id("<EOS>")
    teacher_scores = _teacher_forced_scores_batched(model, examples, device)
    generations = _generate_grouped(model, examples, device, eos_id, max_new_tokens)
    for index, example in enumerate(examples):
        total_loss -= teacher_scores[index]
        token_count += example.answer_end - example.answer_start
        sequence, finished = generations[index]
        prediction = _prediction_from_sequence(tokenizer, sequence)
        result = verify_answer(example.record, prediction)
        typ = str(example.record.get("answer_type", "unknown"))
        by_type[typ]["count"] += 1
        by_type[typ]["verifier"][result["status"]] += 1
        exact = normalize_answer_text(prediction) == normalize_answer_text(example.record["answer"])
        exact_correct += int(exact)
        by_type[typ]["exact_correct"] += int(exact)
        generated += 1
        statuses[result["status"]] += 1
        records.append({"id": example.record.get("id"), "prediction": prediction, "gold": example.record["answer"], "verifier": result, "finished": finished})
    return {
        "examples": len(examples), "answer_tokens": token_count,
        "answer_nll": total_loss / max(token_count, 1),
        "answer_log_likelihood": -total_loss,
        "exact_match": exact_correct / max(generated, 1),
        "verifier_status": dict(statuses), "by_answer_type": {k: {"count": v["count"], "exact_correct": v["exact_correct"], "verifier": dict(v["verifier"])} for k, v in by_type.items()},
        "generation_failure_rate": sum(1 for r in records if not r["finished"]) / max(generated, 1),
        "answer_policy_version": ANSWER_POLICY_VERSION,
        "records": records,
    }


def oracle_scores(model: CausalTransformer, record: dict[str, Any], tokenizer: FrozenTokenizer, max_seq_len: int, device: torch.device, active_k: int = 16) -> list[dict[str, Any]]:
    scores = []
    for index in range(active_k):
        token = tokenizer.codebook_token(index)
        example = encode_record(record, tokenizer, max_seq_len, token)
        scores.append({"codebook_index": index, "token": token, "answer_log_likelihood": teacher_forced_score(model, example, device)})
    return scores


@torch.no_grad()
def evaluate_oracle(
    model: CausalTransformer,
    records: list[dict[str, Any]],
    tokenizer: FrozenTokenizer,
    max_seq_len: int,
    device: torch.device,
    active_k: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Evaluate the gold-answer oracle separately from ordinary generation."""
    candidate_rows: list[tuple[int, int, Any]] = []
    for row_index, record in enumerate(records):
        for codebook_index in range(active_k):
            token = tokenizer.codebook_token(codebook_index)
            candidate_rows.append((row_index, codebook_index, encode_record(record, tokenizer, max_seq_len, token)))
    candidate_scores = _teacher_forced_scores_batched(model, [item[2] for item in candidate_rows], device)
    all_scores: list[list[dict[str, Any]]] = [[] for _ in records]
    for candidate_index, (row_index, codebook_index, _) in enumerate(candidate_rows):
        all_scores[row_index].append({
            "codebook_index": codebook_index,
            "token": tokenizer.codebook_token(codebook_index),
            "answer_log_likelihood": candidate_scores[candidate_index],
        })
    candidates: list[tuple[int, int, Any]] = []
    best: list[dict[str, Any]] = []
    for row_index, record in enumerate(records):
        winner = max(all_scores[row_index], key=lambda item: item["answer_log_likelihood"])
        best.append({"id": record.get("id"), "scores": all_scores[row_index], "selected": winner})
        example = encode_record(record, tokenizer, max_seq_len, winner["token"])
        candidates.append((row_index, winner["codebook_index"], example))

    generated = _generate_grouped(model, [item[2] for item in candidates], device, tokenizer.id("<EOS>"), max_new_tokens)
    exact_correct = 0
    statuses = Counter()
    oracle_records = []
    for candidate_index, (row_index, _, example) in enumerate(candidates):
        sequence, finished = generated[candidate_index]
        prediction = _prediction_from_sequence(tokenizer, sequence)
        result = verify_answer(example.record, prediction)
        exact = normalize_answer_text(prediction) == normalize_answer_text(example.record["answer"])
        exact_correct += int(exact)
        statuses[result["status"]] += 1
        best[candidate_index].update({"prediction": prediction, "gold": example.record["answer"], "verifier": result, "finished": finished})
        oracle_records.append(best[candidate_index])
    gains = [item["selected"]["answer_log_likelihood"] for item in oracle_records]
    return {
        "active_k": active_k,
        "examples": len(records),
        "exact_match": exact_correct / max(len(records), 1),
        "verifier_status": dict(statuses),
        "generation_failure_rate": sum(not item["finished"] for item in oracle_records) / max(len(records), 1),
        "mean_best_answer_log_likelihood": sum(gains) / max(len(gains), 1),
        "records": oracle_records,
        "warning": "oracle selection uses the gold answer and is not deployable; codebook tokens must be trained for a meaningful intervention",
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a pilot checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", action="append", choices=("valid", "test_interpolate", "test_extrapolate"), help="split to score; repeat to score multiple splits into one JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--codebook-mode", choices=("none", "fixed", "random"), default="none")
    parser.add_argument("--codebook-index", type=int, default=0)
    parser.add_argument("--oracle-k", type=int, choices=(1, 16, 32))
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state["model_config"]
    model = CausalTransformer(ModelConfig(**{k: v for k, v in config.items() if k in ModelConfig.__dataclass_fields__})).to(device)
    model.load_state_dict(state["model"])
    tokenizer = FrozenTokenizer(args.tokenizer_dir)
    splits = args.split or ["valid"]
    eos_loss_weight = float(state.get("train_config", {}).get("eos_loss_weight", 0.0))
    split_results = {}
    for split in splits:
        dataset = MathDataset(args.dataset_dir / f"{split}.jsonl", tokenizer, model.config.max_seq_len, codebook_mode=args.codebook_mode, codebook_index=args.codebook_index)
        if args.limit is not None:
            dataset._encoded = dataset._encoded[:args.limit]
            dataset.records = dataset.records[:args.limit]
        result = evaluate_dataset(model, dataset, tokenizer, device, args.max_new_tokens)
        result["split"] = split
        result["codebook_mode"] = args.codebook_mode
        result["codebook_index"] = args.codebook_index if args.codebook_mode == "fixed" else None
        result["checkpoint"] = str(args.checkpoint)
        result["eos_loss_weight"] = eos_loss_weight
        if eos_loss_weight == 0:
            result["generation_warning"] = "EOS had no training loss in this run; generation failure rate may reflect the answer-only objective rather than answer competence"
        if args.oracle_k is not None:
            result["oracle"] = evaluate_oracle(
                model, dataset.records, tokenizer, model.config.max_seq_len, device,
                args.oracle_k, args.max_new_tokens,
            )
            result["oracle_checkpoint_codebook_mode"] = state.get("manifest", {}).get("codebook_mode")
        split_results[split] = result
    result = split_results[splits[0]] if len(splits) == 1 else {"checkpoint": str(args.checkpoint), "splits": split_results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    summary = {split: {"examples": item["examples"], "exact_match": item["exact_match"]} for split, item in split_results.items()}
    print(json.dumps({"splits": summary, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
