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


@torch.no_grad()
def teacher_forced_score(model: CausalTransformer, example, device: torch.device) -> float:
    inputs = torch.tensor(example.input_ids, dtype=torch.long, device=device)[None, :]
    logits = model(inputs)[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    values = [log_probs[i - 1, example.input_ids[i]].item() for i in range(example.answer_start, example.answer_end)]
    return float(sum(values))


@torch.no_grad()
def evaluate_dataset(model: CausalTransformer, dataset: MathDataset, tokenizer: FrozenTokenizer, device: torch.device,
                     max_new_tokens: int = 64) -> dict[str, Any]:
    model.eval()
    total_loss, token_count = 0.0, 0
    exact_correct, generated = 0, 0
    statuses = Counter()
    by_type = defaultdict(lambda: {"count": 0, "exact_correct": 0, "verifier": Counter()})
    records = []
    answer_id, eos_id = tokenizer.id("<ANSWER>"), tokenizer.id("<EOS>")
    for example in dataset:
        inputs = torch.tensor(example.input_ids, dtype=torch.long, device=device)[None, :]
        logits = model(inputs)[0]
        for i in range(example.answer_start, example.answer_end):
            total_loss -= float(torch.log_softmax(logits[i - 1], dim=-1)[example.input_ids[i]])
            token_count += 1
        prefix = torch.tensor(example.input_ids[:example.answer_start], dtype=torch.long, device=device)[None, :]
        sequence, finished = generate(model, prefix, eos_id, max_new_tokens)
        decoded = tokenizer.decode(sequence[0].tolist())
        prediction = decoded.split("<ANSWER>", 1)[-1].split("<EOS>", 1)[0].strip() if "<ANSWER>" in decoded else ""
        result = verify_answer(example.record, prediction)
        typ = str(example.record.get("answer_type", "unknown"))
        by_type[typ]["count"] += 1
        by_type[typ]["verifier"][result["status"]] += 1
        exact = normalize_answer_text(prediction) == normalize_answer_text(example.record["answer"])
        exact_correct += int(exact)
        by_type[typ]["exact_correct"] += int(exact)
        generated += 1
        statuses[result["status"]] += 1
        records.append({"id": example.record.get("id"), "prediction": prediction, "gold": example.record["answer"], "verifier": result, "finished": bool(finished[0])})
    return {
        "examples": len(dataset), "answer_tokens": token_count,
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


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a pilot checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test_interpolate", "test_extrapolate"), default="valid")
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
    dataset = MathDataset(args.dataset_dir / f"{args.split}.jsonl", tokenizer, model.config.max_seq_len, codebook_mode=args.codebook_mode, codebook_index=args.codebook_index)
    if args.limit is not None:
        dataset._encoded = dataset._encoded[:args.limit]
        dataset.records = dataset.records[:args.limit]
    result = evaluate_dataset(model, dataset, tokenizer, device, args.max_new_tokens)
    result["split"] = args.split
    result["codebook_mode"] = args.codebook_mode
    result["codebook_index"] = args.codebook_index if args.codebook_mode == "fixed" else None
    result["checkpoint"] = str(args.checkpoint)
    if args.oracle_k is not None:
        result["oracle"] = [{"id": row.get("id"), "scores": oracle_scores(model, row, tokenizer, model.config.max_seq_len, device, args.oracle_k)} for row in dataset.records]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"split": args.split, "examples": result["examples"], "exact_match": result["exact_match"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
