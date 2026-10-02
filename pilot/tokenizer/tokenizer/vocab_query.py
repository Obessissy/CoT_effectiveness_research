"""Read-only lookup table for a trained tokenizer (no sentencepiece required).

Reads `tokenizer.vocab` directly, so it works even when the model file cannot be
opened, and it never modifies any artifact.

ID convention: the file has no ID column, so ID = line_number - 1.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "v1_sp_bpe_8k_seed1234"
BYTE_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def load_vocab(model_dir: Path) -> list[tuple[str, float]]:
    vocab_path = model_dir / "tokenizer.vocab"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"missing vocab file: {vocab_path}")
    entries: list[tuple[str, float]] = []
    for line_number, line in enumerate(vocab_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            piece, score = line.rsplit("\t", 1)
            entries.append((piece, float(score)))
        except ValueError as exc:
            raise ValueError(f"{vocab_path}:{line_number}: expected 'piece<TAB>score'") from exc
    return entries


def show_escaped(piece: str) -> str:
    """Render the piece with visible markers for whitespace-only characters."""
    return piece.replace("\u2581", "[SP]").replace("\t", "\\t")


def describe_layout(vocab: list[tuple[str, float]]) -> dict[str, object]:
    groups: list[dict[str, object]] = []
    start = 0
    for index in range(1, len(vocab) + 1):
        if index == len(vocab) or kind(vocab[index][0]) != kind(vocab[start][0]):
            groups.append({
                "kind": kind(vocab[start][0]),
                "first_id": start,
                "last_id": index - 1,
                "count": index - start,
                "example": show_escaped(vocab[start][0]),
            })
            start = index
    return {"piece_count": len(vocab), "groups": groups}


def kind(piece: str) -> str:
    if piece == "<unk>":
        return "unk"
    if piece.startswith("<Z_"):
        return "codebook_reserved"
    if BYTE_RE.match(piece):
        return "byte_fallback"
    if piece.startswith("<") and piece.endswith(">"):
        return "special"
    return "learned"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--id", type=int, nargs="+", help="look up one or more token IDs")
    mode.add_argument("--piece", nargs="+", help="look up the ID of one or more exact pieces")
    mode.add_argument("--search", help="list every piece containing this substring")
    mode.add_argument("--range", type=int, nargs=2, metavar=("START", "END"), help="list an ID range")
    mode.add_argument("--layout", action="store_true", help="summarize the ID regions")
    parser.add_argument("--limit", type=int, default=100, help="max rows for --search")
    parser.add_argument("--raw", action="store_true", help="do not escape the whitespace marker")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    vocab = load_vocab(args.model_dir.resolve())
    render = (lambda piece: piece) if args.raw else show_escaped
    rows: list[dict[str, object]] = []
    truncated = False

    if args.layout:
        payload = describe_layout(vocab)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.id is not None:
        for token_id in args.id:
            if not 0 <= token_id < len(vocab):
                rows.append({"id": token_id, "error": "out of range"})
                continue
            piece, score = vocab[token_id]
            rows.append({"id": token_id, "piece": render(piece), "score": score, "kind": kind(piece)})
    elif args.piece is not None:
        for wanted in args.piece:
            search_for = wanted if args.raw else wanted.replace("[SP]", "\u2581")
            matches = [i for i, (piece, _) in enumerate(vocab) if piece == search_for]
            if matches:
                for token_id in matches:
                    piece, score = vocab[token_id]
                    rows.append({"id": token_id, "piece": render(piece), "score": score, "kind": kind(piece)})
            else:
                rows.append({"piece": render(wanted), "error": "not in vocabulary",
                             "hint": "use the exact string, e.g. <Z_017>; the space marker is [SP] (U+2581)"})
    elif args.search is not None:
        needle = args.search if args.raw else args.search.replace("[SP]", "\u2581")
        for token_id, (piece, score) in enumerate(vocab):
            if needle in piece:
                rows.append({"id": token_id, "piece": render(piece), "score": score, "kind": kind(piece)})
                if len(rows) > args.limit:
                    truncated = True
                    break
    elif args.range is not None:
        start, end = args.range
        for token_id in range(max(start, 0), min(end, len(vocab) - 1) + 1):
            piece, score = vocab[token_id]
            rows.append({"id": token_id, "piece": render(piece), "score": score, "kind": kind(piece)})

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    for row in rows:
        if "error" in row:
            print(f"  !! {row.get('id', row.get('piece'))}: {row['error']}  ({row.get('hint', '')})")
        else:
            print(f"  id={row['id']:<6} score={row['score']:<10.4g} kind={row['kind']:<18} piece={row['piece']}")
    if truncated:
        print(f"  ... truncated at {args.limit} rows; raise --limit or narrow the search")


if __name__ == "__main__":
    main()
