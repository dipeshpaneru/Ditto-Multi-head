"""Tune weighted Ditto on STS-B development and evaluate on SICK-R test."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch
from transformers import AutoModel, AutoTokenizer

from evaluate_multihead import STSBenchmarkEval, cosine, encode, paired_bootstrap
from senteval.sts import SICKRelatednessEval

HEADS = ((0, 9), (1, 11), (10, 10))
EQUAL = (1/3, 1/3, 1/3)
WEIGHT_GRID = (EQUAL, (0.50, 0.25, 0.25), (0.25, 0.50, 0.25), (0.25, 0.25, 0.50))


def predict(left, right, weights):
    left = np.einsum("h,nhd->nd", np.asarray(weights, dtype=left.dtype), left)
    right = np.einsum("h,nhd->nd", np.asarray(weights, dtype=right.dtype), right)
    return np.array([np.nan_to_num(cosine(a, b)) for a, b in zip(left, right)])


def score(predicted, gold):
    return float(spearmanr(predicted, gold).statistic * 100)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        args.model, local_files_only=True, attn_implementation="eager"
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    data = Path(__file__).resolve().parent / "SentEval/data/downstream"

    left, right, gold = STSBenchmarkEval(str(data / "STS/STSBenchmark")).data["dev"]
    print(f"Tuning weights on {len(gold)} STS-B development pairs", flush=True)
    dev_left = encode(left, HEADS, model, tokenizer, device)
    dev_right = encode(right, HEADS, model, tokenizer, device)
    dev_scores = [score(predict(dev_left, dev_right, w), gold) for w in WEIGHT_GRID]
    best = int(np.argmax(dev_scores))
    weights = WEIGHT_GRID[best]
    print(f"Development scores: {list(zip(WEIGHT_GRID, dev_scores))}", flush=True)
    print(f"Frozen weights: {weights}", flush=True)

    left, right, gold = SICKRelatednessEval(str(data / "SICK")).data["test"]
    print(f"Evaluating on {len(gold)} SICK-R test pairs", flush=True)
    norm_left, raw_left = encode(left, HEADS, model, tokenizer, device, return_raw=True)
    norm_right, raw_right = encode(right, HEADS, model, tokenizer, device, return_raw=True)
    predictions = {
        "single": predict(raw_left, raw_right, (1, 0, 0)),
        "equal_normalized": predict(norm_left, norm_right, EQUAL),
        "selected_normalized": predict(norm_left, norm_right, weights),
        "equal_raw": predict(raw_left, raw_right, EQUAL),
    }
    comparisons = (
        ("selected_normalized", "equal_normalized"),
        ("equal_normalized", "equal_raw"),
        ("equal_normalized", "single"),
    )
    scores = {name: score(predicted, gold) for name, predicted in predictions.items()}
    result = {
        "metric": "Spearman correlation × 100",
        "heads_zero_based": HEADS,
        "tuning_split": "STS-B development",
        "development_grid": [{"weights": w, "score": s} for w, s in zip(WEIGHT_GRID, dev_scores)],
        "selected_weights": weights,
        "test_split": "SICK-R test",
        "test_pairs": len(gold),
        "test_scores": scores,
        "paired_comparisons": {
            f"{a}_minus_{b}": {
                "delta": scores[a] - scores[b],
                **paired_bootstrap(predictions[b], predictions[a], gold),
            }
            for a, b in comparisons
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
