"""Add heads to fixed Ditto pooling using STS-B dev and check on STS14."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch
from transformers import AutoModel, AutoTokenizer

from evaluate_multihead import STSBenchmarkEval, encode, paired_bootstrap, predictions, score
from senteval.sts import STS14Eval

BASE = ((0, 9), (1, 11), (10, 10))


def pair_key(left, right):
    return tuple(sorted((" ".join(left).casefold(), " ".join(right).casefold())))


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
    data = Path(__file__).resolve().parent / "SentEval/data/downstream/STS"
    left, right, gold = STSBenchmarkEval(str(data / "STSBenchmark")).data["dev"]
    dev_pairs = {pair_key(a, b) for a, b in zip(left, right)}

    # Screen the remaining heads in the same three layers as the baseline.
    heads = BASE + tuple((layer, head) for layer, _ in BASE
                         for head in range(model.config.num_attention_heads)
                         if (layer, head) not in BASE)
    print(f"Screening {len(heads) - len(BASE)} extra heads on STS-B development", flush=True)
    dl = encode(left, heads, model, tokenizer, device)
    dr = encode(right, heads, model, tokenizer, device)
    chosen = list(range(len(BASE)))
    selections = {3: chosen.copy()}
    dev_scores = {3: score(dl, dr, gold, chosen)}
    for k in (4, 5):
        best_score, index = max(
            (score(dl, dr, gold, [*chosen, i]), -i)
            for i in range(len(heads)) if i not in chosen
        )
        chosen.append(-index)
        selections[k] = chosen.copy()
        dev_scores[k] = best_score
    frozen = [heads[i] for i in chosen]
    print(f"Development scores: {dev_scores}; frozen heads: {frozen}", flush=True)

    sts14 = STS14Eval(str(data / "STS14-en-test"))
    pooled = {k: [] for k in selections}
    test_gold = []
    skipped = 0
    for domain, (left, right, gold) in sts14.data.items():
        pairs = [(a, b, y) for a, b, y in zip(left, right, gold) if pair_key(a, b) not in dev_pairs]
        skipped += len(gold) - len(pairs)
        print(f"Evaluating {domain}: {len(pairs)} pairs", flush=True)
        left, right, gold = zip(*pairs)
        tl = encode(left, frozen, model, tokenizer, device)
        tr = encode(right, frozen, model, tokenizer, device)
        test_gold.extend(gold)
        for k in pooled:
            pooled[k].extend(predictions(tl, tr, list(range(k))))
    pooled = {k: np.array(values) for k, values in pooled.items()}
    test_scores = {k: float(spearmanr(values, test_gold).statistic * 100)
                   for k, values in pooled.items()}
    result = {
        "metric": "Pooled Spearman correlation × 100",
        "heads_zero_based_by_k": {k: [heads[i] for i in indices]
                                  for k, indices in selections.items()},
        "development_split": "STS-B development",
        "development_scores": dev_scores,
        "test_split": "STS14 excluding exact STS-B development pairs (either order)",
        "test_pairs": len(test_gold),
        "excluded_pairs": skipped,
        "test_scores": test_scores,
        "comparisons_vs_three": {
            k: {"delta": test_scores[k] - test_scores[3],
                **paired_bootstrap(pooled[3], pooled[k], test_gold)}
            for k in (4, 5)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = json.dumps(result, indent=2, ensure_ascii=False)
    args.output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
