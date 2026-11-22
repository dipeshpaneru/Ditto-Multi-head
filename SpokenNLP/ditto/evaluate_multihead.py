"""Select a normalized multi-head Ditto pooler on STS-B development data."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent / "SentEval"))
from senteval.sts import STSBenchmarkEval
from senteval.utils import cosine

# Table 4 uses one-based indexing.
HEADS = ((0, 9), (1, 11), (10, 10), (0, 6))


@torch.inference_mode()
def encode(sentences, heads, model, tokenizer, device, *, return_raw=False):
    """Return one normalized Ditto embedding per sentence and attention head."""
    batches = []
    raw_batches = []
    for start in range(0, len(sentences), 128):
        batch = tokenizer(
            [" ".join(tokens) for tokens in sentences[start:start + 128]],
            padding=True,
            return_tensors="pt",
        ).to(device)
        outputs = model(**batch, output_hidden_states=True, output_attentions=True)
        states = (outputs.hidden_states[0] + outputs.hidden_states[-1]) / 2
        weights = torch.stack([
            outputs.attentions[layer][:, head].diagonal(dim1=-2, dim2=-1)
            for layer, head in heads
        ], dim=1)
        weights = weights * batch.attention_mask[:, None, :]
        if return_raw:
            raw_batches.append(torch.bmm(weights, states).cpu().numpy())
        total = weights.sum(dim=-1, keepdim=True)
        uniform = batch.attention_mask / batch.attention_mask.sum(dim=-1, keepdim=True)
        weights = torch.where(total > 0, weights / total.clamp_min(torch.finfo(weights.dtype).tiny),
                              uniform[:, None, :])
        batches.append(torch.bmm(weights, states).cpu().numpy())
    normalized = np.concatenate(batches)
    if return_raw:
        return normalized, np.concatenate(raw_batches)
    return normalized


def predictions(left, right, indices):
    # Averaging these embeddings is equivalent to averaging the token weights.
    left = left[:, indices].mean(axis=1)
    right = right[:, indices].mean(axis=1)
    return np.array([np.nan_to_num(cosine(a, b)) for a, b in zip(left, right)])


def score(left, right, gold, indices):
    return float(spearmanr(predictions(left, right, indices), gold).statistic * 100)


def paired_bootstrap(single, multi, gold, replicates=10000, seed=42):
    rng = np.random.default_rng(seed)
    gold = np.asarray(gold)
    deltas = []
    for _ in range(replicates):
        sample = rng.integers(len(gold), size=len(gold))
        a = spearmanr(single[sample], gold[sample]).statistic
        b = spearmanr(multi[sample], gold[sample]).statistic
        deltas.append((b - a) * 100)
    return {
        "replicates": replicates,
        "seed": seed,
        "confidence_interval_95": np.percentile(deltas, [2.5, 97.5]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", action="store_true",
                        help="Estimate a paired confidence interval on the test score difference")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        args.model, local_files_only=True, attn_implementation="eager"
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    data_path = Path(__file__).resolve().parent / "SentEval/data/downstream/STS/STSBenchmark"
    dataset = STSBenchmarkEval(str(data_path))

    left, right, gold = dataset.data["dev"]
    print(f"Development evaluation on {len(gold)} pairs ({device})", flush=True)
    dev_left = encode(left, HEADS, model, tokenizer, device)
    dev_right = encode(right, HEADS, model, tokenizer, device)
    singles = [score(dev_left, dev_right, gold, [i]) for i in range(len(HEADS))]
    ranking = sorted(range(len(HEADS)), key=lambda i: (-singles[i], i))
    candidates = {k: score(dev_left, dev_right, gold, ranking[:k]) for k in (2, 3, 4)}
    best_k = max(candidates, key=lambda k: (candidates[k], -k))
    selected = [HEADS[i] for i in ranking[:best_k]]
    print(f"Development single heads: {list(zip(HEADS, singles))}", flush=True)
    print(f"Development ensembles: {candidates}; selected {selected}", flush=True)

    left, right, gold = dataset.data["test"]
    print(f"Test evaluation on {len(gold)} pairs", flush=True)
    test_heads = list(dict.fromkeys([HEADS[0], *selected]))
    test_left = encode(left, test_heads, model, tokenizer, device)
    test_right = encode(right, test_heads, model, tokenizer, device)
    baseline = score(test_left, test_right, gold, [0])
    multihead = score(test_left, test_right, gold, [test_heads.index(h) for h in selected])
    result = {
        "metric": "Spearman correlation × 100",
        "head_indexing": "zero-based layer, head",
        "development_single_heads": [{"head": head, "score": value}
                                     for head, value in zip(HEADS, singles)],
        "development_ensembles_by_k": candidates,
        "selected_heads": selected,
        "test_single_head": baseline,
        "test_multihead": multihead,
        "test_delta": multihead - baseline,
    }
    if args.bootstrap:
        single = predictions(test_left, test_right, [0])
        multi = predictions(test_left, test_right, [test_heads.index(h) for h in selected])
        result["paired_bootstrap"] = paired_bootstrap(single, multi, gold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
