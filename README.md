# Ditto reproduction and multi-head evaluation

This project reproduces the original Ditto comparison on the Semantic Textual Similarity Benchmark (STS-B), then evaluates a multi-head extension using the same pretrained BERT checkpoint. Neither run requires training.

Run the commands below from the project root, the folder containing this README.

## 1. Set up the environment and inputs

The project was verified on an Apple Silicon Mac with Python 3.12. If the project already has a working `.venv`, use it. To create the environment on an Apple Silicon Mac with Python 3.12 installed, run:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos-arm64.txt
```

Both evaluations use these local inputs:

- **BERT checkpoint** in `models/bert-base-uncased/`, including `config.json`, `vocab.txt`, and `model.safetensors`. Git ignores the weights file, so it may be missing from a fresh checkout.
- **STS-B data** in `SpokenNLP/ditto/SentEval/data/downstream/STS/STSBenchmark/`, including `sts-dev.csv` and `sts-test.csv`.

If any inputs are missing, see [the reproduction notes](meta/REPRODUCE.txt) for the checkpoint and SentEval download and extraction instructions.

## 2. Run the original STS-B benchmark

```bash
DITTO_PYTHON="$PWD/.venv/bin/python" bash meta/reproduce.sh stsb
```

The benchmark prints results in the terminal and writes a log for each pooling method. Read the `STSBenchmark` column. The evaluator's `Avg.` includes zeros for skipped tasks.

| Pooling method | Expected local STS-B test score | Output |
| --- | ---: | --- |
| First–last average pooling | 53.88 | `results/stsb-avg_first_last.log` |
| First–last single-head Ditto | 66.76 | `results/stsb-att_first_last.log` |

## 3. Run the basic multi-head evaluation

```bash
OMP_NUM_THREADS=4 .venv/bin/python SpokenNLP/ditto/evaluate_multihead.py \
    --model models/bert-base-uncased \
    --output results/stsb-multihead.json
```

The script ranks the candidate heads on the STS-B development split, compares ensembles of two, three, and four heads, and evaluates the selected ensemble against single-head Ditto on the held-out test split. It prints the results and saves them to `results/stsb-multihead.json`.

| JSON field | Expected value, rounded |
| --- | ---: |
| `test_single_head` | 66.76 |
| `test_multihead` | 69.59 |
| `test_delta` | +2.83 points |

Both runs report Spearman correlation multiplied by 100. Rerunning these commands overwrites the output files listed above.
