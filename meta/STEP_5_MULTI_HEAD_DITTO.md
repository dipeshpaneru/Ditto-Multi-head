# Step 5. Improving Ditto with multi-head attention pooling

## Motivation

Ditto builds a sentence embedding by weighting BERT token representations with the diagonal of a single attention head. Each diagonal entry measures how much a token attends to itself in that head. Averaging several heads could combine complementary estimates of token importance and reduce the influence of any one head. The experiment tested whether this change improves agreement with human sentence similarity judgments.

## Implementation

The extension reuses attention heads from pretrained BERT-base-uncased. It keeps Ditto’s token representation, the mean of the input embedding and final hidden state. For each head, the code extracts its diagonal self-attention weights, masks padding, and normalizes the weights to sum to one. A weighted sum produces one sentence vector per head. The equal-weight mean of these vectors is the final sentence embedding. This is equivalent to averaging the normalized token weights before pooling. The extension uses one BERT forward pass per batch and introduces no trainable parameters.

The pooling calculation is shown below. [Evaluation script](../SpokenNLP/ditto/evaluate_multihead.py)

```python
states = (outputs.hidden_states[0] + outputs.hidden_states[-1]) / 2
weights = torch.stack([
    outputs.attentions[layer][:, head].diagonal(dim1=-2, dim2=-1)
    for layer, head in heads
], dim=1)
weights = weights * pooling_mask[:, None, :]
total = weights.sum(dim=-1, keepdim=True)
uniform = pooling_mask / pooling_mask.sum(dim=-1, keepdim=True)
weights = torch.where(
    total > 0,
    weights / total.clamp_min(torch.finfo(weights.dtype).tiny),
    uniform[:, None, :],
)
batches.append(torch.bmm(weights, states).cpu().numpy())
```

The evaluation ranked the four candidate heads from Ditto’s Table 4 by their individual STS-B development scores and compared ensembles of the top two, three, and four heads. For each configuration, it averaged the selected head embeddings and computed cosine similarity between the resulting sentence vectors.

```python
# Inside predictions(), indices identifies the selected heads.
left = left[:, indices].mean(axis=1)
right = right[:, indices].mean(axis=1)
return np.array([np.nan_to_num(cosine(a, b)) for a, b in zip(left, right)])
```

## Evaluation against the reproduced baseline

Scores are Spearman rank correlation with human similarity judgments, multiplied by 100. Head identifiers use zero-based (layer, head) indexing.

| Pooling configuration | STS-B development | STS-B test |
| --- | ---: | ---: |
| Reproduced single-head Ditto (0, 9) | 74.56 | 66.76 |
| Top two heads | 75.47 | 69.07 |
| Top three heads | **75.74** | **69.59** |
| Top four heads | 75.58 | — |

The development split selected heads (0, 9), (1, 11), and (10, 10). On 1,379 held-out STS-B test pairs, the three-head ensemble improved on the reproduced baseline by **2.83 Spearman points**. A paired bootstrap with 10,000 resamples gave a 95% confidence interval of **[+1.45, +4.29]** for the gain, holding the selected heads fixed. The two-head test score came from a follow-up on the same split. Testing all six two-head combinations on development recovered the same top pair. On test, two heads scored 0.52 points below three, with a 95% paired interval of [−1.28, +0.23], leaving the difference uncertain. [Main evaluation](../results/stsb-multihead-bootstrap.json) · [Two-head follow-up](../results/stsb-all-head-pairs.json)

## Extended experiments

- **Transfer to SICK-R.** On 4,927 test pairs, single-head Ditto scored 61.64 and the same equal-weight, normalized three-head ensemble scored **64.29**, a gain of **2.66 points** with a paired 95% interval of [+2.11, +3.20]. Weights of 0.50, 0.25, and 0.25 improved STS-B development performance to 75.91, but the weighted ensemble scored 63.94 on SICK-R. Equal weighting with raw attention diagonals scored 64.55. [SICK-R results](../results/sickr-weighted-heads.json)
- **Adding fourth and fifth heads.** Extra heads were selected greedily on STS-B development. On 3,557 STS14 pairs after removing 193 exact overlaps with STS-B development, the three-, four-, and five-head variants scored **62.83**, 62.26, and 61.65. Adding heads reduced performance in this experiment. One possible explanation is that equal averaging gave less informative token weights more influence. [STS14 results](../results/sts14-extra-heads.json)
- **Excluding special tokens from pooling.** With the same three heads, excluding [CLS] and [SEP] from the pooling weights raised the STS-B test score from 69.59 to **70.52** (+0.93) in a follow-up on the previously evaluated split. Both tokens still went through BERT. On 1,347 STS13 pairs after removing 153 exact overlaps with STS-B development or test, the change raised pooled Spearman from 70.96 to **72.10** (+1.14, paired 95% interval [+0.76, +1.52]). [STS-B ablation](../results/stsb-special-token-ablation.json) · [STS13 ablation](../results/sts13-special-token-ablation.json)

The three-head extension improved on single-head Ditto on STS-B and SICK-R without retraining BERT. The follow-ups show that performance depends on which heads and tokens contribute to pooling.
