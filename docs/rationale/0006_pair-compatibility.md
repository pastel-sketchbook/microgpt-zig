# 0006: Pair Compatibility (궁합)

**Status**: Proposed
**Branch**: `saju`
**Date**: 2026-03-07
**Depends on**: 0001, 0002, 0005

## Problem

The previous improvements make the model generate valid, element-annotated
four-pillar sequences. But generation alone has limited practical value —
saju's most common real-world application is **compatibility reading (궁합,
gunghap)**: comparing two people's four pillars to assess relationship
harmony.

A model that can score or classify saju pair compatibility would be
genuinely useful to practitioners.

## Background: Saju Compatibility

Traditional saju compatibility analysis examines how two people's four
pillars interact across several dimensions:

### 1. Day master (일간) harmony

The day pillar's stem (일간) represents the person's core identity. Two
day masters are evaluated for:

- **Same element**: Similar personalities, easy understanding but potential
  stagnation
- **Generating relationship** (상생): One element supports the other
  (e.g., Wood generates Fire). Nurturing dynamic.
- **Controlling relationship** (상극): One element controls the other
  (e.g., Water controls Fire). Tension but also balance.

### 2. Branch interactions (지지 관계)

The 12 branches have specific interaction patterns:

- **Three harmonies** (삼합, samhap): Branches 4 apart form a harmonious
  triangle (e.g., 申子辰 = Metal-Water-Earth triangle → Water harmony)
- **Six harmonies** (육합, yukhap): Specific branch pairs that attract
  (子丑, 寅亥, 卯戌, 辰酉, 巳申, 午未)
- **Clashes** (충, chung): Opposing branches (子午, 丑未, 寅申, 卯酉, 辰戌, 巳亥)
- **Harms** (해, hae): Specific destructive branch pairs

### 3. Element balance

Compare the combined element distribution of both people:
- Does person B supply elements that person A lacks?
- Are there excessive controlling relationships?
- Is there mutual generation (상생) flow?

### Scoring

Traditional practitioners assign scores across these dimensions. Common
scoring systems use 100-point scales or categorical ratings:
- 상 (sang, excellent): Strong harmony, mutual support
- 중 (jung, moderate): Mixed interactions, manageable
- 하 (ha, poor): Significant clashes, requires effort

## Decision

Train the model on **saju pair data with compatibility labels**. Each
training document contains two four-pillar sequences and a compatibility
score.

### Data format

```
壬申庚戌癸酉乙卯|甲子丙寅戊辰庚午|상
```

Structure: `<person A 8 chars>|<person B 8 chars>|<label>`

Labels: `상` (good), `중` (moderate), `하` (poor)

### Sequence length

With codepoint tokenizer: 8 + 1 + 8 + 1 + 1 + 2 (BOS) = **21 tokens**

This requires `block_size >= 21`. Set to 24 for headroom.

### Vocabulary additions

New characters: `상` (U+C0C1), `중` (U+C911), `하` (U+D558)

These are Korean hangul, distinct from the Hanja characters. Vocab size
increases by 3: 29 (from 0005) + 3 = **32**.

## Data Generation

### Approach

1. Sample pairs of four-pillar lines from the existing 452K unique lines.
2. For each pair, compute a compatibility score using deterministic rules
   based on the interactions described above.
3. Output the labeled pair.

### Compatibility scoring algorithm

Implement in `gen_saju_data.zig` (or a new `gen_gunghap_data.zig`):

```
fn scoreCompatibility(a: FourPillars, b: FourPillars) Label {
    var score: i32 = 0;

    // Day master interaction (heaviest weight)
    score += dayMasterScore(a.day.stem, b.day.stem) * 3;

    // Branch harmonies across all pillar positions
    for each pillar position:
        score += branchInteractionScore(a[pos].branch, b[pos].branch);

    // Element balance complementarity
    score += elementBalanceScore(a, b);

    // Map to category
    if (score >= threshold_good) return .sang;
    if (score >= threshold_moderate) return .jung;
    return .ha;
}
```

### Day master scoring

| Interaction | Score |
|-------------|-------|
| Six harmony (육합) stem pair | +5 |
| Generating (상생) | +3 |
| Same element | +1 |
| Controlling (상극) | -2 |
| Clash | -4 |

### Branch interaction scoring

| Interaction | Score |
|-------------|-------|
| Six harmony (육합) | +3 |
| Three harmony (삼합) | +2 |
| Neutral | 0 |
| Harm (해) | -2 |
| Clash (충) | -3 |

### Dataset size

With 452K unique pillar lines, there are ~10^11 possible pairs — far too
many. Sample strategy:
- Generate 50K random pairs
- Ensure roughly balanced labels (stratified sampling or threshold tuning)
- Output to `data/saju_gunghap.txt`

## Training considerations

### Separate model vs. same model

**Option A**: Train a separate model on gunghap data only.
- Pro: Clean separation, tuned hyperparameters
- Con: Loses the pillar-level knowledge from single-pillar training

**Option B**: Fine-tune the pillar model on gunghap data.
- Pro: Transfer learning — model already knows pillar structure
- Con: Catastrophic forgetting of generation capability

**Option C**: Train from scratch on mixed data (single pillars + gunghap pairs).
- Pro: Multi-task learning
- Con: Different sequence lengths and structures may confuse a tiny model

**Recommended**: Option A (separate model) for simplicity. The model is tiny
enough that training from scratch is fast.

### Model adjustments

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `block_size` | 24 | Fits 21-token gunghap sequences |
| `n_embd` | 32 | Same as pillar model |
| `n_layer` | 1 | Same — keep minimal |
| `vocab_size` | ~32 | 22 Hanja + 5 elements + `|` + 3 labels + BOS |

## Expected Impact

- **Practical utility**: First genuinely useful inference mode. Users can
  input two saju and get a compatibility reading.
- **Verifiable output**: Compatibility labels are computed deterministically,
  so model accuracy can be measured precisely against the scoring function.
- **Domain completeness**: Covers the most common real-world saju use case.

## Risks

- **Label quality**: The compatibility scoring is a simplified heuristic.
  Real saju compatibility is nuanced and involves practitioner judgment.
  The model learns our heuristic, not "true" compatibility. This should be
  documented clearly.
- **Class imbalance**: Random pairs might skew heavily toward `중` (moderate).
  May need stratified sampling or oversampling of `상`/`하` pairs.
- **Model capacity**: A 1-layer, 32-dim transformer may not have enough
  capacity to learn both pillar structure AND compatibility scoring. If
  accuracy is poor, consider `n_layer=2` or `n_embd=64`.
- **Sequence length**: 21 tokens is 2x longer than single-pillar data (10
  tokens). Training time increases proportionally.

## Evaluation

- **Accuracy**: What fraction of generated labels match the deterministic
  scoring function? Target: >80% on held-out pairs.
- **Confusion matrix**: Are errors biased toward a particular label?
- **Inference mode**: Given two pillars (16 Hanja + separator), does the
  model predict the correct label? This is essentially using the model as
  a classifier by looking at the probability of `상` vs. `중` vs. `하` at
  the label position.

## Future Extensions

- **Detailed readout**: Instead of a single label, generate a multi-line
  interpretation explaining which interactions contribute to the score.
  Requires much more training data and model capacity.
- **Ten Gods analysis (십신)**: Map the relationship between each pillar
  position and the day master using the ten gods framework. This is the
  next level of saju interpretation beyond element analysis.
- **Daeun (대운) timeline**: Generate the 10-year luck cycles for a given
  saju. This is a separate computation but could be integrated into the
  model's output format.
