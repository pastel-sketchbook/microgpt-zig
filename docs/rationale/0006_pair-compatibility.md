# 0006: Pair Compatibility (궁합)

**Status**: Implemented
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

This requires `block_size >= 21`. Current block_size of 24 fits with
headroom.

### Vocabulary additions

New characters: `상` (U+C0C1), `중` (U+C911), `하` (U+D558)

These are Korean hangul, distinct from the Hanja characters.

## Data Generation

### Approach

Implemented in `zig-saju/src/gen_gunghap_data.zig`:

1. Collect all 452,904 unique four-pillar combinations (years 1900-2050)
2. Shuffle with seed 42
3. Pair consecutive items, score each pair, output labeled lines
4. Generate 50K pairs, sample 30K for training

### Compatibility scoring algorithm

```
scoreCompatibility(a, b) =
    dayMasterScore(a.day.stem, b.day.stem) * 3
  + branchScore(a.year.branch, b.year.branch)
  + branchScore(a.month.branch, b.month.branch)
  + branchScore(a.day.branch, b.day.branch)
  + branchScore(a.hour.branch, b.hour.branch)
  + elementBalanceScore(a, b)
```

### Day master scoring

| Interaction | Score |
|-------------|-------|
| Stem combination (천간합, diff by 5) | +5 |
| Generating (상생) | +3 |
| Same element | +1 |
| Controlling (상극) | -2 |

Weight: ×3

### Branch interaction scoring

| Interaction | Score |
|-------------|-------|
| Six harmony (육합) | +3 |
| Three harmony (삼합, same %4) | +2 |
| Clash (충, diff by 6) | -3 |
| Neutral | 0 |

### Element balance scoring

- Count all 5 elements across 16 positions (8 per person, stem+branch)
- All 5 present → +2
- Any single element count ≥ 6 → -1

### Label thresholds (tuned for balance)

| Label | Threshold | Count | Percentage |
|-------|-----------|-------|------------|
| 상 (good) | score ≥ 12 | 17,239 | 34.5% |
| 중 (moderate) | 2 ≤ score < 12 | 18,044 | 36.1% |
| 하 (poor) | score < 2 | 14,717 | 29.4% |

Score range: [-16, 29], mean: 7.1

Initial thresholds (≥8, ≥0) produced 53.3% 상, requiring tuning.
Final thresholds (≥12, ≥2) achieve roughly uniform distribution.

### Dataset

- 50K pairs generated, 30K sampled for training
- File: `data/saju_gunghap.txt` (1.6 MB)
- 54 UTF-8 bytes per line (16 Hanja × 3 + 2 separators + 3-byte label + newline)

## Implementation

### Changes to microgpt-zig

1. **`validateGunghap()`**: New validation function for 19-codepoint
   gunghap sequences. Checks separators at positions 8 and 17, validates
   both pillar subsequences via `validateSaju()`, and verifies the label
   is one of 상/중/하.

2. **Relaxed prefix validation**: Removed the even-codepoint and max-8
   constraints from prefix parsing. Now only checks: non-empty, within
   block_size, and all characters in vocab. This allows gunghap prefix
   completion (e.g., providing 17 chars to classify a pair).

3. **Inference validation**: Updated to detect format by codepoint count:
   19 → gunghap, 17 → annotated saju, 8 → plain saju.

4. **8 new unit tests** for `validateGunghap()`.

### Changes to zig-saju

1. **`src/gen_gunghap_data.zig`**: New data generator with scoring
   functions (`dayMasterScore`, `branchScore`, `elementBalanceScore`,
   `scoreCompatibility`, `classify`).

2. **`build.zig`**: Added `gen-gunghap` build step.

### Taskfile additions

- `demo:gunghap` — Train and generate from gunghap data
- `demo:gunghap:classify` — Classify a specific pair via prefix completion

## Actual Results

### Training metrics

| Metric | Value |
|--------|-------|
| Vocab size | 27 (22 Hanja + | + 상중하 + BOS) |
| Parameters | 14,784 |
| Initial loss | 3.38 |
| Final loss (step 1000) | ~1.69 |
| Loss at step 100 | ~1.77 |
| Loss at step 500 | ~1.73 |

### Inference (unconstrained generation)

The model learned the format perfectly — all 20 samples produce exactly
19 codepoints with separators in the correct positions and valid labels.
However, structural validity of the generated pillar sequences is low:
**1/20 (5%) fully valid**.

This is expected: the model must simultaneously satisfy saju validity
rules (parity, five-tiger, five-rat) for BOTH pillar sequences
independently. With single-pillar models achieving ~85% per-pillar
validity, the joint probability is ~72% per attempt, but the gunghap
model was trained from scratch without the benefit of single-pillar
pretraining.

### What the model learned

1. **Perfect format**: 8 Hanja + | + 8 Hanja + | + label (100%)
2. **Correct character classes**: Stems appear at even positions, branches
   at odd positions in both pillar subsequences
3. **Label distribution**: Generated labels roughly match training
   distribution
4. **Pillar structure**: Partial learning of inter-pillar constraints
   (parity, stem-branch pairing) but not yet the complex five-tiger/rat
   rules that span across pillars

### Classification mode (prefix completion)

The primary practical use: provide two complete four-pillar sequences as
a prefix and let the model predict the compatibility label. This uses
teacher-forced prefix with a single sampled token for the label position.

## Risks (confirmed)

- **Pillar validity**: As predicted, the tiny model struggles to learn
  both pillar structure AND compatibility scoring simultaneously. The
  classification use case (prefix completion) is more practical than
  free generation.
- **Model capacity**: A 1-layer, 32-dim transformer has limited capacity
  for this more complex task. Potential improvements: increase to
  `n_layer=2` or `n_embd=64`, or train longer.

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
- **Increased model capacity**: Try `n_layer=2` or `n_embd=64` to
  improve both pillar validity and label accuracy in free generation.
