# 0005: Element-Annotated Training Data

**Status**: Implemented
**Branch**: `saju`
**Date**: 2026-03-07
**Depends on**: 0001, 0002

## Problem

The current training data encodes only the raw four-pillar Hanja sequence.
While the model can learn structural patterns (which stems pair with which
branches), it has no signal about the **semantic layer** that saju
practitioners actually work with: the five elements (오행).

In saju reading, the primary analytical framework is element interaction:
- Which elements are present and in what balance
- Whether elements support (생, saeng) or control (극, geuk) each other
- Which element is the "day master" (일간, ilgan) — the day pillar's stem

Without element information in the training data, the model cannot learn
these higher-level patterns.

## Background: Five Elements (오행)

Each stem and branch maps to one of five elements:

### Stem elements

| Stem | Hanja | Element |
|------|-------|---------|
| 甲   | gap   | 木 Wood |
| 乙   | eul   | 木 Wood |
| 丙   | byeong| 火 Fire |
| 丁   | jeong | 火 Fire |
| 戊   | mu    | 土 Earth|
| 己   | gi    | 土 Earth|
| 庚   | gyeong| 金 Metal|
| 辛   | sin   | 金 Metal|
| 壬   | im    | 水 Water|
| 癸   | gye   | 水 Water|

### Branch elements

| Branch | Hanja | Element |
|--------|-------|---------|
| 子     | ja    | 水 Water|
| 丑     | chuk  | 土 Earth|
| 寅     | in    | 木 Wood |
| 卯     | myo   | 木 Wood |
| 辰     | jin   | 土 Earth|
| 巳     | sa    | 火 Fire |
| 午     | o     | 火 Fire |
| 未     | mi    | 土 Earth|
| 申     | sin   | 金 Metal|
| 酉     | yu    | 金 Metal|
| 戌     | sul   | 土 Earth|
| 亥     | hae   | 水 Water|

### Element interaction cycles

The **generating cycle** (상생, sangsaeng):
Wood -> Fire -> Earth -> Metal -> Water -> Wood

The **controlling cycle** (상극, sanggeuk):
Wood -> Earth -> Water -> Fire -> Metal -> Wood

## Decision

Augment the training data with element annotations appended after a
separator character. The model learns to generate both the pillar sequence
and its element decomposition.

### Data format

```
壬申庚戌癸酉乙卯|水金金土水金木木
```

Structure: `<8 Hanja pillars>|<8 element Hanja>`

The element sequence maps 1:1 to the pillar characters:
- Position 0 (壬, stem): 水 (Water)
- Position 1 (申, branch): 金 (Metal)
- Position 2 (庚, stem): 金 (Metal)
- Position 3 (戌, branch): 土 (Earth)
- ...and so on

### Why include element annotations

1. **Richer learning signal**. The model sees explicit element labels
   alongside pillar characters. This creates a multi-task learning effect:
   the model must predict both the next pillar character AND its element.

2. **Element balance patterns**. With elements in the training data, the
   model can learn distributional patterns like "configurations with 3+
   water elements are rare" or "wood-heavy saju tend to have specific
   branch patterns."

3. **Foundation for interpretation**. Element annotations are the first
   step toward the model learning saju interpretation (reading). Future
   work could add interaction labels (생/극), strength assessments, or
   even traditional reading text.

### Vocabulary impact

New characters added to vocab:

| Element | Hanja | UTF-8 bytes |
|---------|-------|-------------|
| Wood    | 木    | E6 9C A8    |
| Fire    | 火    | E7 81 AB    |
| Earth   | 土    | E5 9C 9F    |
| Metal   | 金    | E9 87 91    |
| Water   | 水    | E6 B0 B4    |

Plus the separator: `|` (0x7C, 1 byte)

Vocab size: 23 (current) + 5 (elements) + 1 (separator) = **29**

Sequence length per document: 8 (pillars) + 1 (separator) + 8 (elements) +
2 (BOS) = **19 tokens** (with codepoint tokenizer)

This fits within `block_size=16`... no, 19 > 16. Need to increase
`block_size` to 24 or 32.

### Hyperparameter adjustments

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `block_size` | 16 | 24 | 19 tokens per doc needs headroom |
| `n_embd` | 32 | 32 | Unchanged — vocab increase is modest |

## Data Generation

Modify `gen_saju_data.zig` to append element annotations:

```zig
// For each pillar pair (stem, branch):
const stem_element = stem.element().hanja();  // e.g., "水"
const branch_element = branch.element().hanja();  // e.g., "金"
```

The `Stem.element()` and `Branch.element()` methods already exist in
zig-saju's `types.zig`.

### Pipeline

```
zig build gen-data-elements 2>/dev/null | sort -u > all_pillars_elements.txt
shuf all_pillars_elements.txt | head -30000 > saju_pillars_elements.txt
```

## Expected Impact

- **Element awareness**: Model learns the stem/branch -> element mapping
  implicitly through co-occurrence.
- **Richer generation**: Generated samples include element annotations,
  making output immediately interpretable.
- **Foundation for 0006**: Element-annotated data is a prerequisite for
  meaningful compatibility analysis.

## Risks

- **Longer sequences**: 19 tokens vs. 10 tokens means ~2x more computation
  per step. Training time roughly doubles.
- **Deterministic mapping**: The element annotation is a deterministic
  function of the pillar characters. The model could learn this mapping
  perfectly, which means the element tokens add no new information — they
  just make the existing information explicit. This is arguably still
  valuable for downstream use (the model "shows its work") but doesn't
  increase the entropy of the training distribution.
- **Separator token**: The `|` character creates a fixed position in the
  sequence. The model must learn that position 8 is always `|`. This is
  trivial to learn and shouldn't cause issues.

## Alternatives Considered

1. **Element-only data** (just the 8 element characters, no pillars):
   loses the pillar identity, making the data too abstract.

2. **Separate element channel**: Train a second model on elements only,
   combine at inference. More complex with no clear benefit over a single
   model.

3. **Yin-yang annotations** (e.g., `壬申庚戌癸酉乙卯|양양양양음양음양`):
   Yin-yang is already encoded in the stem/branch parity and adds less
   information than elements. Could be added later as a third annotation
   section if needed.

## Results

### Training metrics

| Metric       | Before (0002) | After (0005) | Notes |
|-------------|---------------|--------------|-------|
| Vocab size   | 23            | 29           | +5 elements +1 separator |
| Params       | 14,272        | 14,912       | +640 from larger vocab embeddings |
| Tokens/doc   | 8             | 17           | 8 pillars + "|" + 8 elements |
| Loss (start) | 3.48          | 3.54         | Similar (log(29) ≈ 3.37) |
| Loss (end)   | ~1.76         | ~0.88        | **2x lower!** |
| block_size   | 16            | 24           | Increased for 19-token docs |
| step_buf     | 128 MB        | 256 MB       | Doubled for larger computation graphs |

The per-token loss dropped dramatically (1.76 → 0.88) because the element
tokens after the separator are near-deterministic given the pillar portion.
The model learns the pillar→element mapping perfectly.

### Inference (20 samples)

- **19/20 valid** (95% acceptance rate)
- **1 rejection**: five-rat violation (same failure mode as before)
- **0 element mismatches**: every valid sample had correct element annotations
- All samples generated the complete `pillars|elements` format

### Sample output

```
sample  1: 戊辰庚子辛未丙午|土土金水金土火火
sample  2: 乙亥壬申辛酉甲辰|木水水金金金木土
sample  3: 癸卯甲子辛亥甲辰|水木木水金水木土
sample  4: 戊辰戊戌乙巳壬申|土土土土木火水金
sample  5: 乙亥乙亥甲辰乙亥|木水木水木土木水
...
sample 11: 壬戌辛酉癸亥己巳|水土金金水水土火  [five-rat violation]
```

### Key observations

1. **Element mapping learned perfectly**: The model never produced an
   incorrect element for a given pillar character. This validates that
   even a small GPT can learn simple deterministic mappings from data.

2. **Lower per-token loss**: The easy element predictions bring the average
   loss down significantly. The pillar tokens still have the same entropy,
   but the element tokens (deterministic given pillars) approach zero loss.

3. **Same saju validity rate**: The pillar generation quality is unchanged
   (95% valid), confirming that the added element tokens don't interfere
   with pillar pattern learning.

4. **Separator learned trivially**: The model always produces `|` at
   position 8, as predicted.
