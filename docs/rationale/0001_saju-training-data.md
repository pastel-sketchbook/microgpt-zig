# 0001: Saju Four-Pillar Training Data

**Status**: Implemented  
**Branch**: `saju`  
**Date**: 2026-03-07

## Context

microgpt-zig is a pure-Zig port of Karpathy's microgpt.py -- a minimal
autograd + GPT implementation that trains a character-level language model on
a names dataset (~32K English names). The model learns to generate plausible
new names after training.

We want to repurpose this model to learn the structure of Korean Four Pillars
(사주, saju) -- a system that maps a birth date/time to eight Hanja
characters representing four pillars (year, month, day, hour). Each pillar
is a pair of one Heavenly Stem (천간) and one Earthly Branch (지지).

## Decision

Replace the English names training data with pre-generated Hanja four-pillar
sequences. Each training document is a single line of 8 Hanja characters
(4 stem-branch pairs), e.g.:

```
壬申庚戌癸酉乙卯
```

### Why Saju data is a good fit for microgpt

1. **Fixed-length sequences**. Every document is exactly 8 Hanja characters
   (24 UTF-8 bytes). This is comparable to the original names dataset (average
   ~6 characters) and fits comfortably within the block size.

2. **Small, structured vocabulary**. There are only 10 Heavenly Stems and
   12 Earthly Branches = 22 unique Hanja characters. At the byte level
   (microgpt's tokenizer operates on raw `u8` bytes), this produces ~35
   unique byte values + 1 BOS token = vocab size 36. The original names
   dataset had 27 (a-z + BOS).

3. **Learnable constraints**. The data is not random -- it encodes real
   calendrical rules:
   - Sexagenary cycle parity: yang stems pair with yang branches, yin with yin
   - Month stem is determined by year stem (five-tiger escape rule)
   - Hour stem is determined by day stem (five-rat escape rule)
   - Branch positions within month/hour follow fixed astronomical ordering
   - A 1-layer, 32-dim transformer should be able to capture these regularities

4. **Verifiable output**. Generated samples can be validated against known
   saju rules, giving a concrete correctness signal beyond just loss numbers.

## Data Generation

### Source

The [zig-saju](https://github.com/nicholasgasior/zig-saju) engine provides
`calculateFourPillars(year, month, day, hour, minute)` which returns the
four pillars for any given date/time.

A data generation tool (`zig-saju/src/gen_saju_data.zig`) was created:

- Iterates years 1900-2050, all months (1-12), all days (1-31), and 12
  representative hours (one per two-hour 시 period: 0, 2, 4, ..., 22)
- Skips invalid dates (e.g., Feb 30) via error handling
- Outputs each four-pillar result as a single line of 8 Hanja characters
- Pipes through `sort -u` to deduplicate

### Pipeline

```
zig build gen-data 2>/dev/null | sort -u > all_pillars.txt    # 452,904 unique lines (11.3 MB)
shuf all_pillars.txt | head -30000 > saju_pillars.txt         # 30,000 sampled lines (750 KB)
```

The 30K sample size was chosen to roughly match the original names dataset
(~32K lines) so that training dynamics remain comparable.

### Data file

`data/saju_pillars.txt` -- 30,000 unique Hanja four-pillar lines, checked
into the repository. The file is 750 KB of UTF-8 text.

## Hyperparameter Changes

| Parameter       | Original (names) | Saju          | Rationale                                           |
|----------------|-------------------|---------------|-----------------------------------------------------|
| `n_embd`        | 16                | 32            | More byte patterns to learn (CJK multi-byte)        |
| `block_size`    | 16                | 32            | 24 bytes per doc + BOS padding needs headroom        |
| `step_buf`      | 32 MB             | 256 MB        | Computation graph grows from ~38K to ~200-300K nodes |
| `estimated_nodes`| 38,000           | 300,000       | Matches observed graph size                          |
| `n_layer`       | 1                 | 1             | Unchanged                                           |
| `n_head`        | 4                 | 4             | Unchanged                                           |
| `num_steps`     | 1000              | 1000          | Unchanged                                           |
| `lr`            | 0.01              | 0.01          | Unchanged                                           |

Total parameters: 15,616 (up from 4,192).

## Training Results

- **Loss**: 3.52 (step 1) to ~0.64 (step 1000)
- **Inference**: All 20 generated samples produced valid-looking 8-character
  Hanja sequences with correct CJK rendering
- **Vocab size**: 36 (35 unique bytes from 22 Hanja chars + 1 BOS)

## Byte-Level Tokenization Detail

microgpt's tokenizer works on raw `u8` bytes, not Unicode codepoints. Each
Hanja character is 3 UTF-8 bytes. For example, `壬` = `[0xE5, 0xA3, 0xAC]`.

This means the model sees each pillar pair as 6 byte tokens, not 2 character
tokens. The full 8-character sequence becomes a 24-token sequence (plus BOS
delimiters). The model must learn:

1. Which 3-byte subsequences form valid Hanja characters
2. Which Hanja pairs (stem + branch) are valid
3. The positional constraints between pillars

The ~0.64 loss suggests the model has learned much of this structure.

## Alternatives Considered

1. **Unicode codepoint tokenizer**: Would reduce sequence length from 24 to 8
   tokens per document but requires rewriting the tokenizer. The byte-level
   approach works and matches the original microgpt design.

2. **Larger dataset (full 452K lines)**: Would provide more coverage but
   increases training time proportionally. 30K is sufficient for the model
   to learn the core patterns.

3. **Korean hangul names**: Already exists in `data/korean_names.txt` (263
   names) but is too small and doesn't have the structured, verifiable
   properties of saju data.
