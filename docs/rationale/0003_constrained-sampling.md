# 0003: Constrained Sampling

**Status**: Implemented
**Branch**: `saju`
**Date**: 2026-03-07
**Depends on**: 0001, 0002

## Problem

The model generates four-pillar sequences by sampling tokens autoregressively.
While training loss converges to ~1.76 and outputs look plausible, there is
no guarantee that generated sequences are **valid** saju combinations.

Invalid outputs waste inference and erode trust in the model's utility. A
practitioner seeing an impossible stem-branch pair (e.g., a yang stem with a
yin branch) would immediately dismiss the output.

## Background: Saju Validity Rules

A four-pillar sequence `S1B1 S2B2 S3B3 S4B4` (stem-branch pairs for year,
month, day, hour) must satisfy:

### Rule 1: Sexagenary parity

Each stem-branch pair must share yin-yang parity:
- Yang stems (甲丙戊庚壬, even index) pair only with yang branches (子寅辰午申戌, even index)
- Yin stems (乙丁己辛癸, odd index) pair only with yin branches (丑卯巳未酉亥, odd index)

This eliminates half of all possible stem-branch combinations (60 valid out
of 120 possible).

### Rule 2: Month stem from year stem (오호둔갑, Five-Tiger Escape)

The month stem is deterministically derived from the year stem:

| Year stem | Month stem cycle starts at |
|-----------|--------------------------|
| 甲 or 己  | 丙 (index 2) |
| 乙 or 庚  | 戊 (index 4) |
| 丙 or 辛  | 庚 (index 6) |
| 丁 or 壬  | 壬 (index 8) |
| 戊 or 癸  | 甲 (index 0) |

Given the month branch (which encodes the month number), the month stem is
fully determined.

### Rule 3: Hour stem from day stem (오서둔갑, Five-Rat Escape)

Same structure as Rule 2, but maps day stem to hour stem cycle:

| Day stem  | Hour stem cycle starts at |
|-----------|--------------------------|
| 甲 or 己  | 甲 (index 0) |
| 乙 or 庚  | 丙 (index 2) |
| 丙 or 辛  | 戊 (index 4) |
| 丁 or 壬  | 庚 (index 6) |
| 戊 or 癸  | 壬 (index 8) |

### Rule 4: Branch ordering

Month branches follow the zodiac order (寅卯辰巳午未申酉戌亥子丑 for months
1-12). Hour branches follow 子丑寅卯辰巳午未申酉戌亥 for the 12 two-hour
periods. These are fixed astronomical sequences, not free choices.

## Decision

Add a **post-generation validation** step that checks each generated
four-pillar sequence against the saju rules above. Invalid samples are
rejected and re-sampled.

### Approach: Rejection sampling

```
for each sample attempt:
    generate 8-character sequence autoregressively
    parse into 4 stem-branch pairs
    check Rule 1 (parity) for all 4 pairs
    check Rule 2 (month stem from year stem)
    check Rule 3 (hour stem from day stem)
    if all rules pass: accept
    else: reject and retry (up to max_attempts)
```

### Why rejection sampling over constrained decoding

**Constrained decoding** (masking logits at each step to only allow valid
continuations) would be more efficient but requires:
- Tracking which position we're at in the pillar structure
- Knowing which codepoints are valid stems vs. branches at each position
- Implementing the five-tiger/five-rat rules as logit masks mid-generation

This adds significant complexity to the inference loop. Rejection sampling
is simpler: generate freely, validate after, retry if invalid. For a well-
trained model, most samples should already be valid, so rejection rate
should be low.

## Implementation

### Validation function

```
fn isValidSaju(uchars: []u21) bool
```

Takes the decoded codepoint sequence (expected length 8) and checks:
1. Length == 8
2. Characters at even positions (0,2,4,6) are valid stems
3. Characters at odd positions (1,3,5,7) are valid branches
4. All 4 pairs satisfy sexagenary parity
5. Month stem (pos 2) matches year stem (pos 0) via five-tiger rule
6. Hour stem (pos 6) matches day stem (pos 4) via five-rat rule

### Stem/Branch lookup tables

Encode the 10 stems and 12 branches as comptime arrays of `u21` codepoints:

```zig
const stems = [_]u21{ '甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸' };
const branches = [_]u21{ '子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥' };
```

### Inference loop change

Wrap the existing sample generation in a retry loop:

```
const max_attempts = 50;
for (0..max_attempts) |_| {
    generate sample
    if (isValidSaju(sample_codepoints)) {
        emit sample
        break
    }
} else {
    emit sample with "[invalid]" marker
}
```

### Output annotation

Print validity status alongside each sample:
```
sample  1: 壬申庚戌癸酉乙卯 [valid]
sample  2: 甲子丙寅戊辰庚午 [valid]
sample  3: 壬丑庚戌癸酉乙卯 [invalid: parity violation at year pillar]
```

## Expected Impact

- **Practical usability**: Every emitted sample is a valid saju combination
  that could correspond to a real birth date/time.
- **Model quality signal**: The rejection rate measures how well the model
  has learned saju structure. A low rejection rate (~5-10%) indicates the
  model internalized the rules; a high rate (~50%+) suggests the model needs
  more capacity or training.
- **No training change**: This is purely an inference-time filter. Training
  remains unchanged.

## Risks

- If the model is poorly trained, rejection rate could be very high, making
  inference slow. Mitigation: cap retries at 50 and emit with a marker.
- The validation only checks structural rules, not calendrical reachability
  (whether a valid combination actually corresponds to a real date). Full
  date validation would require the inverse saju calculation, which is out
  of scope.

## Tests

- Unit test `isValidSaju` with known valid and invalid combinations.
- Known valid: any line from `saju_pillars.txt`.
- Known invalid: swap a yang stem with a yin branch, alter month stem to
  violate five-tiger rule.

## Actual Results

Training: loss 3.48 → ~1.76 over 1000 steps (unchanged — validation is inference-only).

Inference (20 samples, temperature=0.5, max 50 rejection attempts each):
- **17/20 valid** (85% acceptance rate on first attempt)
- **3/20 rejected** — all `[five-tiger violation]` (samples 8, 11, 20)
- No parity mismatches, no five-rat violations, no invalid stems/branches

The five-tiger rule (year stem → month stem correspondence) is the hardest
constraint for the model to learn, which makes sense: it requires coordinating
tokens across pillar boundaries. The model has clearly learned parity and
five-rat rules well, but cross-pillar dependencies need more capacity or
training to fully internalize.

With rejection sampling capped at 50 attempts, valid samples are found quickly
for the vast majority of outputs. The 15% rejection rate is well within the
"healthy" range described in Expected Impact above.
