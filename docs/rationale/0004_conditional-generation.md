# 0004: Conditional Generation (Pillar Completion)

**Status**: Implemented
**Branch**: `saju`
**Date**: 2026-03-07
**Depends on**: 0001, 0002

## Problem

Unconditional generation (start from BOS, sample everything) produces random
valid-looking pillars but has no practical application. A user cannot ask
"given year pillar 壬申, what are plausible completions?"

The most useful mode for a saju tool is **completion**: provide known pillars
and let the model generate the rest. This turns the model from a curiosity
into a practical completion engine.

## Use Cases

1. **Year-to-full**: Given a year pillar (e.g., 壬申 = water monkey year),
   generate plausible month+day+hour combinations. Useful for exploring
   what saju configurations are common for a given birth year.

2. **Year+month+day to hour**: Given 3 pillars, predict the hour pillar.
   Since hour stem is deterministic from day stem, the model should learn
   to produce the correct hour stem with high probability.

3. **Partial debugging**: Feed a partial saju and see what the model
   considers likely completions — reveals what structural patterns it has
   learned.

## Decision

Add a **prefix mode** to inference: accept an optional Hanja prefix string
via CLI argument, encode it as codepoint tokens, and continue autoregressive
generation from that prefix.

### CLI interface

```
# Unconditional (default)
zig build run -Doptimize=ReleaseFast

# Conditional: complete from year pillar
zig build run -Doptimize=ReleaseFast -- data/saju_pillars.txt 壬申

# Conditional: complete from year+month
zig build run -Doptimize=ReleaseFast -- data/saju_pillars.txt 壬申庚戌
```

The second positional argument (after the dataset filename) is the prefix.

### Taskfile additions

```yaml
demo:complete:
  desc: Complete a partial saju from a given prefix
  cmds:
    - zig build run -Doptimize=ReleaseFast -- data/saju_pillars.txt {{.PREFIX}}
  vars:
    PREFIX: '{{.PREFIX | default "壬申"}}'
```

## Implementation

### Inference loop modification

Currently:
```
token_id = BOS
for position 0..block_size:
    logits = gpt(token_id, position)
    sample next token
```

With prefix:
```
prefix_tokens = tokenizer.encode(prefix)  // without BOS wrapper
token_id = BOS
for position 0..block_size:
    if position < prefix_tokens.len:
        token_id = prefix_tokens[position]  // force prefix token
    else:
        logits = gpt(token_id, position)
        sample next token
```

The key insight: during prefix positions, we still run the forward pass
(to populate the KV cache) but ignore the logits and force the known token.
This is standard "prompt processing" in LLM inference.

Actually, a simpler approach: just run the forward pass for all prefix
tokens to warm up the KV cache, then switch to sampling:

```
// Process prefix (teacher-forced)
token_id = BOS
for prefix positions:
    logits = gpt(token_id, pos)  // builds KV cache
    token_id = prefix_tokens[pos]

// Generate completion (sampled)
for remaining positions:
    logits = gpt(token_id, pos)
    token_id = sample(logits)
```

### Output format

```
prefix:  壬申
sample  1: 壬申庚戌癸酉乙卯
sample  2: 壬申丙寅戊辰庚午
...
```

The prefix is shown once, then each sample includes the full sequence
(prefix + generated completion).

## Expected Impact

- **Practical utility**: Users can explore saju completions interactively.
- **Model evaluation**: Comparing completions against known valid
  combinations reveals how well the model has learned the deterministic
  rules (five-tiger, five-rat).
- **No training change**: Prefix mode is purely an inference-time feature.

## Interaction with constrained sampling (0003)

Constrained sampling (if implemented) applies to the generated portion.
The prefix is assumed valid (user-provided). Validation checks the full
8-character output including the prefix.

## Risks

- If the prefix contains characters not in the vocabulary, `encode` will
  return `error.UnknownChar`. This should be caught and reported with a
  clear error message.
- The prefix length must be even (since pillars are stem-branch pairs).
  Odd-length prefixes would split a pillar mid-pair. Validate this upfront.

## Tests

- Prefix encoding: verify that known Hanja prefixes encode correctly.
- Forward pass with prefix: verify KV cache is populated and subsequent
  sampling produces valid continuations.

## Actual Results

Prefix completion with `壬申` (year pillar = water monkey):
- 19/20 samples valid (95% acceptance with prefix constraint)
- 1/20 five-tiger violation (rejected by constrained sampling)
- All 20 samples correctly start with the forced prefix `壬申`
- The model produces diverse completions for month/day/hour pillars

Prefix validation catches errors early:
- Odd-length prefix: clear error message ("must have even number of codepoints")
- Unknown characters: reports the specific unknown character
- Empty prefix: clear error message
- Prefix too long (>8 codepoints): clear error message
