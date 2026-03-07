# 0002: Codepoint-Level Tokenizer

**Status**: Implemented
**Branch**: `saju`
**Date**: 2026-03-07
**Depends on**: 0001

## Problem

The current tokenizer operates on raw `u8` bytes. Each Hanja character is
3 UTF-8 bytes, so an 8-character four-pillar sequence becomes a 24-token
sequence (plus 2 BOS delimiters = 26 tokens total).

This creates two problems:

1. **Wasted model capacity**. The model must learn that certain 3-byte
   subsequences (e.g., `[0xE5, 0xA3, 0xAC]` = `壬`) form valid characters.
   This is UTF-8 encoding mechanics, not saju structure. A 1-layer transformer
   spending parameters on byte-level recombination has fewer parameters left
   for learning actual pillar relationships.

2. **Inflated sequence length**. 24 tokens per document means the model
   processes 3x more positions than necessary. This directly increases the
   computation graph size (~200-300K nodes per step), training time, and
   memory usage.

## Decision

Replace the byte-level tokenizer with a **Unicode codepoint-level tokenizer**.
Each Hanja character becomes a single token.

### Before (byte-level)

```
Input:  壬申庚戌癸酉乙卯
Bytes:  [E5 A3 AC] [E7 94 B3] [E5 BA 9A] [E6 88 8C] [E7 99 B8] [E9 85 89] [E4 B9 99] [E5 8D AF]
Tokens: BOS b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 BOS
Length: 26 tokens
Vocab:  36 (35 unique bytes + BOS)
```

### After (codepoint-level)

```
Input:  壬申庚戌癸酉乙卯
Tokens: BOS 壬 申 庚 戌 癸 酉 乙 卯 BOS
Length: 10 tokens
Vocab:  23 (22 unique Hanja + BOS)
```

## Implementation

### Tokenizer changes

The `Tokenizer` struct changes from `u8` to `u21` (Zig's Unicode codepoint type):

| Field / Method | Before | After |
|---------------|--------|-------|
| `uchars` | `[]const u8` | `[]u21` |
| `init` | Iterates bytes | Iterates UTF-8 codepoints via `std.unicode.Utf8View` |
| `encode` | Maps bytes to indices | Maps codepoints to indices |
| `decode` | Returns `?u8` | Returns `?u21` |

The encode/decode approach stays the same (sorted unique vocabulary + BOS
sentinel), just at the codepoint granularity.

### Inference output

The inference loop must convert decoded codepoints back to UTF-8 bytes for
output. Use `std.unicode.utf8Encode(cp, &buf)` to write 1-4 bytes per
codepoint into the sample buffer.

### Hyperparameter adjustments

| Parameter | Before (byte) | After (codepoint) | Rationale |
|-----------|---------------|-------------------|-----------|
| `block_size` | 32 | 16 | 10 tokens/doc fits in 16; matches original names config |
| `n_embd` | 32 | 32 | Keep — sufficient for 23-token vocab |
| `step_buf` | 256 MB | 128 MB | ~3x fewer nodes per step |
| `estimated_nodes` | 300,000 | 150,000 | ~3x fewer positions to process |

### Backward compatibility

The codepoint tokenizer handles ASCII transparently (ASCII codepoints are
single-byte UTF-8). Passing `data/input.txt` (English names) or
`data/korean_names.txt` still works — the tokenizer just sees 1-byte or
3-byte codepoints respectively.

## Expected Impact

- **Sequence length**: 26 -> 10 tokens (2.6x reduction)
- **Vocab size**: 36 -> 23 (1.6x reduction)
- **Computation graph**: ~200-300K -> ~80-120K nodes per step
- **Model focus**: Entirely on pillar-level structure, zero capacity wasted on
  UTF-8 byte recombination
- **Parameter count**: ~15,616 -> ~14,272 (slight reduction from smaller
  positional embedding and vocab projection)

## Actual Results

| Metric | Byte-level | Codepoint |
|--------|-----------|-----------|
| Vocab size | 36 | 23 |
| Parameters | 15,616 | 14,272 |
| Tokens per doc | 24 | 8 |
| Final per-token loss | ~0.64 | ~1.76 |
| Per-document loss | 24 x 0.64 = 15.4 | 8 x 1.76 = 14.1 |

The per-token loss is higher because every codepoint token is a meaningful
choice among 22 Hanja characters. With the byte-level tokenizer, most
tokens were "easy" UTF-8 continuation bytes (given the first byte of a
3-byte sequence, the continuation bytes are heavily constrained). This
artificially deflated the average per-token loss.

The per-document loss is comparable or slightly lower, confirming the model
performs at least as well on the actual prediction task. Generated samples
are valid-looking 8-character Hanja sequences.

## Risks

- None significant. The change is a strict improvement for any multi-byte
  UTF-8 dataset. For ASCII datasets, behavior is identical since each ASCII
  codepoint is one byte.
- The `std.unicode.Utf8View` API must be verified against Zig 0.15.x, but
  UTF-8 view/iterator has been stable across Zig versions.

## Tests

- Existing tokenizer test (ASCII "abc" / "bca") still passes — ASCII
  codepoints produce the same vocabulary and encoding.
- Add a new test with Hanja input to verify codepoint-level tokenization:
  vocab size = char count + BOS, encode length = char count + 2.
