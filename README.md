# microgpt-zig

A pure Zig port of Karpathy's [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — the most atomic way to train and run a GPT, with zero dependencies.

Everything lives in a single file: autograd, tokenizer, transformer, training loop, and inference. No frameworks, no libraries, just Zig.

## How It Works

The program trains a tiny character-level GPT on a dataset of names, then generates new ones. Here's the full pipeline:

![microgpt-zig training and inference pipeline](docs/pipeline.png)

## Quick Start

```bash
# Requires: Zig 0.15.x

# Build and run (downloads names dataset on first run)
zig build run -Doptimize=ReleaseFast

# Or use the task runner
task demo            # English names
task demo:korean     # Korean (Hangul) names
```

## Sample Output

```
num docs: 32033
vocab size: 27
num params: 4192
step 1000 / 1000 | loss 2.0760
--- inference (new, hallucinated names) ---
sample  1: vemare
sample  2: rileen
sample  3: ariren
sample  4: amelen
sample  5: kasir
```

Korean output (`task demo:korean`):
```
num docs: 263
vocab size: 58
num params: 5184
step 1000 / 1000 | loss 1.0310
--- inference (new, hallucinated names) ---
sample  1: 서원
sample  2: 수윤
sample  3: 지예
sample  4: 지준
sample  5: 태영
```

## Architecture

Everything is in [`src/main.zig`](src/main.zig):

| Component | Description |
|-----------|-------------|
| `Value` | Autograd node: data, grad, children, local_grads, backward() |
| `Tokenizer` | Character ↔ token ID mapping with BOS token |
| `StateDict` | All model parameters: embeddings, attention, MLP weights |
| `gpt()` | Forward pass: embed → RMSNorm → attention → MLP → logits |
| `backward()` | Topological sort + reverse gradient accumulation |
| `main()` | Training loop (Adam + LR decay) → inference (temperature sampling) |

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_layer` | 1 | Transformer layers |
| `n_embd` | 16 | Embedding dimension |
| `block_size` | 16 | Max context length |
| `n_head` | 4 | Attention heads |
| `num_steps` | 1000 | Training steps |
| `lr` | 0.01 | Learning rate (linear decay) |

## Custom Datasets

Pass any text file (one name per line) as argument:

```bash
zig build run -Doptimize=ReleaseFast -- my_names.txt
```

## Reference

Port of Karpathy's [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — *"The most atomic way to train and run inference for a GPT in pure, dependency-free Python."*

This is the Zig equivalent. Everything else is just efficiency.
