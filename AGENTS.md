# ROLES AND EXPERTISE

## Implementor Role

You are a senior Zig engineer porting Karpathy's microgpt.py to idiomatic Zig. You implement changes with attention to numerical correctness, memory management, and clean algorithmic structure.

**Responsibilities:**
- Write idiomatic Zig with explicit allocator usage and proper error handling
- Maintain numerical equivalence with the Python reference implementation
- Ensure all allocations are properly freed (no leaks)
- Keep the code readable as an educational reference

## Reviewer Role

You are a senior engineer who evaluates changes for correctness and adherence to Zig best practices.

**Responsibilities:**
- Verify numerical correctness against the Python reference
- Check that all allocations are freed (defer patterns, arena usage)
- Ensure error handling is comprehensive (no silent failures)
- Run `zig build` and `zig build test`

# SCOPE OF THIS REPOSITORY

This repository contains `microgpt-zig`, a pure Zig port of Karpathy's microgpt.py. It implements:

- **Autograd engine**: `Value` type with forward/backward pass (add, mul, pow, log, exp, relu)
- **Character-level tokenizer**: maps unique characters to token IDs with a BOS token
- **GPT model**: embedding, RMSNorm, multi-head attention, MLP, linear projection
- **Adam optimizer**: with linear learning rate decay
- **Training loop**: trains on a character-level dataset (names)
- **Inference**: generates new samples via temperature-controlled sampling

**Reference**: https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

**Runtime requirements:**
- Zig 0.15.x toolchain
- No external dependencies (pure Zig, no C libraries)
- Downloads `data/input.txt` (names dataset) on first run

# ARCHITECTURE

```
microgpt-zig/
├── build.zig           # Build configuration
├── build.zig.zon       # Package metadata
├── src/
│   └── main.zig        # Single-file app: autograd, model, training, inference
├── Taskfile.yml        # Task runner: build, run, test
├── AGENTS.md           # This file
└── .editorconfig       # Editor settings
```

**Key types (all in `main.zig`):**
- `Value` — Autograd node: data, grad, children, local_grads, backward()
- `Tokenizer` — Character-to-token mapping with BOS support
- `StateDict` — Model parameters (embeddings, attention weights, MLP weights)
- `GPT` — Forward pass: token → logits

**Data flow:**
1. Load/download dataset (names.txt)
2. Build tokenizer from unique characters
3. Initialize random parameters (Gaussian)
4. Training loop: tokenize doc → forward GPT → compute loss → backward → Adam update
5. Inference: sample tokens autoregressively with temperature

# MODEL HYPERPARAMETERS (matching Python reference)

| Parameter    | Value | Description                    |
|-------------|-------|--------------------------------|
| `n_layer`    | 1     | Number of transformer layers   |
| `n_embd`     | 16    | Embedding dimension            |
| `block_size` | 16    | Maximum context length         |
| `n_head`     | 4     | Number of attention heads      |
| `head_dim`   | 4     | Dimension per head (n_embd/n_head) |
| `num_steps`  | 1000  | Training steps                 |
| `lr`         | 0.01  | Initial learning rate          |

# CORE DEVELOPMENT PRINCIPLES

- **Pure Zig**: No external dependencies. Use only the standard library.
- **Single File**: Everything lives in `main.zig` for simplicity and educational clarity.
- **Numerical Fidelity**: Match the Python reference as closely as possible.
- **Explicit Memory**: Use allocators explicitly. Prefer arena allocators for per-step allocations. Free everything.
- **No Panics in Logic**: Use error returns for fallible operations. `@panic` only for true invariant violations.

# ZIG-SPECIFIC GUIDELINES

## Memory Management
- Use `std.heap.GeneralPurposeAllocator` for long-lived allocations
- Use arena allocators for per-training-step computation graphs (allocate, run, free all at once)
- Always `defer` cleanup immediately after allocation

## Error Handling
- Return `!void` or `!T` for fallible functions
- Use `try` to propagate errors
- Provide meaningful error context where possible

## Floating Point
- Use `f64` throughout to match Python's float64 semantics
- Use `@log`, `@exp`, `@sqrt` builtins where available, else `std.math`

## Random Number Generation
- Use `std.Random` with a fixed seed (42) for reproducibility

# CODE STYLE

- 4-space indentation (per .editorconfig)
- No line width limit enforced, but keep reasonable (~120 chars)
- Unix newlines
- Group code into logical sections with comment headers matching the Python structure

# COMMIT CONVENTIONS

Use the following prefixes:
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code improvement without behavior change
- `test`: Adding or improving tests
- `docs`: Documentation changes
- `chore`: Tooling, dependencies, configuration

# CODE REVIEW CHECKLIST

- Does the code match the Python reference numerically?
- Are all allocations properly freed?
- Does `zig build` succeed with no warnings?
- Does `zig build test` pass?
- Is the autograd backward pass correct?

# OUT OF SCOPE / ANTI-PATTERNS

- External dependencies or C bindings
- GPU acceleration or SIMD optimization (keep it simple)
- Multi-file module structure (keep single file)
- Batched training (single document at a time, matching Python)

# SUMMARY MANTRA

Autograd. GPT. Training. Inference. Pure Zig. One file.
