const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;

// ============================================================================
// Hyperparameters (matching Python reference)
// ============================================================================
const n_layer: usize = 1;
const n_embd: usize = 16;
const block_size: usize = 16;
const n_head: usize = 4;
const head_dim: usize = n_embd / n_head;
const num_steps: usize = 1000;
const learning_rate: f64 = 0.01;
const beta1: f64 = 0.85;
const beta2: f64 = 0.99;
const eps_adam: f64 = 1e-8;
const init_std: f64 = 0.08;

// ============================================================================
// Dataset
// ============================================================================
const dataset_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt";

fn loadOrDownloadDataset(allocator: Allocator, filename: []const u8) ![][]const u8 {
    // Try to read from local file first, download if not found
    const data = std.fs.cwd().readFileAlloc(allocator, filename, 10 * 1024 * 1024) catch |err| switch (err) {
        error.FileNotFound => blk: {
            // Only auto-download the default dataset
            if (!std.mem.eql(u8, filename, "data/input.txt")) {
                std.debug.print("Error: file '{s}' not found\n", .{filename});
                return err;
            }
            std.debug.print("Downloading dataset...\n", .{});
            std.fs.cwd().makePath("data") catch {};
            var child = std.process.Child.init(
                &.{ "curl", "-sL", "-o", "data/input.txt", dataset_url },
                allocator,
            );
            _ = try child.spawnAndWait();
            break :blk try std.fs.cwd().readFileAlloc(allocator, "data/input.txt", 10 * 1024 * 1024);
        },
        else => return err,
    };

    // Split into lines, filter empty
    var docs: std.ArrayList([]const u8) = .empty;
    var iter = std.mem.splitSequence(u8, data, "\n");
    while (iter.next()) |line| {
        const trimmed = std.mem.trim(u8, line, &[_]u8{ ' ', '\r', '\t' });
        if (trimmed.len > 0) {
            try docs.append(allocator, trimmed);
        }
    }
    return try docs.toOwnedSlice(allocator);
}

// ============================================================================
// Tokenizer
// ============================================================================
const Tokenizer = struct {
    uchars: []const u8, // sorted unique characters
    bos: usize, // BOS token id
    vocab_size: usize,

    fn init(allocator: Allocator, docs: []const []const u8) !Tokenizer {
        // Collect unique characters
        var char_set = std.AutoHashMap(u8, void).init(allocator);
        defer char_set.deinit();
        for (docs) |doc| {
            for (doc) |ch| {
                try char_set.put(ch, {});
            }
        }
        // Sort them
        var chars: std.ArrayList(u8) = .empty;
        var it = char_set.keyIterator();
        while (it.next()) |key| {
            try chars.append(allocator, key.*);
        }
        const sorted = try chars.toOwnedSlice(allocator);
        std.mem.sort(u8, sorted, {}, std.sort.asc(u8));

        const bos = sorted.len;
        return .{
            .uchars = sorted,
            .bos = bos,
            .vocab_size = sorted.len + 1,
        };
    }

    fn encode(self: *const Tokenizer, allocator: Allocator, doc: []const u8) ![]usize {
        // BOS + char tokens + BOS
        var tokens: std.ArrayList(usize) = try .initCapacity(allocator, doc.len + 2);
        try tokens.append(allocator, self.bos);
        for (doc) |ch| {
            const idx = std.mem.indexOf(u8, self.uchars, &[_]u8{ch}) orelse return error.UnknownChar;
            try tokens.append(allocator, idx);
        }
        try tokens.append(allocator, self.bos);
        return try tokens.toOwnedSlice(allocator);
    }

    fn decode(self: *const Tokenizer, token_id: usize) ?u8 {
        if (token_id >= self.uchars.len) return null; // BOS
        return self.uchars[token_id];
    }
};

// ============================================================================
// Autograd: Value
// ============================================================================
const MAX_CHILDREN = 2;

const Value = struct {
    data: f64,
    grad: f64,
    children: [MAX_CHILDREN]?*Value,
    local_grads: [MAX_CHILDREN]f64,
    n_children: u8,
    gen: u32 = 0, // generation counter: replaces HashMap in backward()

    fn create(allocator: Allocator, data: f64) !*Value {
        const v = try allocator.create(Value);
        v.* = .{
            .data = data,
            .grad = 0,
            .children = .{ null, null },
            .local_grads = .{ 0, 0 },
            .n_children = 0,
        };
        return v;
    }

    fn add(allocator: Allocator, a: *Value, b: *Value) !*Value {
        const v = try allocator.create(Value);
        v.* = .{
            .data = a.data + b.data,
            .grad = 0,
            .children = .{ a, b },
            .local_grads = .{ 1.0, 1.0 },
            .n_children = 2,
        };
        return v;
    }

    fn mul(allocator: Allocator, a: *Value, b: *Value) !*Value {
        const v = try allocator.create(Value);
        v.* = .{
            .data = a.data * b.data,
            .grad = 0,
            .children = .{ a, b },
            .local_grads = .{ b.data, a.data },
            .n_children = 2,
        };
        return v;
    }

    fn pow(allocator: Allocator, base: *Value, exponent: f64) !*Value {
        const v = try allocator.create(Value);
        v.* = .{
            .data = math.pow(f64, base.data, exponent),
            .grad = 0,
            .children = .{ base, null },
            .local_grads = .{ exponent * math.pow(f64, base.data, exponent - 1.0), 0 },
            .n_children = 1,
        };
        return v;
    }

    fn log(allocator: Allocator, a: *Value) !*Value {
        const v = try allocator.create(Value);
        v.* = .{
            .data = @log(a.data),
            .grad = 0,
            .children = .{ a, null },
            .local_grads = .{ 1.0 / a.data, 0 },
            .n_children = 1,
        };
        return v;
    }

    fn exp(allocator: Allocator, a: *Value) !*Value {
        const e = @exp(a.data);
        const v = try allocator.create(Value);
        v.* = .{
            .data = e,
            .grad = 0,
            .children = .{ a, null },
            .local_grads = .{ e, 0 },
            .n_children = 1,
        };
        return v;
    }

    fn relu(allocator: Allocator, a: *Value) !*Value {
        const v = try allocator.create(Value);
        v.* = .{
            .data = @max(0.0, a.data),
            .grad = 0,
            .children = .{ a, null },
            .local_grads = .{ if (a.data > 0) 1.0 else 0.0, 0 },
            .n_children = 1,
        };
        return v;
    }

    fn neg(allocator: Allocator, a: *Value) !*Value {
        const v = try allocator.create(Value);
        v.* = .{
            .data = -a.data,
            .grad = 0,
            .children = .{ a, null },
            .local_grads = .{ -1.0, 0 },
            .n_children = 1,
        };
        return v;
    }

    fn sub(allocator: Allocator, a: *Value, b: *Value) !*Value {
        const v = try allocator.create(Value);
        v.* = .{
            .data = a.data - b.data,
            .grad = 0,
            .children = .{ a, b },
            .local_grads = .{ 1.0, -1.0 },
            .n_children = 2,
        };
        return v;
    }

    fn div(allocator: Allocator, a: *Value, b: *Value) !*Value {
        const v = try allocator.create(Value);
        v.* = .{
            .data = a.data / b.data,
            .grad = 0,
            .children = .{ a, b },
            // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
            .local_grads = .{ 1.0 / b.data, -a.data / (b.data * b.data) },
            .n_children = 2,
        };
        return v;
    }

    fn addScalar(allocator: Allocator, a: *Value, s: f64) !*Value {
        const sv = try Value.create(allocator, s);
        return Value.add(allocator, a, sv);
    }

    fn mulScalar(allocator: Allocator, a: *Value, s: f64) !*Value {
        const sv = try Value.create(allocator, s);
        return Value.mul(allocator, a, sv);
    }

    /// Backward pass: iterative topological sort then reverse accumulation
    fn backward(self: *Value, allocator: Allocator, generation: u32) !void {
        // Build topological order (iterative DFS)
        var topo: std.ArrayList(*Value) = .empty;
        defer topo.deinit(allocator);

        const Frame = struct { node: *Value, phase: enum { enter, exit } };
        var stack: std.ArrayList(Frame) = .empty;
        defer stack.deinit(allocator);
        try stack.append(allocator, .{ .node = self, .phase = .enter });

        while (stack.items.len > 0) {
            const item = stack.pop().?;
            switch (item.phase) {
                .enter => {
                    if (item.node.gen == generation) continue;
                    item.node.gen = generation;
                    try stack.append(allocator, .{ .node = item.node, .phase = .exit });
                    for (0..item.node.n_children) |i| {
                        if (item.node.children[i]) |child| {
                            if (child.gen != generation) {
                                try stack.append(allocator, .{ .node = child, .phase = .enter });
                            }
                        }
                    }
                },
                .exit => {
                    try topo.append(allocator, item.node);
                },
            }
        }

        // Backward pass
        self.grad = 1.0;
        var idx = topo.items.len;
        while (idx > 0) {
            idx -= 1;
            const v = topo.items[idx];
            for (0..v.n_children) |ci| {
                if (v.children[ci]) |child| {
                    child.grad += v.local_grads[ci] * v.grad;
                }
            }
        }
    }
};

// ============================================================================
// Neural Network Operations
// ============================================================================

/// linear(x, w) = [sum(wi * xi) for wo in w]
/// in_dim is comptime so LLVM can unroll the inner dot-product loop.
fn linear(comptime in_dim: usize, allocator: Allocator, x: []const *Value, w: []const []const *Value) ![]*Value {
    const result = try allocator.alloc(*Value, w.len);
    for (w, 0..) |row, i| {
        var sum_val = try Value.create(allocator, 0.0);
        for (0..in_dim) |j| {
            const prod = try Value.mul(allocator, row[j], x[j]);
            sum_val = try Value.add(allocator, sum_val, prod);
        }
        result[i] = sum_val;
    }
    return result;
}

/// softmax with numerical stability (subtract max)
fn softmax(allocator: Allocator, logits: []const *Value) ![]*Value {
    var max_val: f64 = -math.inf(f64);
    for (logits) |v| {
        if (v.data > max_val) max_val = v.data;
    }

    const exps = try allocator.alloc(*Value, logits.len);
    var total = try Value.create(allocator, 0.0);
    for (logits, 0..) |v, i| {
        const shifted = try Value.addScalar(allocator, v, -max_val);
        exps[i] = try Value.exp(allocator, shifted);
        total = try Value.add(allocator, total, exps[i]);
    }

    const probs = try allocator.alloc(*Value, logits.len);
    for (exps, 0..) |e, i| {
        probs[i] = try Value.div(allocator, e, total);
    }
    return probs;
}

/// RMS normalization
fn rmsnorm(allocator: Allocator, x: []const *Value) ![]*Value {
    const n: f64 = @floatFromInt(x.len);
    var ms = try Value.create(allocator, 0.0);
    for (x) |xi| {
        const sq = try Value.mul(allocator, xi, xi);
        ms = try Value.add(allocator, ms, sq);
    }
    ms = try Value.mulScalar(allocator, ms, 1.0 / n);
    const scale = try Value.addScalar(allocator, ms, 1e-5);
    const scale_inv = try Value.pow(allocator, scale, -0.5);

    const result = try allocator.alloc(*Value, x.len);
    for (x, 0..) |xi, i| {
        result[i] = try Value.mul(allocator, xi, scale_inv);
    }
    return result;
}

// ============================================================================
// State Dict (model parameters)
// ============================================================================
const StateDict = struct {
    wte: [][]const *Value, // [vocab_size][n_embd]
    wpe: [][]const *Value, // [block_size][n_embd]
    lm_head: [][]const *Value, // [vocab_size][n_embd]

    // Per-layer attention weights
    attn_wq: [n_layer][][]const *Value, // [n_embd][n_embd]
    attn_wk: [n_layer][][]const *Value,
    attn_wv: [n_layer][][]const *Value,
    attn_wo: [n_layer][][]const *Value,

    // Per-layer MLP weights
    mlp_fc1: [n_layer][][]const *Value, // [4*n_embd][n_embd]
    mlp_fc2: [n_layer][][]const *Value, // [n_embd][4*n_embd]

    params: []*Value, // flat list of all parameters

    fn init(allocator: Allocator, vocab_size: usize, rng: std.Random) !StateDict {
        var all_params: std.ArrayList(*Value) = .empty;

        const wte = try initMatrix(allocator, vocab_size, n_embd, rng, &all_params);
        const wpe = try initMatrix(allocator, block_size, n_embd, rng, &all_params);
        const lm_head = try initMatrix(allocator, vocab_size, n_embd, rng, &all_params);

        var sd: StateDict = undefined;
        sd.wte = wte;
        sd.wpe = wpe;
        sd.lm_head = lm_head;

        for (0..n_layer) |i| {
            sd.attn_wq[i] = try initMatrix(allocator, n_embd, n_embd, rng, &all_params);
            sd.attn_wk[i] = try initMatrix(allocator, n_embd, n_embd, rng, &all_params);
            sd.attn_wv[i] = try initMatrix(allocator, n_embd, n_embd, rng, &all_params);
            sd.attn_wo[i] = try initMatrix(allocator, n_embd, n_embd, rng, &all_params);
            sd.mlp_fc1[i] = try initMatrix(allocator, 4 * n_embd, n_embd, rng, &all_params);
            sd.mlp_fc2[i] = try initMatrix(allocator, n_embd, 4 * n_embd, rng, &all_params);
        }

        sd.params = try all_params.toOwnedSlice(allocator);
        return sd;
    }

    fn initMatrix(
        allocator: Allocator,
        nout: usize,
        nin: usize,
        rng: std.Random,
        all_params: *std.ArrayList(*Value),
    ) ![][]const *Value {
        const rows = try allocator.alloc([]const *Value, nout);
        for (0..nout) |i| {
            const row = try allocator.alloc(*Value, nin);
            for (0..nin) |j| {
                const val = try Value.create(allocator, rng.floatNorm(f64) * init_std);
                row[j] = val;
                try all_params.append(allocator, val);
            }
            rows[i] = row;
        }
        return rows;
    }
};

// ============================================================================
// GPT Forward Pass
// ============================================================================

/// KV cache for attention (per layer, accumulated over positions)
const KVCache = struct {
    keys: [n_layer]std.ArrayList([]*Value),
    values: [n_layer]std.ArrayList([]*Value),

    fn init() KVCache {
        var kv: KVCache = undefined;
        for (0..n_layer) |i| {
            kv.keys[i] = .empty;
            kv.values[i] = .empty;
        }
        return kv;
    }

    fn deinit(self: *KVCache, allocator: Allocator) void {
        for (0..n_layer) |i| {
            self.keys[i].deinit(allocator);
            self.values[i].deinit(allocator);
        }
    }
};

fn gpt(allocator: Allocator, sd: *const StateDict, token_id: usize, pos_id: usize, kv: *KVCache) ![]*Value {
    // Token + position embedding
    const tok_emb = sd.wte[token_id];
    const pos_emb = sd.wpe[pos_id];
    var x = try allocator.alloc(*Value, n_embd);
    for (0..n_embd) |i| {
        x[i] = try Value.add(allocator, tok_emb[i], pos_emb[i]);
    }

    // Pre-attention RMSNorm (note: not redundant due to backward pass via residual)
    x = try rmsnorm(allocator, x);

    for (0..n_layer) |li| {
        // 1) Multi-head Attention
        const x_residual = x;
        x = try rmsnorm(allocator, x);

        const q = try linear(n_embd, allocator, x, sd.attn_wq[li]);
        const k = try linear(n_embd, allocator, x, sd.attn_wk[li]);
        const v = try linear(n_embd, allocator, x, sd.attn_wv[li]);

        try kv.keys[li].append(allocator, k);
        try kv.values[li].append(allocator, v);

        const x_attn = try allocator.alloc(*Value, n_embd);
        for (0..n_head) |h| {
            const hs = h * head_dim;

            // Slice q for this head
            const q_h = q[hs .. hs + head_dim];

            // Gather k, v for all cached positions for this head
            const n_pos = kv.keys[li].items.len;
            const attn_logits = try allocator.alloc(*Value, n_pos);

            for (0..n_pos) |t| {
                const k_t = kv.keys[li].items[t];
                var dot = try Value.create(allocator, 0.0);
                for (0..head_dim) |j| {
                    const prod = try Value.mul(allocator, q_h[j], k_t[hs + j]);
                    dot = try Value.add(allocator, dot, prod);
                }
                attn_logits[t] = try Value.mulScalar(allocator, dot, 1.0 / @sqrt(@as(f64, head_dim)));
            }

            const attn_weights = try softmax(allocator, attn_logits);

            for (0..head_dim) |j| {
                var head_out_j = try Value.create(allocator, 0.0);
                for (0..n_pos) |t| {
                    const v_t = kv.values[li].items[t];
                    const prod = try Value.mul(allocator, attn_weights[t], v_t[hs + j]);
                    head_out_j = try Value.add(allocator, head_out_j, prod);
                }
                x_attn[hs + j] = head_out_j;
            }
        }

        x = try linear(n_embd, allocator, x_attn, sd.attn_wo[li]);
        for (0..n_embd) |i| {
            x[i] = try Value.add(allocator, x[i], x_residual[i]);
        }

        // 2) MLP
        const x_residual2 = x;
        x = try rmsnorm(allocator, x);
        x = try linear(n_embd, allocator, x, sd.mlp_fc1[li]);
        for (0..4 * n_embd) |i| {
            x[i] = try Value.relu(allocator, x[i]);
        }
        x = try linear(4 * n_embd, allocator, x, sd.mlp_fc2[li]);
        for (0..n_embd) |i| {
            x[i] = try Value.add(allocator, x[i], x_residual2[i]);
        }
    }

    return try linear(n_embd, allocator, x, sd.lm_head);
}

// ============================================================================
// Shuffle (matching Python's random.shuffle with seed 42)
// ============================================================================
fn shuffle(comptime T: type, items: []T, rng: std.Random) void {
    if (items.len <= 1) return;
    var i = items.len - 1;
    while (i > 0) : (i -= 1) {
        const j = rng.intRangeAtMost(usize, 0, i);
        const tmp = items[i];
        items[i] = items[j];
        items[j] = tmp;
    }
}

// ============================================================================
// Weighted random sampling
// ============================================================================
fn weightedSample(weights: []const f64, rng: std.Random) usize {
    var total: f64 = 0;
    for (weights) |w| total += w;
    var r = rng.float(f64) * total;
    for (weights, 0..) |w, i| {
        r -= w;
        if (r <= 0) return i;
    }
    return weights.len - 1;
}

// ============================================================================
// Main: Training + Inference
// ============================================================================
pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const print = std.debug.print;

    // --- Parse args ---
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.skip(); // skip program name
    const input_file = args.next() orelse "data/input.txt";

    // --- Dataset ---
    const docs = try loadOrDownloadDataset(allocator, input_file);
    print("num docs: {d}\n", .{docs.len});

    // Use a PRNG with seed 42
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Shuffle docs
    shuffle([]const u8, docs, rng);

    // --- Tokenizer ---
    const tok = try Tokenizer.init(allocator, docs);
    print("vocab size: {d}\n", .{tok.vocab_size});

    // --- Initialize parameters ---
    const sd = try StateDict.init(allocator, tok.vocab_size, rng);
    print("num params: {d}\n", .{sd.params.len});

    // --- Adam optimizer buffers ---
    const m_buf = try allocator.alloc(f64, sd.params.len);
    defer allocator.free(m_buf);
    @memset(m_buf, 0.0);
    const v_buf = try allocator.alloc(f64, sd.params.len);
    defer allocator.free(v_buf);
    @memset(v_buf, 0.0);

    // Pre-allocate a reusable buffer for per-step computation graphs.
    // FixedBufferAllocator avoids mmap/munmap syscalls that page_allocator
    // would issue on every ArenaAllocator chunk request.
    const step_buf = try allocator.alloc(u8, 32 * 1024 * 1024); // 32 MB
    defer allocator.free(step_buf);

    // --- Training loop ---
    for (0..num_steps) |step| {
        // Per-step arena backed by pre-allocated buffer (no syscalls)
        var fba = std.heap.FixedBufferAllocator.init(step_buf);
        var arena = std.heap.ArenaAllocator.init(fba.allocator());
        defer arena.deinit();
        const step_alloc = arena.allocator();

        const doc = docs[step % docs.len];
        const tokens = try tok.encode(step_alloc, doc);
        const n: usize = @min(block_size, tokens.len - 1);

        var kv_cache = KVCache.init();
        defer kv_cache.deinit(step_alloc);

        const losses = try step_alloc.alloc(*Value, n);
        for (0..n) |pos_id| {
            const token_id = tokens[pos_id];
            const target_id = tokens[pos_id + 1];
            const logits = try gpt(step_alloc, &sd, token_id, pos_id, &kv_cache);
            const probs = try softmax(step_alloc, logits);
            const log_prob = try Value.log(step_alloc, probs[target_id]);
            losses[pos_id] = try Value.neg(step_alloc, log_prob);
        }

        // Average loss
        var total_loss = try Value.create(step_alloc, 0.0);
        for (losses) |l| {
            total_loss = try Value.add(step_alloc, total_loss, l);
        }
        const n_f: f64 = @floatFromInt(n);
        const loss = try Value.mulScalar(step_alloc, total_loss, 1.0 / n_f);

        // Backward
        try loss.backward(step_alloc, @intCast(step + 1));

        // Adam update
        const lr_t = learning_rate * (1.0 - @as(f64, @floatFromInt(step)) / @as(f64, @floatFromInt(num_steps)));
        const step_f: f64 = @floatFromInt(step + 1);
        for (sd.params, 0..) |p, i| {
            m_buf[i] = beta1 * m_buf[i] + (1.0 - beta1) * p.grad;
            v_buf[i] = beta2 * v_buf[i] + (1.0 - beta2) * p.grad * p.grad;
            const m_hat = m_buf[i] / (1.0 - math.pow(f64, beta1, step_f));
            const v_hat = v_buf[i] / (1.0 - math.pow(f64, beta2, step_f));
            p.data -= lr_t * m_hat / (@sqrt(v_hat) + eps_adam);
            p.grad = 0;
        }

        print("step {d:4} / {d:4} | loss {d:.4}\r", .{ step + 1, num_steps, loss.data });
    }

    // --- Inference ---
    const temperature: f64 = 0.5;
    print("\n--- inference (new, hallucinated names) ---\n", .{});

    for (0..20) |sample_idx| {
        var fba = std.heap.FixedBufferAllocator.init(step_buf);
        var arena = std.heap.ArenaAllocator.init(fba.allocator());
        defer arena.deinit();
        const inf_alloc = arena.allocator();

        var kv_cache = KVCache.init();
        defer kv_cache.deinit(inf_alloc);
        var token_id: usize = tok.bos;
        var sample: std.ArrayList(u8) = .empty;

        for (0..block_size) |pos_id| {
            const logits = try gpt(inf_alloc, &sd, token_id, pos_id, &kv_cache);

            // Apply temperature
            const scaled = try inf_alloc.alloc(*Value, logits.len);
            for (logits, 0..) |l, i| {
                scaled[i] = try Value.mulScalar(inf_alloc, l, 1.0 / temperature);
            }
            const probs = try softmax(inf_alloc, scaled);

            // Extract probabilities as f64 for sampling
            const weights = try inf_alloc.alloc(f64, probs.len);
            for (probs, 0..) |p, i| {
                weights[i] = p.data;
            }

            token_id = weightedSample(weights, rng);
            if (token_id == tok.bos) break;
            if (tok.decode(token_id)) |ch| {
                try sample.append(inf_alloc, ch);
            }
        }

        print("sample {d:2}: {s}\n", .{ sample_idx + 1, sample.items });
    }
}

// ============================================================================
// Tests
// ============================================================================
test "Value add" {
    const allocator = std.testing.allocator;
    const a = try Value.create(allocator, 2.0);
    defer allocator.destroy(a);
    const b = try Value.create(allocator, 3.0);
    defer allocator.destroy(b);
    const c = try Value.add(allocator, a, b);
    defer allocator.destroy(c);
    try std.testing.expectApproxEqAbs(5.0, c.data, 1e-10);
}

test "Value mul" {
    const allocator = std.testing.allocator;
    const a = try Value.create(allocator, 2.0);
    defer allocator.destroy(a);
    const b = try Value.create(allocator, 3.0);
    defer allocator.destroy(b);
    const c = try Value.mul(allocator, a, b);
    defer allocator.destroy(c);
    try std.testing.expectApproxEqAbs(6.0, c.data, 1e-10);
}

test "Value backward simple" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const a = try Value.create(alloc, 2.0);
    const b = try Value.create(alloc, 3.0);
    const c = try Value.mul(alloc, a, b); // c = a * b = 6
    const d = try Value.add(alloc, c, a); // d = c + a = 8
    try d.backward(alloc, 1);

    // dd/da = dc/da + 1 = b + 1 = 4
    try std.testing.expectApproxEqAbs(4.0, a.grad, 1e-10);
    // dd/db = dc/db = a = 2
    try std.testing.expectApproxEqAbs(2.0, b.grad, 1e-10);
}

test "Tokenizer encode decode" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const docs = [_][]const u8{ "abc", "bca" };
    const tok = try Tokenizer.init(alloc, &docs);

    try std.testing.expectEqual(@as(usize, 4), tok.vocab_size); // a,b,c + BOS
    try std.testing.expectEqual(@as(usize, 3), tok.bos);

    const encoded = try tok.encode(alloc, "abc");
    try std.testing.expectEqual(@as(usize, 5), encoded.len); // BOS + a + b + c + BOS
    try std.testing.expectEqual(@as(usize, 3), encoded[0]); // BOS
    try std.testing.expectEqual(@as(usize, 3), encoded[4]); // BOS
}
