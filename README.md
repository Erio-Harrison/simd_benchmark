# Rust SIMD Benchmark

Performance comparison of three SIMD implementations: Scalar vs std::simd vs NEON

**Tested on Apple M4 (MacBook Pro 2024)**

## Requirements

- Rust nightly (for `std::simd` / portable SIMD)
- Apple Silicon Mac (M1/M2/M3/M4) for NEON intrinsics

## Setup

```bash
rustup install nightly
rustup override set nightly  # Use nightly in current directory
```

## Running Benchmarks

### Quick Test

```bash
cargo run --release
```

Runs every scenario, checks that all three implementations agree, and prints median per-iteration times. Each implementation gets 100 ms of warmup and 500 ms of timed samples, taken round-robin across the three implementations in 5 rounds so that clock drift and background noise affect them equally. Expect run-to-run medians to agree within a few percent; the whole run takes about 20 seconds.

### Criterion Detailed Benchmark

```bash
cargo bench
```

Results are saved in `target/criterion/` directory with HTML reports.

## Project Structure

```
src/
├── lib.rs          # Three implementations: scalar, portable_simd, neon
└── main.rs         # Simple benchmark runner
benches/
└── simd_bench.rs   # Criterion benchmarks
```

## Comparison Rules

The point of this repository is to compare the two SIMD *APIs*, not two different algorithms. For every scenario the `portable_simd` and `neon` implementations must therefore be structurally equivalent:

1. **Same elements per iteration.** If `std::simd` uses `f32x8`, the NEON loop processes 8 floats per iteration too (two 128-bit registers), and vice versa.
2. **Same number of accumulators / dependency chains.** A single NEON accumulator against an `f32x8` accumulator (which the compiler lowers to two independent chains) measures FMA latency, not the API.
3. **Same reduction frequency.** Horizontal operations (`reduce_sum`, `all()`, `to_bitmask`, `vminvq`, `vaddvq`) run the same number of times per element in both versions.
4. **Same algorithm.** Pick one approach per scenario (for example "shift the previous element in with `simd_swizzle!` / `vextq`") and implement it with both APIs.

When sending a PR that speeds up one side, apply the equivalent change to the other side, or explain in the PR why no equivalent exists. The scalar version is a plain baseline and is allowed to be whatever the compiler makes of idiomatic Rust, including auto-vectorized code.

## Test Scenarios

| Category | Scenario | Description |
|----------|----------|-------------|
| Image Processing | RGB to Grayscale | 1920x1080 image conversion |
| Audio Processing | Volume Adjustment | 5.3M samples (60s stereo) with gain control |
| Audio Processing | Two-track Mixing | 5.3M samples (60s stereo) mixing |
| String Search | Count Bytes | Count newlines in 10MB text |
| String Search | Find Byte | Find character position |
| Numerical | Dot Product | 10M element float vectors |
| Numerical | Matrix-Vector Multiply | 1024x1024 matrix |
| Validation | Range Check | Verify values in range |
| Validation | Sorted Check | Verify array is sorted |

## Blog Post

For detailed experimental analysis and code explanations, please see [BLOG.md](BLOG.md)
