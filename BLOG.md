# Rust SIMD Benchmark: std::simd vs NEON on Apple M4

A friend shared Sylvain Kerkour's post [SIMD programming in pure Rust](https://kerkour.com/simd-rust), which covers AVX-512 on AMD Zen 5. That got me curious about ARM's side of the story, specifically how `std::simd` compares to hand-written NEON intrinsics. My main development machine is a MacBook Pro with Apple M4, so I ran the benchmarks there.

I ran 9 benchmarks comparing three approaches: scalar, `std::simd`, and NEON. All numbers below are median per-iteration times from `cargo run --release`, averaged over 3 runs.

**Key finding**: once the `std::simd` and NEON implementations are made structurally equivalent, they perform the same. Across all 9 scenarios the two APIs are within 5% of each other, except byte search where NEON keeps a 9% lead that traces to load addressing, not to SIMD. What actually moved the numbers was never the choice of API. It was vector width, the number of accumulators, how often the loop does a horizontal reduction, and the shape of the scalar loop around the SIMD code.

## A Note on the Previous Version of This Post

The first version of this post claimed that `std::simd` ranged from 9x faster to 7.7x slower than scalar code, with interleaved data (RGB, stereo audio) exposing "limitations in the portable SIMD abstraction." That conclusion was wrong, for two separate reasons.

First, the original `std::simd` implementations deinterleaved RGB, widened i16 to i32, and narrowed back with per-element scalar loops and `from_array`. Those loops were the whole cost. Readers fixed them: [nicofff](https://github.com/nicofff) caught that the first table's 4x NEON figure for RGB was wrong ([#2](https://github.com/Erio-Harrison/simd_benchmark/issues/2)), rewrote the audio code with `from_slice`, `cast` and `copy_to_slice`, and moved the scalar baselines to fixed-point and halving-add (PRs #3, #4, #6, #7); [programmerjake](https://github.com/programmerjake) proposed the `simd_swizzle!` deinterleave for RGB and pointed out that it compiles to a single `ld3` ([#9](https://github.com/Erio-Harrison/simd_benchmark/issues/9)); [sutajo](https://github.com/sutajo) proposed the `simd_swizzle!` approach for the sorted check ([#10](https://github.com/Erio-Harrison/simd_benchmark/issues/10)). With those in, the "7.7x slower" cases disappeared.

Second, even after that, the comparison was not fair. The `std::simd` dot product used `f32x8` while the NEON version used a single `f32x4` accumulator. The `std::simd` byte counter used 32 lanes while NEON used 16. The sorted check used two different algorithms. Every remaining gap in the table traced back to one of these mismatches, not to the API.

This rewrite uses the current code, where both SIMD versions of each scenario follow the same rules (see "Comparison Rules" in the README): same elements per iteration, same number of accumulators, same reduction frequency, same algorithm.

## Test Environment

- **CPU**: Apple M4 (MacBook Pro 2024)
- **Rust**: rustc 1.95.0-nightly (2026-02-01)
- **OS**: macOS
- **Command**: `cargo run --release`, 3 runs averaged
- **Harness**: each implementation gets 100 ms of warmup, then 500 ms of timed iterations spread over 5 rounds. The three implementations of a scenario are measured round-robin (Scalar, std::simd, NEON, Scalar, ...) so clock drift and background noise hit all of them equally. The reported figure is the median per-iteration time, which gives between 100 samples (dot product) and 5,500 samples (RGB) per implementation.
- **Code**: commit `a6e7bd6` and later

Three implementations per scenario:

| Approach | Stability | Portability |
|----------|-----------|-------------|
| Scalar | stable | everywhere |
| std::simd | nightly | cross-platform |
| std::arch (NEON) | stable | ARM64 only |

A note on the harness, because it changed the numbers: the first version of this post ran each implementation 10 times in sequence and reported the mean. For a 0.1 ms workload that is 1 ms of measurement, before the CPU has ramped its clock or the scheduler has settled the process on a performance core, and the sub-millisecond rows varied by up to 2x between runs. With the time-budgeted, interleaved harness the medians agree to within 3% across runs, and the audio scenarios were enlarged from 10 s to 60 s of stereo so that one iteration is a few hundred microseconds and the working set comes from memory rather than L2, like every other scenario.

## Results Summary

| Scenario | Scalar | std::simd | NEON | std::simd | NEON |
|----------|--------|-----------|------|-----------|------|
| RGB→Grayscale (2M px) | 0.090ms | 0.090ms | 0.090ms | **1.00x** | **1.00x** |
| Volume Adjust (5.3M) | 0.447ms | 0.450ms | 0.452ms | **0.99x** | **0.99x** |
| Audio Mixing (5.3M) | 0.279ms | 0.280ms | 0.278ms | **1.00x** | **1.00x** |
| Count Newlines (10MB) | 1.228ms | 0.193ms | 0.192ms | **6.36x** | **6.40x** |
| Find Byte (10MB) | 2.494ms | 0.181ms | 0.165ms | **13.8x** | **15.1x** |
| Dot Product (10M) | 4.987ms | 1.029ms | 1.059ms | **4.85x** | **4.71x** |
| Matrix-Vec Mul (1024²) | 0.433ms | 0.077ms | 0.077ms | **5.62x** | **5.62x** |
| Range Check (10M) | 2.486ms | 0.773ms | 0.775ms | **3.22x** | **3.21x** |
| Sorted Check (10M) | 2.473ms | 0.548ms | 0.569ms | **4.51x** | **4.35x** |

Two things stand out. The two SIMD columns are nearly identical. And the three "interleaved data" scenarios show no speedup at all, for either API. The rest of this post explains both.

For reference, here is what the same scenarios looked like one commit earlier, before the implementations were aligned (single run, same machine):

| Scenario | std::simd | NEON | What was different |
|----------|-----------|------|--------------------|
| Dot Product | 4.72x | 2.61x | `f32x8` vs one `f32x4` accumulator |
| Matrix-Vec Mul | 6.60x | 3.15x | same, reused dot product |
| Count Newlines | 4.68x | 7.54x | `u8x32` vs `u8x16` |
| Range Check | 3.07x | 2.67x | `i32x8` vs `i32x4` |
| Sorted Check | 4.70x | 3.58x | swizzle on 16 lanes vs overlapping loads on 4 |

Each of these gaps looked like a statement about the API. None of them was.

---

## Finding 1: f32x8 Is Secretly Two Accumulators

**Scenario**: dot product of two 10M-element f32 vectors.

The original NEON implementation was the textbook version:

```rust
let mut acc = vdupq_n_f32(0.0);
for i in (0..simd_len).step_by(4) {
    let va = vld1q_f32(a.as_ptr().add(i));
    let vb = vld1q_f32(b.as_ptr().add(i));
    acc = vfmaq_f32(acc, va, vb);
}
vaddvq_f32(acc)
```

Every iteration depends on the previous `acc`. The loop can run no faster than one FMA latency per 4 floats, regardless of how many FMA units the core has.

The `std::simd` version used `f32x8`:

```rust
let mut acc = f32x8::splat(0.0);
for (a_chunk, b_chunk) in chunks_a.zip(chunks_b) {
    let va = f32x8::from_slice(a_chunk);
    let vb = f32x8::from_slice(b_chunk);
    acc = va.mul_add(vb, acc);
}
acc.reduce_sum()
```

NEON registers are 128 bits, so there is no 8-wide f32 register. The compiler lowers `f32x8` to two registers and the loop body to two independent FMAs. That is two accumulation chains, and the CPU overlaps them. The "portable abstraction" was not generating smarter code. It was unrolled by two, and the hand-written version was not.

Measured on the probe harness (median of 15, two separate runs):

| Variant | Run 1 | Run 2 |
|---------|-------|-------|
| std::simd f32x4, 1 accumulator | 1.49ms | 1.50ms |
| std::simd f32x8 (2 chains) | 1.06ms | 1.19ms |
| std::simd f32x16 (4 chains) | 1.06ms | 1.20ms |
| NEON f32x4, 1 accumulator | 2.59ms | 1.95ms |
| NEON f32x4, 2 accumulators | 1.18ms | 1.09ms |
| NEON f32x4, 4 accumulators | 1.11ms | 1.10ms |

Two accumulators is where both APIs hit the memory bandwidth ceiling (80MB in about 1.1ms is roughly 75GB/s on one core). Going wider changes nothing. The current NEON code uses two explicit accumulators:

```rust
let mut acc0 = vdupq_n_f32(0.0);
let mut acc1 = vdupq_n_f32(0.0);
for i in (0..simd_len).step_by(8) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a.as_ptr().add(i + 4)), vld1q_f32(b.as_ptr().add(i + 4)));
}
vaddvq_f32(vaddq_f32(acc0, acc1))
```

One more correction from the first version: the `std::simd` code used to read `acc += va * vb` with a comment calling it FMA. Rust never contracts a separate multiply and add into a fused one. If you want FMA you write `mul_add` (from `std::simd::StdFloat`). In this memory-bound loop it makes no measurable difference, but the comment was wrong and the numerics differ from the NEON version.

---

## Finding 2: `to_bitmask` on 32 Lanes Is a Trap on NEON

**Scenario**: count `\n` bytes in 10MB of text.

The core loop is the same in both APIs: compare 16 bytes against the target, turn the mask into a count, accumulate.

```rust
// std::simd
let mask = u8x16::from_slice(chunk).simd_eq(target_vec);
count += mask.to_bitmask().count_ones() as usize;

// NEON
let eq = vceqq_u8(v, target_vec);          // 0xFF where equal
let ones = vshrq_n_u8(eq, 7);              // 0xFF -> 1
count += vaddvq_u8(ones) as usize;
```

The first version of this post used `u8x32` for the `std::simd` side and explained its lead over NEON as "256-bit vectors versus 128-bit." The measurement said otherwise:

| Variant | Run 1 | Run 2 |
|---------|-------|-------|
| std::simd u8x32 | 0.287ms | 0.318ms |
| std::simd u8x16 | 0.193ms | 0.197ms |
| NEON u8x16 | 0.181ms | 0.197ms |

Wider was slower. x86 has `movemask`, which turns a byte mask into a bitmask in one instruction. NEON has nothing like it, so a 32-lane bitmask has to be assembled from two registers on every iteration. I did not keep the 32-lane assembly, so treat the mechanism as the likely explanation and the timing as the fact.

At 16 lanes the two APIs do not just perform the same, they compile to the same thing. In the release assembly both `count_byte` loops contain one `sdot` and one `addv`: LLVM recognized both the `to_bitmask().count_ones()` idiom and the `vshrq_n_u8` + `vaddvq_u8` idiom and replaced them with a dot-product popcount. Neither version runs the instructions its source code names.

### A 15% gap that had nothing to do with SIMD

The byte search scenario was the one place where NEON kept a lead after alignment: 0.157ms against 0.181ms. Both loops test "any lane set?" each iteration and only extract the position on a hit, and both compile that test to a single `umaxv`. The hot loops were eight instructions each. So where did 15% come from?

Swapping the loop *shapes* between the two APIs answered it (probe harness, median of 7 rounds):

| Variant | Time |
|---------|------|
| std::simd, `chunks_exact().enumerate()` (old code) | 0.194ms |
| std::simd, `chunks_exact()` with a manual offset | 0.169ms |
| std::simd, index loop with raw pointer loads | 0.157ms |
| std::simd, index loop with `&data[i..i+16]` | 0.229ms |
| NEON, index loop with raw pointer loads (repo) | 0.157ms |
| NEON, `chunks_exact()` iterator | 0.174ms |

The gap follows the loop shape, not the API. Give `std::simd` the NEON loop and it runs at NEON speed; give NEON the iterator loop and it slows down to the `std::simd` number. Two things were going on:

- `.enumerate()` adds a second loop counter. In a loop that already runs at about one iteration per cycle (10MB in 0.16ms is 655K iterations in roughly 630K cycles), one extra integer op per iteration is a measurable 13%. Tracking the offset by hand removes it.
- `chunks_exact` lowers to a pointer-bump loop with a post-indexed load (`ldr q, [x], #16`), while the index loop lowers to base-plus-index addressing (`ldr q, [base, idx]`). On M4 the writeback form is a further 5 to 7% slower in this loop.

The bounds-checked slice variant is the slowest of all: `from_slice` asserts the slice length on every iteration and LLVM does not prove it away.

The repository now uses the manual-offset form, which is still safe code. That removes the counter but keeps the post-indexed load, so NEON stays about 9% ahead in the results table (0.165ms vs 0.181ms). Closing the last few percent would mean raw pointer loads on the `std::simd` side, and I would rather keep that version safe and document the difference. The lesson generalizes beyond SIMD: when a loop is already at one iteration per cycle, iterator adaptors that look free are not, and neither is the addressing mode the compiler picks for you.

---

## Finding 3: `vextq` Is `simd_swizzle!`

**Scenario**: check that 10M i32 values are sorted ascending.

The first version of this post had `std::simd` iterating over every `windows(9)`, stepping by one element and so doing eight times the necessary comparisons; that is where the original 0.68x came from. qdot3 diagnosed it in [#11](https://github.com/Erio-Harrison/simd_benchmark/issues/11) and proposed `step_by(8)`, which would have fixed it on its own. I went with sutajo's suggestion from #10 instead: build a "previous element" vector with `simd_swizzle!` and compare it to the current chunk, 16 lanes at a time, one load per iteration. NEON meanwhile loaded overlapping windows (`data[i..i+4]` and `data[i+1..i+5]`), 4 lanes at a time, so the two implementations used different algorithms. `std::simd` won, 4.70x to 3.58x, and the gap looked like an API difference.

It was two differences: algorithm and width. NEON has a direct equivalent of the swizzle. `vextq_s32(a, b, 3)` concatenates two vectors and extracts four lanes starting at index 3, giving `[a3, b0, b1, b2]`. That is exactly "shift the previous chunk's last element in front of the current chunk."

```rust
// std::simd: mask = [2*LANES-1, 0, 1, ..., LANES-2]
let prev_current = simd_swizzle!(current, prev, get_swizzle_mask());
if !prev_current.simd_le(current).all() { return false; }
prev = current;

// NEON, 16 elements per iteration
let s0 = vextq_s32(prev, c0, 3);
let s1 = vextq_s32(c0, c1, 3);
let s2 = vextq_s32(c1, c2, 3);
let s3 = vextq_s32(c2, c3, 3);
let ok = vandq_u32(
    vandq_u32(vcleq_s32(s0, c0), vcleq_s32(s1, c1)),
    vandq_u32(vcleq_s32(s2, c2), vcleq_s32(s3, c3)),
);
if vminvq_u32(ok) == 0 { return false; }
prev = c3;
```

This is not just an analogy. In the release assembly the `std::simd` function contains four `ext` instructions and one `umaxv`, and so does the NEON function. The swizzle lowered to exactly the instruction the NEON code names.

With the same algorithm at the same width, the two APIs are indistinguishable:

| Variant | Time |
|---------|------|
| std::simd overlapping loads, 4 lanes | 0.673ms |
| NEON overlapping loads, 4 lanes | 0.691ms |
| std::simd swizzle, 16 lanes | 0.513ms |
| std::simd overlapping loads, 16 lanes | 0.544ms |
| NEON `vextq`, 16 lanes | 0.554ms |

The remaining 6% in the results table is within run-to-run noise.

---

## Finding 4: The Scalar Baseline Is Already SIMD

**Scenarios**: volume adjustment and two-track mixing on 5.3M i16 samples (60 s of 44.1 kHz stereo).

Both audio scenarios show 1.00x for both SIMD APIs. The first version of this post reported 0.40x and 0.13x for `std::simd` and read it as the portable API lacking widening, narrowing, and halving-add instructions. The current `std::simd` code uses `cast` for the conversions and runs at exactly the same speed as NEON. But it also runs at exactly the same speed as the scalar code, and that deserves an explanation.

The scalar mixer is one line:

```rust
*out = ((*a as i32 + *b as i32) >> 1) as i16;
```

Here is what the release build turns it into (instruction histogram of the function body):

```
   4 shadd.8h
   4 ldp
   2 stp
   ...
```

`shadd.8h` is the signed halving add on eight 16-bit lanes. It is the same instruction the NEON version calls through `vhaddq_s16`. LLVM auto-vectorized the scalar loop into the hand-written NEON code. It went one step further with the `std::simd` version, too: that code widens to `i32x8`, adds, shifts, and narrows back, and LLVM folded the whole sequence into `shadd` as well. All three mixers are the same instruction.

The volume adjustment is the same story: the scalar function compiles to `mul.4s`, `sshll.4s`, `sqshrn.4h`, which is widen, multiply, saturating narrow. And the RGB conversion's scalar loop contains `ld3` followed by `umull.8h` and `umlal2.8h`: the compiler found the deinterleaving load on its own.

So in those three rows, the table compares NEON code to NEON code to NEON code. nicofff called this early: in [#5](https://github.com/Erio-Harrison/simd_benchmark/issues/5) he asked whether the scalar versions should be written to auto-vectorize and said he would love to see the three implementations compile to the same code, and his PRs #3 and #4 (fixed-point volume, `>> 1` instead of `/ 2` in the mixer) are what let LLVM get there. Five different hand-written variants of the mixer, including one that widens to i32 and one that stays in i16 lanes, all measure 0.051ms on the probe harness at 882K samples. The loop is memory-bound and the instruction choice does not matter.

The lesson is not "SIMD does not help audio." It is that the scalar baseline in a SIMD benchmark is not what it looks like, and a 1.0x result can mean the compiler got there first. If you want to know whether explicit SIMD helps, check the baseline's assembly before drawing conclusions.

---

## The Scenarios

Full code for all three implementations of every scenario is in [`src/lib.rs`](https://github.com/Erio-Harrison/simd_benchmark/blob/master/src/lib.rs). Below are the parts that matter for each one.

### RGB to Grayscale

1920×1080 image, fixed-point BT.601: `Gray = (77*R + 150*G + 29*B) >> 8`. Both SIMD versions process 16 pixels (48 bytes) per iteration.

NEON has `vld3q_u8`, which loads 48 bytes and deinterleaves them into three registers in one instruction. `std::simd` does the same thing with a 48-lane load and three `simd_swizzle!` calls using a compile-time "every third byte" mask:

```rust
let rgb = Simd::<u8, 48>::from_slice(rgb_chunk);
let r: u16x16 = simd_swizzle!(rgb, every_third(0)).cast();
let g: u16x16 = simd_swizzle!(rgb, every_third(1)).cast();
let b: u16x16 = simd_swizzle!(rgb, every_third(2)).cast();
let gray_u16 = (r * weight_r + g * weight_g + b * weight_b) >> Simd::splat(8);
let gray_u8: Simd<u8, 16> = gray_u16.cast();
```

I expected the swizzle to lower to table lookups. It does not: LLVM recognizes the every-third-byte pattern and emits `ld3`, the same instruction the NEON code calls through `vld3q_u8`. programmerjake spotted this on godbolt when proposing the change in #9; the release assembly of this repository confirms it. The scalar version is auto-vectorized with `ld3` as well (see Finding 4), which is why all three rows are within a few percent of each other.

### Volume Adjustment and Mixing

See Finding 4. Both SIMD versions process 8 samples per iteration. The `std::simd` volume code widens with `cast`, multiplies in i32, shifts, clamps, and narrows with `cast`. NEON does the same with `vmovl`, `vmulq`, `vshrq_n`, `vqmovn`.

### Count Newlines and Find Byte

See Finding 2. Both APIs use 16 lanes.

### Dot Product and Matrix-Vector Multiplication

See Finding 1. Matrix-vector multiplication calls the dot product once per row, so it inherits the same behavior: identical speed once the accumulator count matches.

### Range Check

Both versions process 8 i32 per iteration with one horizontal reduction. `std::simd`:

```rust
let v = i32x8::from_slice(chunk);
if !(v.simd_ge(min_vec) & v.simd_le(max_vec)).all() { return false; }
```

NEON, two registers, one `vminvq`:

```rust
let ok0 = vandq_u32(vcgeq_s32(v0, min_vec), vcleq_s32(v0, max_vec));
let ok1 = vandq_u32(vcgeq_s32(v1, min_vec), vcleq_s32(v1, max_vec));
if vminvq_u32(vandq_u32(ok0, ok1)) == 0 { return false; }
```

Before alignment the NEON loop did 4 elements and a reduction per iteration and came in at 2.67x against 3.07x. Now they are 3.20x and 3.24x.

### Sorted Check

See Finding 3.

---

## What the Comparison Actually Measures

Here is the rule this repository now follows, and the one I would suggest for any `std::simd` versus intrinsics comparison:

1. Same elements per iteration in both versions.
2. Same number of accumulators and dependency chains.
3. Same number of horizontal reductions per element.
4. Same algorithm.

If you hold those fixed, the choice between `std::simd` and `std::arch` on aarch64 is not a performance choice. It is a choice about portability, stability (nightly versus stable), and how the code reads. Every performance difference I found in nine scenarios was explained by a violation of one of the four rules, or, in the byte search case, by the scalar loop machinery and load addressing around the SIMD instructions.

That cuts both ways. The things that made `std::simd` look bad in the first version (scalar conversion loops) and the things that made it look good (an implicit unroll the NEON code did not have) were both artifacts of how the two sides were written.

## Recommendations

| Situation | Approach |
|-----------|----------|
| Need stable Rust | `std::arch` intrinsics behind `cfg(target_arch)` |
| Need one code path for ARM and x86 | `std::simd` |
| Choosing a lane count | Match the hardware register, or a small multiple of it. Wider is not free: `to_bitmask` on 32 lanes is slower than on 16 on NEON. |
| Reductions in a loop | Count them. `f32x8` on NEON already gives you two chains; a single intrinsic accumulator does not. |
| Interpreting "1.0x" | Read the scalar assembly first. The compiler may have vectorized it. |
| Loops near one iteration per cycle | Iterator adaptors are not free. `.enumerate()` on `chunks_exact` cost 13% in the byte search. |

## NEON Instruction Reference

Naming pattern:
```
vld3q_u8
│││││└─ u8: data type
││││└── q: 128-bit register
│││└─── 3: 3-way deinterleave
││└──── ld: load
│└───── v: vector
```

| Instruction | Operation | std::simd equivalent |
|-------------|-----------|----------------------|
| `vld1q_u8` | Load 16 bytes | `Simd::from_slice` |
| `vld3q_u8` | Load + deinterleave RGB | `from_slice` + three `simd_swizzle!` (lowers to `ld3`) |
| `vextq_s32(a, b, 3)` | `[a3, b0, b1, b2]` | `simd_swizzle!(b, a, [7, 0, 1, 2])` |
| `vmull_u8` | Widening multiply u8→u16 | `cast::<u16>()` then `*` |
| `vmovl_s16` | Widen i16→i32 | `cast::<i32>()` |
| `vqmovn_s32` | Saturating narrow i32→i16 | `simd_clamp` then `cast::<i16>()` |
| `vfmaq_f32` | Fused multiply-add | `mul_add` (not `a * b + c`) |
| `vhaddq_s16` | Halving add | `((a.cast::<i32>() + b.cast::<i32>()) >> 1).cast()` (lowers to `shadd`) |
| `vaddvq_f32` | Horizontal sum | `reduce_sum` |
| `vminvq_u32` on a mask | "All lanes true?" | `Mask::all` |
| `vmaxvq_u8` on a mask | "Any lane true?" | `Mask::any` |

## Reproducing

```bash
rustup override set nightly
cargo run --release    # correctness checks + quick timings
cargo bench            # criterion, HTML reports in target/criterion/
```

The numbers in the first version of this post can be reproduced from commit `ca8a510` and earlier. The aligned implementations are in `a6e7bd6` and later.

## References

- [Rust std::simd documentation](https://doc.rust-lang.org/std/simd/index.html) - Portable SIMD module (nightly)
- [Rust std::arch::aarch64](https://doc.rust-lang.org/std/arch/aarch64/index.html) - ARM64 intrinsics in Rust
- [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/) - Official ARM intrinsics search
- [ARM NEON Programmer's Guide](https://developer.arm.com/documentation/den0018/a/) - NEON architecture overview

## Source

Full source code: [github.com/Erio-Harrison/simd_benchmark](https://github.com/Erio-Harrison/simd_benchmark)
