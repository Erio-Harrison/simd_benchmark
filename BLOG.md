# Rust SIMD Benchmark: std::simd vs NEON on Apple M4

A friend shared Sylvain Kerkour's post [SIMD programming in pure Rust](https://kerkour.com/simd-rust) which covers AVX-512 on AMD Zen 5. That got me curious about ARM's side of the story—specifically how `std::simd` compares to hand-written NEON intrinsics. My main development machine is a MacBook Pro with Apple M4, so I ran the benchmarks there.

I ran 9 benchmarks comparing three approaches: scalar, `std::simd`, and NEON. All numbers below are averaged from 3 runs.

Key finding: `std::simd` ranged from **9x faster** to **7.7x slower** than scalar code. NEON always delivered speedups (1.2x–4.3x). The difference comes down to data layout—interleaved data like RGB images and stereo audio exposed limitations in the portable SIMD abstraction.

## Test Environment

- **CPU**: Apple M4 (MacBook Pro 2024)
- **Rust**: rustc 1.94.0-nightly (2026-01-14)
- **OS**: macOS
- **Command**: `cargo run --release`

Three implementations per scenario:

| Approach | Stability | Portability |
|----------|-----------|-------------|
| Scalar | stable | everywhere |
| std::simd | nightly | cross-platform |
| std::arch (NEON) | stable | ARM64 only |

## Results Summary

| Scenario | Scalar | std::simd | NEON | std::simd | NEON |
|----------|--------|-----------|------|-----------|------|
| RGB→Grayscale | 6.16ms | 18.51ms | 1.45ms | **0.33x** | **1.26x** |
| Volume Adjust | 1.39ms | 3.42ms | 0.82ms | **0.40x** | **1.70x** |
| Audio Mixing | 0.63ms | 4.84ms | 0.53ms | **0.13x** | **1.20x** |
| Count Newlines | 14.10ms | 3.22ms | 3.87ms | **4.38x** | **3.65x** |
| Find Byte | 24.07ms | 2.61ms | 2.68ms | **9.23x** | **9.00x** |
| Dot Product | 51.20ms | 10.99ms | 20.14ms | **4.66x** | **2.54x** |
| Matrix-Vec Mul | 4.48ms | 0.69ms | 1.35ms | **6.53x** | **3.32x** |
| Range Check | 24.93ms | 8.24ms | 9.33ms | **3.02x** | **2.67x** |
| Sorted Check | 25.14ms | 37.18ms | 6.87ms | **0.68x** | **3.66x** |

Three scenarios showed `std::simd` slower than scalar: RGB conversion (0.33x), audio mixing (0.13x), and sorted check (0.68x).

---

## Scenario 1: RGB to Grayscale

**Task**: Convert 1920×1080 RGB image to grayscale using ITU-R BT.601 formula.

I used fixed-point arithmetic to avoid floating-point operations—much faster on integer pipelines: `Gray = (77*R + 150*G + 29*B) >> 8`

### Scalar

The scalar version is straightforward with `chunks_exact(3)`:

```rust
pub fn rgb_to_grayscale(rgb: &[u8], gray: &mut [u8]) {
    for (i, chunk) in rgb.chunks_exact(3).enumerate() {
        let r = chunk[0] as u32;
        let g = chunk[1] as u32;
        let b = chunk[2] as u32;
        gray[i] = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
    }
}
```

### std::simd

```rust
pub fn rgb_to_grayscale(rgb: &[u8], gray: &mut [u8]) {
    let chunks = rgb.chunks_exact(48);
    let weight_r = u16x16::splat(77);
    let weight_g = u16x16::splat(150);
    let weight_b = u16x16::splat(29);

    let mut out_idx = 0;
    for chunk in chunks {
        // No deinterleave instruction available
        // Must use scalar loop: 48 memory accesses per iteration
        let mut r_vals = [0u8; 16];
        let mut g_vals = [0u8; 16];
        let mut b_vals = [0u8; 16];

        for i in 0..16 {
            r_vals[i] = chunk[i * 3];
            g_vals[i] = chunk[i * 3 + 1];
            b_vals[i] = chunk[i * 3 + 2];
        }

        let r = u16x16::from_array(r_vals.map(|x| x as u16));
        let g = u16x16::from_array(g_vals.map(|x| x as u16));
        let b = u16x16::from_array(b_vals.map(|x| x as u16));

        let gray_u16 = (r * weight_r + g * weight_g + b * weight_b) >> Simd::splat(8);

        let gray_u8: [u8; 16] = gray_u16.to_array().map(|x| x as u8);
        gray[out_idx..out_idx + 16].copy_from_slice(&gray_u8);
        out_idx += 16;
    }
}
```

### NEON

```rust
pub fn rgb_to_grayscale(rgb: &[u8], gray: &mut [u8]) {
    unsafe {
        let weight_r = vdupq_n_u8(77);
        let weight_g = vdupq_n_u8(150);
        let weight_b = vdupq_n_u8(29);

        for i in (0..simd_len).step_by(16) {
            // vld3q_u8: load + deinterleave in one instruction
            // [R0,G0,B0,R1,G1,B1,...] → R[], G[], B[]
            let rgb_data = vld3q_u8(rgb.as_ptr().add(i * 3));
            let r = rgb_data.0;
            let g = rgb_data.1;
            let b = rgb_data.2;

            // Widening multiply: u8 × u8 → u16
            let r_lo = vmull_u8(vget_low_u8(r), vget_low_u8(weight_r));
            let r_hi = vmull_high_u8(r, weight_r);
            let g_lo = vmull_u8(vget_low_u8(g), vget_low_u8(weight_g));
            let g_hi = vmull_high_u8(g, weight_g);
            let b_lo = vmull_u8(vget_low_u8(b), vget_low_u8(weight_b));
            let b_hi = vmull_high_u8(b, weight_b);

            let sum_lo = vaddq_u16(vaddq_u16(r_lo, g_lo), b_lo);
            let sum_hi = vaddq_u16(vaddq_u16(r_hi, g_hi), b_hi);

            // Shift right + narrow: u16 → u8
            let gray_lo = vshrn_n_u16(sum_lo, 8);
            let gray_hi = vshrn_n_u16(sum_hi, 8);
            let gray_vec = vcombine_u8(gray_lo, gray_hi);

            vst1q_u8(gray.as_mut_ptr().add(i), gray_vec);
        }
    }
}
```

### Results

| Implementation | Time | vs Scalar |
|----------------|------|-----------|
| Scalar | 6.16ms | 1.0x |
| std::simd | 18.51ms | 0.33x |
| NEON | 1.45ms | 4.26x |

**Analysis**: `vld3q_u8` performs load and 3-way deinterleave in one instruction. `std::simd` lacks this capability, requiring a 16-iteration scalar loop that dominates execution time.

---

## Scenario 2: Audio Volume Adjustment

**Task**: Multiply 880K audio samples (i16) by gain factor 0.8.

I used fixed-point here too—multiplying by 256 and shifting right 8 bits keeps everything in integer domain.

### Scalar

```rust
pub fn adjust_volume(samples: &mut [i16], volume: f32) {
    for sample in samples.iter_mut() {
        let adjusted = (*sample as f32 * volume) as i32;
        *sample = adjusted.clamp(-32768, 32767) as i16;
    }
}
```

### std::simd

```rust
pub fn adjust_volume(samples: &mut [i16], volume: f32) {
    let vol_fixed = (volume * 256.0) as i32;
    let vol_vec = i32x8::splat(vol_fixed);

    for i in (0..simd_len).step_by(8) {
        // No i16 → i32 widening instruction
        let mut vals = [0i32; 8];
        for j in 0..8 {
            vals[j] = samples[i + j] as i32;
        }
        let v = i32x8::from_array(vals);

        let adjusted = (v * vol_vec) >> Simd::splat(8);
        let clamped = adjusted.simd_clamp(i32x8::splat(-32768), i32x8::splat(32767));

        // No i32 → i16 narrowing instruction
        for (j, &val) in clamped.to_array().iter().enumerate() {
            samples[i + j] = val as i16;
        }
    }
}
```

### NEON

```rust
pub fn adjust_volume(samples: &mut [i16], volume: f32) {
    let vol_fixed = (volume * 256.0) as i32;

    unsafe {
        let vol_vec = vdupq_n_s32(vol_fixed);

        for i in (0..simd_len).step_by(8) {
            let v = vld1q_s16(samples.as_ptr().add(i));

            // vmovl: widen i16 → i32
            let v_lo = vmovl_s16(vget_low_s16(v));
            let v_hi = vmovl_high_s16(v);

            let mul_lo = vmulq_s32(v_lo, vol_vec);
            let mul_hi = vmulq_s32(v_hi, vol_vec);

            let shifted_lo = vshrq_n_s32(mul_lo, 8);
            let shifted_hi = vshrq_n_s32(mul_hi, 8);

            // vqmovn: saturating narrow i32 → i16
            let result_lo = vqmovn_s32(shifted_lo);
            let result_hi = vqmovn_s32(shifted_hi);
            let result = vcombine_s16(result_lo, result_hi);

            vst1q_s16(samples.as_mut_ptr().add(i), result);
        }
    }
}
```

### Results

| Implementation | Time | vs Scalar |
|----------------|------|-----------|
| Scalar | 1.39ms | 1.0x |
| std::simd | 3.42ms | 0.40x |
| NEON | 0.82ms | 1.70x |

**Analysis**: `std::simd` lacks type conversion instructions. `vmovl` (widen) and `vqmovn` (saturating narrow) are single instructions in NEON but require scalar loops in `std::simd`.

---

## Scenario 3: Audio Mixing

**Task**: Mix two audio tracks using `(a + b) / 2` to prevent overflow.

Division by 2 is the classic way to mix audio without clipping. Simple but requires widening to i32 first.

### Scalar

```rust
pub fn mix_tracks(track_a: &[i16], track_b: &[i16], output: &mut [i16]) {
    for ((a, b), out) in track_a.iter().zip(track_b.iter()).zip(output.iter_mut()) {
        *out = ((*a as i32 + *b as i32) / 2) as i16;
    }
}
```

### std::simd

```rust
pub fn mix_tracks(track_a: &[i16], track_b: &[i16], output: &mut [i16]) {
    for i in (0..simd_len).step_by(8) {
        let mut a_vals = [0i32; 8];
        let mut b_vals = [0i32; 8];
        for j in 0..8 {
            a_vals[j] = track_a[i + j] as i32;
            b_vals[j] = track_b[i + j] as i32;
        }

        let a = i32x8::from_array(a_vals);
        let b = i32x8::from_array(b_vals);
        let mixed = (a + b) >> Simd::splat(1);

        for (j, &val) in mixed.to_array().iter().enumerate() {
            output[i + j] = val as i16;
        }
    }
}
```

### NEON

```rust
pub fn mix_tracks(track_a: &[i16], track_b: &[i16], output: &mut [i16]) {
    unsafe {
        for i in (0..simd_len).step_by(8) {
            let a = vld1q_s16(track_a.as_ptr().add(i));
            let b = vld1q_s16(track_b.as_ptr().add(i));

            // vhaddq: halving add, computes (a + b) / 2 without overflow
            let mixed = vhaddq_s16(a, b);

            vst1q_s16(output.as_mut_ptr().add(i), mixed);
        }
    }
}
```

### Results

| Implementation | Time | vs Scalar |
|----------------|------|-----------|
| Scalar | 0.63ms | 1.0x |
| std::simd | 4.84ms | 0.13x |
| NEON | 0.53ms | 1.20x |

**Analysis**: `vhaddq_s16` performs halving add in one instruction—designed specifically for audio mixing. `std::simd` requires widening to i32, adding, shifting, and narrowing back, plus three scalar loops for type conversions.

---

## Scenario 4: Counting Newlines

**Task**: Count occurrences of `\n` in 10MB text.

This is where SIMD really shines—contiguous data with simple comparison. I used `filter().count()` for scalar, which Rust's iterator makes clean.

### Scalar

```rust
pub fn count_byte(data: &[u8], target: u8) -> usize {
    data.iter().filter(|&&b| b == target).count()
}
```

### std::simd

```rust
pub fn count_byte(data: &[u8], target: u8) -> usize {
    let target_vec = u8x32::splat(target);
    let chunks = data.chunks_exact(32);
    let mut count = 0usize;

    for chunk in chunks {
        let v = u8x32::from_slice(chunk);
        let mask = v.simd_eq(target_vec);
        count += mask.to_bitmask().count_ones() as usize;
    }

    count += chunks.remainder().iter().filter(|&&b| b == target).count();
    count
}
```

### NEON

```rust
pub fn count_byte(data: &[u8], target: u8) -> usize {
    let mut count = 0usize;

    unsafe {
        let target_vec = vdupq_n_u8(target);

        for i in (0..simd_len).step_by(16) {
            let v = vld1q_u8(data.as_ptr().add(i));
            let eq = vceqq_u8(v, target_vec);
            let ones = vshrq_n_u8(eq, 7);
            count += vaddvq_u8(ones) as usize;
        }
    }

    count
}
```

### Results

| Implementation | Time | vs Scalar |
|----------------|------|-----------|
| Scalar | 14.10ms | 1.0x |
| std::simd | 3.22ms | 4.38x |
| NEON | 3.87ms | 3.65x |

**Analysis**: Contiguous data, simple comparison, no type conversions. Both SIMD approaches perform well. `std::simd` uses 256-bit vectors, NEON uses 128-bit, explaining the slight `std::simd` advantage.

---

## Scenario 5: Finding a Byte

**Task**: Find position of `@` in 10MB data (target near end).

I placed the target byte near the end to simulate worst-case search. The `position()` iterator is nice for scalar.

### Scalar

```rust
pub fn find_byte(data: &[u8], target: u8) -> Option<usize> {
    data.iter().position(|&b| b == target)
}
```

### std::simd

```rust
pub fn find_byte(data: &[u8], target: u8) -> Option<usize> {
    let target_vec = u8x32::splat(target);
    let chunks = data.chunks_exact(32);

    for (chunk_idx, chunk) in chunks.enumerate() {
        let v = u8x32::from_slice(chunk);
        let mask = v.simd_eq(target_vec);
        let bitmask = mask.to_bitmask();
        if bitmask != 0 {
            return Some(chunk_idx * 32 + bitmask.trailing_zeros() as usize);
        }
    }

    None
}
```

### NEON

```rust
pub fn find_byte(data: &[u8], target: u8) -> Option<usize> {
    let len = data.len();
    let simd_len = len - (len % 16);

    unsafe {
        let target_vec = vdupq_n_u8(target);

        for i in (0..simd_len).step_by(16) {
            let v = vld1q_u8(data.as_ptr().add(i));
            let eq = vceqq_u8(v, target_vec);
            // vmaxvq_u8: if any 0xFF exists, returns 0xFF
            if vmaxvq_u8(eq) != 0 {
                // Found match, linear search for exact position
                for j in 0..16 {
                    if data[i + j] == target {
                        return Some(i + j);
                    }
                }
            }
        }
    }

    None
}
```

### Results

| Implementation | Time | vs Scalar |
|----------------|------|-----------|
| Scalar | 24.07ms | 1.0x |
| std::simd | 2.61ms | 9.23x |
| NEON | 2.68ms | 9.00x |

**Analysis**: Best-case SIMD scenario. Compare 32 bytes per iteration, early exit on match.

---

## Scenario 6: Dot Product

**Task**: Dot product of two 10M-element f32 vectors.

Classic numerical computing workload. Rust's iterator chain makes the scalar version readable.

### Scalar

```rust
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
```

### std::simd

```rust
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);

    let mut acc = f32x8::splat(0.0);

    for (a_chunk, b_chunk) in chunks_a.zip(chunks_b) {
        let va = f32x8::from_slice(a_chunk);
        let vb = f32x8::from_slice(b_chunk);
        acc += va * vb;
    }

    acc.reduce_sum()
}
```

### NEON

```rust
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let mut acc = vdupq_n_f32(0.0);

        for i in (0..simd_len).step_by(4) {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            acc = vfmaq_f32(acc, va, vb);
        }

        vaddvq_f32(acc)
    }
}
```

### Results

| Implementation | Time | vs Scalar |
|----------------|------|-----------|
| Scalar | 51.20ms | 1.0x |
| std::simd | 10.99ms | 4.66x |
| NEON | 20.14ms | 2.54x |

**Analysis**: `std::simd` outperforms hand-written NEON. `std::simd` uses f32x8 (256-bit), while the NEON implementation uses f32x4 (128-bit). The compiler generates efficient code from the portable abstraction.

---

## Scenario 7: Matrix-Vector Multiplication

**Task**: 1024×1024 matrix times 1024-element vector.

I reused the dot product implementation row by row—keeps the code simple.

### Scalar

```rust
pub fn matrix_vector_mul(matrix: &[f32], vector: &[f32], result: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let row_start = i * cols;
        result[i] = matrix[row_start..row_start + cols]
            .iter()
            .zip(vector.iter())
            .map(|(m, v)| m * v)
            .sum();
    }
}
```

### std::simd

```rust
pub fn matrix_vector_mul(matrix: &[f32], vector: &[f32], result: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let row = &matrix[i * cols..(i + 1) * cols];
        result[i] = dot_product(row, vector);  // reuse SIMD dot product
    }
}
```

### NEON

```rust
pub fn matrix_vector_mul(matrix: &[f32], vector: &[f32], result: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let row = &matrix[i * cols..(i + 1) * cols];
        result[i] = dot_product(row, vector);  // reuse NEON dot product
    }
}
```

### Results

| Implementation | Time | vs Scalar |
|----------------|------|-----------|
| Scalar | 4.48ms | 1.0x |
| std::simd | 0.69ms | 6.53x |
| NEON | 1.35ms | 3.32x |

**Analysis**: Regular memory access pattern, same characteristics as dot product.

---

## Scenario 8: Range Check

**Task**: Verify all 10M i32 values are in `[0, 100)`.

Early exit on failure makes this fast when data is invalid. I kept all values in range to measure worst-case (full scan).

### Scalar

```rust
pub fn all_in_range(data: &[i32], min: i32, max: i32) -> bool {
    data.iter().all(|&x| x >= min && x <= max)
}
```

### std::simd

```rust
pub fn all_in_range(data: &[i32], min: i32, max: i32) -> bool {
    let min_vec = i32x8::splat(min);
    let max_vec = i32x8::splat(max);

    for chunk in data.chunks_exact(8) {
        let v = i32x8::from_slice(chunk);
        let ge_min = v.simd_ge(min_vec);
        let le_max = v.simd_le(max_vec);
        if !(ge_min & le_max).all() {
            return false;
        }
    }

    true
}
```

### NEON

```rust
pub fn all_in_range(data: &[i32], min: i32, max: i32) -> bool {
    let len = data.len();
    let simd_len = len - (len % 4);

    unsafe {
        let min_vec = vdupq_n_s32(min);
        let max_vec = vdupq_n_s32(max);

        for i in (0..simd_len).step_by(4) {
            let v = vld1q_s32(data.as_ptr().add(i));
            let ge_min = vcgeq_s32(v, min_vec);
            let le_max = vcleq_s32(v, max_vec);
            let both = vandq_u32(ge_min, le_max);
            // vminvq: if any 0 exists, returns 0
            if vminvq_u32(both) == 0 {
                return false;
            }
        }
    }

    true
}
```

### Results

| Implementation | Time | vs Scalar |
|----------------|------|-----------|
| Scalar | 24.93ms | 1.0x |
| std::simd | 8.24ms | 3.02x |
| NEON | 9.33ms | 2.67x |

**Analysis**: Simple comparisons on contiguous i32 data. No type conversions needed.

---

## Scenario 9: Sorted Check

**Task**: Verify 10M i32 values are sorted ascending.

I used pre-sorted data to measure full-scan performance. The `windows()` iterator is convenient but has overhead.

### Scalar

```rust
pub fn is_sorted(data: &[i32]) -> bool {
    data.windows(2).all(|w| w[0] <= w[1])
}
```

### std::simd

```rust
pub fn is_sorted(data: &[i32]) -> bool {
    for window in data.windows(9) {
        let current = i32x8::from_slice(&window[0..8]);
        let next = i32x8::from_slice(&window[1..9]);
        if !current.simd_le(next).all() {
            return false;
        }
    }
    true
}
```

### NEON

```rust
pub fn is_sorted(data: &[i32]) -> bool {
    unsafe {
        let mut i = 0;
        while i + 4 < data.len() {
            let current = vld1q_s32(data.as_ptr().add(i));
            let next = vld1q_s32(data.as_ptr().add(i + 1));
            let le = vcleq_s32(current, next);
            if vminvq_u32(le) == 0 {
                return false;
            }
            i += 4;
        }
    }
    true
}
```

### Results

| Implementation | Time | vs Scalar |
|----------------|------|-----------|
| Scalar | 25.14ms | 1.0x |
| std::simd | 37.18ms | 0.68x |
| NEON | 6.87ms | 3.66x |

**Analysis**: `windows(9)` iterator creates overlapping slices with overhead. NEON uses direct pointer arithmetic.

---

## Root Cause Analysis

After digging into the assembly, the pattern became clear. `std::simd` exposes only operations available across all target platforms. ARM-specific instructions cannot be represented:

| Operation | std::simd | NEON |
|-----------|-----------|------|
| Deinterleave load | scalar loop | `vld3q_u8` |
| Widen i16→i32 | scalar loop | `vmovl_s16` |
| Saturating narrow i32→i16 | manual clamp | `vqmovn_s32` |
| Halving add | add + shift + narrow | `vhaddq_s16` |

When these operations are needed, `std::simd` falls back to scalar loops, negating SIMD benefits and adding overhead.

---

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

| Instruction | Operation |
|-------------|-----------|
| `vld1q_u8` | Load 16 bytes |
| `vld3q_u8` | Load + deinterleave RGB |
| `vmull_u8` | Widening multiply u8→u16 |
| `vmovl_s16` | Widen i16→i32 |
| `vqmovn_s32` | Saturating narrow i32→i16 |
| `vfmaq_f32` | Fused multiply-add |
| `vhaddq_s16` | Halving add |
| `vaddvq_f32` | Horizontal sum |

---

## Recommendations

| Data Pattern | Approach |
|--------------|----------|
| Contiguous f32/i32 arrays | std::simd |
| Interleaved data (RGB, stereo audio) | Platform intrinsics |
| Operations requiring type conversion | Platform intrinsics |
| Cross-platform library | std::simd + scalar fallback |
| Maximum ARM performance | NEON intrinsics |

---

## Conclusion

My takeaway: `std::simd` is great for numerical workloads on contiguous data—often better than hand-written intrinsics because the compiler knows optimization tricks I don't.

But for image and audio processing, `std::simd` falls short. The portable abstraction cannot express instructions like `vld3q_u8` or `vhaddq_s16` that ARM provides specifically for these workloads.

If we're targeting ARM and working with interleaved data, NEON intrinsics remain the way to go.

---

## References

- [Rust std::simd documentation](https://doc.rust-lang.org/std/simd/index.html) - Portable SIMD module (nightly)
- [Rust std::arch::aarch64](https://doc.rust-lang.org/std/arch/aarch64/index.html) - ARM64 intrinsics in Rust
- [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/) - Official ARM intrinsics search
- [ARM NEON Programmer's Guide](https://developer.arm.com/documentation/den0018/a/) - NEON architecture overview

---

## Source

Full source code: [github.com/Erio-Harrison/simd_benchmark](https://github.com/Erio-Harrison/simd_benchmark)