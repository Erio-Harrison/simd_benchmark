//! # Rust SIMD Complete Tutorial and Performance Benchmark
//!
//! This project demonstrates SIMD usage across multiple real-world scenarios,
//! comparing three implementation approaches:
//! 1. Scalar - Baseline implementation
//! 2. std::simd (nightly) - Cross-platform portable SIMD
//! 3. std::arch (stable) - Platform-specific intrinsics (ARM NEON)
//!
//! ## SIMD Fundamentals
//!
//! SIMD = Single Instruction, Multiple Data
//!
//! Traditional scalar processing:
//! ```text
//! a[0] + b[0] → c[0]  (1 instruction)
//! a[1] + b[1] → c[1]  (1 instruction)
//! a[2] + b[2] → c[2]  (1 instruction)
//! a[3] + b[3] → c[3]  (1 instruction)
//! Total: 4 instructions
//! ```
//!
//! SIMD parallel processing:
//! ```text
//! [a0,a1,a2,a3] + [b0,b1,b2,b3] → [c0,c1,c2,c3]  (1 instruction)
//! Total: 1 instruction, theoretical 4x speedup
//! ```

#![feature(portable_simd)]

// ============================================================================
// Scenario 1: Image Processing - RGB to Grayscale
// ============================================================================
//
// This is a classic SIMD use case. Each pixel's computation is completely
// independent, making it ideal for parallel processing.
//
// Grayscale formula (ITU-R BT.601 standard):
// Gray = 0.299 * R + 0.587 * G + 0.114 * B
//
// To avoid floating-point operations, we use fixed-point arithmetic:
// Gray = (77 * R + 150 * G + 29 * B) >> 8
// where 77 ≈ 0.299 * 256, 150 ≈ 0.587 * 256, 29 ≈ 0.114 * 256

pub mod image_processing {
    pub mod scalar {
        /// Scalar implementation: process pixel by pixel
        ///
        /// Input: RGB data, 3 bytes per pixel [R,G,B,R,G,B,...]
        /// Output: Grayscale data, 1 byte per pixel
        pub fn rgb_to_grayscale(rgb: &[u8], gray: &mut [u8]) {
            // Process one pixel per iteration (3 bytes)
            for (i, chunk) in rgb.chunks_exact(3).enumerate() {
                let r = chunk[0] as u32;
                let g = chunk[1] as u32;
                let b = chunk[2] as u32;
                // Fixed-point calculation: avoids floating-point operations
                gray[i] = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
            }
        }
    }

    pub mod portable_simd {
        use std::simd::*;

        /// std::simd implementation: process 16 pixels at once
        ///
        /// ## Core Syntax Explained
        ///
        /// ### 1. Type Aliases
        /// - `u8x16`: Vector of 16 u8 values (128-bit)
        /// - `u16x16`: Vector of 16 u16 values (256-bit)
        /// - `u32x4`: Vector of 4 u32 values (128-bit)
        ///
        /// ### 2. Common Methods
        /// - `splat(v)`: Fill entire vector with the same value
        /// - `from_slice(s)`: Load data from a slice
        /// - `copy_to_slice(s)`: Write vector to a slice
        /// - `cast::<T>()`: Type conversion (preserving bit width)
        ///
        /// ### 3. Swizzle (Data Rearrangement)
        /// One of SIMD's most powerful operations for rearranging vector elements
        pub fn rgb_to_grayscale(rgb: &[u8], gray: &mut [u8]) {
            let chunks = rgb.chunks_exact(48); // 48 bytes = 16 pixels × 3 channels
            let remainder = chunks.remainder();

            // Weight vectors for R, G, B channels
            let weight_r = u16x16::splat(77);
            let weight_g = u16x16::splat(150);
            let weight_b = u16x16::splat(29);

            let mut out_idx = 0;
            for chunk in chunks {
                // Load 48 bytes of RGB data
                // Memory layout: [R0,G0,B0, R1,G1,B1, R2,G2,B2, ...]

                // We need to deinterleave RGB into separate R, G, B vectors
                // Using index extraction for each channel
                let mut r_vals = [0u8; 16];
                let mut g_vals = [0u8; 16];
                let mut b_vals = [0u8; 16];

                for i in 0..16 {
                    r_vals[i] = chunk[i * 3];
                    g_vals[i] = chunk[i * 3 + 1];
                    b_vals[i] = chunk[i * 3 + 2];
                }

                // Convert to u16 to avoid multiplication overflow
                // u8 max is 255, 255 * 150 = 38250, requires u16
                let r = u16x16::from_array(r_vals.map(|x| x as u16));
                let g = u16x16::from_array(g_vals.map(|x| x as u16));
                let b = u16x16::from_array(b_vals.map(|x| x as u16));

                // Parallel computation: process 16 pixels at once
                let gray_u16 = (r * weight_r + g * weight_g + b * weight_b) >> Simd::splat(8);

                // Convert back to u8 and store
                let gray_u8: [u8; 16] = gray_u16.to_array().map(|x| x as u8);
                gray[out_idx..out_idx + 16].copy_from_slice(&gray_u8);
                out_idx += 16;
            }

            // Handle remaining pixels (less than 16)
            for chunk in remainder.chunks_exact(3) {
                let r = chunk[0] as u32;
                let g = chunk[1] as u32;
                let b = chunk[2] as u32;
                gray[out_idx] = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
                out_idx += 1;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub mod neon {
        use std::arch::aarch64::*;

        /// ARM NEON implementation
        ///
        /// ## NEON Instruction Naming Convention
        ///
        /// ```text
        /// v    ld3q   _u8
        /// │     │      │
        /// │     │      └── Data type: u8, u16, u32, f32, s8(signed)...
        /// │     └── Operation + width: ld3=load+deinterleave, q=128-bit
        /// └── Vector prefix
        /// ```
        ///
        /// ## Common Instruction Reference
        ///
        /// | Instruction | Function | Example |
        /// |-------------|----------|---------|
        /// | vld1q_u8 | Load 16 u8 | Read contiguous memory |
        /// | vld3q_u8 | Load + deinterleave | Separate RGB into R,G,B |
        /// | vst1q_u8 | Store 16 u8 | Write to contiguous memory |
        /// | vmull_u8 | Widening multiply u8→u16 | Avoid overflow |
        /// | vaddq_u16 | 16-bit addition | Vector addition |
        /// | vshrn_n_u16 | Narrowing shift u16→u8 | Compress result |
        pub fn rgb_to_grayscale(rgb: &[u8], gray: &mut [u8]) {
            let len = rgb.len() / 3;
            let simd_len = len - (len % 16);

            unsafe {
                // Weight vectors
                let weight_r = vdupq_n_u8(77);
                let weight_g = vdupq_n_u8(150);
                let weight_b = vdupq_n_u8(29);

                for i in (0..simd_len).step_by(16) {
                    let rgb_idx = i * 3;

                    // vld3q_u8: Load and auto-deinterleave!
                    // Input: [R0,G0,B0,R1,G1,B1,R2,G2,B2,...]
                    // Output: .0=[R0,R1,R2,...], .1=[G0,G1,G2,...], .2=[B0,B1,B2,...]
                    // This instruction is much faster than manual deinterleaving
                    let rgb_data = vld3q_u8(rgb.as_ptr().add(rgb_idx));
                    let r = rgb_data.0;
                    let g = rgb_data.1;
                    let b = rgb_data.2;

                    // vmull: Widening multiply, u8 × u8 → u16 (lower 8 elements)
                    // vmull_high: Process upper 8 elements
                    let r_lo = vmull_u8(vget_low_u8(r), vget_low_u8(weight_r));
                    let r_hi = vmull_high_u8(r, weight_r);
                    let g_lo = vmull_u8(vget_low_u8(g), vget_low_u8(weight_g));
                    let g_hi = vmull_high_u8(g, weight_g);
                    let b_lo = vmull_u8(vget_low_u8(b), vget_low_u8(weight_b));
                    let b_hi = vmull_high_u8(b, weight_b);

                    // Accumulate
                    let sum_lo = vaddq_u16(vaddq_u16(r_lo, g_lo), b_lo);
                    let sum_hi = vaddq_u16(vaddq_u16(r_hi, g_hi), b_hi);

                    // vshrn: Shift right and narrow u16 → u8
                    // vcombine: Merge two 64-bit vectors into 128-bit
                    let gray_lo = vshrn_n_u16(sum_lo, 8);
                    let gray_hi = vshrn_n_u16(sum_hi, 8);
                    let gray_vec = vcombine_u8(gray_lo, gray_hi);

                    // Store result
                    vst1q_u8(gray.as_mut_ptr().add(i), gray_vec);
                }
            }

            // Handle remaining pixels
            for i in simd_len..len {
                let rgb_idx = i * 3;
                let r = rgb[rgb_idx] as u32;
                let g = rgb[rgb_idx + 1] as u32;
                let b = rgb[rgb_idx + 2] as u32;
                gray[i] = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    pub mod neon {
        pub fn rgb_to_grayscale(_rgb: &[u8], _gray: &mut [u8]) {
            panic!("NEON is only available on aarch64");
        }
    }
}

// ============================================================================
// Scenario 2: Audio Processing - Volume Adjustment and Mixing
// ============================================================================
//
// Audio data is typically 16-bit samples (i16), range -32768 to 32767
// SIMD is ideal for this kind of batch numerical processing

pub mod audio_processing {
    pub mod scalar {

        /// Volume adjustment: multiply all samples by volume coefficient
        /// volume: 0.0 = mute, 1.0 = original volume, 2.0 = double volume
        pub fn adjust_volume(samples: &mut [i16], volume: f32) {
            // Avoid the float conversion (as the neon implementation does)
            // We scale the volume by 256, and then shift right by 8 bits to divide by 256
            let vol_fixed = (volume * 256.0) as i32;
            for sample in samples.iter_mut() {
                let adjusted = (*sample as i32 * vol_fixed) >> 8;
                *sample = adjusted.max(-32768).min(32767) as i16;
            }
        }

        /// Audio mixing: blend two tracks together
        /// Simple addition divided by 2 to avoid overflow
        pub fn mix_tracks(track_a: &[i16], track_b: &[i16], output: &mut [i16]) {
            for ((a, b), out) in track_a.iter().zip(track_b.iter()).zip(output.iter_mut()) {
                *out = ((*a as i32 + *b as i32) / 2) as i16;
            }
        }
    }

    pub mod portable_simd {
        use std::simd::*;
        use std::simd::cmp::SimdOrd;

        /// std::simd volume adjustment implementation
        ///
        /// ## Key Concept: Saturating Arithmetic
        ///
        /// SIMD provides saturating operations that auto-clamp to type range:
        /// - `saturating_add`: Saturating addition
        /// - `saturating_sub`: Saturating subtraction
        /// - Multiplication requires manual handling
        pub fn adjust_volume(samples: &mut [i16], volume: f32) {
            // Use fixed-point: volume * 256 as integer multiplier
            let vol_fixed = (volume * 256.0) as i32;
            let vol_vec = i32x8::splat(vol_fixed);

            let len = samples.len();
            let simd_len = len - (len % 8);

            for i in (0..simd_len).step_by(8) {
                // i16 → i32 (avoid multiplication overflow)
                let mut vals = [0i32; 8];
                for j in 0..8 {
                    vals[j] = samples[i + j] as i32;
                }
                let v = i32x8::from_array(vals);

                // Multiply then shift right 8 bits (divide by 256)
                let adjusted = (v * vol_vec) >> Simd::splat(8);

                // Clamp to i16 range
                let clamped = adjusted.simd_clamp(
                    i32x8::splat(-32768),
                    i32x8::splat(32767),
                );

                // Write back
                for (j, &val) in clamped.to_array().iter().enumerate() {
                    samples[i + j] = val as i16;
                }
            }

            // Handle remainder
            for i in simd_len..len {
                let adjusted = (samples[i] as f32 * volume) as i32;
                samples[i] = adjusted.clamp(-32768, 32767) as i16;
            }
        }

        /// Audio mixing implementation
        pub fn mix_tracks(track_a: &[i16], track_b: &[i16], output: &mut [i16]) {
            let len = track_a.len().min(track_b.len()).min(output.len());
            let simd_len = len - (len % 8);

            for i in (0..simd_len).step_by(8) {
                // Load and convert to i32
                let mut a_vals = [0i32; 8];
                let mut b_vals = [0i32; 8];
                for j in 0..8 {
                    a_vals[j] = track_a[i + j] as i32;
                    b_vals[j] = track_b[i + j] as i32;
                }

                let a = i32x8::from_array(a_vals);
                let b = i32x8::from_array(b_vals);

                // Mix: (a + b) / 2
                let mixed = (a + b) >> Simd::splat(1);

                // Write back
                for (j, &val) in mixed.to_array().iter().enumerate() {
                    output[i + j] = val as i16;
                }
            }

            // Handle remainder
            for i in simd_len..len {
                output[i] = ((track_a[i] as i32 + track_b[i] as i32) / 2) as i16;
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub mod neon {
        use std::arch::aarch64::*;

        /// NEON volume adjustment implementation
        ///
        /// ## NEON Instructions Used
        ///
        /// - `vld1q_s16`: Load 8 i16 values
        /// - `vmovl_s16`: Sign-extend i16 → i32
        /// - `vmulq_s32`: 32-bit signed multiplication
        /// - `vshrq_n_s32`: Arithmetic right shift
        /// - `vqmovn_s32`: Saturating narrow i32 → i16
        pub fn adjust_volume(samples: &mut [i16], volume: f32) {
            let vol_fixed = (volume * 256.0) as i32;
            let len = samples.len();
            let simd_len = len - (len % 8);

            unsafe {
                let vol_vec = vdupq_n_s32(vol_fixed);

                for i in (0..simd_len).step_by(8) {
                    // Load 8 i16 values
                    let v = vld1q_s16(samples.as_ptr().add(i));

                    // Extend to i32 (lower 4 and upper 4 separately)
                    let v_lo = vmovl_s16(vget_low_s16(v));
                    let v_hi = vmovl_high_s16(v);

                    // Multiply
                    let mul_lo = vmulq_s32(v_lo, vol_vec);
                    let mul_hi = vmulq_s32(v_hi, vol_vec);

                    // Shift right 8 bits
                    let shifted_lo = vshrq_n_s32(mul_lo, 8);
                    let shifted_hi = vshrq_n_s32(mul_hi, 8);

                    // vqmovn: Saturating narrow, auto-clamps to i16 range!
                    let result_lo = vqmovn_s32(shifted_lo);
                    let result_hi = vqmovn_s32(shifted_hi);
                    let result = vcombine_s16(result_lo, result_hi);

                    // Store
                    vst1q_s16(samples.as_mut_ptr().add(i), result);
                }
            }

            // Handle remainder
            for i in simd_len..len {
                let adjusted = (samples[i] as f32 * volume) as i32;
                samples[i] = adjusted.clamp(-32768, 32767) as i16;
            }
        }

        /// NEON audio mixing implementation
        pub fn mix_tracks(track_a: &[i16], track_b: &[i16], output: &mut [i16]) {
            let len = track_a.len().min(track_b.len()).min(output.len());
            let simd_len = len - (len % 8);

            unsafe {
                for i in (0..simd_len).step_by(8) {
                    let a = vld1q_s16(track_a.as_ptr().add(i));
                    let b = vld1q_s16(track_b.as_ptr().add(i));

                    // vhaddq: Halving add, (a + b) / 2, handles overflow automatically!
                    let mixed = vhaddq_s16(a, b);

                    vst1q_s16(output.as_mut_ptr().add(i), mixed);
                }
            }

            for i in simd_len..len {
                output[i] = ((track_a[i] as i32 + track_b[i] as i32) / 2) as i16;
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    pub mod neon {
        pub fn adjust_volume(_samples: &mut [i16], _volume: f32) {
            panic!("NEON is only available on aarch64");
        }
        pub fn mix_tracks(_a: &[i16], _b: &[i16], _out: &mut [i16]) {
            panic!("NEON is only available on aarch64");
        }
    }
}

// ============================================================================
// Scenario 3: String Search - Count Character Occurrences
// ============================================================================
//
// This is a common operation in parsers and text processing
// SIMD can compare multiple characters at once

pub mod string_search {
    pub mod scalar {
        /// Count byte occurrences (e.g., counting newlines '\n')
        pub fn count_byte(data: &[u8], target: u8) -> usize {
            data.iter().filter(|&&b| b == target).count()
        }

        /// Find first occurrence position of a byte
        pub fn find_byte(data: &[u8], target: u8) -> Option<usize> {
            data.iter().position(|&b| b == target)
        }
    }

    pub mod portable_simd {
        use std::simd::*;
        use std::simd::cmp::SimdPartialEq;

        /// std::simd byte counting implementation
        ///
        /// ## Core Technique: Compare + Mask
        ///
        /// SIMD comparison returns a mask (true/false per element)
        /// Then count the number of true values
        ///
        /// ```text
        /// data:   [a, \n, b, c, \n, d, ...]
        /// target: [\n,\n,\n,\n,\n,\n, ...]
        /// compare:[0, -1, 0, 0, -1, 0, ...]  // -1 = all 1s = true
        /// convert:[0,  1, 0, 0,  1, 0, ...]
        /// sum:    2
        /// ```
        pub fn count_byte(data: &[u8], target: u8) -> usize {
            let target_vec = u8x32::splat(target);
            let chunks = data.chunks_exact(32);
            let remainder = chunks.remainder();

            let mut count = 0usize;

            for chunk in chunks {
                let v = u8x32::from_slice(chunk);
                // simd_eq returns a mask
                let mask = v.simd_eq(target_vec);
                // to_bitmask converts mask to bitmap, then count 1s
                count += mask.to_bitmask().count_ones() as usize;
            }

            // Handle remainder
            count += remainder.iter().filter(|&&b| b == target).count();

            count
        }

        /// Find byte position
        pub fn find_byte(data: &[u8], target: u8) -> Option<usize> {
            let target_vec = u8x32::splat(target);
            let chunks = data.chunks_exact(32);
            let remainder_start = data.len() - chunks.remainder().len();

            for (chunk_idx, chunk) in chunks.enumerate() {
                let v = u8x32::from_slice(chunk);
                let mask = v.simd_eq(target_vec);
                let bitmask = mask.to_bitmask();
                if bitmask != 0 {
                    // trailing_zeros gives position of first 1
                    return Some(chunk_idx * 32 + bitmask.trailing_zeros() as usize);
                }
            }

            // Check remainder
            for (i, &b) in data[remainder_start..].iter().enumerate() {
                if b == target {
                    return Some(remainder_start + i);
                }
            }

            None
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub mod neon {
        use std::arch::aarch64::*;

        /// NEON byte counting implementation
        ///
        /// ## NEON Comparison Instructions
        ///
        /// - `vceqq_u8`: Compare equal, returns 0x00 (not equal) or 0xFF (equal)
        /// - `vcntq_u8`: Count 1 bits in each byte (popcount)
        /// - `vaddvq_u8`: Horizontal sum
        pub fn count_byte(data: &[u8], target: u8) -> usize {
            let len = data.len();
            let simd_len = len - (len % 16);
            let mut count = 0usize;

            unsafe {
                let target_vec = vdupq_n_u8(target);

                for i in (0..simd_len).step_by(16) {
                    let v = vld1q_u8(data.as_ptr().add(i));
                    // Compare: equal returns 0xFF, not equal returns 0x00
                    let eq = vceqq_u8(v, target_vec);
                    // 0xFF has 8 ones, so shift right 7 bits to get 0 or 1
                    let ones = vshrq_n_u8(eq, 7); // 0xFF >> 7 = 1
                    count += vaddvq_u8(ones) as usize;
                }
            }

            // Handle remainder
            count += data[simd_len..].iter().filter(|&&b| b == target).count();

            count
        }

        /// Find byte position
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

            // Check remainder
            for (i, &b) in data[simd_len..].iter().enumerate() {
                if b == target {
                    return Some(simd_len + i);
                }
            }

            None
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    pub mod neon {
        pub fn count_byte(_data: &[u8], _target: u8) -> usize {
            panic!("NEON is only available on aarch64");
        }
        pub fn find_byte(_data: &[u8], _target: u8) -> Option<usize> {
            panic!("NEON is only available on aarch64");
        }
    }
}

// ============================================================================
// Scenario 4: Numerical Computing - Dot Product and Matrix Operations
// ============================================================================
//
// Fundamental operations for machine learning and scientific computing

pub mod numerical {
    pub mod scalar {
        /// Vector dot product
        pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        }

        /// Matrix-vector multiplication
        /// matrix: row-major, rows × cols
        /// vector: length = cols
        /// result: length = rows
        pub fn matrix_vector_mul(
            matrix: &[f32],
            vector: &[f32],
            result: &mut [f32],
            rows: usize,
            cols: usize,
        ) {
            for i in 0..rows {
                let row_start = i * cols;
                result[i] = matrix[row_start..row_start + cols]
                    .iter()
                    .zip(vector.iter())
                    .map(|(m, v)| m * v)
                    .sum();
            }
        }
    }

    pub mod portable_simd {
        use std::simd::*;
        use std::simd::num::SimdFloat;

        /// std::simd dot product implementation
        ///
        /// ## Key Operation: Horizontal Reduction
        ///
        /// `reduce_sum()`: Sum all elements in a vector
        /// ```text
        /// [1.0, 2.0, 3.0, 4.0].reduce_sum() = 10.0
        /// ```
        pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
            let chunks_a = a.chunks_exact(8);
            let chunks_b = b.chunks_exact(8);
            let remainder_a = chunks_a.remainder();
            let remainder_b = chunks_b.remainder();

            // Use f32x8 (256-bit) for better throughput
            let mut acc = f32x8::splat(0.0);

            for (a_chunk, b_chunk) in chunks_a.zip(chunks_b) {
                let va = f32x8::from_slice(a_chunk);
                let vb = f32x8::from_slice(b_chunk);
                // FMA: Fused Multiply-Add, more accurate and faster
                // acc = acc + va * vb
                acc += va * vb;
            }

            let mut result = acc.reduce_sum();

            for (x, y) in remainder_a.iter().zip(remainder_b.iter()) {
                result += x * y;
            }

            result
        }

        /// Matrix-vector multiplication
        pub fn matrix_vector_mul(
            matrix: &[f32],
            vector: &[f32],
            result: &mut [f32],
            rows: usize,
            cols: usize,
        ) {
            for i in 0..rows {
                let row = &matrix[i * cols..(i + 1) * cols];
                result[i] = dot_product(row, vector);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub mod neon {
        use std::arch::aarch64::*;

        /// NEON dot product implementation
        ///
        /// ## Using FMA Instruction
        ///
        /// `vfmaq_f32(acc, a, b)`: acc + a * b
        /// FMA is faster and more precise than separate multiply + add
        pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
            let len = a.len();
            let simd_len = len - (len % 4);

            unsafe {
                let mut acc = vdupq_n_f32(0.0);

                for i in (0..simd_len).step_by(4) {
                    let va = vld1q_f32(a.as_ptr().add(i));
                    let vb = vld1q_f32(b.as_ptr().add(i));
                    // FMA: Fused Multiply-Add
                    acc = vfmaq_f32(acc, va, vb);
                }

                // Horizontal sum
                let mut result = vaddvq_f32(acc);

                for i in simd_len..len {
                    result += a[i] * b[i];
                }

                result
            }
        }

        /// Matrix-vector multiplication
        pub fn matrix_vector_mul(
            matrix: &[f32],
            vector: &[f32],
            result: &mut [f32],
            rows: usize,
            cols: usize,
        ) {
            for i in 0..rows {
                let row = &matrix[i * cols..(i + 1) * cols];
                result[i] = dot_product(row, vector);
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    pub mod neon {
        pub fn dot_product(_a: &[f32], _b: &[f32]) -> f32 {
            panic!("NEON is only available on aarch64");
        }
        pub fn matrix_vector_mul(
            _matrix: &[f32],
            _vector: &[f32],
            _result: &mut [f32],
            _rows: usize,
            _cols: usize,
        ) {
            panic!("NEON is only available on aarch64");
        }
    }
}

// ============================================================================
// Scenario 5: Data Validation - Check if Array is Sorted
// ============================================================================
//
// Demonstrates SIMD usage in conditional checking scenarios

pub mod validation {
    pub mod scalar {
        /// Check if array is sorted in ascending order
        pub fn is_sorted(data: &[i32]) -> bool {
            data.windows(2).all(|w| w[0] <= w[1])
        }

        /// Check if all elements are within range
        pub fn all_in_range(data: &[i32], min: i32, max: i32) -> bool {
            data.iter().all(|&x| x >= min && x <= max)
        }
    }

    pub mod portable_simd {
        use std::simd::*;
        use std::simd::cmp::SimdPartialOrd;

        /// Check if all elements are within range
        ///
        /// ## Technique: Early Exit
        ///
        /// Return false immediately when finding an element that doesn't satisfy
        /// the condition. This saves significant time when data doesn't meet criteria.
        pub fn all_in_range(data: &[i32], min: i32, max: i32) -> bool {
            let min_vec = i32x8::splat(min);
            let max_vec = i32x8::splat(max);

            let chunks = data.chunks_exact(8);
            let remainder = chunks.remainder();

            for chunk in chunks {
                let v = i32x8::from_slice(chunk);
                // simd_ge: greater than or equal comparison
                // simd_le: less than or equal comparison
                let ge_min = v.simd_ge(min_vec);
                let le_max = v.simd_le(max_vec);
                // all(): returns true only if all elements satisfy condition
                if !(ge_min & le_max).all() {
                    return false;
                }
            }

            remainder.iter().all(|&x| x >= min && x <= max)
        }

        /// Check if array is sorted in ascending order
        ///
        /// ## Technique: Offset Comparison
        ///
        /// Compare [a0,a1,a2,a3] with [a1,a2,a3,a4]
        pub fn is_sorted(data: &[i32]) -> bool {
            if data.len() < 2 {
                return true;
            }

            for window in data.windows(9) {
                let current = i32x8::from_slice(&window[0..8]);
                let next = i32x8::from_slice(&window[1..9]);
                if !current.simd_le(next).all() {
                    return false;
                }
            }

            // Handle boundary cases
            let remainder_start = (data.len() - 1) / 8 * 8;
            for i in remainder_start.max(1)..data.len() {
                if data[i - 1] > data[i] {
                    return false;
                }
            }

            true
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub mod neon {
        use std::arch::aarch64::*;

        /// NEON range check
        pub fn all_in_range(data: &[i32], min: i32, max: i32) -> bool {
            let len = data.len();
            let simd_len = len - (len % 4);

            unsafe {
                let min_vec = vdupq_n_s32(min);
                let max_vec = vdupq_n_s32(max);

                for i in (0..simd_len).step_by(4) {
                    let v = vld1q_s32(data.as_ptr().add(i));
                    // vcgeq: greater than or equal comparison
                    // vcleq: less than or equal comparison
                    let ge_min = vcgeq_s32(v, min_vec);
                    let le_max = vcleq_s32(v, max_vec);
                    // vand: bitwise AND
                    let both = vandq_u32(ge_min, le_max);
                    // vminvq: if any 0 exists, returns 0
                    if vminvq_u32(both) == 0 {
                        return false;
                    }
                }
            }

            data[simd_len..].iter().all(|&x| x >= min && x <= max)
        }

        /// NEON sorted check
        pub fn is_sorted(data: &[i32]) -> bool {
            if data.len() < 2 {
                return true;
            }

            let len = data.len();

            unsafe {
                let mut i = 0;
                while i + 4 < len {
                    let current = vld1q_s32(data.as_ptr().add(i));
                    let next = vld1q_s32(data.as_ptr().add(i + 1));
                    let le = vcleq_s32(current, next);
                    if vminvq_u32(le) == 0 {
                        return false;
                    }
                    i += 4;
                }

                // Check remaining elements
                for j in i..len - 1 {
                    if data[j] > data[j + 1] {
                        return false;
                    }
                }
            }

            true
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    pub mod neon {
        pub fn all_in_range(_data: &[i32], _min: i32, _max: i32) -> bool {
            panic!("NEON is only available on aarch64");
        }
        pub fn is_sorted(_data: &[i32]) -> bool {
            panic!("NEON is only available on aarch64");
        }
    }
}
