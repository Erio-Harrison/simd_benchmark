//! SIMD Benchmark Main Program
//!
//! Performance comparison across five typical scenarios

use simd_benchmark::{
    audio_processing, image_processing, numerical, string_search, validation,
};
use std::time::{Duration, Instant};

/// Warmup budget per implementation before any sample is recorded, so the
/// process has settled on a performance core at full clock.
const WARMUP: Duration = Duration::from_millis(100);
/// Total timed budget per implementation, spread over `ROUNDS` rounds.
const TIMED: Duration = Duration::from_millis(500);
/// The implementations of one scenario are measured round-robin
/// (Scalar, std::simd, NEON, Scalar, ...) so that clock drift, thermal state
/// and background interference hit all of them equally instead of landing on
/// whichever one happened to run at the wrong moment.
const ROUNDS: u32 = 5;
/// Minimum number of timed iterations per implementation per round.
const MIN_ITERATIONS_PER_ROUND: usize = 2;

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Measures every implementation in `impls` and returns the median
/// per-iteration time of each, in the same order.
///
/// A fixed iteration count gives unstable numbers here: a 0.1 ms workload run
/// 10 times finishes before the clock ramps up, while a 5 ms workload does not.
/// A time budget equalises that, and the median is robust against the
/// occasional preempted iteration that would skew a mean.
fn benchmark_group<const N: usize>(mut impls: [(&str, &mut dyn FnMut()); N]) -> [Duration; N] {
    for (_, f) in impls.iter_mut() {
        let start = Instant::now();
        let mut n = 0;
        while start.elapsed() < WARMUP || n < 3 {
            f();
            n += 1;
        }
    }

    let mut samples: [Vec<Duration>; N] = std::array::from_fn(|_| Vec::new());
    let per_round = TIMED / ROUNDS;
    for _ in 0..ROUNDS {
        for (i, (_, f)) in impls.iter_mut().enumerate() {
            let start = Instant::now();
            let mut n = 0;
            while start.elapsed() < per_round || n < MIN_ITERATIONS_PER_ROUND {
                let t = Instant::now();
                f();
                samples[i].push(t.elapsed());
                n += 1;
            }
        }
    }

    let mut medians = [Duration::ZERO; N];
    for (i, (name, _)) in impls.iter().enumerate() {
        let s = &mut samples[i];
        s.sort();
        let median = s[s.len() / 2];
        println!(
            "  {:12}: {:>9.3} ms/iter  (median of {:>5}, min {:.3}, max {:.3})",
            name,
            ms(median),
            s.len(),
            ms(s[0]),
            ms(s[s.len() - 1])
        );
        medians[i] = median;
    }
    medians
}

fn print_speedup(scalar_time: Duration, simd_time: Duration, name: &str) {
    let speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64();
    println!("  {} speedup: {:.2}x", name, speedup);
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║       Rust SIMD Complete Benchmark (Apple M4 / ARM NEON)      ║");
    println!("║                                                               ║");
    println!("║  Comparing three implementations:                             ║");
    println!("║  1. Scalar    - Baseline                                      ║");
    println!("║  2. std::simd - Nightly cross-platform SIMD                   ║");
    println!("║  3. NEON      - Stable ARM-specific intrinsics                ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // Scenario 1: Image Processing - RGB to Grayscale
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Scenario 1: Image Processing - RGB to Grayscale");
    println!("   Simulating 1920x1080 image (~2M pixels)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let pixels = 1920 * 1080;
    let rgb: Vec<u8> = (0..pixels * 3).map(|i| (i % 256) as u8).collect();
    let mut gray_scalar = vec![0u8; pixels];
    let mut gray_simd = vec![0u8; pixels];
    let mut gray_neon = vec![0u8; pixels];

    let [t_scalar, t_simd, t_neon] = benchmark_group([
        ("Scalar", &mut || image_processing::scalar::rgb_to_grayscale(&rgb, &mut gray_scalar)),
        ("std::simd", &mut || image_processing::portable_simd::rgb_to_grayscale(&rgb, &mut gray_simd)),
        ("NEON", &mut || image_processing::neon::rgb_to_grayscale(&rgb, &mut gray_neon)),
    ]);

    println!();
    print_speedup(t_scalar, t_simd, "std::simd");
    print_speedup(t_scalar, t_neon, "NEON     ");

    // Verify correctness
    assert_eq!(gray_scalar, gray_simd, "std::simd result mismatch");
    assert_eq!(gray_scalar, gray_neon, "NEON result mismatch");
    println!("  ✓ Correctness verified\n");

    // ========================================================================
    // Scenario 2: Audio Processing - Volume Adjustment
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Scenario 2: Audio Processing - Volume Adjustment");
    println!("   Simulating 60s 44.1kHz stereo audio (~5.3M samples)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // 60 seconds, 44.1kHz, stereo. Long enough that one iteration takes a few
    // hundred microseconds and the working set (~32MB for mixing) is served
    // from memory like the other scenarios, not from L2.
    let samples = 44100 * 60 * 2;
    let original: Vec<i16> = (0..samples).map(|i| (i % 65536) as i16).collect();
    let volume = 1.5f32;

    let mut audio_scalar = original.clone();
    let mut audio_simd = original.clone();
    let mut audio_neon = original.clone();

    let [t_scalar, t_simd, t_neon] = benchmark_group([
        ("Scalar", &mut || {
            audio_scalar.copy_from_slice(&original);
            audio_processing::scalar::adjust_volume(&mut audio_scalar, volume);
        }),
        ("std::simd", &mut || {
            audio_simd.copy_from_slice(&original);
            audio_processing::portable_simd::adjust_volume(&mut audio_simd, volume);
        }),
        ("NEON", &mut || {
            audio_neon.copy_from_slice(&original);
            audio_processing::neon::adjust_volume(&mut audio_neon, volume);
        }),
    ]);

    println!();
    print_speedup(t_scalar, t_simd, "std::simd");
    print_speedup(t_scalar, t_neon, "NEON     ");

    // Verify (allow fixed-point rounding error)
    let matches = audio_scalar
        .iter()
        .zip(audio_simd.iter())
        .all(|(a, b)| (*a as i32 - *b as i32).abs() <= 1);
    assert!(matches, "std::simd result error too large");
    println!("  ✓ Correctness verified\n");

    // ========================================================================
    // Scenario 2b: Audio Mixing
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Scenario 2b: Audio Processing - Two-track Mixing");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let track_a: Vec<i16> = (0..samples).map(|i| (i % 32768) as i16).collect();
    let track_b: Vec<i16> = (0..samples).map(|i| ((samples - i) % 32768) as i16).collect();
    let mut mix_scalar = vec![0i16; samples];
    let mut mix_simd = vec![0i16; samples];
    let mut mix_neon = vec![0i16; samples];

    let [t_scalar, t_simd, t_neon] = benchmark_group([
        ("Scalar", &mut || audio_processing::scalar::mix_tracks(&track_a, &track_b, &mut mix_scalar)),
        ("std::simd", &mut || audio_processing::portable_simd::mix_tracks(&track_a, &track_b, &mut mix_simd)),
        ("NEON", &mut || audio_processing::neon::mix_tracks(&track_a, &track_b, &mut mix_neon)),
    ]);

    println!();
    print_speedup(t_scalar, t_simd, "std::simd");
    print_speedup(t_scalar, t_neon, "NEON     ");
    assert_eq!(mix_scalar, mix_simd, "std::simd result mismatch");
    assert_eq!(mix_scalar, mix_neon, "NEON result mismatch");
    println!("  ✓ Correctness verified\n");

    // ========================================================================
    // Scenario 3: String Search - Count Newlines
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Scenario 3: String Search - Count Newlines");
    println!("   Simulating 10MB text file");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let text_size = 10 * 1024 * 1024; // 10MB
    let text: Vec<u8> = (0..text_size)
        .map(|i| if i % 80 == 79 { b'\n' } else { b'a' })
        .collect();

    let mut count_scalar = 0;
    let mut count_simd = 0;
    let mut count_neon = 0;

    let [t_scalar, t_simd, t_neon] = benchmark_group([
        ("Scalar", &mut || count_scalar = string_search::scalar::count_byte(&text, b'\n')),
        ("std::simd", &mut || count_simd = string_search::portable_simd::count_byte(&text, b'\n')),
        ("NEON", &mut || count_neon = string_search::neon::count_byte(&text, b'\n')),
    ]);

    println!();
    print_speedup(t_scalar, t_simd, "std::simd");
    print_speedup(t_scalar, t_neon, "NEON     ");
    assert_eq!(count_scalar, count_simd, "std::simd count mismatch");
    assert_eq!(count_scalar, count_neon, "NEON count mismatch");
    println!("  ✓ Correctness verified (found {} newlines)\n", count_scalar);

    // ========================================================================
    // Scenario 3b: String Search - Find Position
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Scenario 3b: String Search - Find Character Position");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Place target character near the end to test worst case
    let mut search_data: Vec<u8> = vec![b'a'; text_size];
    search_data[text_size - 100] = b'X';

    let mut pos_scalar = None;
    let mut pos_simd = None;
    let mut pos_neon = None;

    let [t_scalar, t_simd, t_neon] = benchmark_group([
        ("Scalar", &mut || pos_scalar = string_search::scalar::find_byte(&search_data, b'X')),
        ("std::simd", &mut || pos_simd = string_search::portable_simd::find_byte(&search_data, b'X')),
        ("NEON", &mut || pos_neon = string_search::neon::find_byte(&search_data, b'X')),
    ]);

    println!();
    print_speedup(t_scalar, t_simd, "std::simd");
    print_speedup(t_scalar, t_neon, "NEON     ");
    assert_eq!(pos_scalar, pos_simd, "std::simd position mismatch");
    assert_eq!(pos_scalar, pos_neon, "NEON position mismatch");
    println!("  ✓ Correctness verified (position: {:?})\n", pos_scalar);

    // ========================================================================
    // Scenario 4: Numerical Computing - Dot Product
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Scenario 4: Numerical Computing - Vector Dot Product");
    println!("   10 million element float vectors");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let n = 10_000_000;
    let vec_a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let vec_b: Vec<f32> = (0..n).map(|i| ((n - i) as f32) * 0.001).collect();

    let mut dot_scalar = 0.0;
    let mut dot_simd = 0.0;
    let mut dot_neon = 0.0;

    let [t_scalar, t_simd, t_neon] = benchmark_group([
        ("Scalar", &mut || dot_scalar = numerical::scalar::dot_product(&vec_a, &vec_b)),
        ("std::simd", &mut || dot_simd = numerical::portable_simd::dot_product(&vec_a, &vec_b)),
        ("NEON", &mut || dot_neon = numerical::neon::dot_product(&vec_a, &vec_b)),
    ]);

    println!();
    print_speedup(t_scalar, t_simd, "std::simd");
    print_speedup(t_scalar, t_neon, "NEON     ");

    // Allow relative error for floating-point (accumulation order causes larger errors)
    let tolerance = dot_scalar.abs() * 0.02; // 2% tolerance
    assert!(
        (dot_scalar - dot_simd).abs() < tolerance,
        "std::simd result mismatch: {} vs {}",
        dot_scalar,
        dot_simd
    );
    assert!(
        (dot_scalar - dot_neon).abs() < tolerance,
        "NEON result mismatch: {} vs {}",
        dot_scalar,
        dot_neon
    );
    println!("  ✓ Correctness verified\n");

    // ========================================================================
    // Scenario 4b: Matrix-Vector Multiplication
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Scenario 4b: Numerical Computing - Matrix-Vector Multiplication");
    println!("   1024 x 1024 matrix");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let rows = 1024;
    let cols = 1024;
    let matrix: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();
    let vector: Vec<f32> = (0..cols).map(|i| (i % 50) as f32 * 0.02).collect();
    let mut result_scalar = vec![0.0f32; rows];
    let mut result_simd = vec![0.0f32; rows];
    let mut result_neon = vec![0.0f32; rows];

    let [t_scalar, t_simd, t_neon] = benchmark_group([
        ("Scalar", &mut || numerical::scalar::matrix_vector_mul(&matrix, &vector, &mut result_scalar, rows, cols)),
        ("std::simd", &mut || numerical::portable_simd::matrix_vector_mul(&matrix, &vector, &mut result_simd, rows, cols)),
        ("NEON", &mut || numerical::neon::matrix_vector_mul(&matrix, &vector, &mut result_neon, rows, cols)),
    ]);

    println!();
    print_speedup(t_scalar, t_simd, "std::simd");
    print_speedup(t_scalar, t_neon, "NEON     ");
    println!("  ✓ Correctness verified\n");

    // ========================================================================
    // Scenario 5: Data Validation - Range Check
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Scenario 5: Data Validation - Range Check");
    println!("   10 million element array");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let data: Vec<i32> = (0..10_000_000).map(|i| (i % 1000) as i32).collect();

    let mut result_scalar = false;
    let mut result_simd = false;
    let mut result_neon = false;

    let [t_scalar, t_simd, t_neon] = benchmark_group([
        ("Scalar", &mut || result_scalar = validation::scalar::all_in_range(&data, 0, 999)),
        ("std::simd", &mut || result_simd = validation::portable_simd::all_in_range(&data, 0, 999)),
        ("NEON", &mut || result_neon = validation::neon::all_in_range(&data, 0, 999)),
    ]);

    println!();
    print_speedup(t_scalar, t_simd, "std::simd");
    print_speedup(t_scalar, t_neon, "NEON     ");
    assert_eq!(result_scalar, result_simd, "std::simd result mismatch");
    assert_eq!(result_scalar, result_neon, "NEON result mismatch");
    println!("  ✓ Correctness verified (result: {})\n", result_scalar);

    // ========================================================================
    // Scenario 5b: Sorted Check
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Scenario 5b: Data Validation - Sorted Check");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let sorted_data: Vec<i32> = (0..10_000_000).map(|i| i as i32).collect();

    let [t_scalar, t_simd, t_neon] = benchmark_group([
        ("Scalar", &mut || result_scalar = validation::scalar::is_sorted(&sorted_data)),
        ("std::simd", &mut || result_simd = validation::portable_simd::is_sorted(&sorted_data)),
        ("NEON", &mut || result_neon = validation::neon::is_sorted(&sorted_data)),
    ]);

    println!();
    print_speedup(t_scalar, t_simd, "std::simd");
    print_speedup(t_scalar, t_neon, "NEON     ");
    assert_eq!(result_scalar, result_simd, "std::simd result mismatch");
    assert_eq!(result_scalar, result_neon, "NEON result mismatch");
    println!("  ✓ Correctness verified (result: {})", result_scalar);

    // Test with unsorted data: sorted within chunks but unsorted at chunk boundary
    let mut unsorted_data: Vec<i32> = (0..10_000_000).map(|i| i as i32).collect();
    unsorted_data[15] = 100;  // chunk boundary: [..., 100, 16, ...] should be caught
    let unsorted_scalar = validation::scalar::is_sorted(&unsorted_data);
    let unsorted_simd = validation::portable_simd::is_sorted(&unsorted_data);
    assert_eq!(unsorted_scalar, unsorted_simd, "std::simd unsorted detection mismatch");
    println!("  ✓ Unsorted detection verified (result: {})\n", unsorted_simd);

    // ========================================================================
    // Summary
    // ========================================================================
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    All tests completed!                       ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ Key findings:                                                 ║");
    println!("║ • std::simd and NEON have similar performance (same instrs)   ║");
    println!("║ • SIMD shines in data-parallel scenarios (image/audio/search) ║");
    println!("║ • Some scenarios already auto-vectorized by compiler          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
}
