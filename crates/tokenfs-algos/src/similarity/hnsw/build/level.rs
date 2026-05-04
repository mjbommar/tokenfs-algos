//! Logarithmic level assignment per HNSW paper §4.1 (Eq. 4):
//!
//! `l = floor(-ln(unif(0, 1)) * mL)`
//!
//! drawn on the **open** interval (since `ln(0)` is undefined). With
//! `mL = 1 / ln(M)` this is equivalent to a geometric distribution
//! where each level retains ~`1/M` of the previous level's elements.
//!
//! For determinism (per `docs/hnsw/research/DETERMINISM.md`):
//!
//! - Uses [`rand_chacha::ChaCha8Rng`] seeded from
//!   [`super::BuildConfig::seed`] — explicitly NOT std's
//!   `DefaultHasher` or `thread_rng`.
//! - The unif draw clamps `1.0` away (avoids `ln(1.0) == 0` →
//!   `level 0` bias) and `0.0` away (avoids `ln(0) == -∞`).
//! - Output is capped at `BuildConfig::max_level` to bound graph
//!   tape sizes.

#![cfg(feature = "hnsw-build")]

use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};

/// Construct a deterministic ChaCha8 RNG seeded from the build seed.
pub fn seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Sample a single level value per HNSW paper §4.1.
///
/// `level_mult = 1 / ln(M)` per `BuildConfig::level_mult`. `max_level`
/// is the hard cap.
///
/// Algorithm:
/// 1. Sample `u ∈ (0, 1)` from the open interval (clamps the RNG
///    output away from both endpoints).
/// 2. Compute `l = floor(-ln(u) * level_mult)`.
/// 3. Cap at `max_level`.
pub fn random_level(rng: &mut ChaCha8Rng, level_mult: f64, max_level: u8) -> u8 {
    // Draw u ∈ (0, 1) by sampling u32 → f64 in (0, 1).
    // (raw + 1) / (2^32 + 1) ∈ (0, 1) — avoids both endpoints.
    let raw = u64::from(rng.next_u32());
    let u = (raw + 1) as f64 / (u64::from(u32::MAX) + 2) as f64;
    let l = (-u.ln() * level_mult).floor();
    if l < 0.0 {
        return 0;
    }
    let l = l as u32;
    let cap = u32::from(max_level);
    u8::try_from(l.min(cap)).unwrap_or(max_level)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn random_level_is_deterministic_per_seed() {
        let mut rng_a = seeded_rng(42);
        let mut rng_b = seeded_rng(42);
        for _ in 0..100 {
            let la = random_level(&mut rng_a, 1.0 / (16.0_f64).ln(), 16);
            let lb = random_level(&mut rng_b, 1.0 / (16.0_f64).ln(), 16);
            assert_eq!(la, lb);
        }
    }

    #[test]
    fn random_level_different_seeds_differ() {
        let mut rng_a = seeded_rng(42);
        let mut rng_b = seeded_rng(43);
        let level_mult = 1.0 / (16.0_f64).ln();
        let mut any_differ = false;
        for _ in 0..100 {
            let la = random_level(&mut rng_a, level_mult, 16);
            let lb = random_level(&mut rng_b, level_mult, 16);
            if la != lb {
                any_differ = true;
                break;
            }
        }
        assert!(any_differ);
    }

    #[test]
    fn random_level_respects_cap() {
        let mut rng = seeded_rng(0xDEAD_BEEF);
        // level_mult = 100 → produces astronomically high levels;
        // cap at 5.
        let level_mult = 100.0;
        for _ in 0..1_000 {
            let l = random_level(&mut rng, level_mult, 5);
            assert!(l <= 5, "level {l} exceeds cap 5");
        }
    }

    #[test]
    fn random_level_distribution_majority_zero() {
        // For M = 16 (level_mult ≈ 0.3606), expected fraction of
        // level-0 nodes is `1 - 1/M ≈ 93.75%`. Empirically over a
        // few thousand draws this should hold within a few percent.
        let mut rng = seeded_rng(1);
        let level_mult = 1.0 / (16.0_f64).ln();
        let mut zero_count = 0;
        let total = 5_000;
        for _ in 0..total {
            let l = random_level(&mut rng, level_mult, 16);
            if l == 0 {
                zero_count += 1;
            }
        }
        let frac_zero = zero_count as f64 / total as f64;
        // Loose 89%-99% band to absorb sampling variance at N=5000.
        assert!(
            (0.89..=0.99).contains(&frac_zero),
            "expected ~93.75% level-0 for M=16, got {frac_zero}"
        );
    }

    #[test]
    fn random_level_never_negative_or_panics() {
        let mut rng = seeded_rng(0);
        let level_mult = 1.0 / (16.0_f64).ln();
        for _ in 0..10_000 {
            let _ = random_level(&mut rng, level_mult, 16);
        }
    }

    #[test]
    fn random_level_zero_level_mult_returns_zero() {
        let mut rng = seeded_rng(0);
        for _ in 0..100 {
            assert_eq!(random_level(&mut rng, 0.0, 16), 0);
        }
    }
}
