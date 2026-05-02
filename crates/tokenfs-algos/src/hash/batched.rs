//! Batched cryptographic hashing helpers for many small messages.
//!
//! These wrappers fan a slice of message references through a per-message
//! cryptographic digest. They are the canonical Merkle-leaf workload: hash N
//! independent payloads (typically extents, chunks, or content-addressed
//! blobs) and write each digest into a caller-provided output array.
//!
//! # Naming convention
//!
//! Each algorithm exposes two variants:
//!
//! - `*_batch_st`: single-thread loop. Kernel-safe (subject to the
//!   underlying primitive — `sha256_batch_st` is no_std + alloc clean,
//!   while `blake3_batch_st_32` requires the `blake3` cargo feature which
//!   in turn requires `std`).
//! - `*_batch_par`: rayon-parallel form. Userspace-only; gated on the
//!   `parallel` cargo feature. Falls back to the `_st` path for batches
//!   smaller than [`BATCH_PARALLEL_THRESHOLD`] to avoid rayon thread-pool
//!   overhead on small inputs.
//!
//! # Why batched
//!
//! Cryptographic hashing of one large message already saturates a single
//! core; batching adds nothing. The win is at the other end of the size
//! distribution: many small (sub-page) messages, where per-message
//! function-call overhead and underused SIMD pipelines dominate. Rayon
//! distributes that work across cores, while a single-thread batched
//! variant amortises the call-site over a tight loop without surrendering
//! kernel-safety.
//!
//! # Safety contract
//!
//! All four functions panic if `messages.len() != out.len()`. This is a
//! precondition violation (mismatched caller-allocated buffers), not an
//! input-validation failure, so it surfaces as a clear panic rather than
//! a `Result`. Hot paths perform the length check once and never allocate.
//!
//! # Examples
//!
//! ```
//! use tokenfs_algos::hash::sha256_batch_st;
//! let messages: [&[u8]; 3] = [b"a", b"bc", b"def"];
//! let mut digests = [[0_u8; 32]; 3];
//! sha256_batch_st(&messages, &mut digests);
//! assert_eq!(digests[0], tokenfs_algos::hash::sha256::sha256(b"a"));
//! ```

use crate::hash::sha256::DIGEST_BYTES as SHA256_DIGEST_BYTES;
use crate::hash::sha256::sha256 as sha256_one;

#[cfg(feature = "blake3")]
use crate::hash::blake3::DIGEST_BYTES as BLAKE3_DIGEST_BYTES;
#[cfg(feature = "blake3")]
use crate::hash::blake3::blake3 as blake3_one;

/// Minimum batch size at which the `_par` variants fan out via rayon.
///
/// Below this threshold the parallel variants delegate to their
/// single-thread counterparts. The value is intentionally conservative —
/// rayon's per-task scheduling overhead is on the order of a microsecond
/// per work item, which dominates a sub-microsecond hash of a small
/// payload until the batch is large enough to amortise pool wake-up.
pub const BATCH_PARALLEL_THRESHOLD: usize = 256;

#[inline]
#[track_caller]
fn assert_batch_lengths<T>(messages: &[&[u8]], out: &[T]) {
    assert_eq!(
        messages.len(),
        out.len(),
        "hash batch length mismatch: messages.len()={} but out.len()={}",
        messages.len(),
        out.len()
    );
}

// =============================================================================
// SHA-256
// =============================================================================

/// Hash `messages.len()` byte slices into `out` using SHA-256, single-thread.
///
/// This is the kernel-safe entry point. It compiles in no_std + alloc, has no
/// rayon dependency, and uses the same hardware-accelerated SHA-256 backend
/// (x86 SHA-NI, AArch64 FEAT_SHA2) as [`crate::hash::sha256::sha256`].
///
/// # Panics
///
/// Panics if `messages.len() != out.len()`. This is a precondition violation
/// on the caller-provided output buffer, not an input-validation failure.
///
/// # Examples
///
/// ```
/// use tokenfs_algos::hash::sha256_batch_st;
/// let messages: [&[u8]; 2] = [b"hello", b"world"];
/// let mut digests = [[0_u8; 32]; 2];
/// sha256_batch_st(&messages, &mut digests);
/// ```
pub fn sha256_batch_st(messages: &[&[u8]], out: &mut [[u8; SHA256_DIGEST_BYTES]]) {
    assert_batch_lengths(messages, out);
    for (msg, dst) in messages.iter().zip(out.iter_mut()) {
        *dst = sha256_one(msg);
    }
}

/// Hash `messages.len()` byte slices into `out` using SHA-256, in parallel.
///
/// Userspace-only: requires the `parallel` cargo feature. Distributes work
/// across the global rayon thread pool when `messages.len()` is at least
/// [`BATCH_PARALLEL_THRESHOLD`]; below that, falls back to the single-thread
/// path to avoid pool wake-up cost on small batches. Kernel modules and
/// other contexts where rayon is forbidden must use [`sha256_batch_st`].
///
/// Output is bit-exact to [`sha256_batch_st`] — only the work distribution
/// differs.
///
/// # Panics
///
/// Panics if `messages.len() != out.len()`.
#[cfg(feature = "parallel")]
pub fn sha256_batch_par(messages: &[&[u8]], out: &mut [[u8; SHA256_DIGEST_BYTES]]) {
    assert_batch_lengths(messages, out);
    if messages.len() < BATCH_PARALLEL_THRESHOLD {
        for (msg, dst) in messages.iter().zip(out.iter_mut()) {
            *dst = sha256_one(msg);
        }
        return;
    }
    use rayon::prelude::*;
    messages
        .par_iter()
        .zip(out.par_iter_mut())
        .for_each(|(msg, dst)| {
            *dst = sha256_one(msg);
        });
}

// =============================================================================
// BLAKE3 (gated on `blake3` feature, which in turn requires `std`)
// =============================================================================

/// Hash `messages.len()` byte slices into `out` using BLAKE3, single-thread.
///
/// Userspace-only by virtue of the `blake3` cargo feature already requiring
/// `std`. Wraps [`crate::hash::blake3::blake3`] in a tight loop. For batches
/// large enough to amortise rayon pool wake-up, prefer [`blake3_batch_par_32`].
///
/// # Panics
///
/// Panics if `messages.len() != out.len()`.
#[cfg(feature = "blake3")]
pub fn blake3_batch_st_32(messages: &[&[u8]], out: &mut [[u8; BLAKE3_DIGEST_BYTES]]) {
    assert_batch_lengths(messages, out);
    for (msg, dst) in messages.iter().zip(out.iter_mut()) {
        *dst = blake3_one(msg);
    }
}

/// Hash `messages.len()` byte slices into `out` using BLAKE3, in parallel.
///
/// Userspace-only: requires both the `blake3` and `parallel` cargo features.
/// Distributes work across the global rayon thread pool when `messages.len()`
/// is at least [`BATCH_PARALLEL_THRESHOLD`]; below that, falls back to the
/// single-thread path to avoid pool wake-up cost on small batches.
///
/// Output is bit-exact to [`blake3_batch_st_32`] — only the work distribution
/// differs.
///
/// # Panics
///
/// Panics if `messages.len() != out.len()`.
#[cfg(all(feature = "blake3", feature = "parallel"))]
pub fn blake3_batch_par_32(messages: &[&[u8]], out: &mut [[u8; BLAKE3_DIGEST_BYTES]]) {
    assert_batch_lengths(messages, out);
    if messages.len() < BATCH_PARALLEL_THRESHOLD {
        for (msg, dst) in messages.iter().zip(out.iter_mut()) {
            *dst = blake3_one(msg);
        }
        return;
    }
    use rayon::prelude::*;
    messages
        .par_iter()
        .zip(out.par_iter_mut())
        .for_each(|(msg, dst)| {
            *dst = blake3_one(msg);
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::sha256::sha256;

    /// Deterministic pseudo-random byte generator for repeatable tests.
    fn fill_pseudo_random(buf: &mut [u8], seed: u64) {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        for byte in buf {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *byte = (state >> 33) as u8;
        }
    }

    fn make_message(len: usize, seed: u64) -> Vec<u8> {
        let mut buf = vec![0_u8; len];
        fill_pseudo_random(&mut buf, seed);
        buf
    }

    // -------------------------------------------------------------------------
    // SHA-256 parity
    // -------------------------------------------------------------------------

    #[test]
    fn sha256_batch_st_single_message_matches_one_shot() {
        let msg = b"hello, world";
        let messages: [&[u8]; 1] = [msg];
        let mut out = [[0_u8; 32]; 1];
        sha256_batch_st(&messages, &mut out);
        assert_eq!(out[0], sha256(msg));
    }

    #[test]
    fn sha256_batch_st_empty_batch_is_no_op() {
        let messages: [&[u8]; 0] = [];
        let mut out: [[u8; 32]; 0] = [];
        sha256_batch_st(&messages, &mut out);
    }

    #[test]
    fn sha256_batch_st_empty_message() {
        let messages: [&[u8]; 1] = [b""];
        let mut out = [[0_u8; 32]; 1];
        sha256_batch_st(&messages, &mut out);
        assert_eq!(out[0], sha256(b""));
    }

    #[test]
    fn sha256_batch_st_mixed_sizes() {
        let m0 = make_message(0, 1);
        let m1 = make_message(1, 2);
        let m2 = make_message(63, 3);
        let m3 = make_message(64, 4);
        let m4 = make_message(65, 5);
        let m5 = make_message(1024, 6);
        let m6 = make_message(16_384, 7);
        let messages: [&[u8]; 7] = [&m0, &m1, &m2, &m3, &m4, &m5, &m6];
        let mut out = [[0_u8; 32]; 7];
        sha256_batch_st(&messages, &mut out);
        for (i, msg) in messages.iter().enumerate() {
            assert_eq!(out[i], sha256(msg), "digest mismatch at index {i}");
        }
    }

    #[test]
    fn sha256_batch_st_large_single_message() {
        let m = make_message(1 << 20, 0xDEAD_BEEF);
        let messages: [&[u8]; 1] = [&m];
        let mut out = [[0_u8; 32]; 1];
        sha256_batch_st(&messages, &mut out);
        assert_eq!(out[0], sha256(&m));
    }

    #[test]
    #[should_panic(expected = "hash batch length mismatch")]
    fn sha256_batch_st_panics_on_length_mismatch() {
        let messages: [&[u8]; 2] = [b"a", b"b"];
        let mut out = [[0_u8; 32]; 1];
        sha256_batch_st(&messages, &mut out);
    }

    #[test]
    fn sha256_batch_st_property_random_batches() {
        for &batch_size in &[1_usize, 2, 7, 100, 500] {
            let messages_owned: Vec<Vec<u8>> = (0..batch_size)
                .map(|i| make_message((i * 13) % 1024, (i as u64).wrapping_mul(31)))
                .collect();
            let messages: Vec<&[u8]> = messages_owned.iter().map(Vec::as_slice).collect();
            let mut out = vec![[0_u8; 32]; batch_size];
            sha256_batch_st(&messages, &mut out);
            for (i, msg) in messages.iter().enumerate() {
                assert_eq!(
                    out[i],
                    sha256(msg),
                    "batch_size={batch_size}: digest mismatch at index {i}"
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // SHA-256 parallel parity (gated on `parallel`)
    // -------------------------------------------------------------------------

    #[cfg(feature = "parallel")]
    #[test]
    fn sha256_batch_par_matches_st_small_batch() {
        // Below threshold: should still match.
        let messages_owned: Vec<Vec<u8>> =
            (0..16).map(|i| make_message(64 + i, i as u64)).collect();
        let messages: Vec<&[u8]> = messages_owned.iter().map(Vec::as_slice).collect();
        let mut out_st = vec![[0_u8; 32]; messages.len()];
        let mut out_par = vec![[0_u8; 32]; messages.len()];
        sha256_batch_st(&messages, &mut out_st);
        sha256_batch_par(&messages, &mut out_par);
        assert_eq!(out_st, out_par);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn sha256_batch_par_matches_st_above_threshold() {
        // Above threshold: should fan out and still match.
        let n = BATCH_PARALLEL_THRESHOLD * 4;
        let messages_owned: Vec<Vec<u8>> = (0..n)
            .map(|i| make_message(((i * 7) % 256) + 1, i as u64))
            .collect();
        let messages: Vec<&[u8]> = messages_owned.iter().map(Vec::as_slice).collect();
        let mut out_st = vec![[0_u8; 32]; n];
        let mut out_par = vec![[0_u8; 32]; n];
        sha256_batch_st(&messages, &mut out_st);
        sha256_batch_par(&messages, &mut out_par);
        assert_eq!(out_st, out_par);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn sha256_batch_par_empty_batch_is_no_op() {
        let messages: [&[u8]; 0] = [];
        let mut out: [[u8; 32]; 0] = [];
        sha256_batch_par(&messages, &mut out);
    }

    #[cfg(feature = "parallel")]
    #[test]
    #[should_panic(expected = "hash batch length mismatch")]
    fn sha256_batch_par_panics_on_length_mismatch() {
        let messages: [&[u8]; 1] = [b"a"];
        let mut out = [[0_u8; 32]; 2];
        sha256_batch_par(&messages, &mut out);
    }

    // -------------------------------------------------------------------------
    // BLAKE3 parity (gated on `blake3`)
    // -------------------------------------------------------------------------

    #[cfg(feature = "blake3")]
    #[test]
    fn blake3_batch_st_single_message_matches_one_shot() {
        use crate::hash::blake3::blake3;
        let msg = b"hello, world";
        let messages: [&[u8]; 1] = [msg];
        let mut out = [[0_u8; 32]; 1];
        blake3_batch_st_32(&messages, &mut out);
        assert_eq!(out[0], blake3(msg));
    }

    #[cfg(feature = "blake3")]
    #[test]
    fn blake3_batch_st_empty_batch_is_no_op() {
        let messages: [&[u8]; 0] = [];
        let mut out: [[u8; 32]; 0] = [];
        blake3_batch_st_32(&messages, &mut out);
    }

    #[cfg(feature = "blake3")]
    #[test]
    fn blake3_batch_st_empty_message() {
        use crate::hash::blake3::blake3;
        let messages: [&[u8]; 1] = [b""];
        let mut out = [[0_u8; 32]; 1];
        blake3_batch_st_32(&messages, &mut out);
        assert_eq!(out[0], blake3(b""));
    }

    #[cfg(feature = "blake3")]
    #[test]
    fn blake3_batch_st_mixed_sizes() {
        use crate::hash::blake3::blake3;
        let m0 = make_message(0, 1);
        let m1 = make_message(1, 2);
        let m2 = make_message(63, 3);
        let m3 = make_message(64, 4);
        let m4 = make_message(65, 5);
        let m5 = make_message(1024, 6);
        let m6 = make_message(16_384, 7);
        let messages: [&[u8]; 7] = [&m0, &m1, &m2, &m3, &m4, &m5, &m6];
        let mut out = [[0_u8; 32]; 7];
        blake3_batch_st_32(&messages, &mut out);
        for (i, msg) in messages.iter().enumerate() {
            assert_eq!(out[i], blake3(msg), "digest mismatch at index {i}");
        }
    }

    #[cfg(feature = "blake3")]
    #[test]
    fn blake3_batch_st_large_single_message() {
        use crate::hash::blake3::blake3;
        let m = make_message(1 << 20, 0xDEAD_BEEF);
        let messages: [&[u8]; 1] = [&m];
        let mut out = [[0_u8; 32]; 1];
        blake3_batch_st_32(&messages, &mut out);
        assert_eq!(out[0], blake3(&m));
    }

    #[cfg(feature = "blake3")]
    #[test]
    #[should_panic(expected = "hash batch length mismatch")]
    fn blake3_batch_st_panics_on_length_mismatch() {
        let messages: [&[u8]; 2] = [b"a", b"b"];
        let mut out = [[0_u8; 32]; 1];
        blake3_batch_st_32(&messages, &mut out);
    }

    #[cfg(feature = "blake3")]
    #[test]
    fn blake3_batch_st_property_random_batches() {
        use crate::hash::blake3::blake3;
        for &batch_size in &[1_usize, 2, 7, 100, 500] {
            let messages_owned: Vec<Vec<u8>> = (0..batch_size)
                .map(|i| make_message((i * 13) % 1024, (i as u64).wrapping_mul(31)))
                .collect();
            let messages: Vec<&[u8]> = messages_owned.iter().map(Vec::as_slice).collect();
            let mut out = vec![[0_u8; 32]; batch_size];
            blake3_batch_st_32(&messages, &mut out);
            for (i, msg) in messages.iter().enumerate() {
                assert_eq!(
                    out[i],
                    blake3(msg),
                    "batch_size={batch_size}: digest mismatch at index {i}"
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // BLAKE3 parallel parity (gated on `blake3` + `parallel`)
    // -------------------------------------------------------------------------

    #[cfg(all(feature = "blake3", feature = "parallel"))]
    #[test]
    fn blake3_batch_par_matches_st_small_batch() {
        let messages_owned: Vec<Vec<u8>> =
            (0..16).map(|i| make_message(64 + i, i as u64)).collect();
        let messages: Vec<&[u8]> = messages_owned.iter().map(Vec::as_slice).collect();
        let mut out_st = vec![[0_u8; 32]; messages.len()];
        let mut out_par = vec![[0_u8; 32]; messages.len()];
        blake3_batch_st_32(&messages, &mut out_st);
        blake3_batch_par_32(&messages, &mut out_par);
        assert_eq!(out_st, out_par);
    }

    #[cfg(all(feature = "blake3", feature = "parallel"))]
    #[test]
    fn blake3_batch_par_matches_st_above_threshold() {
        let n = BATCH_PARALLEL_THRESHOLD * 4;
        let messages_owned: Vec<Vec<u8>> = (0..n)
            .map(|i| make_message(((i * 7) % 256) + 1, i as u64))
            .collect();
        let messages: Vec<&[u8]> = messages_owned.iter().map(Vec::as_slice).collect();
        let mut out_st = vec![[0_u8; 32]; n];
        let mut out_par = vec![[0_u8; 32]; n];
        blake3_batch_st_32(&messages, &mut out_st);
        blake3_batch_par_32(&messages, &mut out_par);
        assert_eq!(out_st, out_par);
    }

    #[cfg(all(feature = "blake3", feature = "parallel"))]
    #[test]
    fn blake3_batch_par_empty_batch_is_no_op() {
        let messages: [&[u8]; 0] = [];
        let mut out: [[u8; 32]; 0] = [];
        blake3_batch_par_32(&messages, &mut out);
    }

    #[cfg(all(feature = "blake3", feature = "parallel"))]
    #[test]
    #[should_panic(expected = "hash batch length mismatch")]
    fn blake3_batch_par_panics_on_length_mismatch() {
        let messages: [&[u8]; 1] = [b"a"];
        let mut out = [[0_u8; 32]; 2];
        blake3_batch_par_32(&messages, &mut out);
    }
}
