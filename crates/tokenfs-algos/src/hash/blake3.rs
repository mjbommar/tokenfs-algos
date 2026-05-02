//! BLAKE3 — content-addressable cryptographic hash with parallel tree mode.
//!
//! BLAKE3 is the modern de-facto choice for content addressing: it is a
//! Merkle tree of 1024-byte chunks, naturally parallelizable, supports
//! arbitrary-length output (XOF), and ships with hand-tuned SIMD backends
//! (SSE4.1 / AVX2 / AVX-512 on x86, NEON on AArch64). On every architecture
//! the upstream reference implementation is faster than SHA-256.
//!
//! # When to choose BLAKE3 over SHA-256
//!
//! Pick **BLAKE3** when:
//! - You control both producer and consumer (no protocol pinning to SHA-2).
//! - You want raw throughput (≥ 2× SHA-256 on most CPUs, much more with
//!   parallel `update_rayon`).
//! - You need variable-length output (HKDF replacement, XOF, sponge use).
//! - You need a keyed MAC primitive without a separate HMAC layer.
//!
//! Pick [`crate::hash::sha256`] when:
//! - The protocol or compliance regime mandates SHA-2 (TLS, JWT, FIPS,
//!   bitcoin headers, OpenSSH host keys, etc.).
//! - Cross-language compatibility is required and the receiver only knows
//!   SHA-256.
//!
//! # Backend
//!
//! This module is a thin, deterministic wrapper around the upstream
//! [`blake3`](https://docs.rs/blake3) crate. The upstream crate is the
//! reference implementation maintained by the BLAKE3 authors; it already
//! contains the SIMD backends we would otherwise re-implement, so wrapping
//! it (rather than porting from scratch) is the right call for a v0.x
//! release. Output is bit-exact for every input.
//!
//! The `blake3` cargo feature on this crate gates both the dependency and
//! this entire module. Builds without `--features blake3` do not link the
//! `blake3` crate at all.
//!
//! # API summary
//!
//! - [`blake3()`][blake3()]: one-shot 32-byte digest.
//! - [`blake3_xof`]: arbitrary-length output written into a caller buffer.
//! - [`Hasher`]: streaming, mirrors the shape of
//!   [`crate::hash::sha256`] callers.
//! - [`blake3_keyed`]: 32-byte keyed MAC (replacement for HMAC-BLAKE3).
//! - [`blake3_derive_key`]: HKDF-style key derivation with a static
//!   context string.
//!
//! All public functions are `#[must_use]` for digests and consume input by
//! reference so the wrapper introduces no extra copies.

use blake3 as upstream;

/// Output digest size in bytes for the standard BLAKE3 hash.
///
/// BLAKE3 supports any output length via [`blake3_xof`] / [`Hasher::finalize_xof`];
/// 32 bytes (256 bits) is the default referenced by all standard test vectors.
pub const DIGEST_BYTES: usize = 32;

/// BLAKE3 internal block size in bytes.
///
/// BLAKE3 hashes data in 64-byte blocks grouped into 1024-byte chunks at
/// the leaves of its Merkle tree. Exposed for callers that want to pick a
/// natural buffer size when feeding [`Hasher::update`].
pub const BLOCK_BYTES: usize = 64;

/// Compute the standard 32-byte BLAKE3 digest of `bytes`.
///
/// One-shot wrapper around the upstream [`blake3::hash`]. Equivalent to
/// constructing a [`Hasher`], pushing `bytes` once, and calling
/// [`Hasher::finalize`].
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "blake3")]
/// # {
/// use tokenfs_algos::hash::blake3::blake3;
/// let d = blake3(b"abc");
/// // First few bytes of the canonical "abc" digest.
/// assert_eq!(&d[..4], &[0x64, 0x37, 0xb3, 0xac]);
/// # }
/// ```
#[must_use]
pub fn blake3(bytes: &[u8]) -> [u8; DIGEST_BYTES] {
    *upstream::hash(bytes).as_bytes()
}

/// Variable-length BLAKE3 (XOF mode), writing `out.len()` bytes into `out`.
///
/// BLAKE3 is natively a XOF; the first 32 bytes of any XOF stream are the
/// standard digest, and longer streams continue deterministically from
/// there. The output is well-defined for `out.len() == 0` (no bytes are
/// written).
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "blake3")]
/// # {
/// use tokenfs_algos::hash::blake3::{blake3, blake3_xof, DIGEST_BYTES};
/// let mut buf = [0_u8; 64];
/// blake3_xof(b"abc", &mut buf);
/// assert_eq!(&buf[..DIGEST_BYTES], &blake3(b"abc"));
/// # }
/// ```
pub fn blake3_xof(bytes: &[u8], out: &mut [u8]) {
    let mut h = upstream::Hasher::new();
    h.update(bytes);
    h.finalize_xof().fill(out);
}

/// Keyed-mode BLAKE3, replacing HMAC-BLAKE3 for MAC use cases.
///
/// `key` is exactly 32 bytes of secret keying material. Keyed mode is a
/// distinct hash family from the unkeyed one — keyed and unkeyed digests
/// of the same input differ.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "blake3")]
/// # {
/// use tokenfs_algos::hash::blake3::{blake3, blake3_keyed};
/// let key = [0_u8; 32];
/// assert_ne!(blake3_keyed(&key, b"abc"), blake3(b"abc"));
/// # }
/// ```
#[must_use]
pub fn blake3_keyed(key: &[u8; 32], bytes: &[u8]) -> [u8; DIGEST_BYTES] {
    *upstream::keyed_hash(key, bytes).as_bytes()
}

/// HKDF-style key derivation. `context` is a hard-coded application-domain
/// string (see the BLAKE3 spec §5.3 for the recommended convention of
/// `"<app> <year>-<month>-<day> <purpose>"`); `key_material` is the secret
/// input. Returns 32 bytes of derived key material.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "blake3")]
/// # {
/// use tokenfs_algos::hash::blake3::blake3_derive_key;
/// let k1 = blake3_derive_key("example.com 2026-05-01 session-key", b"shared secret");
/// let k2 = blake3_derive_key("example.com 2026-05-01 session-key", b"shared secret");
/// assert_eq!(k1, k2);
/// # }
/// ```
#[must_use]
pub fn blake3_derive_key(context: &str, key_material: &[u8]) -> [u8; DIGEST_BYTES] {
    upstream::derive_key(context, key_material)
}

/// Streaming BLAKE3 hasher.
///
/// Mirrors the shape of [`crate::hash::sha256`]'s digest path: construct
/// with [`Hasher::new`], feed bytes via [`Hasher::update`] in any chunk
/// size, then [`Hasher::finalize`] for the standard 32-byte digest, or
/// [`Hasher::finalize_xof`] for variable-length output. The XOF stream
/// begins with the same bytes that [`Hasher::finalize`] would have
/// returned, so callers can switch between the two without re-hashing.
///
/// `Hasher` is `Clone` — cloning lets a caller branch the same prefix into
/// multiple finalizations without re-hashing the prefix.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "blake3")]
/// # {
/// use tokenfs_algos::hash::blake3::{blake3, Hasher};
/// let mut h = Hasher::new();
/// h.update(b"a");
/// h.update(b"bc");
/// assert_eq!(h.finalize(), blake3(b"abc"));
/// # }
/// ```
#[derive(Clone)]
pub struct Hasher {
    inner: upstream::Hasher,
}

impl Hasher {
    /// Construct an unkeyed BLAKE3 hasher with the standard initial state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: upstream::Hasher::new(),
        }
    }

    /// Construct a keyed BLAKE3 hasher (MAC mode). `key` is exactly 32 bytes.
    #[must_use]
    pub fn new_keyed(key: &[u8; 32]) -> Self {
        Self {
            inner: upstream::Hasher::new_keyed(key),
        }
    }

    /// Construct a derive-key BLAKE3 hasher. `context` is a hard-coded
    /// application-domain string (see [`blake3_derive_key`] for the
    /// recommended convention).
    #[must_use]
    pub fn new_derive_key(context: &str) -> Self {
        Self {
            inner: upstream::Hasher::new_derive_key(context),
        }
    }

    /// Feed `bytes` into the hasher. Any chunk size is accepted; output is
    /// independent of the chunking.
    pub fn update(&mut self, bytes: &[u8]) {
        self.inner.update(bytes);
    }

    /// Finalize and return the standard 32-byte digest. Consumes the
    /// hasher; for repeated finalization use [`Hasher::finalize_reset`] or
    /// [`Clone`].
    #[must_use]
    pub fn finalize(self) -> [u8; DIGEST_BYTES] {
        *self.inner.finalize().as_bytes()
    }

    /// Finalize into the caller-provided buffer using BLAKE3 XOF mode.
    /// Writes exactly `out.len()` bytes. The first 32 bytes of any XOF
    /// stream equal the standard [`finalize`](Self::finalize) digest.
    pub fn finalize_xof(self, out: &mut [u8]) {
        self.inner.finalize_xof().fill(out);
    }

    /// Finalize and reset to the initial state in one operation. Useful
    /// for hashing a stream of independent messages without reallocating.
    /// Returns the standard 32-byte digest of everything fed since
    /// construction or the previous reset.
    pub fn finalize_reset(&mut self) -> [u8; DIGEST_BYTES] {
        let digest = *self.inner.finalize().as_bytes();
        self.inner.reset();
        digest
    }

    /// Reset to the initial state, dropping any buffered input. The mode
    /// (unkeyed / keyed / derive-key) is preserved.
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BLOCK_BYTES, DIGEST_BYTES, Hasher, blake3, blake3_derive_key, blake3_keyed, blake3_xof,
    };

    fn hex(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    /// Sanity: exposed sizes match the BLAKE3 spec.
    #[test]
    fn digest_block_constants() {
        assert_eq!(DIGEST_BYTES, 32);
        assert_eq!(BLOCK_BYTES, 64);
    }

    /// Empty-input digest from the BLAKE3 official test-vectors file
    /// (`test_vectors/test_vectors.json`, key="").
    #[test]
    fn empty_digest_canonical() {
        assert_eq!(
            hex(&blake3(b"")),
            "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
        );
    }

    /// "abc" digest — also referenced in the BLAKE3 paper §A.1.
    #[test]
    fn abc_digest_canonical() {
        assert_eq!(
            hex(&blake3(b"abc")),
            "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"
        );
    }

    /// 1 MiB of zero bytes — pinned digest computed via the upstream
    /// reference implementation. Exercises the parallel multi-chunk path.
    #[test]
    fn one_mib_zeros() {
        let bytes = vec![0_u8; 1024 * 1024];
        assert_eq!(
            hex(&blake3(&bytes)),
            "488de202f73bd976de4e7048f4e1f39a776d86d582b7348ff53bf432b987fca8"
        );
    }

    /// 1 MiB of 0x42 — second large fixed-content vector, pinned via the
    /// upstream reference.
    #[test]
    fn one_mib_0x42() {
        let bytes = vec![0x42_u8; 1024 * 1024];
        assert_eq!(
            hex(&blake3(&bytes)),
            "bcfe133d6462a3f8d7608158ef428616b98c72c8c630926e3be1db7166d174b4"
        );
    }

    /// Streaming chunked into a variety of sizes (including chunks above
    /// and below the 64-byte internal block boundary) must match the
    /// one-shot digest.
    #[test]
    fn streaming_matches_one_shot() {
        // Use a moderately-large fixed message so multiple BLAKE3 chunks
        // (1024 bytes each) are exercised.
        let mut msg = Vec::with_capacity(8 * 1024);
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        while msg.len() < 8 * 1024 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            msg.extend_from_slice(&state.to_le_bytes());
        }
        msg.truncate(8 * 1024);

        let one_shot = blake3(&msg);
        for &chunk_size in &[1_usize, 17, 64, 65, 1024] {
            let mut h = Hasher::new();
            for window in msg.chunks(chunk_size) {
                h.update(window);
            }
            assert_eq!(
                h.finalize(),
                one_shot,
                "streaming chunk={chunk_size} differs from one-shot"
            );
        }
    }

    /// Keyed mode produces a stable digest distinct from the unkeyed one.
    /// Pinned vector via the upstream reference.
    #[test]
    fn keyed_canonical() {
        let key = [0x42_u8; 32];
        let d = blake3_keyed(&key, b"abc");
        assert_eq!(
            hex(&d),
            "86ecc4fc472a9d0f5e29bc2864865a14a24d36e68f97a149ccae807cfb5cdbf5"
        );
        assert_ne!(d, blake3(b"abc"));
    }

    /// Derive-key mode is deterministic and depends on both context and
    /// material. Pinned vector via the upstream reference.
    #[test]
    fn derive_key_canonical() {
        let dk = blake3_derive_key(
            "tokenfs-algos blake3 test 2026-05-01",
            b"input key material",
        );
        assert_eq!(
            hex(&dk),
            "09dda92e97944ecdf8d0d1a9d1e07e1e0c549b6dad85f291277cb741c1373f01"
        );
        // Different material must change the output.
        let dk2 = blake3_derive_key(
            "tokenfs-algos blake3 test 2026-05-01",
            b"different material",
        );
        assert_ne!(dk, dk2);
        // Different context must change the output.
        let dk3 = blake3_derive_key("a different context", b"input key material");
        assert_ne!(dk, dk3);
    }

    /// XOF outputs at lengths 16, 32, 33, 64, 1024 must equal a single
    /// 1024-byte XOF stream truncated to those lengths. This proves the
    /// XOF prefix is consistent regardless of the requested length.
    #[test]
    fn xof_prefix_consistency() {
        let mut full = [0_u8; 1024];
        blake3_xof(b"abc", &mut full);
        // XOF at 32 bytes equals the standard digest.
        assert_eq!(&full[..DIGEST_BYTES], &blake3(b"abc"));
        for &len in &[16_usize, 32, 33, 64, 1024] {
            let mut buf = vec![0_u8; len];
            blake3_xof(b"abc", &mut buf);
            assert_eq!(buf, full[..len], "xof len={len} prefix mismatch");
        }
    }

    /// Streaming XOF via the [`Hasher::finalize_xof`] route matches the
    /// one-shot [`blake3_xof`] for the same input.
    #[test]
    fn hasher_xof_matches_one_shot_xof() {
        let mut h = Hasher::new();
        h.update(b"abc");
        let mut from_hasher = [0_u8; 256];
        h.finalize_xof(&mut from_hasher);
        let mut from_func = [0_u8; 256];
        blake3_xof(b"abc", &mut from_func);
        assert_eq!(from_hasher, from_func);
    }

    /// `Hasher::finalize_reset` returns the same digest the consuming
    /// `finalize` would have, and leaves the hasher in a usable state for
    /// the next message.
    #[test]
    fn hasher_finalize_reset_round_trip() {
        let mut h = Hasher::new();
        h.update(b"abc");
        let d1 = h.finalize_reset();
        assert_eq!(d1, blake3(b"abc"));
        h.update(b"defgh");
        let d2 = h.finalize();
        assert_eq!(d2, blake3(b"defgh"));
    }

    /// `Hasher::reset` drops state without producing a digest; subsequent
    /// updates start a fresh message.
    #[test]
    fn hasher_reset_clears_state() {
        let mut h = Hasher::new();
        h.update(b"junk we will throw away");
        h.reset();
        h.update(b"abc");
        assert_eq!(h.finalize(), blake3(b"abc"));
    }

    /// Cloning the hasher branches the prefix without re-hashing it.
    #[test]
    fn hasher_clone_branches_prefix() {
        let mut a = Hasher::new();
        a.update(b"common-prefix-");
        let mut b = a.clone();
        a.update(b"left");
        b.update(b"right");
        assert_eq!(a.finalize(), blake3(b"common-prefix-left"));
        assert_eq!(b.finalize(), blake3(b"common-prefix-right"));
    }

    /// `Default` impl matches `new()`.
    #[test]
    fn hasher_default_matches_new() {
        let mut a = Hasher::new();
        let mut b = Hasher::default();
        a.update(b"abc");
        b.update(b"abc");
        assert_eq!(a.finalize(), b.finalize());
    }

    /// Keyed `Hasher` matches the one-shot keyed function.
    #[test]
    fn hasher_keyed_matches_function() {
        let key = [0x55_u8; 32];
        let mut h = Hasher::new_keyed(&key);
        h.update(b"abc");
        assert_eq!(h.finalize(), blake3_keyed(&key, b"abc"));
    }

    /// Derive-key `Hasher` matches the one-shot derive-key function.
    #[test]
    fn hasher_derive_key_matches_function() {
        const CTX: &str = "tokenfs-algos blake3 hasher derive 2026-05-01";
        let mut h = Hasher::new_derive_key(CTX);
        h.update(b"abc");
        let mut out = [0_u8; DIGEST_BYTES];
        h.finalize_xof(&mut out);
        assert_eq!(out, blake3_derive_key(CTX, b"abc"));
    }

    /// XOF on a zero-length output must be a no-op (does not panic).
    #[test]
    fn xof_zero_length() {
        let mut buf: [u8; 0] = [];
        blake3_xof(b"abc", &mut buf);
    }
}
