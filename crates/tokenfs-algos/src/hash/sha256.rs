//! SHA-256 (FIPS 180-4) with portable scalar, x86 SHA-NI, and AArch64
//! FEAT_SHA2 backends.
//!
//! The public [`sha256`] function picks the fastest available backend at
//! runtime. Pinned reference paths live under [`kernels::scalar`] for
//! reproducibility; pinned hardware paths live under [`kernels::x86_shani`]
//! and `kernels::aarch64_sha2` respectively.
//!
//! All backends produce bit-exact identical output. This is enforced by
//! parity tests in this module and a long-input stress vector.

const H0: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// Block size in bytes for SHA-256.
pub const BLOCK_BYTES: usize = 64;

/// Output digest size in bytes for SHA-256.
pub const DIGEST_BYTES: usize = 32;

/// Computes the SHA-256 digest of `bytes` using the fastest available backend.
///
/// # Panics
///
/// Panics if `bytes.len() * 8 > u64::MAX` — i.e. inputs larger than
/// `u64::MAX / 8` bytes (~2 EiB) would overflow the FIPS 180-4 bit-length
/// field. Realistic inputs are nowhere near this cap; kernel/FUSE callers
/// that handle attacker-supplied buffers MUST use [`try_sha256`] (audit-R10
/// #4), which surfaces the overflow as `Sha256LengthOverflow` instead.
///
/// Available only with `feature = "userspace"` — kernel-default builds
/// reach [`try_sha256`] only.
#[cfg(feature = "userspace")]
#[must_use]
pub fn sha256(bytes: &[u8]) -> [u8; DIGEST_BYTES] {
    try_sha256(bytes).expect("sha256: input length exceeds 2^64 bits (~2 EiB)")
}

/// Fallible one-shot SHA-256.
///
/// Returns [`Sha256LengthOverflow`] if `bytes.len() * 8` would not fit
/// in `u64` — i.e. inputs larger than ~2 EiB. Otherwise computes the
/// SHA-256 digest of `bytes` using the fastest available backend
/// (audit-R10 #4 closeout).
///
/// The kernel/FUSE/Postgres-extension entry point for one-shot SHA-256:
/// the per-backend `kernels::*::sha256` use `wrapping_mul(8)` to compute
/// the FIPS bit-length field, so feeding them an input above the cap
/// would silently produce an incorrect digest.
pub fn try_sha256(bytes: &[u8]) -> Result<[u8; DIGEST_BYTES], Sha256LengthOverflow> {
    // FIPS 180-4 bit-length field is u64; reject inputs that would wrap
    // it before any kernel touches the data.
    if (bytes.len() as u64).checked_mul(8).is_none() {
        return Err(Sha256LengthOverflow {
            current_bits: 0,
            attempted_chunk_bytes: bytes.len(),
        });
    }
    Ok(kernels::auto::sha256(bytes))
}

/// Pinned SHA-256 kernels.
pub mod kernels {
    /// Runtime-dispatched SHA-256 entry points.
    pub mod auto {
        use super::super::DIGEST_BYTES;

        /// Computes the SHA-256 digest of `bytes` using the fastest available
        /// backend. Falls back to the portable scalar kernel when no
        /// hardware-accelerated path is enabled.
        #[must_use]
        pub fn sha256(bytes: &[u8]) -> [u8; DIGEST_BYTES] {
            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
            {
                if super::x86_shani::is_available() {
                    // SAFETY: availability checked immediately above.
                    return unsafe { super::x86_shani::sha256(bytes) };
                }
            }
            #[cfg(all(feature = "std", target_arch = "aarch64"))]
            {
                if super::aarch64_sha2::is_available() {
                    // SAFETY: availability checked immediately above.
                    return unsafe { super::aarch64_sha2::sha256(bytes) };
                }
            }
            super::scalar::sha256(bytes)
        }
    }

    /// Portable scalar SHA-256 implementation (FIPS 180-4 reference).
    #[cfg(feature = "arch-pinned-kernels")]
    pub mod scalar;
    #[cfg(not(feature = "arch-pinned-kernels"))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod scalar;

    /// x86 SHA-NI accelerated SHA-256 (Goldmont+, Zen+).
    ///
    /// The compressor follows Intel's published reference for the SHA-NI
    /// extension (see "Intel SHA Extensions" whitepaper and the Linux
    /// kernel's `arch/x86/crypto/sha256_ni_asm.S`). The state is held in
    /// two `__m128i` registers in (ABEF, CDGH) order, where each lane is
    /// a 32-bit word laid out *little-endian* in the register (so a lane
    /// memory dump is `[F, E, B, A]` for the ABEF register).
    #[cfg(all(
        feature = "arch-pinned-kernels",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    pub mod x86_shani;
    #[cfg(all(
        not(feature = "arch-pinned-kernels"),
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod x86_shani;

    /// AArch64 FEAT_SHA2 accelerated SHA-256.
    ///
    /// The block compressor is a direct port of the canonical
    /// noloader / Jeffrey Walton SHA-Intrinsics reference
    /// (`sha256-arm.c`, public domain) which is itself based on ARM /
    /// mbedTLS code by Johannes Schneiders, Skip Hovsmith, and Barry
    /// O'Rourke. The same pattern is used by OpenSSL's
    /// `crypto/sha/asm/sha256-armv8.pl` and Apple's CommonCrypto
    /// `SHA256_Update_ARM`.
    ///
    /// Each 4-round burst follows this fused pattern:
    ///
    /// ```text
    ///     MSGn = vsha256su0q_u32(MSGn, MSGn+1)         // partial schedule
    ///     TMP2 = STATE0
    ///     TMPnext = vaddq_u32(MSGn+1, K[i+4..i+8])     // pipeline next burst
    ///     STATE0 = vsha256hq_u32(STATE0, STATE1, TMPcur)
    ///     STATE1 = vsha256h2q_u32(STATE1, TMP2, TMPcur)
    ///     MSGn = vsha256su1q_u32(MSGn, MSGn+2, MSGn+3) // finish schedule
    /// ```
    ///
    /// The first burst's TMP is precomputed before the body. Bursts
    /// alternate writing TMP0 / TMP1 so each burst consumes the value
    /// produced by the previous burst — this pipeline staging matches
    /// every published canonical implementation.
    ///
    /// Deviations from noloader's reference: (1) we load the 16-byte
    /// chunks via `vld1q_u8` then `vrev32q_u8` (functionally identical
    /// to the noloader `vld1q_u32 + vreinterpretq_u8_u32 + vrev32q_u8`
    /// sequence — both produce big-endian 32-bit message words on
    /// little-endian ARM), (2) we delegate the FIPS 180-4 padding
    /// block(s) to the scalar kernel for bit-exact cross-backend
    /// parity rather than re-implementing the tail in NEON.
    #[cfg(all(feature = "arch-pinned-kernels", target_arch = "aarch64"))]
    pub mod aarch64_sha2;
    #[cfg(all(not(feature = "arch-pinned-kernels"), target_arch = "aarch64"))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod aarch64_sha2;
}

/// Streaming SHA-256 hasher.
///
/// Mirrors the FIPS 180-4 chaining contract: bytes can arrive in any-sized
/// chunks via [`Hasher::update`], partial blocks accumulate in an internal
/// 64-byte buffer, and full blocks are routed to whichever per-backend
/// `compress_blocks` is fastest on the host (detected once at [`Hasher::new`]).
/// [`Hasher::finalize`] performs the canonical padding (0x80, zero-fill,
/// big-endian 64-bit bit length) and emits the 32-byte digest. The streaming
/// path is bit-exact with the one-shot [`sha256`] entry point for any chunking
/// pattern.
///
/// We deliberately do **not** implement [`core::hash::Hasher`]: that trait
/// returns a `u64`, while SHA-256 produces a 256-bit digest. Squeezing the
/// digest down to 64 bits would silently weaken cryptographic strength for
/// every caller that grabbed it through the trait, which is the exact bug
/// `tokenfs-algos` exists to avoid.
///
/// # Example
///
/// ```
/// use tokenfs_algos::hash::sha256::{Hasher, sha256};
///
/// let mut h = Hasher::new();
/// h.update(b"hello, ");
/// h.update(b"world");
/// assert_eq!(h.finalize(), sha256(b"hello, world"));
/// ```
/// Error returned by [`Hasher::try_update`] when the cumulative
/// SHA-256 message length would exceed FIPS 180-4's `2^64 - 1` bit
/// cap. Past the cap the padding length field would wrap and the
/// digest would collide with a shorter different input — a content-ID
/// hazard rather than a memory hazard.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Sha256LengthOverflow {
    /// Cumulative bit length already absorbed before the failed call.
    pub current_bits: u64,
    /// Length in bytes of the chunk that would have pushed past the cap.
    pub attempted_chunk_bytes: usize,
}

impl core::fmt::Display for Sha256LengthOverflow {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "SHA-256 stream length overflow: {} bits already absorbed + {} more bytes would exceed 2^64 - 1 bits",
            self.current_bits, self.attempted_chunk_bytes
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Sha256LengthOverflow {}

/// Streaming SHA-256 hasher.
///
/// See the doctest at the top of this module for the canonical
/// `new() / update() / finalize()` shape. The detected backend is
/// cached on the struct so repeated `update` calls pay no dispatch
/// cost. Length tracking is checked: a cumulative message length
/// past `2^64 - 1` bits panics in [`Self::update`] (or returns
/// [`Sha256LengthOverflow`] from [`Self::try_update`]).
#[derive(Clone)]
pub struct Hasher {
    state: [u32; 8],
    buffer: [u8; BLOCK_BYTES],
    buffered: u8,
    total_bits: u64,
    backend: HasherBackend,
}

/// SHA-256 backend selected for a streaming [`Hasher`] instance.
///
/// The variant is decided once at [`Hasher::new`] via the same runtime feature
/// detection used by the one-shot dispatcher; subsequent updates pay no
/// per-call detection cost.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HasherBackend {
    /// Portable scalar fallback. Always available.
    Scalar,
    /// x86 SHA-NI (`sha` + `sse4.1` + `ssse3`).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Shani,
    /// AArch64 FEAT_SHA2 (`sha2`).
    #[cfg(target_arch = "aarch64")]
    AArch64Sha2,
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher {
    /// Construct a fresh streaming SHA-256 hasher seeded with the FIPS 180-4
    /// initial state. The fastest available backend is detected here and
    /// cached on the struct so [`Hasher::update`] pays no dispatch cost.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: H0,
            buffer: [0_u8; BLOCK_BYTES],
            buffered: 0,
            total_bits: 0,
            backend: detect_backend(),
        }
    }

    /// Returns the backend selected for this hasher instance.
    #[must_use]
    pub const fn backend(&self) -> HasherBackend {
        self.backend
    }

    /// Reset the hasher to its initial state. Backend selection is preserved.
    pub fn reset(&mut self) {
        self.state = H0;
        self.buffer = [0_u8; BLOCK_BYTES];
        self.buffered = 0;
        self.total_bits = 0;
    }

    /// Feed `bytes` into the hash. Calls may be of any length, including
    /// empty.
    ///
    /// # Length limit (panics)
    ///
    /// FIPS 180-4 caps the SHA-256 message length at `2^64 - 1` bits
    /// (= ~2 EiB). This streaming Hasher panics if `update` would push
    /// the cumulative length past that bound. Past the cap the padding
    /// length field would wrap and the digest would collide with a
    /// shorter, different input — a content-ID hazard, not a memory
    /// hazard.
    ///
    /// # Kernel/FUSE callers
    ///
    /// Use [`Self::try_update`] instead. Untrusted callers can supply
    /// adversarial byte streams approaching the 2 EiB cap (over multiple
    /// syscalls feeding the same hasher); a panic at the kernel
    /// boundary is a DoS vector. `try_update` returns
    /// [`Sha256LengthOverflow`] on overflow with both the current bit
    /// count and the attempted chunk size for diagnostics.
    ///
    /// Available only with `feature = "userspace"` since v0.4.3
    /// (audit-R9 #8). Default kernel-safe builds reach `try_update`
    /// only.
    #[cfg(feature = "userspace")]
    pub fn update(&mut self, bytes: &[u8]) {
        self.try_update(bytes)
            .expect("SHA-256 stream length exceeded 2^64 bits");
    }

    /// Fallible variant of [`Self::update`]: returns
    /// [`Sha256LengthOverflow`] if the cumulative bit length would
    /// exceed FIPS 180-4's `2^64 - 1` bit cap. Successful calls leave
    /// the hasher state advanced; failed calls leave it unchanged.
    pub fn try_update(&mut self, bytes: &[u8]) -> Result<(), Sha256LengthOverflow> {
        if bytes.is_empty() {
            return Ok(());
        }
        let added_bits = (bytes.len() as u64)
            .checked_mul(8)
            .ok_or(Sha256LengthOverflow {
                current_bits: self.total_bits,
                attempted_chunk_bytes: bytes.len(),
            })?;
        self.total_bits = self
            .total_bits
            .checked_add(added_bits)
            .ok_or(Sha256LengthOverflow {
                current_bits: self.total_bits,
                attempted_chunk_bytes: bytes.len(),
            })?;

        let mut input = bytes;
        let buffered = self.buffered as usize;

        // 1. Top off any partially-filled buffer first.
        if buffered != 0 {
            let need = BLOCK_BYTES - buffered;
            if input.len() < need {
                self.buffer[buffered..buffered + input.len()].copy_from_slice(input);
                self.buffered = (buffered + input.len()) as u8;
                return Ok(());
            }
            self.buffer[buffered..BLOCK_BYTES].copy_from_slice(&input[..need]);
            input = &input[need..];
            self.buffered = 0;
            // SAFETY: pointer is to a valid 64-byte array.
            unsafe {
                self.compress_buffer();
            }
        }

        // 2. Compress whole 64-byte blocks straight from the input. Routing the
        //    entire run through one `compress_blocks` call lets the HW backends
        //    amortize state load/store across the chunk.
        let full_blocks = input.len() / BLOCK_BYTES;
        if full_blocks != 0 {
            let consumed = full_blocks * BLOCK_BYTES;
            // SAFETY: bounds checked above; the per-backend compress is gated
            // by the runtime detection performed in `detect_backend()`.
            unsafe {
                self.compress_blocks_dispatch(input.as_ptr(), full_blocks);
            }
            input = &input[consumed..];
        }

        // 3. Stash any tail bytes for the next update / finalize.
        if !input.is_empty() {
            self.buffer[..input.len()].copy_from_slice(input);
            self.buffered = input.len() as u8;
        }
        Ok(())
    }

    /// Consume the hasher and emit the 32-byte digest.
    #[must_use]
    pub fn finalize(mut self) -> [u8; DIGEST_BYTES] {
        self.finalize_in_place()
    }

    /// Emit the digest and reset the hasher to its initial state. Useful for
    /// hash-of-hash trees / Merkle constructions where the same hasher is
    /// reused across many sibling digests.
    pub fn finalize_reset(&mut self) -> [u8; DIGEST_BYTES] {
        let digest = self.finalize_in_place();
        self.reset();
        digest
    }

    fn finalize_in_place(&mut self) -> [u8; DIGEST_BYTES] {
        // Build the FIPS 180-4 padding into the existing buffer (plus, if
        // needed, one extra block-sized scratch). The buffer already contains
        // the unconsumed tail bytes; we append 0x80, zero-fill, and write the
        // big-endian 64-bit bit length into the last 8 bytes of the final
        // padding block.
        let buffered = self.buffered as usize;
        let mut last = [0_u8; BLOCK_BYTES * 2];
        last[..buffered].copy_from_slice(&self.buffer[..buffered]);
        last[buffered] = 0x80;

        let total = if buffered + 1 + 8 <= BLOCK_BYTES {
            BLOCK_BYTES
        } else {
            BLOCK_BYTES * 2
        };
        let length_off = total - 8;
        last[length_off..total].copy_from_slice(&self.total_bits.to_be_bytes());

        // Padding is small and rare (one or two blocks per finalize), so
        // delegating to the scalar reference here keeps the per-backend code
        // small without measurable cost.
        let pad_block: &[u8; BLOCK_BYTES] = (&last[..BLOCK_BYTES])
            .try_into()
            .expect("BLOCK_BYTES slice");
        kernels::scalar::compress(&mut self.state, pad_block);
        if total == BLOCK_BYTES * 2 {
            let pad2: &[u8; BLOCK_BYTES] = (&last[BLOCK_BYTES..total])
                .try_into()
                .expect("BLOCK_BYTES slice");
            kernels::scalar::compress(&mut self.state, pad2);
        }

        let mut digest = [0_u8; DIGEST_BYTES];
        for (i, word) in self.state.iter().enumerate() {
            digest[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
        }
        digest
    }

    /// Dispatch one full-block compress through the cached backend. Used to
    /// flush the internal accumulator when it tops off mid-`update`.
    ///
    /// # Safety
    ///
    /// Requires that `self.backend` was selected by `detect_backend()`, which
    /// performs the appropriate runtime feature check.
    unsafe fn compress_buffer(&mut self) {
        let ptr = self.buffer.as_ptr();
        // SAFETY: backend matches a runtime-detected capability.
        unsafe {
            self.compress_blocks_dispatch(ptr, 1);
        }
    }

    /// Dispatch `n_blocks` full-block compressions through the cached backend.
    ///
    /// # Safety
    ///
    /// `block_ptr` must point to at least `n_blocks * 64` readable bytes and
    /// `self.backend` must match a runtime-detected capability.
    unsafe fn compress_blocks_dispatch(&mut self, block_ptr: *const u8, n_blocks: usize) {
        match self.backend {
            HasherBackend::Scalar => {
                for i in 0..n_blocks {
                    // SAFETY: caller guarantees `block_ptr + i*64 + 63` is
                    // readable.
                    let block_ref: &[u8; BLOCK_BYTES] =
                        unsafe { &*(block_ptr.add(i * BLOCK_BYTES).cast::<[u8; BLOCK_BYTES]>()) };
                    kernels::scalar::compress(&mut self.state, block_ref);
                }
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            HasherBackend::Shani => {
                // SAFETY: backend variant is only stored when
                // `kernels::x86_shani::is_available()` returned true.
                unsafe {
                    kernels::x86_shani::compress_blocks(&mut self.state, block_ptr, n_blocks);
                }
            }
            #[cfg(target_arch = "aarch64")]
            HasherBackend::AArch64Sha2 => {
                // SAFETY: backend variant is only stored when
                // `kernels::aarch64_sha2::is_available()` returned true.
                unsafe {
                    kernels::aarch64_sha2::compress_blocks(&mut self.state, block_ptr, n_blocks);
                }
            }
        }
    }
}

impl core::fmt::Debug for Hasher {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Avoid leaking partial-input bytes via Debug; only expose shape.
        f.debug_struct("Hasher")
            .field("backend", &self.backend)
            .field("buffered", &self.buffered)
            .field("total_bits", &self.total_bits)
            .finish_non_exhaustive()
    }
}

fn detect_backend() -> HasherBackend {
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if kernels::x86_shani::is_available() {
            return HasherBackend::Shani;
        }
    }
    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    {
        if kernels::aarch64_sha2::is_available() {
            return HasherBackend::AArch64Sha2;
        }
    }
    HasherBackend::Scalar
}

#[cfg(test)]
mod tests {
    use super::{Hasher, HasherBackend, kernels, try_sha256};
    // `Vec`, `String`, and the `vec!` / `format!` macros are not in the
    // no-std prelude; alias them from `alloc` for the alloc-only build
    // (audit-R6 finding #164).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::format;
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::string::String;
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec;
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec::Vec;

    fn hex(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    /// FIPS 180-4 § B.2 (empty message).
    #[test]
    fn nist_empty() {
        let d = kernels::scalar::sha256(b"");
        assert_eq!(
            hex(&d),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    /// `try_sha256` returns the same digest as the kernel auto path for
    /// realistic inputs (audit-R10 #4).
    #[test]
    fn try_sha256_matches_dispatch_for_normal_input() {
        let cases: &[&[u8]] = &[b"", b"abc", b"the quick brown fox jumps over the lazy dog"];
        for &msg in cases {
            let direct = kernels::auto::sha256(msg);
            let fallible = try_sha256(msg).expect("normal input within bit-length cap");
            assert_eq!(direct, fallible);
        }
    }

    /// `try_sha256` Err path for the ~2 EiB cap is unreachable on real
    /// allocations; verify the cap arithmetic via a constructed byte
    /// slice with `len() > usize::MAX / 8` would have wrapped.
    /// Instead we exercise the boundary check by asserting a small
    /// input does NOT trip the err.
    #[test]
    fn try_sha256_does_not_reject_realistic_inputs() {
        let msg = vec![0u8; 1 << 20]; // 1 MiB
        assert!(try_sha256(&msg).is_ok());
    }

    /// FIPS 180-4 § B.1 ("abc").
    #[test]
    fn nist_abc() {
        let d = kernels::scalar::sha256(b"abc");
        assert_eq!(
            hex(&d),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    /// FIPS 180-4 § B.2 (multi-block sample fitting in two blocks).
    #[test]
    fn nist_two_block() {
        let d =
            kernels::scalar::sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        assert_eq!(
            hex(&d),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    /// 10000 'a' bytes — many full blocks plus a partial tail.
    /// Reference computed by feeding the same input through OpenSSL
    /// `sha256sum` and locked in here.
    #[test]
    fn long_input_stress() {
        let bytes = vec![b'a'; 10_000];
        let d = kernels::scalar::sha256(&bytes);
        assert_eq!(
            hex(&d),
            "27dd1f61b867b6a0f6e9d8a41c43231de52107e53ae424de8f847b821db4b711"
        );
    }

    #[test]
    fn public_dispatch_matches_scalar() {
        let cases: [&[u8]; 6] = [
            b"",
            b"abc",
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            b"a",
            &[0x42_u8; 64],
            &[0xa5_u8; 1000],
        ];
        for case in cases {
            assert_eq!(
                kernels::auto::sha256(case),
                kernels::scalar::sha256(case),
                "auto vs scalar for len={}",
                case.len(),
            );
        }
    }

    // The runtime-availability tests below print a skip notice via
    // `eprintln!` (only in `std` builds) when the SIMD path is missing on
    // the host; gate them on `feature = "std"` so the alloc-only build
    // compiles without pulling in stdio (audit-R6 #164).
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn x86_shani_parity_matches_scalar() {
        if !kernels::x86_shani::is_available() {
            eprintln!("skipping: SHA-NI not available on this host");
            return;
        }
        let cases: [&[u8]; 8] = [
            b"",
            b"abc",
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            b"a",
            &[0x42_u8; 55],
            &[0x42_u8; 56],
            &[0x42_u8; 64],
            &[0xa5_u8; 10_000],
        ];
        for case in cases {
            // SAFETY: availability checked above.
            let hw = unsafe { kernels::x86_shani::sha256(case) };
            assert_eq!(
                hw,
                kernels::scalar::sha256(case),
                "shani vs scalar for len={}",
                case.len(),
            );
        }
    }

    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    #[test]
    fn aarch64_sha2_parity_matches_scalar() {
        if !kernels::aarch64_sha2::is_available() {
            eprintln!("skipping: FEAT_SHA2 not available on this host");
            return;
        }
        let cases: [&[u8]; 8] = [
            b"",
            b"abc",
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            b"a",
            &[0x42_u8; 55],
            &[0x42_u8; 56],
            &[0x42_u8; 64],
            &[0xa5_u8; 10_000],
        ];
        for case in cases {
            // SAFETY: availability checked above.
            let hw = unsafe { kernels::aarch64_sha2::sha256(case) };
            assert_eq!(
                hw,
                kernels::scalar::sha256(case),
                "sha2 vs scalar for len={}",
                case.len(),
            );
        }
    }

    /// FEAT_SHA2 path direct-vs-NIST check. Pinned alongside the
    /// scalar `nist_*` cases so a same-direction regression in both
    /// kernels would still be caught. The 64-byte case is the one
    /// that historically failed on real ARM silicon (CI run
    /// 25241406257); locking it in here makes the bug-class
    /// non-recurring.
    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    #[test]
    fn aarch64_sha2_known_vectors() {
        if !kernels::aarch64_sha2::is_available() {
            eprintln!("skipping: FEAT_SHA2 not available on this host");
            return;
        }
        // Each row: (input, expected SHA-256 hex).
        let vectors: &[(&[u8], &str)] = &[
            (
                b"",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                b"abc",
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
            (
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            ),
            // 64-byte all-0x42 input — the historical failure case.
            // Reference computed via Python `hashlib.sha256(b'B'*64)`.
            (
                &[0x42_u8; 64],
                "c422e7070cb1cb455b5de9afee0d975e303d0239c72030cd7414ab5c382d3ae8",
            ),
        ];
        for (input, expected_hex) in vectors {
            // SAFETY: availability checked above.
            let hw = unsafe { kernels::aarch64_sha2::sha256(input) };
            assert_eq!(
                hex(&hw),
                *expected_hex,
                "sha2 known vector mismatch for len={}",
                input.len(),
            );
        }
    }

    // ----- streaming Hasher tests -------------------------------------------

    /// A pseudo-random byte stream for parameterized streaming tests. Same
    /// LCG used by `examples/bench_compare.rs::make_random_bytes` so the
    /// inputs match the calibration corpus.
    fn random_bytes(n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n);
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        while out.len() < n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out.extend_from_slice(&state.to_le_bytes());
        }
        out.truncate(n);
        out
    }

    #[test]
    fn hasher_empty_matches_one_shot() {
        let h = Hasher::new();
        assert_eq!(h.finalize(), kernels::auto::sha256(b""));
    }

    #[test]
    fn hasher_single_call_matches_one_shot() {
        let payload = b"the quick brown fox jumps over the lazy dog";
        let mut h = Hasher::new();
        h.try_update(payload)
            .expect("test sha256 update within bounds");
        assert_eq!(h.finalize(), kernels::auto::sha256(payload));
    }

    #[test]
    fn hasher_nist_abc_matches_one_shot() {
        let mut h = Hasher::new();
        h.try_update(b"abc")
            .expect("test sha256 update within bounds");
        assert_eq!(
            hex(&h.finalize()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn hasher_nist_two_block_matches_one_shot() {
        let mut h = Hasher::new();
        h.try_update(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
            .expect("test sha256 update within bounds");
        assert_eq!(
            hex(&h.finalize()),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn hasher_chunked_matches_one_shot_for_all_chunk_sizes() {
        // 100 KiB random payload, chunked into 1, 17, 64, 65, 1024-byte
        // updates. Must produce the same digest as a single-shot call.
        let payload = random_bytes(100 * 1024);
        let expected = kernels::auto::sha256(&payload);
        for &chunk in &[1_usize, 17, 63, 64, 65, 127, 128, 1024, 4096] {
            let mut h = Hasher::new();
            for block in payload.chunks(chunk) {
                h.try_update(block)
                    .expect("test sha256 update within bounds");
            }
            assert_eq!(h.finalize(), expected, "mismatch for chunk={chunk}");
        }
    }

    #[test]
    fn hasher_empty_updates_are_no_ops() {
        let payload = b"hello, world";
        let expected = kernels::auto::sha256(payload);
        let mut h = Hasher::new();
        h.try_update(b"").expect("test sha256 update within bounds");
        h.try_update(payload)
            .expect("test sha256 update within bounds");
        h.try_update(b"").expect("test sha256 update within bounds");
        assert_eq!(h.finalize(), expected);
    }

    #[test]
    fn hasher_finalize_reset_matches_finalize_then_new() {
        let payload = b"reset me and try again";
        let mut h = Hasher::new();
        h.try_update(payload)
            .expect("test sha256 update within bounds");
        let d1 = h.finalize_reset();
        assert_eq!(d1, kernels::auto::sha256(payload));
        // After reset, the hasher should produce the empty digest.
        assert_eq!(h.clone().finalize(), kernels::auto::sha256(b""));
        h.try_update(payload)
            .expect("test sha256 update within bounds");
        assert_eq!(h.finalize(), d1);
    }

    #[test]
    fn hasher_reset_clears_state() {
        let mut h = Hasher::new();
        h.try_update(b"garbage")
            .expect("test sha256 update within bounds");
        h.reset();
        h.try_update(b"abc")
            .expect("test sha256 update within bounds");
        assert_eq!(h.finalize(), kernels::auto::sha256(b"abc"));
    }

    /// All NIST § B and the long-stress vector via the streaming path.
    #[test]
    fn hasher_nist_vectors_stream() {
        let vectors: &[(&[u8], &str)] = &[
            (
                b"",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                b"abc",
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
            (
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            ),
        ];
        for (input, expected_hex) in vectors {
            // Try every plausible chunking pattern.
            for &chunk in &[1_usize, 7, 32, 56, 64] {
                let mut h = Hasher::new();
                for block in input.chunks(chunk) {
                    h.try_update(block)
                        .expect("test sha256 update within bounds");
                }
                assert_eq!(
                    hex(&h.finalize()),
                    *expected_hex,
                    "stream NIST vector mismatch for len={} chunk={}",
                    input.len(),
                    chunk,
                );
            }
        }

        // 10000 'a' bytes — the stress vector, fed in 1000-byte chunks.
        let bytes = vec![b'a'; 10_000];
        let mut h = Hasher::new();
        for block in bytes.chunks(1000) {
            h.try_update(block)
                .expect("test sha256 update within bounds");
        }
        assert_eq!(
            hex(&h.finalize()),
            "27dd1f61b867b6a0f6e9d8a41c43231de52107e53ae424de8f847b821db4b711"
        );
    }

    /// Force the streaming hasher onto every available backend and confirm
    /// they all produce the bit-exact digest for the same chunked input.
    #[test]
    fn hasher_cross_backend_parity() {
        let payload = random_bytes(8_192);
        // Reference: scalar one-shot.
        let expected = kernels::scalar::sha256(&payload);

        // Every backend variant we can construct on this host.
        let mut backends: Vec<HasherBackend> = vec![HasherBackend::Scalar];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if kernels::x86_shani::is_available() {
            backends.push(HasherBackend::Shani);
        }
        #[cfg(target_arch = "aarch64")]
        if kernels::aarch64_sha2::is_available() {
            backends.push(HasherBackend::AArch64Sha2);
        }

        for backend in backends {
            for &chunk in &[1_usize, 17, 64, 65, 4096] {
                let mut h = Hasher::new();
                // Force the variant we want to exercise; new() picked the
                // host's fastest, so we override here for parity coverage.
                h.backend = backend;
                for block in payload.chunks(chunk) {
                    h.try_update(block)
                        .expect("test sha256 update within bounds");
                }
                assert_eq!(h.finalize(), expected, "backend={backend:?} chunk={chunk}");
            }
        }
    }

    #[test]
    fn hasher_default_matches_new() {
        let a = Hasher::default().finalize();
        let b = Hasher::new().finalize();
        assert_eq!(a, b);
    }

    #[test]
    fn hasher_try_update_detects_2_to_64_bit_length_overflow() {
        // Drive total_bits to just under 2^64. We can't actually feed
        // 2 EiB of bytes through the hasher in a test, so simulate by
        // manipulating the field directly via a sequence that gets
        // close to the cap, then ask try_update to push past it.
        let mut h = Hasher::new();
        // Set the cumulative bit count to 2^64 - 16 by directly
        // mutating the field via reset+manual update of the public
        // surface. We do this by hand-setting through unsafe — for
        // a test, the cleanest path is to use the public surface to
        // get to a known state, then the next try_update with a
        // chunk whose bit-length pushes past 2^64 must error.
        //
        // The easiest reproducible setup: ask try_update for a chunk
        // whose `len * 8` itself overflows. That triggers the
        // `checked_mul(8)` path. usize::MAX bytes is unrepresentable
        // as a slice we can allocate, but on 64-bit we can pass a
        // synthetic slice header pointing at a tiny buffer with a
        // big length using an empty wrapper. Instead, exercise the
        // `checked_add` path by setting total_bits via direct field
        // access — gated by the test itself living in the same crate
        // module.
        h.total_bits = u64::MAX - 7; // one more byte = +8 bits = wraps
        let err = h.try_update(b"x").expect_err("should overflow");
        assert_eq!(err.current_bits, u64::MAX - 7);
        assert_eq!(err.attempted_chunk_bytes, 1);
        // The hasher state must NOT have advanced — total_bits unchanged.
        assert_eq!(h.total_bits, u64::MAX - 7);
        // A fresh hasher accepts the same byte without panicking.
        let mut h2 = Hasher::new();
        assert!(h2.try_update(b"x").is_ok());
    }

    #[cfg(feature = "userspace")]
    #[test]
    #[should_panic(expected = "SHA-256 stream length exceeded 2^64 bits")]
    fn hasher_update_panics_on_overflow() {
        // Mirrors the try_update test but exercises the panicking
        // wrapper (gated on `userspace` per audit-R9 #8) so its
        // documented panic behavior is pinned.
        let mut h = Hasher::new();
        h.total_bits = u64::MAX - 7;
        h.update(b"x");
    }
}
