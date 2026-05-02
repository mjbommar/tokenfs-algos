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
#[must_use]
pub fn sha256(bytes: &[u8]) -> [u8; DIGEST_BYTES] {
    kernels::auto::sha256(bytes)
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
    pub mod scalar {
        use super::super::{BLOCK_BYTES, DIGEST_BYTES, H0, K};

        /// Computes SHA-256 of `bytes` with the portable scalar kernel.
        #[must_use]
        pub fn sha256(bytes: &[u8]) -> [u8; DIGEST_BYTES] {
            let mut state = H0;

            let full_blocks = bytes.len() / BLOCK_BYTES;
            let tail_start = full_blocks * BLOCK_BYTES;

            for block_index in 0..full_blocks {
                let off = block_index * BLOCK_BYTES;
                let block: &[u8; BLOCK_BYTES] = (&bytes[off..off + BLOCK_BYTES])
                    .try_into()
                    .expect("BLOCK_BYTES slice");
                compress(&mut state, block);
            }

            // Padding: append 0x80, zero-fill, then big-endian 64-bit bit length.
            let tail = &bytes[tail_start..];
            let bit_len = (bytes.len() as u64).wrapping_mul(8);
            let mut last = [0_u8; BLOCK_BYTES * 2];
            last[..tail.len()].copy_from_slice(tail);
            last[tail.len()] = 0x80;

            let total = if tail.len() + 1 + 8 <= BLOCK_BYTES {
                BLOCK_BYTES
            } else {
                BLOCK_BYTES * 2
            };
            let length_off = total - 8;
            last[length_off..total].copy_from_slice(&bit_len.to_be_bytes());

            let last_block: &[u8; BLOCK_BYTES] = (&last[..BLOCK_BYTES])
                .try_into()
                .expect("BLOCK_BYTES slice");
            compress(&mut state, last_block);
            if total == BLOCK_BYTES * 2 {
                let next_block: &[u8; BLOCK_BYTES] = (&last[BLOCK_BYTES..total])
                    .try_into()
                    .expect("BLOCK_BYTES slice");
                compress(&mut state, next_block);
            }

            let mut digest = [0_u8; DIGEST_BYTES];
            for (i, word) in state.iter().enumerate() {
                digest[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
            }
            digest
        }

        /// Compress one 64-byte block with the FIPS 180-4 reference path.
        ///
        /// This is exposed (crate-private) so backend modules and tests
        /// can reuse the bit-exact reference for the padding block step.
        pub fn compress(state: &mut [u32; 8], block: &[u8; BLOCK_BYTES]) {
            let mut w = [0_u32; 64];
            for (i, slot) in w.iter_mut().enumerate().take(16) {
                let off = i * 4;
                *slot = u32::from_be_bytes([
                    block[off],
                    block[off + 1],
                    block[off + 2],
                    block[off + 3],
                ]);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }

            let mut a = state[0];
            let mut b = state[1];
            let mut c = state[2];
            let mut d = state[3];
            let mut e = state[4];
            let mut f = state[5];
            let mut g = state[6];
            let mut h = state[7];

            for i in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ (!e & g);
                let temp1 = h
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let temp2 = s0.wrapping_add(maj);

                h = g;
                g = f;
                f = e;
                e = d.wrapping_add(temp1);
                d = c;
                c = b;
                b = a;
                a = temp1.wrapping_add(temp2);
            }

            state[0] = state[0].wrapping_add(a);
            state[1] = state[1].wrapping_add(b);
            state[2] = state[2].wrapping_add(c);
            state[3] = state[3].wrapping_add(d);
            state[4] = state[4].wrapping_add(e);
            state[5] = state[5].wrapping_add(f);
            state[6] = state[6].wrapping_add(g);
            state[7] = state[7].wrapping_add(h);
        }
    }

    /// x86 SHA-NI accelerated SHA-256 (Goldmont+, Zen+).
    ///
    /// The compressor follows Intel's published reference for the SHA-NI
    /// extension (see "Intel SHA Extensions" whitepaper and the Linux
    /// kernel's `arch/x86/crypto/sha256_ni_asm.S`). The state is held in
    /// two `__m128i` registers in (ABEF, CDGH) order, where each lane is
    /// a 32-bit word laid out *little-endian* in the register (so a lane
    /// memory dump is `[F, E, B, A]` for the ABEF register).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub mod x86_shani {
        use super::super::{BLOCK_BYTES, DIGEST_BYTES, H0};
        use super::scalar;

        #[cfg(target_arch = "x86")]
        use core::arch::x86::{
            __m128i, _mm_add_epi32, _mm_alignr_epi8, _mm_blend_epi16, _mm_loadu_si128,
            _mm_set_epi64x, _mm_sha256msg1_epu32, _mm_sha256msg2_epu32, _mm_sha256rnds2_epu32,
            _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_storeu_si128,
        };
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{
            __m128i, _mm_add_epi32, _mm_alignr_epi8, _mm_blend_epi16, _mm_loadu_si128,
            _mm_set_epi64x, _mm_sha256msg1_epu32, _mm_sha256msg2_epu32, _mm_sha256rnds2_epu32,
            _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_storeu_si128,
        };

        /// Returns true when SHA-NI is available at runtime.
        ///
        /// The compressor needs `sha`, `ssse3` (for `pshufb`/`alignr`),
        /// and `sse4.1` (for `pblendw`).
        #[cfg(feature = "std")]
        #[must_use]
        pub fn is_available() -> bool {
            std::is_x86_feature_detected!("sha")
                && std::is_x86_feature_detected!("sse4.1")
                && std::is_x86_feature_detected!("ssse3")
        }

        /// Returns false because runtime SHA-NI detection requires `std`.
        #[cfg(not(feature = "std"))]
        #[must_use]
        pub const fn is_available() -> bool {
            false
        }

        /// Computes SHA-256 of `bytes` using the SHA-NI extension for
        /// every full block, then a scalar fallback for the padding
        /// block(s) so the bit-length tail is handled identically across
        /// backends.
        ///
        /// # Safety
        ///
        /// Caller must ensure SHA-NI plus SSE4.1 + SSSE3 are available.
        #[target_feature(enable = "sha,sse4.1,ssse3")]
        #[must_use]
        pub unsafe fn sha256(bytes: &[u8]) -> [u8; DIGEST_BYTES] {
            let mut state = H0;

            let full_blocks = bytes.len() / BLOCK_BYTES;
            let tail_start = full_blocks * BLOCK_BYTES;

            if full_blocks > 0 {
                // SAFETY: caller's target_feature contract guarantees SHA-NI;
                // we read `full_blocks * BLOCK_BYTES` bytes and write 8 u32s.
                unsafe {
                    compress_blocks(&mut state, bytes.as_ptr(), full_blocks);
                }
            }

            let tail = &bytes[tail_start..];
            let bit_len = (bytes.len() as u64).wrapping_mul(8);
            let mut last = [0_u8; BLOCK_BYTES * 2];
            last[..tail.len()].copy_from_slice(tail);
            last[tail.len()] = 0x80;

            let total = if tail.len() + 1 + 8 <= BLOCK_BYTES {
                BLOCK_BYTES
            } else {
                BLOCK_BYTES * 2
            };
            let length_off = total - 8;
            last[length_off..total].copy_from_slice(&bit_len.to_be_bytes());

            let pad_block: &[u8; BLOCK_BYTES] = (&last[..BLOCK_BYTES])
                .try_into()
                .expect("BLOCK_BYTES slice");
            scalar::compress(&mut state, pad_block);
            if total == BLOCK_BYTES * 2 {
                let pad2: &[u8; BLOCK_BYTES] = (&last[BLOCK_BYTES..total])
                    .try_into()
                    .expect("BLOCK_BYTES slice");
                scalar::compress(&mut state, pad2);
            }

            let mut digest = [0_u8; DIGEST_BYTES];
            for (i, word) in state.iter().enumerate() {
                digest[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
            }
            digest
        }

        /// Compress `n_blocks` consecutive 64-byte blocks read from `block_ptr`
        /// into the scalar `state`. Loads the (ABEF, CDGH) vector pair once,
        /// runs the SHA-NI block compressor `n_blocks` times, then writes the
        /// scalar state back. This is the entry point used by the streaming
        /// [`super::super::Hasher`] so a 4 KiB chunk pays the
        /// `state_from_words`/`state_to_words` cost once instead of 64 times.
        ///
        /// # Safety
        ///
        /// `block_ptr` must point to at least `n_blocks * 64` readable bytes.
        /// SHA-NI plus SSE4.1 + SSSE3 must be available; this is enforced by
        /// the `#[target_feature]` gate plus the caller's runtime check.
        #[target_feature(enable = "sha,sse4.1,ssse3")]
        pub(crate) unsafe fn compress_blocks(
            state: &mut [u32; 8],
            block_ptr: *const u8,
            n_blocks: usize,
        ) {
            if n_blocks == 0 {
                return;
            }
            // SAFETY: state is a valid 8-dword array.
            let (mut state0, mut state1) = unsafe { state_from_words(state) };

            // Byte-swap mask for converting little-endian xmm dword loads
            // into big-endian message words.
            let bswap = _mm_set_epi64x(
                0x0c0d_0e0f_0809_0a0b_u64 as i64,
                0x0405_0607_0001_0203_u64 as i64,
            );

            for block_index in 0..n_blocks {
                // SAFETY: caller guarantees `block_ptr + i*64 + 63` is readable.
                let (s0, s1) = unsafe {
                    compress_block_shani(
                        state0,
                        state1,
                        block_ptr.add(block_index * BLOCK_BYTES),
                        bswap,
                    )
                };
                state0 = s0;
                state1 = s1;
            }

            // SAFETY: state is a valid 8-dword writable destination.
            unsafe { state_to_words(state, state0, state1) };
        }

        // -- helpers ------------------------------------------------------

        /// Pack scalar `state` into the (ABEF, CDGH) xmm pair the SHA-NI
        /// round intrinsics consume.
        ///
        /// The byte pattern in the ABEF register is `[F, E, B, A]` with
        /// the lowest dword first; CDGH is `[H, G, D, C]`. This matches
        /// Intel's published example and is also the layout used by the
        /// Linux kernel's `sha256_ni_asm.S`.
        ///
        /// # Safety
        ///
        /// Requires SHA + SSE4.1 + SSSE3 (the function is called inside a
        /// `target_feature`-gated path).
        #[target_feature(enable = "sha,sse4.1,ssse3")]
        #[inline]
        unsafe fn state_from_words(state: &[u32; 8]) -> (__m128i, __m128i) {
            // Load ABCD and EFGH straight from memory; both registers
            // hold (a,b,c,d) and (e,f,g,h) with `a` in the low dword.
            // SAFETY: 16 readable bytes at the pointer.
            let abcd = unsafe { _mm_loadu_si128(state.as_ptr().cast::<__m128i>()) };
            let efgh = unsafe { _mm_loadu_si128(state.as_ptr().add(4).cast::<__m128i>()) };

            // SHA-NI wants ABEF and CDGH where the high dword is the
            // first state word. Following Intel's sample:
            //   tmp     = shuffle32(abcd, 0xB1)  // -> (b,a,d,c)
            //   state1  = shuffle32(efgh, 0x1B)  // -> (h,g,f,e)
            //   state0  = alignr8(tmp, state1, 8) // -> (f,e,b,a) == ABEF lane order
            //   state1  = blend16(state1, tmp, 0xF0) // (h,g,d,c) == CDGH lane order
            let tmp = _mm_shuffle_epi32::<0xB1>(abcd);
            let state1_init = _mm_shuffle_epi32::<0x1B>(efgh);
            let state0 = _mm_alignr_epi8::<8>(tmp, state1_init);
            let state1 = _mm_blend_epi16::<0xF0>(state1_init, tmp);
            (state0, state1)
        }

        /// Reverse of [`state_from_words`].
        ///
        /// # Safety
        ///
        /// Requires SHA + SSE4.1 (provided by the call site's
        /// `target_feature` gate).
        #[target_feature(enable = "sha,sse4.1,ssse3")]
        #[inline]
        unsafe fn state_to_words(state: &mut [u32; 8], state0: __m128i, state1: __m128i) {
            // Reverse the packing in `state_from_words`:
            //   state0 = (f,e,b,a), state1 = (h,g,d,c)
            // We want abcd = (a,b,c,d), efgh = (e,f,g,h).
            //   tmp    = shuffle32(state0, 0x1B)  // -> (a,b,e,f)
            //   state1' = shuffle32(state1, 0xB1) // -> (g,h,c,d)
            //   abcd    = alignr8(tmp, state1', 8) // -> (c,d,a,b)? need to verify
            //
            // The simpler, slower-but-correct path: store both registers
            // and manually re-pack via 8 dword loads. This costs four
            // moves but happens once per digest call so the overhead is
            // negligible relative to compression.
            let mut buf = [0_u32; 8];
            // SAFETY: 16-byte writable destinations on stack.
            unsafe {
                _mm_storeu_si128(buf.as_mut_ptr().cast::<__m128i>(), state0);
                _mm_storeu_si128(buf.as_mut_ptr().add(4).cast::<__m128i>(), state1);
            }
            // buf layout (ABEF, CDGH) lane dump:
            //   buf[0..4] = [F, E, B, A]
            //   buf[4..8] = [H, G, D, C]
            state[0] = buf[3]; // A
            state[1] = buf[2]; // B
            state[2] = buf[7]; // C
            state[3] = buf[6]; // D
            state[4] = buf[1]; // E
            state[5] = buf[0]; // F
            state[6] = buf[5]; // G
            state[7] = buf[4]; // H
        }

        /// Compress one 64-byte block via SHA-NI. Returns the updated
        /// (ABEF, CDGH) state pair.
        ///
        /// # Safety
        ///
        /// `block_ptr` must point to at least 64 readable bytes. SHA-NI
        /// + SSE4.1 + SSSE3 must be available.
        #[target_feature(enable = "sha,sse4.1,ssse3")]
        #[allow(clippy::too_many_lines)]
        #[inline]
        unsafe fn compress_block_shani(
            mut state0: __m128i,
            mut state1: __m128i,
            block_ptr: *const u8,
            bswap: __m128i,
        ) -> (__m128i, __m128i) {
            let abef_save = state0;
            let cdgh_save = state1;

            // Load four 16-byte chunks and byte-swap each 32-bit word so
            // the message words are big-endian per FIPS 180-4.
            // SAFETY: caller guarantees 64 readable bytes.
            let mut msg0 =
                unsafe { _mm_shuffle_epi8(_mm_loadu_si128(block_ptr.cast::<__m128i>()), bswap) };
            let mut msg1 = unsafe {
                _mm_shuffle_epi8(_mm_loadu_si128(block_ptr.add(16).cast::<__m128i>()), bswap)
            };
            let mut msg2 = unsafe {
                _mm_shuffle_epi8(_mm_loadu_si128(block_ptr.add(32).cast::<__m128i>()), bswap)
            };
            let mut msg3 = unsafe {
                _mm_shuffle_epi8(_mm_loadu_si128(block_ptr.add(48).cast::<__m128i>()), bswap)
            };

            // K constants packed four per __m128i (low dword first).
            // `_mm_set_epi64x` is fine inside this `#[target_feature]`
            // function — no further unsafe needed.
            let kv = |hi: u64, lo: u64| -> __m128i { _mm_set_epi64x(hi as i64, lo as i64) };
            // K[0..4] = (0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5)
            let k0 = kv(0xe9b5_dba5_b5c0_fbcf, 0x7137_4491_428a_2f98);
            let k1 = kv(0xab1c_5ed5_923f_82a4, 0x59f1_11f1_3956_c25b);
            let k2 = kv(0x550c_7dc3_2431_85be, 0x1283_5b01_d807_aa98);
            let k3 = kv(0xc19b_f174_9bdc_06a7, 0x80de_b1fe_72be_5d74);
            let k4 = kv(0x240c_a1cc_0fc1_9dc6, 0xefbe_4786_e49b_69c1);
            let k5 = kv(0x76f9_88da_5cb0_a9dc, 0x4a74_84aa_2de9_2c6f);
            let k6 = kv(0xbf59_7fc7_b003_27c8, 0xa831_c66d_983e_5152);
            let k7 = kv(0x1429_2967_06ca_6351, 0xd5a7_9147_c6e0_0bf3);
            let k8 = kv(0x5338_0d13_4d2c_6dfc, 0x2e1b_2138_27b7_0a85);
            let k9 = kv(0x9272_2c85_81c2_c92e, 0x766a_0abb_650a_7354);
            let k10 = kv(0xc76c_51a3_c24b_8b70, 0xa81a_664b_a2bf_e8a1);
            let k11 = kv(0x106a_a070_f40e_3585, 0xd699_0624_d192_e819);
            let k12 = kv(0x34b0_bcb5_2748_774c, 0x1e37_6c08_19a4_c116);
            let k13 = kv(0x682e_6ff3_5b9c_ca4f, 0x4ed8_aa4a_391c_0cb3);
            let k14 = kv(0x8cc7_0208_84c8_7814, 0x78a5_636f_748f_82ee);
            let k15 = kv(0xc671_78f2_bef9_a3f7, 0xa450_6ceb_90be_fffa);

            // The round structure is identical for all 16 message slots,
            // but the schedule update on the last 12 slots differs from
            // the first 4. Pattern from Intel's SHA-NI reference.

            macro_rules! rounds_lo {
                ($msg:expr, $k:expr) => {{
                    let msg = _mm_add_epi32($msg, $k);
                    state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
                    let msg_shuf = _mm_shuffle_epi32::<0x0E>(msg);
                    state0 = _mm_sha256rnds2_epu32(state0, state1, msg_shuf);
                }};
            }

            // Rounds 0-3, no schedule update needed (just consume msg0).
            rounds_lo!(msg0, k0);
            // Rounds 4-7
            rounds_lo!(msg1, k1);
            msg0 = _mm_sha256msg1_epu32(msg0, msg1);
            // Rounds 8-11
            rounds_lo!(msg2, k2);
            msg1 = _mm_sha256msg1_epu32(msg1, msg2);
            // Rounds 12-15: start the cyclic msg-schedule update.
            rounds_lo!(msg3, k3);
            let mut tmp = _mm_alignr_epi8::<4>(msg3, msg2);
            msg0 = _mm_add_epi32(msg0, tmp);
            msg0 = _mm_sha256msg2_epu32(msg0, msg3);
            msg2 = _mm_sha256msg1_epu32(msg2, msg3);

            // Rounds 16-19
            rounds_lo!(msg0, k4);
            tmp = _mm_alignr_epi8::<4>(msg0, msg3);
            msg1 = _mm_add_epi32(msg1, tmp);
            msg1 = _mm_sha256msg2_epu32(msg1, msg0);
            msg3 = _mm_sha256msg1_epu32(msg3, msg0);

            // Rounds 20-23
            rounds_lo!(msg1, k5);
            tmp = _mm_alignr_epi8::<4>(msg1, msg0);
            msg2 = _mm_add_epi32(msg2, tmp);
            msg2 = _mm_sha256msg2_epu32(msg2, msg1);
            msg0 = _mm_sha256msg1_epu32(msg0, msg1);

            // Rounds 24-27
            rounds_lo!(msg2, k6);
            tmp = _mm_alignr_epi8::<4>(msg2, msg1);
            msg3 = _mm_add_epi32(msg3, tmp);
            msg3 = _mm_sha256msg2_epu32(msg3, msg2);
            msg1 = _mm_sha256msg1_epu32(msg1, msg2);

            // Rounds 28-31
            rounds_lo!(msg3, k7);
            tmp = _mm_alignr_epi8::<4>(msg3, msg2);
            msg0 = _mm_add_epi32(msg0, tmp);
            msg0 = _mm_sha256msg2_epu32(msg0, msg3);
            msg2 = _mm_sha256msg1_epu32(msg2, msg3);

            // Rounds 32-35
            rounds_lo!(msg0, k8);
            tmp = _mm_alignr_epi8::<4>(msg0, msg3);
            msg1 = _mm_add_epi32(msg1, tmp);
            msg1 = _mm_sha256msg2_epu32(msg1, msg0);
            msg3 = _mm_sha256msg1_epu32(msg3, msg0);

            // Rounds 36-39
            rounds_lo!(msg1, k9);
            tmp = _mm_alignr_epi8::<4>(msg1, msg0);
            msg2 = _mm_add_epi32(msg2, tmp);
            msg2 = _mm_sha256msg2_epu32(msg2, msg1);
            msg0 = _mm_sha256msg1_epu32(msg0, msg1);

            // Rounds 40-43
            rounds_lo!(msg2, k10);
            tmp = _mm_alignr_epi8::<4>(msg2, msg1);
            msg3 = _mm_add_epi32(msg3, tmp);
            msg3 = _mm_sha256msg2_epu32(msg3, msg2);
            msg1 = _mm_sha256msg1_epu32(msg1, msg2);

            // Rounds 44-47
            rounds_lo!(msg3, k11);
            tmp = _mm_alignr_epi8::<4>(msg3, msg2);
            msg0 = _mm_add_epi32(msg0, tmp);
            msg0 = _mm_sha256msg2_epu32(msg0, msg3);
            msg2 = _mm_sha256msg1_epu32(msg2, msg3);

            // Rounds 48-51
            rounds_lo!(msg0, k12);
            tmp = _mm_alignr_epi8::<4>(msg0, msg3);
            msg1 = _mm_add_epi32(msg1, tmp);
            msg1 = _mm_sha256msg2_epu32(msg1, msg0);
            msg3 = _mm_sha256msg1_epu32(msg3, msg0);

            // Rounds 52-55
            rounds_lo!(msg1, k13);
            tmp = _mm_alignr_epi8::<4>(msg1, msg0);
            msg2 = _mm_add_epi32(msg2, tmp);
            msg2 = _mm_sha256msg2_epu32(msg2, msg1);

            // Rounds 56-59
            rounds_lo!(msg2, k14);
            tmp = _mm_alignr_epi8::<4>(msg2, msg1);
            msg3 = _mm_add_epi32(msg3, tmp);
            msg3 = _mm_sha256msg2_epu32(msg3, msg2);

            // Rounds 60-63
            rounds_lo!(msg3, k15);

            // Combine with starting state (state-add).
            state0 = _mm_add_epi32(state0, abef_save);
            state1 = _mm_add_epi32(state1, cdgh_save);

            (state0, state1)
        }
    }

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
    #[cfg(target_arch = "aarch64")]
    pub mod aarch64_sha2 {
        use super::super::{BLOCK_BYTES, DIGEST_BYTES, H0, K};
        use super::scalar;

        use core::arch::aarch64::{
            uint32x4_t, vaddq_u32, vld1q_u8, vld1q_u32, vreinterpretq_u32_u8, vrev32q_u8,
            vsha256h2q_u32, vsha256hq_u32, vsha256su0q_u32, vsha256su1q_u32, vst1q_u32,
        };

        /// Returns true when AArch64 FEAT_SHA2 is available at runtime.
        ///
        /// **Currently force-disabled pending real-hardware CI
        /// verification.** A previous incarnation of
        /// Returns true when AArch64 FEAT_SHA2 is available at runtime.
        ///
        /// `compress_block_sha2` is a literal port of the canonical
        /// noloader / SHA-Intrinsics reference (also the pattern used
        /// by OpenSSL `sha256-armv8.pl`, mbedTLS, and Apple's
        /// CommonCrypto). The previous hand-rolled macro version was
        /// bit-divergent from scalar on real ARM silicon (Linux
        /// Cobalt-100 / Apple M1 / Windows Cobalt-100, run
        /// 25241406257); QEMU user-mode emulation mirrored the bug,
        /// masking it locally. Re-enabled in 240aab1 / canonical-port
        /// merge after CI verification on real hardware.
        #[cfg(feature = "std")]
        #[must_use]
        pub fn is_available() -> bool {
            std::arch::is_aarch64_feature_detected!("sha2")
        }

        #[cfg(not(feature = "std"))]
        #[must_use]
        pub const fn is_available() -> bool {
            false
        }

        /// Computes SHA-256 of `bytes` using FEAT_SHA2 for full blocks
        /// and the scalar reference for the padding tail (so output is
        /// bit-exact across backends).
        ///
        /// # Safety
        ///
        /// Caller must ensure FEAT_SHA2 (the `sha2` feature) is available.
        #[target_feature(enable = "sha2")]
        #[must_use]
        pub unsafe fn sha256(bytes: &[u8]) -> [u8; DIGEST_BYTES] {
            let mut state = H0;

            let full_blocks = bytes.len() / BLOCK_BYTES;
            let tail_start = full_blocks * BLOCK_BYTES;

            if full_blocks > 0 {
                // SAFETY: caller guarantees FEAT_SHA2; we read full_blocks*64
                // bytes and write 8 u32s into `state`.
                unsafe {
                    compress_blocks(&mut state, bytes.as_ptr(), full_blocks);
                }
            }

            // Padding via the scalar compress for bit-exact output.
            let tail = &bytes[tail_start..];
            let bit_len = (bytes.len() as u64).wrapping_mul(8);
            let mut last = [0_u8; BLOCK_BYTES * 2];
            last[..tail.len()].copy_from_slice(tail);
            last[tail.len()] = 0x80;

            let total = if tail.len() + 1 + 8 <= BLOCK_BYTES {
                BLOCK_BYTES
            } else {
                BLOCK_BYTES * 2
            };
            let length_off = total - 8;
            last[length_off..total].copy_from_slice(&bit_len.to_be_bytes());

            let pad_block: &[u8; BLOCK_BYTES] = (&last[..BLOCK_BYTES])
                .try_into()
                .expect("BLOCK_BYTES slice");
            scalar::compress(&mut state, pad_block);
            if total == BLOCK_BYTES * 2 {
                let pad2: &[u8; BLOCK_BYTES] = (&last[BLOCK_BYTES..total])
                    .try_into()
                    .expect("BLOCK_BYTES slice");
                scalar::compress(&mut state, pad2);
            }

            let mut digest = [0_u8; DIGEST_BYTES];
            for (i, word) in state.iter().enumerate() {
                digest[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
            }
            digest
        }

        /// Compress `n_blocks` consecutive 64-byte blocks read from `block_ptr`
        /// into the scalar `state`. Loads the (ABCD, EFGH) vector pair once,
        /// runs the FEAT_SHA2 block compressor `n_blocks` times, then writes
        /// the scalar state back. Used by the streaming
        /// [`super::super::Hasher`] so a 4 KiB chunk pays the `vld1q_u32` /
        /// `vst1q_u32` cost once instead of 64 times.
        ///
        /// # Safety
        ///
        /// `block_ptr` must point to at least `n_blocks * 64` readable bytes.
        /// FEAT_SHA2 must be available (enforced by the `#[target_feature]`
        /// gate plus the caller's runtime check).
        #[target_feature(enable = "sha2")]
        pub(crate) unsafe fn compress_blocks(
            state: &mut [u32; 8],
            block_ptr: *const u8,
            n_blocks: usize,
        ) {
            if n_blocks == 0 {
                return;
            }
            // SAFETY: state is a valid 8-dword array.
            let mut state_abcd = unsafe { vld1q_u32(state.as_ptr()) };
            let mut state_efgh = unsafe { vld1q_u32(state.as_ptr().add(4)) };

            for block_index in 0..n_blocks {
                // SAFETY: caller guarantees block_ptr+i*64+63 is readable.
                let (na, ne) = unsafe {
                    compress_block_sha2(
                        state_abcd,
                        state_efgh,
                        block_ptr.add(block_index * BLOCK_BYTES),
                    )
                };
                state_abcd = na;
                state_efgh = ne;
            }

            // SAFETY: state is a valid 8-dword writable destination.
            unsafe {
                vst1q_u32(state.as_mut_ptr(), state_abcd);
                vst1q_u32(state.as_mut_ptr().add(4), state_efgh);
            }
        }

        /// Compress one 64-byte block using FEAT_SHA2.
        ///
        /// Direct port of the noloader / SHA-Intrinsics canonical
        /// reference (`sha256-arm.c`). Each round-burst:
        /// 1. starts the partial schedule update for `MSGn`,
        /// 2. saves `STATE0` into `tmp2` (for the h2 input),
        /// 3. computes the *next* burst's `tmp` (pipelined),
        /// 4. runs `vsha256hq_u32` and `vsha256h2q_u32` on the
        ///    *current* burst's `tmp`,
        /// 5. finishes the schedule update for `MSGn`.
        ///
        /// The very first `tmp0` is computed before the burst chain.
        /// `tmp0` and `tmp1` alternate roles so bursts always consume
        /// the value the previous burst produced. Last 16 rounds drop
        /// the schedule update (W[64+] is never needed).
        ///
        /// # Safety
        ///
        /// `block_ptr` must point to at least 64 readable bytes.
        /// FEAT_SHA2 must be available (enforced by the
        /// `#[target_feature]` gate plus the caller's runtime check).
        #[target_feature(enable = "sha2")]
        #[allow(clippy::too_many_lines)]
        #[inline]
        unsafe fn compress_block_sha2(
            mut state0: uint32x4_t,
            mut state1: uint32x4_t,
            block_ptr: *const u8,
        ) -> (uint32x4_t, uint32x4_t) {
            // Save starting state for the per-block add at the end
            // (FIPS 180-4 §6.2.2 step 4: H[i] = a + H[i-1] etc.).
            let abef_save = state0;
            let cdgh_save = state1;

            // Load four 16-byte chunks and byte-swap each 32-bit word
            // so the message words are big-endian per FIPS 180-4. The
            // canonical noloader source loads as u32 then `vrev32q_u8`s;
            // loading as u8 then reversing within each 32-bit lane is
            // equivalent on little-endian AArch64 (both produce the same
            // big-endian 32-bit words in each lane).
            // SAFETY: caller guarantees 64 readable bytes at block_ptr.
            let mut msg0 = unsafe { vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block_ptr))) };
            let mut msg1 = unsafe { vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block_ptr.add(16)))) };
            let mut msg2 = unsafe { vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block_ptr.add(32)))) };
            let mut msg3 = unsafe { vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block_ptr.add(48)))) };

            // K-vector loader. K is `[u32; 64]` so `add(i)` advances by
            // one `u32`; `K[i*4..i*4+4]` is the i-th 4-word block.
            // SAFETY: i is in 0..16, so i*4..i*4+4 fits in K.
            let kv = |i: usize| unsafe { vld1q_u32(K.as_ptr().add(i * 4)) };

            // Pre-compute the first burst's tmp (rounds 0-3 input).
            let mut tmp0 = vaddq_u32(msg0, kv(0));
            let mut tmp1;
            let mut tmp2;

            // Rounds 0-3 — produces W[16..20] in msg0; tmp1 staged for
            // rounds 4-7.
            msg0 = vsha256su0q_u32(msg0, msg1);
            tmp2 = state0;
            tmp1 = vaddq_u32(msg1, kv(1));
            state0 = vsha256hq_u32(state0, state1, tmp0);
            state1 = vsha256h2q_u32(state1, tmp2, tmp0);
            msg0 = vsha256su1q_u32(msg0, msg2, msg3);

            // Rounds 4-7 — produces W[20..24] in msg1; tmp0 staged for
            // rounds 8-11.
            msg1 = vsha256su0q_u32(msg1, msg2);
            tmp2 = state0;
            tmp0 = vaddq_u32(msg2, kv(2));
            state0 = vsha256hq_u32(state0, state1, tmp1);
            state1 = vsha256h2q_u32(state1, tmp2, tmp1);
            msg1 = vsha256su1q_u32(msg1, msg3, msg0);

            // Rounds 8-11.
            msg2 = vsha256su0q_u32(msg2, msg3);
            tmp2 = state0;
            tmp1 = vaddq_u32(msg3, kv(3));
            state0 = vsha256hq_u32(state0, state1, tmp0);
            state1 = vsha256h2q_u32(state1, tmp2, tmp0);
            msg2 = vsha256su1q_u32(msg2, msg0, msg1);

            // Rounds 12-15.
            msg3 = vsha256su0q_u32(msg3, msg0);
            tmp2 = state0;
            tmp0 = vaddq_u32(msg0, kv(4));
            state0 = vsha256hq_u32(state0, state1, tmp1);
            state1 = vsha256h2q_u32(state1, tmp2, tmp1);
            msg3 = vsha256su1q_u32(msg3, msg1, msg2);

            // Rounds 16-19.
            msg0 = vsha256su0q_u32(msg0, msg1);
            tmp2 = state0;
            tmp1 = vaddq_u32(msg1, kv(5));
            state0 = vsha256hq_u32(state0, state1, tmp0);
            state1 = vsha256h2q_u32(state1, tmp2, tmp0);
            msg0 = vsha256su1q_u32(msg0, msg2, msg3);

            // Rounds 20-23.
            msg1 = vsha256su0q_u32(msg1, msg2);
            tmp2 = state0;
            tmp0 = vaddq_u32(msg2, kv(6));
            state0 = vsha256hq_u32(state0, state1, tmp1);
            state1 = vsha256h2q_u32(state1, tmp2, tmp1);
            msg1 = vsha256su1q_u32(msg1, msg3, msg0);

            // Rounds 24-27.
            msg2 = vsha256su0q_u32(msg2, msg3);
            tmp2 = state0;
            tmp1 = vaddq_u32(msg3, kv(7));
            state0 = vsha256hq_u32(state0, state1, tmp0);
            state1 = vsha256h2q_u32(state1, tmp2, tmp0);
            msg2 = vsha256su1q_u32(msg2, msg0, msg1);

            // Rounds 28-31.
            msg3 = vsha256su0q_u32(msg3, msg0);
            tmp2 = state0;
            tmp0 = vaddq_u32(msg0, kv(8));
            state0 = vsha256hq_u32(state0, state1, tmp1);
            state1 = vsha256h2q_u32(state1, tmp2, tmp1);
            msg3 = vsha256su1q_u32(msg3, msg1, msg2);

            // Rounds 32-35.
            msg0 = vsha256su0q_u32(msg0, msg1);
            tmp2 = state0;
            tmp1 = vaddq_u32(msg1, kv(9));
            state0 = vsha256hq_u32(state0, state1, tmp0);
            state1 = vsha256h2q_u32(state1, tmp2, tmp0);
            msg0 = vsha256su1q_u32(msg0, msg2, msg3);

            // Rounds 36-39.
            msg1 = vsha256su0q_u32(msg1, msg2);
            tmp2 = state0;
            tmp0 = vaddq_u32(msg2, kv(10));
            state0 = vsha256hq_u32(state0, state1, tmp1);
            state1 = vsha256h2q_u32(state1, tmp2, tmp1);
            msg1 = vsha256su1q_u32(msg1, msg3, msg0);

            // Rounds 40-43.
            msg2 = vsha256su0q_u32(msg2, msg3);
            tmp2 = state0;
            tmp1 = vaddq_u32(msg3, kv(11));
            state0 = vsha256hq_u32(state0, state1, tmp0);
            state1 = vsha256h2q_u32(state1, tmp2, tmp0);
            msg2 = vsha256su1q_u32(msg2, msg0, msg1);

            // Rounds 44-47.
            msg3 = vsha256su0q_u32(msg3, msg0);
            tmp2 = state0;
            tmp0 = vaddq_u32(msg0, kv(12));
            state0 = vsha256hq_u32(state0, state1, tmp1);
            state1 = vsha256h2q_u32(state1, tmp2, tmp1);
            msg3 = vsha256su1q_u32(msg3, msg1, msg2);

            // Rounds 48-51 — schedule update no longer needed; tmp1
            // staged with msg1+K[0x34] for rounds 52-55.
            tmp2 = state0;
            tmp1 = vaddq_u32(msg1, kv(13));
            state0 = vsha256hq_u32(state0, state1, tmp0);
            state1 = vsha256h2q_u32(state1, tmp2, tmp0);

            // Rounds 52-55.
            tmp2 = state0;
            tmp0 = vaddq_u32(msg2, kv(14));
            state0 = vsha256hq_u32(state0, state1, tmp1);
            state1 = vsha256h2q_u32(state1, tmp2, tmp1);

            // Rounds 56-59.
            tmp2 = state0;
            tmp1 = vaddq_u32(msg3, kv(15));
            state0 = vsha256hq_u32(state0, state1, tmp0);
            state1 = vsha256h2q_u32(state1, tmp2, tmp0);

            // Rounds 60-63 — last burst, no further tmp staging needed.
            tmp2 = state0;
            state0 = vsha256hq_u32(state0, state1, tmp1);
            state1 = vsha256h2q_u32(state1, tmp2, tmp1);

            // Combine with starting state (FIPS 180-4 step 4).
            state0 = vaddq_u32(state0, abef_save);
            state1 = vaddq_u32(state1, cdgh_save);

            (state0, state1)
        }
    }
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
    /// hazard. Use [`Self::try_update`] in callers that want a
    /// `Result` instead of a panic for adversarial input lengths.
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
    use super::{Hasher, HasherBackend, kernels, sha256};
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
                sha256(case),
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
        assert_eq!(h.finalize(), sha256(b""));
    }

    #[test]
    fn hasher_single_call_matches_one_shot() {
        let payload = b"the quick brown fox jumps over the lazy dog";
        let mut h = Hasher::new();
        h.update(payload);
        assert_eq!(h.finalize(), sha256(payload));
    }

    #[test]
    fn hasher_nist_abc_matches_one_shot() {
        let mut h = Hasher::new();
        h.update(b"abc");
        assert_eq!(
            hex(&h.finalize()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn hasher_nist_two_block_matches_one_shot() {
        let mut h = Hasher::new();
        h.update(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
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
        let expected = sha256(&payload);
        for &chunk in &[1_usize, 17, 63, 64, 65, 127, 128, 1024, 4096] {
            let mut h = Hasher::new();
            for block in payload.chunks(chunk) {
                h.update(block);
            }
            assert_eq!(h.finalize(), expected, "mismatch for chunk={chunk}");
        }
    }

    #[test]
    fn hasher_empty_updates_are_no_ops() {
        let payload = b"hello, world";
        let expected = sha256(payload);
        let mut h = Hasher::new();
        h.update(b"");
        h.update(payload);
        h.update(b"");
        assert_eq!(h.finalize(), expected);
    }

    #[test]
    fn hasher_finalize_reset_matches_finalize_then_new() {
        let payload = b"reset me and try again";
        let mut h = Hasher::new();
        h.update(payload);
        let d1 = h.finalize_reset();
        assert_eq!(d1, sha256(payload));
        // After reset, the hasher should produce the empty digest.
        assert_eq!(h.clone().finalize(), sha256(b""));
        h.update(payload);
        assert_eq!(h.finalize(), d1);
    }

    #[test]
    fn hasher_reset_clears_state() {
        let mut h = Hasher::new();
        h.update(b"garbage");
        h.reset();
        h.update(b"abc");
        assert_eq!(h.finalize(), sha256(b"abc"));
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
                    h.update(block);
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
            h.update(block);
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
                    h.update(block);
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

    #[test]
    #[should_panic(expected = "SHA-256 stream length exceeded 2^64 bits")]
    fn hasher_update_panics_on_overflow() {
        // Mirrors the try_update test but exercises the panicking
        // wrapper so the documented behavior is pinned.
        let mut h = Hasher::new();
        h.total_bits = u64::MAX - 7;
        h.update(b"x");
    }
}
