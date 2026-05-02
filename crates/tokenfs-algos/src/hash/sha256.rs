//! SHA-256 (FIPS 180-4) with portable scalar, x86 SHA-NI, and AArch64
//! FEAT_SHA2 backends.
//!
//! The public [`sha256`] function picks the fastest available backend at
//! runtime. Pinned reference paths live under [`kernels::scalar`] for
//! reproducibility; pinned hardware paths live under [`kernels::x86_shani`]
//! and [`kernels::aarch64_sha2`] respectively.
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

            // SAFETY: `state` has 8 dwords; all loads use 4-dword vectors.
            let (mut state0, mut state1) = unsafe { state_from_words(&state) };

            // Byte-swap mask for converting little-endian xmm dword loads
            // into big-endian message words. `_mm_set_epi64x` is safe when
            // its target features are active (we're already inside a
            // `#[target_feature]` function).
            let bswap = _mm_set_epi64x(
                0x0c0d_0e0f_0809_0a0b_u64 as i64,
                0x0405_0607_0001_0203_u64 as i64,
            );

            for block_index in 0..full_blocks {
                let off = block_index * BLOCK_BYTES;
                // SAFETY: bounds checked above; SHA-NI feature gate active.
                let (s0, s1) =
                    unsafe { compress_block_shani(state0, state1, bytes.as_ptr().add(off), bswap) };
                state0 = s0;
                state1 = s1;
            }

            // Convert the SHA-NI state back to scalar order, then run the
            // scalar compress on the padding block(s). This guarantees
            // bit-exact parity with the scalar kernel without re-implementing
            // padding inside the SHA-NI path.
            // SAFETY: SSE2 enabled.
            unsafe { state_to_words(&mut state, state0, state1) };

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
    #[cfg(target_arch = "aarch64")]
    pub mod aarch64_sha2 {
        use super::super::{BLOCK_BYTES, DIGEST_BYTES, H0, K};
        use super::scalar;

        use core::arch::aarch64::{
            uint32x4_t, vaddq_u32, vld1q_u8, vld1q_u32, vreinterpretq_u32_u8, vrev32q_u8,
            vsha256h2q_u32, vsha256hq_u32, vsha256su0q_u32, vsha256su1q_u32, vst1q_u32,
        };

        /// Returns true when AArch64 FEAT_SHA2 is available at runtime.
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

            // SAFETY: state has 8 dwords; we load two 4-dword vectors.
            let mut state_abcd = unsafe { vld1q_u32(state.as_ptr()) };
            let mut state_efgh = unsafe { vld1q_u32(state.as_ptr().add(4)) };

            for block_index in 0..full_blocks {
                let off = block_index * BLOCK_BYTES;
                // SAFETY: bounds checked above; FEAT_SHA2 enabled.
                let (na, ne) =
                    unsafe { compress_block_sha2(state_abcd, state_efgh, bytes.as_ptr().add(off)) };
                state_abcd = na;
                state_efgh = ne;
            }

            // SAFETY: 16-byte stores into 8-element u32 array.
            unsafe {
                vst1q_u32(state.as_mut_ptr(), state_abcd);
                vst1q_u32(state.as_mut_ptr().add(4), state_efgh);
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

        /// Compress one 64-byte block using FEAT_SHA2.
        ///
        /// # Safety
        ///
        /// `block_ptr` must point to at least 64 readable bytes. FEAT_SHA2
        /// must be available.
        #[target_feature(enable = "sha2")]
        #[allow(clippy::too_many_lines)]
        #[inline]
        unsafe fn compress_block_sha2(
            mut state_abcd: uint32x4_t,
            mut state_efgh: uint32x4_t,
            block_ptr: *const u8,
        ) -> (uint32x4_t, uint32x4_t) {
            let abcd_save = state_abcd;
            let efgh_save = state_efgh;

            // Load four 16-byte chunks and byte-swap each 32-bit word so
            // the message words are big-endian per FIPS 180-4.
            // SAFETY: caller guarantees 64 readable bytes.
            let mut msg0 = unsafe { vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block_ptr))) };
            let mut msg1 = unsafe { vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block_ptr.add(16)))) };
            let mut msg2 = unsafe { vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block_ptr.add(32)))) };
            let mut msg3 = unsafe { vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block_ptr.add(48)))) };

            // SAFETY: K has 64 dwords; we load 16 vectors of 4.
            let kv = |i: usize| unsafe { vld1q_u32(K.as_ptr().add(i * 4)) };
            let k0 = kv(0);
            let k1 = kv(1);
            let k2 = kv(2);
            let k3 = kv(3);
            let k4 = kv(4);
            let k5 = kv(5);
            let k6 = kv(6);
            let k7 = kv(7);
            let k8 = kv(8);
            let k9 = kv(9);
            let k10 = kv(10);
            let k11 = kv(11);
            let k12 = kv(12);
            let k13 = kv(13);
            let k14 = kv(14);
            let k15 = kv(15);

            // 4-round bursts. The standard FEAT_SHA2 pattern is:
            //   tmp  = vaddq_u32(msg, k);
            //   prev = state_abcd;
            //   state_abcd = vsha256hq_u32(state_abcd, state_efgh, tmp);
            //   state_efgh = vsha256h2q_u32(state_efgh, prev, tmp);
            // Schedule updates start after the first 16 rounds:
            //   msg0 = vsha256su0q_u32(msg0, msg1);
            //   msg0 = vsha256su1q_u32(msg0, msg2, msg3);

            macro_rules! round4 {
                ($msg:expr, $k:expr) => {{
                    let tmp = vaddq_u32($msg, $k);
                    let prev = state_abcd;
                    state_abcd = vsha256hq_u32(state_abcd, state_efgh, tmp);
                    state_efgh = vsha256h2q_u32(state_efgh, prev, tmp);
                }};
            }

            // Rounds 0-15: schedule starts up.
            round4!(msg0, k0);
            round4!(msg1, k1);
            msg0 = vsha256su0q_u32(msg0, msg1);
            round4!(msg2, k2);
            msg1 = vsha256su0q_u32(msg1, msg2);
            round4!(msg3, k3);
            msg2 = vsha256su0q_u32(msg2, msg3);
            msg0 = vsha256su1q_u32(msg0, msg2, msg3);

            // Rounds 16-31
            round4!(msg0, k4);
            msg3 = vsha256su0q_u32(msg3, msg0);
            msg1 = vsha256su1q_u32(msg1, msg3, msg0);
            round4!(msg1, k5);
            msg0 = vsha256su0q_u32(msg0, msg1);
            msg2 = vsha256su1q_u32(msg2, msg0, msg1);
            round4!(msg2, k6);
            msg1 = vsha256su0q_u32(msg1, msg2);
            msg3 = vsha256su1q_u32(msg3, msg1, msg2);
            round4!(msg3, k7);
            msg2 = vsha256su0q_u32(msg2, msg3);
            msg0 = vsha256su1q_u32(msg0, msg2, msg3);

            // Rounds 32-47
            round4!(msg0, k8);
            msg3 = vsha256su0q_u32(msg3, msg0);
            msg1 = vsha256su1q_u32(msg1, msg3, msg0);
            round4!(msg1, k9);
            msg0 = vsha256su0q_u32(msg0, msg1);
            msg2 = vsha256su1q_u32(msg2, msg0, msg1);
            round4!(msg2, k10);
            msg1 = vsha256su0q_u32(msg1, msg2);
            msg3 = vsha256su1q_u32(msg3, msg1, msg2);
            round4!(msg3, k11);
            msg2 = vsha256su0q_u32(msg2, msg3);
            msg0 = vsha256su1q_u32(msg0, msg2, msg3);

            // Rounds 48-63: no further schedule updates needed for the
            // last 12 rounds (the values are already final).
            round4!(msg0, k12);
            msg3 = vsha256su0q_u32(msg3, msg0);
            msg1 = vsha256su1q_u32(msg1, msg3, msg0);
            round4!(msg1, k13);
            round4!(msg2, k14);
            round4!(msg3, k15);

            state_abcd = vaddq_u32(state_abcd, abcd_save);
            state_efgh = vaddq_u32(state_efgh, efgh_save);

            (state_abcd, state_efgh)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{kernels, sha256};

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

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
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

    #[cfg(target_arch = "aarch64")]
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
}
