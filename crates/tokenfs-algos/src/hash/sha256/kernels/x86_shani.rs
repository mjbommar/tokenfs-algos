use super::super::{BLOCK_BYTES, DIGEST_BYTES, H0};
use super::scalar;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m128i, _mm_add_epi32, _mm_alignr_epi8, _mm_blend_epi16, _mm_loadu_si128, _mm_set_epi64x,
    _mm_sha256msg1_epu32, _mm_sha256msg2_epu32, _mm_sha256rnds2_epu32, _mm_shuffle_epi8,
    _mm_shuffle_epi32, _mm_storeu_si128,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_alignr_epi8, _mm_blend_epi16, _mm_loadu_si128, _mm_set_epi64x,
    _mm_sha256msg1_epu32, _mm_sha256msg2_epu32, _mm_sha256rnds2_epu32, _mm_shuffle_epi8,
    _mm_shuffle_epi32, _mm_storeu_si128,
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
pub(crate) unsafe fn compress_blocks(state: &mut [u32; 8], block_ptr: *const u8, n_blocks: usize) {
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
    let mut msg0 = unsafe { _mm_shuffle_epi8(_mm_loadu_si128(block_ptr.cast::<__m128i>()), bswap) };
    let mut msg1 =
        unsafe { _mm_shuffle_epi8(_mm_loadu_si128(block_ptr.add(16).cast::<__m128i>()), bswap) };
    let mut msg2 =
        unsafe { _mm_shuffle_epi8(_mm_loadu_si128(block_ptr.add(32).cast::<__m128i>()), bswap) };
    let mut msg3 =
        unsafe { _mm_shuffle_epi8(_mm_loadu_si128(block_ptr.add(48).cast::<__m128i>()), bswap) };

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
