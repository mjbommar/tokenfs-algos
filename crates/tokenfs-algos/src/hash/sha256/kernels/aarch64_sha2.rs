use super::super::{BLOCK_BYTES, DIGEST_BYTES, H0, K};
use super::scalar;

use core::arch::aarch64::{
    uint32x4_t, vaddq_u32, vld1q_u8, vld1q_u32, vreinterpretq_u32_u8, vrev32q_u8, vsha256h2q_u32,
    vsha256hq_u32, vsha256su0q_u32, vsha256su1q_u32, vst1q_u32,
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
pub(crate) unsafe fn compress_blocks(state: &mut [u32; 8], block_ptr: *const u8, n_blocks: usize) {
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
