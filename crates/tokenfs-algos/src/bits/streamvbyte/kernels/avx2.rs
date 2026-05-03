use super::super::{GROUP, streamvbyte_control_len};
use super::scalar;
use super::tables::{length_table, shuffle_table};

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m128i, __m256i, _mm_loadu_si128, _mm_shuffle_epi8, _mm256_inserti128_si256,
    _mm256_storeu_si256,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_loadu_si128, _mm_shuffle_epi8, _mm256_inserti128_si256,
    _mm256_storeu_si256,
};

/// Returns true when AVX2 is available at runtime.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("avx2")
}

/// Returns true when AVX2 is available at runtime.
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// AVX2 dual-pumped PSHUFB decode.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn decode_u32(control: &[u8], data: &[u8], n: usize, out: &mut [u32]) -> usize {
    let ctrl_needed = streamvbyte_control_len(n);
    assert!(
        control.len() >= ctrl_needed,
        "control too small: {} < {}",
        control.len(),
        ctrl_needed
    );
    assert!(
        out.len() >= n,
        "decode output buffer too small: {} < {}",
        out.len(),
        n
    );
    // SAFETY: AVX2 availability and buffer-length preconditions
    // are both established above.
    unsafe { decode_u32_unchecked(control, data, n, out) }
}

/// AVX2 dual-pumped PSHUFB decode without bounds-checking
/// asserts.
///
/// # Safety
///
/// Caller must ensure the current CPU supports AVX2,
/// `control.len() >= streamvbyte_control_len(n)`,
/// `out.len() >= n`, and `data.len()` covers the implied byte
/// sum from the control stream.
#[target_feature(enable = "avx2")]
pub unsafe fn decode_u32_unchecked(
    control: &[u8],
    data: &[u8],
    n: usize,
    out: &mut [u32],
) -> usize {
    let shuf = shuffle_table();
    let lens = length_table();

    let full_groups = n / GROUP;
    let mut data_pos = 0_usize;
    let mut g = 0_usize;

    // Dual-pumped: process two control bytes per iteration.
    // Each half needs 16 bytes of safe data; check both before
    // entering the body.
    while g + 2 <= full_groups {
        let c0 = control[g] as usize;
        let c1 = control[g + 1] as usize;
        let len0 = lens[c0] as usize;
        let len1 = lens[c1] as usize;

        // Bounds check both halves against `data` before issuing
        // the unaligned 16-byte loads.
        if data_pos + 16 > data.len() || data_pos + len0 + 16 > data.len() {
            break;
        }

        // SAFETY: bounds checked above; AVX2 (which implies SSSE3)
        // enabled on the enclosing fn; the two output stores fall
        // inside `out[g*4 .. g*4 + 8]`, in-bounds because
        // `g + 2 <= full_groups`, so `g*4 + 8 <= n <= out.len()`.
        unsafe {
            let v0 = _mm_loadu_si128(data.as_ptr().add(data_pos).cast::<__m128i>());
            let s0 = _mm_loadu_si128(shuf[c0].as_ptr().cast::<__m128i>());
            let r0 = _mm_shuffle_epi8(v0, s0);

            let v1 = _mm_loadu_si128(data.as_ptr().add(data_pos + len0).cast::<__m128i>());
            let s1 = _mm_loadu_si128(shuf[c1].as_ptr().cast::<__m128i>());
            let r1 = _mm_shuffle_epi8(v1, s1);

            // Compose two __m128i into one __m256i and store both
            // 16-byte halves in a single 32-byte unaligned store.
            let lo = _mm256_inserti128_si256::<0>(core::mem::zeroed::<__m256i>(), r0);
            let combined = _mm256_inserti128_si256::<1>(lo, r1);
            _mm256_storeu_si256(out.as_mut_ptr().add(g * GROUP).cast::<__m256i>(), combined);
        }

        data_pos += len0 + len1;
        g += 2;
    }

    // Single-group SSSE3 path for the residual full group when
    // `full_groups` is odd.
    while g < full_groups {
        let c = control[g] as usize;
        let len = lens[c] as usize;
        if data_pos + 16 > data.len() {
            break;
        }
        // SAFETY: bounds checked above; AVX2 implies SSSE3 so
        // PSHUFB is available; output lane indexing matches the
        // SSSE3 kernel.
        unsafe {
            let v = _mm_loadu_si128(data.as_ptr().add(data_pos).cast::<__m128i>());
            let s = _mm_loadu_si128(shuf[c].as_ptr().cast::<__m128i>());
            let r = _mm_shuffle_epi8(v, s);
            let dst = out.as_mut_ptr().add(g * GROUP).cast::<__m128i>();
            core::arch::x86_64::_mm_storeu_si128(dst, r);
        }
        data_pos += len;
        g += 1;
    }

    // Scalar tail: residual full groups that didn't have 16
    // bytes of data slack, plus any partial trailing group.
    let written = g * GROUP;
    if written < n {
        // SAFETY: caller upholds the buffer-length preconditions
        // on `control`, `data`, and `out`; the slice subranges
        // share their parents' validity.
        data_pos += unsafe {
            scalar::decode_u32_unchecked(
                &control[g..],
                &data[data_pos..],
                n - written,
                &mut out[written..],
            )
        };
    }

    data_pos
}
