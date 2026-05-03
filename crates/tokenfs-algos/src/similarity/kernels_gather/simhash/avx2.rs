use super::{BITS, Table};

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m128i, __m256i, _mm_loadl_epi64, _mm256_add_epi32, _mm256_cvtepi8_epi32, _mm256_loadu_si256,
    _mm256_storeu_si256,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_loadl_epi64, _mm256_add_epi32, _mm256_cvtepi8_epi32, _mm256_loadu_si256,
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

/// AVX2 implementation of the per-byte SimHash accumulator
/// update.
///
/// For each input byte we load the 64-entry contribution row,
/// sign-extend each 8-byte slice from i8 to i32 (so 64 i8
/// contributions become 8 × `__m256i` of 8 × i32 each), then
/// add into 8 register-resident accumulators.
///
/// # Safety
///
/// Caller must ensure AVX2 is available at runtime.
#[target_feature(enable = "avx2")]
pub unsafe fn update_accumulator(bytes: &[u8], table: &Table, acc: &mut [i32; BITS]) {
    // Load all 8 accumulator lanes (8 × __m256i of 8 × i32 = 64 lanes).
    // SAFETY: `acc` is 256 bytes (64 × i32); AVX2 enabled.
    let mut a0 = unsafe { _mm256_loadu_si256(acc.as_ptr().cast::<__m256i>()) };
    let mut a1 = unsafe { _mm256_loadu_si256(acc.as_ptr().add(8).cast::<__m256i>()) };
    let mut a2 = unsafe { _mm256_loadu_si256(acc.as_ptr().add(16).cast::<__m256i>()) };
    let mut a3 = unsafe { _mm256_loadu_si256(acc.as_ptr().add(24).cast::<__m256i>()) };
    let mut a4 = unsafe { _mm256_loadu_si256(acc.as_ptr().add(32).cast::<__m256i>()) };
    let mut a5 = unsafe { _mm256_loadu_si256(acc.as_ptr().add(40).cast::<__m256i>()) };
    let mut a6 = unsafe { _mm256_loadu_si256(acc.as_ptr().add(48).cast::<__m256i>()) };
    let mut a7 = unsafe { _mm256_loadu_si256(acc.as_ptr().add(56).cast::<__m256i>()) };

    for &b in bytes {
        // SAFETY: row is 64 bytes inside `Table`; reads stay in-bounds.
        let row_base = unsafe { table.as_ptr().add(b as usize) }.cast::<u8>();

        // Sign-extend 8 i8 lanes per gather → __m256i of 8 × i32.
        // SAFETY: row_base..row_base+64 is inside the table.
        let s0 = unsafe { _mm256_cvtepi8_epi32(load64(row_base)) };
        let s1 = unsafe { _mm256_cvtepi8_epi32(load64(row_base.add(8))) };
        let s2 = unsafe { _mm256_cvtepi8_epi32(load64(row_base.add(16))) };
        let s3 = unsafe { _mm256_cvtepi8_epi32(load64(row_base.add(24))) };
        let s4 = unsafe { _mm256_cvtepi8_epi32(load64(row_base.add(32))) };
        let s5 = unsafe { _mm256_cvtepi8_epi32(load64(row_base.add(40))) };
        let s6 = unsafe { _mm256_cvtepi8_epi32(load64(row_base.add(48))) };
        let s7 = unsafe { _mm256_cvtepi8_epi32(load64(row_base.add(56))) };

        a0 = _mm256_add_epi32(a0, s0);
        a1 = _mm256_add_epi32(a1, s1);
        a2 = _mm256_add_epi32(a2, s2);
        a3 = _mm256_add_epi32(a3, s3);
        a4 = _mm256_add_epi32(a4, s4);
        a5 = _mm256_add_epi32(a5, s5);
        a6 = _mm256_add_epi32(a6, s6);
        a7 = _mm256_add_epi32(a7, s7);
    }

    // SAFETY: 256 writable bytes; AVX2 enabled.
    unsafe { _mm256_storeu_si256(acc.as_mut_ptr().cast::<__m256i>(), a0) };
    unsafe { _mm256_storeu_si256(acc.as_mut_ptr().add(8).cast::<__m256i>(), a1) };
    unsafe { _mm256_storeu_si256(acc.as_mut_ptr().add(16).cast::<__m256i>(), a2) };
    unsafe { _mm256_storeu_si256(acc.as_mut_ptr().add(24).cast::<__m256i>(), a3) };
    unsafe { _mm256_storeu_si256(acc.as_mut_ptr().add(32).cast::<__m256i>(), a4) };
    unsafe { _mm256_storeu_si256(acc.as_mut_ptr().add(40).cast::<__m256i>(), a5) };
    unsafe { _mm256_storeu_si256(acc.as_mut_ptr().add(48).cast::<__m256i>(), a6) };
    unsafe { _mm256_storeu_si256(acc.as_mut_ptr().add(56).cast::<__m256i>(), a7) };
}

/// Helper: 64-bit load into the low half of a `__m128i`, upper
/// half zero-filled. `_mm256_cvtepi8_epi32` only sign-extends
/// the low 8 lanes of its input — using a 64-bit load (instead
/// of 128-bit) avoids reading past the end of the 64-byte
/// table row at byte offset 56.
///
/// # Safety
///
/// `ptr..ptr+8` must be readable and inside an allocation.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load64(ptr: *const u8) -> __m128i {
    // SAFETY: caller guarantees 8 readable bytes.
    unsafe { _mm_loadl_epi64(ptr.cast::<__m128i>()) }
}
