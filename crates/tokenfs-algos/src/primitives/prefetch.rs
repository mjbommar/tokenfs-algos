//! Cross-architecture software prefetch hints.
//!
//! These are advisory; the only guarantees are correctness (no UB on any
//! aligned/unaligned/null-ish pointer the compiler hands us) and uniform
//! API across `x86`, `x86_64`, and `aarch64`. On other architectures the
//! functions compile to no-ops so call-sites can stay branch-free.
//!
//! Locality hints follow the conventional naming used by Intel and Arm:
//! `t0` — keep in all cache levels (most local), `t1` — skip L1, `t2` —
//! skip L1 and L2, `nta` — non-temporal (avoid all caches; useful for
//! one-shot streaming reads).
//!
//! ## Why `*const u8`
//!
//! Call sites usually have a typed pointer (e.g. `*const f32`, `*const
//! __m256i`). Asking them to cast to `*const u8` keeps the public surface
//! single-purpose and matches the underlying intrinsics (which prefetch a
//! cache-line worth of bytes regardless of declared element type).

#![allow(dead_code)]

/// Prefetch `ptr` into all cache levels (T0 hint).
///
/// On x86, this is `prefetcht0`. On AArch64, this is `prfm pldl1keep`.
/// On other architectures, this is a no-op. Safe to call with any pointer
/// value — the intrinsics are explicitly hint-only and never fault.
#[inline(always)]
pub(crate) fn prefetch_t0(ptr: *const u8) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{_MM_HINT_T0, _mm_prefetch};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

        // SAFETY: `_mm_prefetch` is a hint-only intrinsic. It never
        // dereferences the pointer to produce architectural state, so
        // any address — even null or unmapped — is acceptable.
        unsafe { _mm_prefetch::<_MM_HINT_T0>(ptr.cast()) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `prfm` is hint-only on AArch64; it never raises a
        // synchronous abort regardless of the address provided.
        unsafe {
            core::arch::asm!(
                "prfm pldl1keep, [{ptr}]",
                ptr = in(reg) ptr,
                options(nostack, preserves_flags, readonly),
            );
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // Other architectures: no portable software-prefetch primitive.
        let _ = ptr;
    }
}

/// Prefetch `ptr` into L2 and below (T1 hint).
///
/// On x86, this is `prefetcht1`. On AArch64, this is `prfm pldl2keep`.
/// On other architectures, this is a no-op.
#[inline(always)]
pub(crate) fn prefetch_t1(ptr: *const u8) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{_MM_HINT_T1, _mm_prefetch};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{_MM_HINT_T1, _mm_prefetch};

        // SAFETY: hint-only intrinsic; see [`prefetch_t0`].
        unsafe { _mm_prefetch::<_MM_HINT_T1>(ptr.cast()) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: hint-only AArch64 instruction; see [`prefetch_t0`].
        unsafe {
            core::arch::asm!(
                "prfm pldl2keep, [{ptr}]",
                ptr = in(reg) ptr,
                options(nostack, preserves_flags, readonly),
            );
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
    }
}

/// Prefetch `ptr` into L3 and below (T2 hint).
///
/// On x86, this is `prefetcht2`. On AArch64, this is `prfm pldl3keep`.
/// On other architectures, this is a no-op.
#[inline(always)]
pub(crate) fn prefetch_t2(ptr: *const u8) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{_MM_HINT_T2, _mm_prefetch};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{_MM_HINT_T2, _mm_prefetch};

        // SAFETY: hint-only intrinsic; see [`prefetch_t0`].
        unsafe { _mm_prefetch::<_MM_HINT_T2>(ptr.cast()) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: hint-only AArch64 instruction; see [`prefetch_t0`].
        unsafe {
            core::arch::asm!(
                "prfm pldl3keep, [{ptr}]",
                ptr = in(reg) ptr,
                options(nostack, preserves_flags, readonly),
            );
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
    }
}

/// Prefetch `ptr` non-temporally (NTA hint).
///
/// On x86, this is `prefetchnta`. On AArch64, this is `prfm pldl1strm`.
/// On other architectures, this is a no-op. Use for one-shot streaming
/// reads where you don't expect to revisit the line.
#[inline(always)]
pub(crate) fn prefetch_nta(ptr: *const u8) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{_MM_HINT_NTA, _mm_prefetch};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{_MM_HINT_NTA, _mm_prefetch};

        // SAFETY: hint-only intrinsic; see [`prefetch_t0`].
        unsafe { _mm_prefetch::<_MM_HINT_NTA>(ptr.cast()) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: hint-only AArch64 instruction; see [`prefetch_t0`].
        unsafe {
            core::arch::asm!(
                "prfm pldl1strm, [{ptr}]",
                ptr = in(reg) ptr,
                options(nostack, preserves_flags, readonly),
            );
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each prefetch flavour must accept a real, in-bounds pointer without
    /// faulting. This is a smoke test; we don't assert any cache effect.
    #[test]
    fn prefetch_all_flavours_are_safe_on_real_data() {
        let bytes = [0_u8; 4096];
        let p = bytes.as_ptr();
        prefetch_t0(p);
        prefetch_t1(p);
        prefetch_t2(p);
        prefetch_nta(p);
    }

    /// Prefetching one-past-the-end is well-defined: the intrinsics are
    /// hint-only and never fault on unmapped addresses on supported
    /// architectures. This guards against accidental dereference if a
    /// future refactor swaps the inline-asm out for a real load.
    #[test]
    fn prefetch_one_past_end_is_safe() {
        let bytes = [0_u8; 64];
        // SAFETY: pointer arithmetic to one-past-end is allowed; we
        // never dereference here.
        let p = unsafe { bytes.as_ptr().add(bytes.len()) };
        prefetch_t0(p);
        prefetch_t1(p);
        prefetch_t2(p);
        prefetch_nta(p);
    }
}
