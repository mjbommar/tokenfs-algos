//! MinHash signatures for set-similarity (Jaccard) estimation.
//!
//! `MinHash` answers "how similar are these two sets?" without storing the
//! sets themselves. The signature is `K` smallest hash values across `K`
//! independent hash functions; the fraction of equal positions in two
//! signatures is an unbiased estimator of the Jaccard similarity.
//!
//! Variants implemented:
//!
//! - **Classic K-min MinHash**: `K` independent seeded hashers; signature
//!   stores `min` per hasher. Most established and statistically clean.
//! - **One-permutation MinHash (OPH)**: a single hash function is partitioned
//!   into `K` buckets; the signature is the per-bucket minimum. Cheaper to
//!   build (one hash per element instead of `K`), with documented degraded
//!   accuracy on sparse inputs (see `densified_one_permutation` for the fix).
//! - **b-bit MinHash**: any of the above signatures truncated to the lowest
//!   `b` bits per slot, trading collision probability for compactness.
//!
//! Hashing uses [`crate::hash::mix64`] with per-slot seeds for deterministic
//! reproducibility.

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::boxed::Box;

use crate::hash::mix64;
use crate::similarity::kernels_gather;

/// Fixed-size MinHash signature. Each slot holds the minimum observed hash
/// for one of `K` independent hash functions (classic) or `K` partitions
/// (OPH).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Signature<const K: usize> {
    slots: [u64; K],
    /// True for slots that have never been updated. Used by densification.
    populated: [bool; K],
}

impl<const K: usize> Signature<K> {
    /// Empty signature; every slot is unpopulated and set to `u64::MAX`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            slots: [u64::MAX; K],
            populated: [false; K],
        }
    }

    /// Number of slots `K`.
    #[must_use]
    pub const fn len(&self) -> usize {
        K
    }

    /// Returns true when no slot has been populated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.populated.iter().all(|&p| !p)
    }

    /// Returns the raw signature slice.
    #[must_use]
    pub const fn slots(&self) -> &[u64; K] {
        &self.slots
    }

    /// Returns the slot at `index`. Out-of-range returns `u64::MAX`.
    #[must_use]
    pub fn slot(&self, index: usize) -> u64 {
        if index >= K {
            u64::MAX
        } else {
            self.slots[index]
        }
    }

    /// Truncate every slot to its lowest `b` bits. Used by b-bit MinHash.
    /// `b` is clamped to `1..=64`.
    #[must_use]
    pub fn b_bit(self, b: u32) -> Self {
        let b = b.clamp(1, 64);
        let mask = if b == 64 { u64::MAX } else { (1_u64 << b) - 1 };
        let mut out = Self::new();
        for i in 0..K {
            out.slots[i] = self.slots[i] & mask;
            out.populated[i] = self.populated[i];
        }
        out
    }
}

impl<const K: usize> Default for Signature<K> {
    fn default() -> Self {
        Self::new()
    }
}

/// Builds a classic K-min MinHash signature from an iterator of `u64`
/// element hashes.
///
/// Each slot `k` is updated with `min(slot_k, mix64_of(element ^ seed_k))`,
/// where `seed_k = base_seed + k as u64`. Iterating elements in any order
/// produces the same signature.
#[must_use]
pub fn classic_from_hashes<I, const K: usize>(elements: I, base_seed: u64) -> Signature<K>
where
    I: IntoIterator<Item = u64>,
{
    let mut sig = Signature::<K>::new();
    for element in elements {
        for k in 0..K {
            // Seed each hasher with base_seed + k; mix the element's hash with
            // that seed via the public mix function used elsewhere.
            let seed_k = base_seed.wrapping_add(k as u64);
            let h = mix_two(element, seed_k);
            if h < sig.slots[k] {
                sig.slots[k] = h;
                sig.populated[k] = true;
            }
        }
    }
    sig
}

/// Builds a classic MinHash signature from raw byte slices.
///
/// Each item's hash is computed via [`crate::hash::mix64`] with `base_seed`
/// before being fed into the signature builder. Useful for n-gram features.
#[must_use]
pub fn classic_from_bytes<'a, I, const K: usize>(items: I, base_seed: u64) -> Signature<K>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    classic_from_hashes::<_, K>(
        items.into_iter().map(|bytes| mix64(bytes, base_seed)),
        base_seed,
    )
}

/// Per-byte MinHash signature backed by a precomputed gather table.
///
/// The table-based representation defines an alternative hash family
/// where each input byte contributes `K` independent hash values via a
/// single row of `T : [u8 -> [u64; K]]`. Building the table from seeds
/// is one-shot (`build_byte_table_from_seeds`); per-byte updates are
/// then table loads instead of per-byte hash evaluations.
///
/// State footprint: `K * 256 * 8` bytes — see
/// [`kernels_gather`] for the L1/L2 trade-off discussion.
///
/// **WARNING (kernel stack)**: returns the table *by value*. At
/// `K = 64` that is 128 KiB on the call frame; at `K = 256` it is 512
/// KiB. Neither is safe on a kernel-adjacent stack (typical 8-16 KiB
/// budget). Callers in those environments must use one of the heap-
/// free siblings:
///
/// - [`build_byte_table_from_seeds_into`] — caller-provided
///   `&mut [[u64; K]; 256]` scratch (zero stack cost beyond the
///   borrow); use whenever the caller already controls the storage
///   (mmap, thread-local pool, postgres memory context, kernel
///   `kmalloc`'d slab);
/// - [`build_byte_table_from_seeds_boxed`] — convenience wrapper that
///   `Box::new_uninit()`-allocates the table on the heap; or
/// - [`kernels_gather::build_table_from_seeds_into`] — the underlying
///   primitive.
///
/// For `K <= MINHASH_TABLE_BY_VALUE_SAFE_K_MAX` (currently `K = 8`,
/// 16 KiB table) the by-value form fits inside the documented kernel
/// stack budget; beyond that, prefer one of the siblings above.
///
/// **Hash family**: the table-based variant is **not** bit-equivalent
/// to [`classic_from_bytes`], which streams whole inputs through
/// [`mix64`]. It defines its own per-byte family:
/// `h_k(byte) = mix_word(byte ^ seeds[k])`. Two callers that build
/// signatures with the same seeds — one via the scalar table-based
/// path, one via the gather kernels — produce **bit-identical**
/// signatures.
#[must_use]
pub fn build_byte_table_from_seeds<const K: usize>(
    seeds: &[u64; K],
) -> [[u64; K]; kernels_gather::TABLE_ROWS] {
    kernels_gather::build_table_from_seeds(seeds)
}

/// Heap-free / kernel-safe sibling of [`build_byte_table_from_seeds`]:
/// writes the per-byte gather table into caller-provided scratch
/// instead of returning the (potentially large) array by value.
///
/// Bit-exact with [`build_byte_table_from_seeds`] for every `K`. The
/// caller decides where the table lives (heap `Box`, mmap, thread-local
/// pool, postgres memory context, kernel `kmalloc`'d slab, etc.) so the
/// kernel-stack hazard documented on [`build_byte_table_from_seeds`]
/// does not apply. This is the recommended entry point for any caller
/// at or below the documented kernel stack budget — including
/// FFI/cgo crossings into shallow goroutines and Postgres backend
/// extensions called from a deep call chain.
///
/// `scratch` is fully overwritten on entry — its prior contents are
/// discarded — so callers may reuse a single buffer across many calls
/// without pre-clearing it.
///
/// Mirrors the §156 caller-provided-scratch convention (audit-R5 fix
/// for `crc32_hash4_bins_pipelined`); `kernels_gather` already exposes
/// the same primitive at [`kernels_gather::build_table_from_seeds_into`]
/// — this wrapper is the public-facing MinHash entry point that
/// matches the by-value [`build_byte_table_from_seeds`] naming
/// convention (audit-R8 #6b).
pub fn build_byte_table_from_seeds_into<const K: usize>(
    seeds: &[u64; K],
    scratch: &mut [[u64; K]; kernels_gather::TABLE_ROWS],
) {
    kernels_gather::build_table_from_seeds_into::<K>(seeds, scratch);
}

/// Heap-allocated companion of [`build_byte_table_from_seeds`].
///
/// Returns a `Box<[[u64; K]; 256]>` so the table never sits on the
/// caller's stack frame. Bit-exact with [`build_byte_table_from_seeds`]
/// for every `K`; mirrors the §77 caller-provided-scratch convention
/// fixed for `crc32_hash4_bins_pipelined` (audit-round-3).
///
/// **Use this whenever the caller is on a constrained stack**
/// (kernel module, postgres backend > shallow-call-chain primitive,
/// FFI/cgo crossing a small Go goroutine, embedded). For the
/// `K = 256` width (512 KiB table) this is the only stack-safe entry
/// point.
///
/// Allocation is performed via `Box::<T>::new_uninit().assume_init()`
/// after fully writing every entry — `Box::new([[0; K]; 256])` would
/// stack-allocate the literal first and defeat the kernel-stack
/// safety we set out to provide.
///
/// For total control over where the table lives (mmap, thread-local
/// pool, postgres memory context, etc.), use
/// [`kernels_gather::build_table_from_seeds_into`] directly with
/// caller-supplied storage.
#[cfg(any(feature = "std", feature = "alloc"))]
#[must_use]
pub fn build_byte_table_from_seeds_boxed<const K: usize>(
    seeds: &[u64; K],
) -> Box<[[u64; K]; kernels_gather::TABLE_ROWS]> {
    use core::mem::MaybeUninit;

    // Reserve heap space for the table without touching the stack:
    // `Box::<T>::new_uninit()` allocates `sizeof::<T>()` on the heap
    // and returns `Box<MaybeUninit<T>>`. We then write every entry
    // through `as_mut_ptr` and `assume_init` once the table is fully
    // initialised. This avoids the `Box::new([[0; K]; 256])` stack
    // copy that, at K = 256, would burn 512 KiB of stack before the
    // heap copy — exactly the hazard this function is designed to
    // dodge.
    let mut uninit: Box<MaybeUninit<[[u64; K]; kernels_gather::TABLE_ROWS]>> = Box::new_uninit();
    // SAFETY: `uninit.as_mut_ptr()` is a valid, properly-aligned,
    // non-null pointer to writable heap storage of exactly
    // `[[u64; K]; TABLE_ROWS]` size. We cast to the raw element-type
    // pointer to write through it (`u64` is `Copy` and trivially
    // initialised). After every byte/k slot is filled,
    // `assume_init` is sound because the entire 256 × K block has
    // been written.
    unsafe {
        let row_ptr = uninit.as_mut_ptr().cast::<u64>();
        let mut byte = 0_usize;
        while byte < kernels_gather::TABLE_ROWS {
            let mut k = 0;
            while k < K {
                row_ptr
                    .add(byte * K + k)
                    .write(crate::hash::mix_word((byte as u64) ^ seeds[k]));
                k += 1;
            }
            byte += 1;
        }
        uninit.assume_init()
    }
}

/// Updates an 8-way `Signature` from a byte slice using the
/// runtime-dispatched gather kernel.
///
/// The signature uses the per-byte hash family defined above. Falls
/// through to the scalar implementation when no SIMD path is
/// available.
pub fn update_bytes_table_8(
    sig: &mut Signature<8>,
    table: &[[u64; 8]; kernels_gather::TABLE_ROWS],
    bytes: &[u8],
) {
    if bytes.is_empty() {
        return;
    }
    kernels_gather::update_minhash_8way_auto(bytes, table, &mut sig.slots);
    for k in 0..8 {
        if sig.slots[k] != u64::MAX {
            sig.populated[k] = true;
        }
    }
}

/// Builds a fresh table-based 8-way MinHash signature from a byte
/// slice and a precomputed gather table. Convenience wrapper over
/// [`update_bytes_table_8`].
///
/// # Stack
///
/// Returns `Signature<8>` (~80 bytes) by value; safe at K=8. The
/// stack-cost concern grows with `K` — for the K-generic siblings see
/// [`signature_simd`] / [`signature_simd_into`].
#[must_use]
pub fn classic_from_bytes_table_8(
    bytes: &[u8],
    table: &[[u64; 8]; kernels_gather::TABLE_ROWS],
) -> Signature<8> {
    let mut sig = Signature::<8>::new();
    update_bytes_table_8(&mut sig, table, bytes);
    sig
}

/// Heap-free / kernel-safe sibling of [`classic_from_bytes_table_8`]:
/// writes the 8-way signature into the caller-provided slot instead of
/// returning it by value.
///
/// Equivalent to `*out = classic_from_bytes_table_8(bytes, table);` but
/// avoids the (small) by-value `Signature<8>` copy. Provided for
/// API-symmetry with [`signature_simd_into`]; useful for batch loops
/// that pre-allocate the output and want every per-byte signature
/// written in place (audit-R8 #6b).
pub fn classic_from_bytes_table_8_into(
    bytes: &[u8],
    table: &[[u64; 8]; kernels_gather::TABLE_ROWS],
    out: &mut Signature<8>,
) {
    *out = Signature::<8>::new();
    update_bytes_table_8(out, table, bytes);
}

/// Updates a `K`-way `Signature` from a byte slice using the runtime-
/// dispatched gather kernel for general `K`.
///
/// Bit-identical with [`update_bytes_table_8`] for `K = 8` and with the
/// per-byte scalar reference at every other `K`. Picks AVX-512 → AVX2
/// → NEON → scalar in priority order.
pub fn update_bytes_table_kway<const K: usize>(
    sig: &mut Signature<K>,
    table: &[[u64; K]; kernels_gather::TABLE_ROWS],
    bytes: &[u8],
) {
    if bytes.is_empty() {
        return;
    }
    kernels_gather::update_minhash_kway_auto::<K>(bytes, table, &mut sig.slots);
    for k in 0..K {
        if sig.slots[k] != u64::MAX {
            sig.populated[k] = true;
        }
    }
}

/// SIMD-accelerated one-shot `K`-way MinHash signature over a byte
/// slice using a precomputed per-byte gather table.
///
/// Equivalent to [`classic_from_bytes_table_8`] generalised over `K`,
/// dispatched to the best available SIMD backend at runtime
/// (AVX-512 → AVX2 → NEON → scalar). The table family is the per-byte
/// hash defined by [`build_byte_table_from_seeds`].
///
/// For `K = 8` this matches [`classic_from_bytes_table_8`] bit-exactly;
/// for general `K` it matches a hand-rolled K-min update over
/// `mix_word(byte ^ seeds[k])` for every `k`.
///
/// Per the spec doc-set
/// (`docs/v0.2_planning/03_EXECUTION_PLAN.md` § Sprint 45-46), this is
/// the SIMD-accelerated entry point for callers that already hold a
/// gather table; pair with [`build_byte_table_from_seeds`] to
/// pre-build the table from seeds.
///
/// # Stack
///
/// Returns `Signature<K>` by value: `K * 8` bytes for the slot array
/// plus `K` bytes for the populated bitmap. At `K = 256` that is
/// ~2.25 KiB on the call frame — small relative to the table itself
/// (`build_byte_table_from_seeds` discusses the table-side hazard) but
/// non-trivial when accumulated across deep call chains in a
/// kernel/FUSE context. The heap-free sibling
/// [`signature_simd_into`] writes the signature into a caller-provided
/// `&mut Signature<K>` slot instead, eliminating the by-value copy
/// entirely (audit-R8 #6b).
///
/// **Allocation note (kernel stack)**: the table itself is borrowed.
/// For `K > kernels_gather::MINHASH_TABLE_BY_VALUE_SAFE_K_MAX`
/// (currently `K > 8`) the table footprint exceeds the kernel stack
/// budget; build it via [`build_byte_table_from_seeds_into`] /
/// [`build_byte_table_from_seeds_boxed`] instead of
/// [`build_byte_table_from_seeds`] so the table lives off-stack.
#[must_use]
pub fn signature_simd<const K: usize>(
    bytes: &[u8],
    table: &[[u64; K]; kernels_gather::TABLE_ROWS],
) -> Signature<K> {
    let mut sig = Signature::<K>::new();
    update_bytes_table_kway::<K>(&mut sig, table, bytes);
    sig
}

/// Heap-free / kernel-safe sibling of [`signature_simd`]: writes the
/// `K`-way signature into the caller-provided `out` slot instead of
/// returning it by value.
///
/// Equivalent to `*out = signature_simd::<K>(bytes, table);` but
/// avoids the by-value `Signature<K>` copy on return — at `K = 256`
/// that copy is ~2.25 KiB on the call frame, which compounds across
/// deep kernel/FUSE call chains. Use this entry point in any
/// kernel-adjacent context, FFI/cgo crossings into shallow goroutines,
/// or postgres backend extensions called from a deep call chain
/// (audit-R8 #6b).
///
/// `out` is fully overwritten on entry — its prior contents (slots,
/// populated flags) are reset before the K-min update runs — so callers
/// may reuse a single signature buffer across many calls without
/// pre-clearing it.
///
/// Bit-exact with [`signature_simd`] for every `K` and every input.
pub fn signature_simd_into<const K: usize>(
    bytes: &[u8],
    table: &[[u64; K]; kernels_gather::TABLE_ROWS],
    out: &mut Signature<K>,
) {
    *out = Signature::<K>::new();
    update_bytes_table_kway::<K>(out, table, bytes);
}

/// Batched form of [`signature_simd`]: compute one MinHash signature
/// per input byte slice, writing into a caller-provided output buffer.
///
/// This is the **panicking** form. Use [`try_signature_batch_simd`] to
/// recover from a length mismatch instead of panicking.
///
/// # Panics
///
/// Panics when `byte_slices.len() != out.len()`. Both lengths must
/// match so the per-slice signatures land in the corresponding output
/// slot.
///
/// Only compiled when the `panicking-shape-apis` Cargo feature is
/// enabled (default). Kernel/FUSE consumers should disable that
/// feature and use [`try_signature_batch_simd`] (audit-R5 #157).
#[cfg(feature = "panicking-shape-apis")]
pub fn signature_batch_simd<const K: usize>(
    byte_slices: &[&[u8]],
    table: &[[u64; K]; kernels_gather::TABLE_ROWS],
    out: &mut [Signature<K>],
) {
    assert_eq!(
        byte_slices.len(),
        out.len(),
        "signature_batch_simd: byte_slices.len() ({}) must match out.len() ({})",
        byte_slices.len(),
        out.len()
    );
    for (slot, bytes) in out.iter_mut().zip(byte_slices.iter()) {
        *slot = signature_simd::<K>(bytes, table);
    }
}

/// Shape error returned by [`try_signature_batch_simd`] when input and
/// output lengths disagree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchShapeError {
    /// Number of input byte slices supplied.
    pub byte_slices_len: usize,
    /// Number of output signature slots supplied.
    pub out_len: usize,
}

impl core::fmt::Display for BatchShapeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "MinHash batch shape mismatch: byte_slices.len() = {}, out.len() = {}",
            self.byte_slices_len, self.out_len
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BatchShapeError {}

/// Fallible form of [`signature_batch_simd`]. Returns a
/// [`BatchShapeError`] when `byte_slices.len() != out.len()`; otherwise
/// computes the batch and returns `Ok(())`.
pub fn try_signature_batch_simd<const K: usize>(
    byte_slices: &[&[u8]],
    table: &[[u64; K]; kernels_gather::TABLE_ROWS],
    out: &mut [Signature<K>],
) -> Result<(), BatchShapeError> {
    if byte_slices.len() != out.len() {
        return Err(BatchShapeError {
            byte_slices_len: byte_slices.len(),
            out_len: out.len(),
        });
    }
    for (slot, bytes) in out.iter_mut().zip(byte_slices.iter()) {
        *slot = signature_simd::<K>(bytes, table);
    }
    Ok(())
}

/// Streaming K-way table-based MinHash signature builder.
///
/// Wraps a [`Signature<K>`] plus a borrowed gather table so callers can feed
/// bytes incrementally — one byte at a time, or in 4 KiB-ish chunks — and
/// snapshot the running signature at any point. The hash family matches the
/// table-based [`classic_from_bytes_table_8`] / [`update_bytes_table_8`]
/// path: `h_k(byte) = mix_word(byte ^ seeds[k])`.
///
/// This is the right primitive for FUSE-style write paths and for any caller
/// that needs both a "live" Jaccard estimate and a final signature without
/// re-hashing the input.
///
/// ## Memory shape
///
/// The table itself is borrowed (`&[[u64; K]; 256]`); the builder owns only
/// `Signature<K>` (16 bytes per slot plus a populated bitmap) so it stays
/// `Copy`-ish and cheap to keep around per write handle.
///
/// ## Bit-exact across chunkings
///
/// Two streams that consume the same bytes (in any order — the hash family
/// is order-independent) produce identical signatures regardless of chunk
/// boundaries. This is verified by `tests::stream_chunking_invariant`.
///
/// ## Example
///
/// ```
/// use tokenfs_algos::similarity::minhash::{
///     build_byte_table_from_seeds, IncrementalSignature,
/// };
///
/// let seeds: [u64; 8] = core::array::from_fn(|i| 0x9E37_79B9_u64 ^ i as u64);
/// let table = build_byte_table_from_seeds::<8>(&seeds);
///
/// let mut builder = IncrementalSignature::<8>::new(&table);
/// for chunk in b"abcdef".chunks(2) {
///     builder.update_bytes(chunk);
/// }
/// let sig = builder.finalize();
/// assert!(!sig.is_empty());
/// ```
#[derive(Debug)]
pub struct IncrementalSignature<'a, const K: usize> {
    sig: Signature<K>,
    table: &'a [[u64; K]; kernels_gather::TABLE_ROWS],
}

impl<'a, const K: usize> IncrementalSignature<'a, K> {
    /// Construct an empty incremental signature backed by the given gather
    /// table. The table is borrowed for the lifetime of the builder; build
    /// it once via [`build_byte_table_from_seeds`] and reuse.
    #[must_use]
    pub fn new(table: &'a [[u64; K]; kernels_gather::TABLE_ROWS]) -> Self {
        Self {
            sig: Signature::<K>::new(),
            table,
        }
    }

    /// Construct an incremental signature seeded from an existing
    /// [`Signature<K>`]. Useful for resuming a partial computation that was
    /// snapshotted with [`Self::snapshot`] or merged via [`Self::merge`].
    #[must_use]
    pub fn from_signature(
        table: &'a [[u64; K]; kernels_gather::TABLE_ROWS],
        sig: Signature<K>,
    ) -> Self {
        Self { sig, table }
    }

    /// Feed a single byte into the running K-min update.
    pub fn update_byte(&mut self, byte: u8) {
        let row = &self.table[byte as usize];
        for (k, &h) in row.iter().enumerate() {
            if h < self.sig.slots[k] {
                self.sig.slots[k] = h;
                self.sig.populated[k] = true;
            }
        }
    }

    /// Feed `bytes` into the running K-min update.
    ///
    /// Dispatches to the runtime-selected gather kernel via
    /// [`kernels_gather::update_minhash_kway_auto`] for every `K`. The
    /// output matches [`classic_from_bytes_table_8`] /
    /// [`update_bytes_table_8`] for the concatenated input at `K = 8`,
    /// and the per-byte scalar reference at every other `K`.
    pub fn update_bytes(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        kernels_gather::update_minhash_kway_auto::<K>(bytes, self.table, &mut self.sig.slots);
        for k in 0..K {
            if self.sig.slots[k] != u64::MAX {
                self.sig.populated[k] = true;
            }
        }
    }

    /// Read the current signature mid-stream without consuming the builder.
    ///
    /// Useful for emitting periodic snapshots while the underlying byte
    /// stream is still being written.
    #[must_use]
    pub fn snapshot(&self) -> Signature<K> {
        self.sig
    }

    /// Return the final signature and consume the builder.
    #[must_use]
    pub fn finalize(self) -> Signature<K> {
        self.sig
    }

    /// Reset the running signature to empty. The table reference is
    /// preserved; reuse the builder for the next stream without re-binding.
    pub fn reset(&mut self) {
        self.sig = Signature::<K>::new();
    }

    /// Merge another signature into the running state via per-slot `min`.
    ///
    /// Lets a caller hash disjoint shards of the same input in parallel
    /// (each shard accumulates its own [`IncrementalSignature`], all sharing
    /// the same table) then fold the results: the final merged signature is
    /// identical to one produced by feeding the concatenated input through a
    /// single builder, because the K-min update is associative and
    /// commutative under `min`.
    pub fn merge(&mut self, other: &Signature<K>) {
        for k in 0..K {
            if other.slots[k] < self.sig.slots[k] {
                self.sig.slots[k] = other.slots[k];
            }
            if other.populated[k] {
                self.sig.populated[k] = true;
            }
        }
    }

    /// Number of slots `K`.
    #[must_use]
    pub const fn len(&self) -> usize {
        K
    }

    /// True when the running signature has no populated slots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sig.is_empty()
    }
}

/// Builds a one-permutation MinHash (OPH) signature.
///
/// One hash function partitions the universe into `K` equal-sized buckets;
/// each slot stores the minimum hash that landed in its bucket. This is `K`x
/// cheaper to build than [`classic_from_hashes`] — single hash per element
/// instead of `K` — at the cost of accuracy on sparse inputs (slots may stay
/// `u64::MAX`). Use [`densified_one_permutation`] to repair.
#[must_use]
pub fn one_permutation_from_hashes<I, const K: usize>(elements: I, seed: u64) -> Signature<K>
where
    I: IntoIterator<Item = u64>,
{
    assert!(K > 0, "MinHash signature size K must be > 0");
    let mut sig = Signature::<K>::new();
    for element in elements {
        let h = mix_two(element, seed);
        let bucket = (h % K as u64) as usize;
        // Within-bucket score: rotate so the bucket index doesn't dominate.
        let score = h.rotate_right((bucket as u32) & 63);
        if score < sig.slots[bucket] {
            sig.slots[bucket] = score;
            sig.populated[bucket] = true;
        }
    }
    sig
}

/// Builds a one-permutation MinHash signature from raw byte slices.
#[must_use]
pub fn one_permutation_from_bytes<'a, I, const K: usize>(items: I, seed: u64) -> Signature<K>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    one_permutation_from_hashes::<_, K>(items.into_iter().map(|bytes| mix64(bytes, seed)), seed)
}

/// Densified one-permutation MinHash: fills empty slots by borrowing from
/// the nearest populated neighbor (round-robin both directions).
///
/// On sparse inputs OPH leaves many slots at `u64::MAX`. Densification
/// preserves the OPH speed advantage while restoring the unbiased Jaccard
/// estimator. Reference: Shrivastava & Li, "Densifying One-Permutation
/// Hashing via Rotation for Fast Near Neighbor Search" (ICML 2014).
///
/// Implementation here is the simple "rotation" variant: for each empty
/// slot, walk forward (mod K) to the first populated slot. Returns the
/// signature unchanged if it was already fully populated or if no slots are
/// populated at all.
#[must_use]
pub fn densified_one_permutation<const K: usize>(mut sig: Signature<K>) -> Signature<K> {
    if sig.is_empty() {
        return sig;
    }
    if sig.populated.iter().all(|&p| p) {
        return sig;
    }
    // For each empty slot, find the next populated slot mod K and copy.
    for i in 0..K {
        if sig.populated[i] {
            continue;
        }
        for offset in 1..K {
            let j = (i + offset) % K;
            if sig.populated[j] {
                sig.slots[i] = sig.slots[j];
                sig.populated[i] = true;
                break;
            }
        }
    }
    sig
}

/// Estimates the Jaccard similarity between two signatures of the same size.
///
/// Returns the fraction of slots where the two signatures agree. For
/// classic MinHash this is an unbiased estimator of `|A ∩ B| / |A ∪ B|`.
#[must_use]
pub fn jaccard_similarity<const K: usize>(a: &Signature<K>, b: &Signature<K>) -> f64 {
    if K == 0 {
        return 0.0;
    }
    // Slots that are unpopulated in both don't contribute information.
    let mut equal = 0_usize;
    let mut total = 0_usize;
    for i in 0..K {
        match (a.populated[i], b.populated[i]) {
            (false, false) => continue,
            _ => {
                total += 1;
                if a.slots[i] == b.slots[i] {
                    equal += 1;
                }
            }
        }
    }
    if total == 0 {
        return 0.0;
    }
    equal as f64 / total as f64
}

/// Estimates Jaccard similarity from b-bit signatures.
///
/// b-bit MinHash collisions are more frequent than classic, so the raw
/// agreement rate over-estimates similarity. The unbiased estimator is
/// `(observed_match - 1/2^b) / (1 - 1/2^b)`, clamped to `[0, 1]`. See
/// Li & König (2010), "b-Bit Minwise Hashing".
#[must_use]
pub fn b_bit_jaccard_similarity<const K: usize>(
    a: &Signature<K>,
    b: &Signature<K>,
    b_bits: u32,
) -> f64 {
    let raw = jaccard_similarity(a, b);
    let b_bits = b_bits.clamp(1, 63);
    let collision = 1.0 / ((1_u64 << b_bits) as f64);
    let est = (raw - collision) / (1.0 - collision);
    est.clamp(0.0, 1.0)
}

/// Mixes two `u64` values into one. Cheaper than [`mix64`] over bytes.
#[inline]
fn mix_two(a: u64, b: u64) -> u64 {
    crate::hash::mix_word(a ^ crate::hash::mix_word(b.wrapping_add(0x9E37_79B9_7F4A_7C15)))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    /// Helper: hash a Rust set into u64 element hashes for test inputs.
    fn elements(items: &[u32]) -> impl Iterator<Item = u64> + '_ {
        items.iter().map(|&x| crate::hash::mix_word(u64::from(x)))
    }

    #[test]
    fn empty_signature_is_max() {
        let sig: Signature<8> = Signature::new();
        for slot in sig.slots() {
            assert_eq!(*slot, u64::MAX);
        }
        assert!(sig.is_empty());
    }

    #[test]
    fn identical_inputs_produce_identical_signatures() {
        let a = (0..100_u32).collect::<Vec<_>>();
        let s1 = classic_from_hashes::<_, 64>(elements(&a), 0xCAFE);
        let s2 = classic_from_hashes::<_, 64>(elements(&a), 0xCAFE);
        assert_eq!(s1, s2);
    }

    #[test]
    fn order_does_not_matter() {
        let mut a = (0..100_u32).collect::<Vec<_>>();
        let s1 = classic_from_hashes::<_, 64>(elements(&a), 0x1234);
        a.reverse();
        let s2 = classic_from_hashes::<_, 64>(elements(&a), 0x1234);
        assert_eq!(s1, s2);
    }

    #[test]
    fn jaccard_estimate_is_close_for_known_overlap() {
        // Two sets with known Jaccard similarity 0.5: {0..200} and {100..300}.
        // Intersection 100, union 300, Jaccard = 1/3.
        let a: Vec<u32> = (0..200).collect();
        let b: Vec<u32> = (100..300).collect();
        let sa = classic_from_hashes::<_, 256>(elements(&a), 0xABCD);
        let sb = classic_from_hashes::<_, 256>(elements(&b), 0xABCD);
        let est = jaccard_similarity(&sa, &sb);
        let expected = 1.0 / 3.0;
        assert!(
            (est - expected).abs() < 0.10,
            "est={est}, expected≈{expected}"
        );
    }

    #[test]
    fn jaccard_estimate_is_close_for_high_overlap() {
        // |A| = 200, |B| = 200, intersection 180, union 220, Jaccard = 180/220.
        let a: Vec<u32> = (0..200).collect();
        let b: Vec<u32> = (20..220).collect();
        let sa = classic_from_hashes::<_, 512>(elements(&a), 0x99);
        let sb = classic_from_hashes::<_, 512>(elements(&b), 0x99);
        let est = jaccard_similarity(&sa, &sb);
        let expected = 180.0 / 220.0;
        assert!(
            (est - expected).abs() < 0.10,
            "est={est}, expected≈{expected}"
        );
    }

    #[test]
    fn jaccard_estimate_is_close_for_disjoint_sets() {
        let a: Vec<u32> = (0..100).collect();
        let b: Vec<u32> = (1000..1100).collect();
        let sa = classic_from_hashes::<_, 256>(elements(&a), 0xDDDD);
        let sb = classic_from_hashes::<_, 256>(elements(&b), 0xDDDD);
        let est = jaccard_similarity(&sa, &sb);
        assert!(est < 0.10, "disjoint est={est}");
    }

    #[test]
    fn one_permutation_with_densification_recovers_jaccard() {
        let a: Vec<u32> = (0..200).collect();
        let b: Vec<u32> = (100..300).collect();
        let sa =
            densified_one_permutation(one_permutation_from_hashes::<_, 256>(elements(&a), 0xBEEF));
        let sb =
            densified_one_permutation(one_permutation_from_hashes::<_, 256>(elements(&b), 0xBEEF));
        let est = jaccard_similarity(&sa, &sb);
        let expected = 1.0 / 3.0;
        assert!(
            (est - expected).abs() < 0.15,
            "OPH est={est}, expected≈{expected}"
        );
    }

    #[test]
    fn b_bit_jaccard_close_to_classic() {
        let a: Vec<u32> = (0..500).collect();
        let b: Vec<u32> = (250..750).collect();
        let sa_full = classic_from_hashes::<_, 512>(elements(&a), 0x4242);
        let sb_full = classic_from_hashes::<_, 512>(elements(&b), 0x4242);
        let full_est = jaccard_similarity(&sa_full, &sb_full);

        let sa = sa_full.b_bit(8);
        let sb = sb_full.b_bit(8);
        let bbit_est = b_bit_jaccard_similarity(&sa, &sb, 8);

        assert!(
            (full_est - bbit_est).abs() < 0.05,
            "b-bit est={bbit_est} drifted from full est={full_est}"
        );
    }

    #[test]
    fn classic_from_bytes_matches_classic_from_hashes() {
        let items: Vec<&[u8]> = vec![b"alpha", b"beta", b"gamma", b"delta", b"epsilon"];
        let s_bytes = classic_from_bytes::<_, 64>(items.iter().copied(), 0x77);
        let s_hashed = classic_from_hashes::<_, 64>(items.iter().map(|b| mix64(b, 0x77)), 0x77);
        assert_eq!(s_bytes, s_hashed);
    }

    #[test]
    fn table_based_8way_matches_per_byte_reference() {
        // Hand-compute the per-byte hash family directly and compare
        // against the dispatched gather path.
        let seeds: [u64; 8] = [
            0x1111_1111_u64,
            0x2222_2222,
            0x3333_3333,
            0x4444_4444,
            0x5555_5555,
            0x6666_6666,
            0x7777_7777,
            0x8888_8888,
        ];
        let table = build_byte_table_from_seeds(&seeds);
        let payload = b"the quick brown fox jumps over the lazy dog 0123456789!@#$%^&*()";

        let actual = classic_from_bytes_table_8(payload, &table);

        let mut expected = [u64::MAX; 8];
        for &b in payload {
            for k in 0..8 {
                let h = crate::hash::mix_word((b as u64) ^ seeds[k]);
                if h < expected[k] {
                    expected[k] = h;
                }
            }
        }
        assert_eq!(actual.slots(), &expected);
        // populated flags should all be true since we updated every slot.
        for k in 0..8 {
            assert!(actual.populated[k], "slot {k} should be populated");
        }
    }

    #[test]
    fn empty_input_does_not_populate_table_signature() {
        let seeds: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let table = build_byte_table_from_seeds(&seeds);
        let sig = classic_from_bytes_table_8(b"", &table);
        for slot in sig.slots() {
            assert_eq!(*slot, u64::MAX);
        }
        assert!(sig.is_empty());
    }

    // ----- IncrementalSignature streaming tests -----------------------------

    fn make_test_seeds_8() -> [u64; 8] {
        [
            0x1111_1111_u64,
            0x2222_2222,
            0x3333_3333,
            0x4444_4444,
            0x5555_5555,
            0x6666_6666,
            0x7777_7777,
            0x8888_8888,
        ]
    }

    fn make_test_seeds_4() -> [u64; 4] {
        [0xAA_u64, 0xBB, 0xCC, 0xDD]
    }

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
    fn incremental_empty_signature_is_empty() {
        let seeds = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);
        let builder = IncrementalSignature::<8>::new(&table);
        let sig = builder.finalize();
        assert!(sig.is_empty());
    }

    #[test]
    fn incremental_update_byte_matches_one_shot() {
        let seeds = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);

        let payload = b"the quick brown fox jumps over the lazy dog";

        let mut builder = IncrementalSignature::<8>::new(&table);
        for &b in payload {
            builder.update_byte(b);
        }
        let stream_sig = builder.finalize();

        let one_shot = classic_from_bytes_table_8(payload, &table);
        assert_eq!(stream_sig.slots(), one_shot.slots());
    }

    #[test]
    fn incremental_update_bytes_matches_one_shot() {
        let seeds = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);

        let payload = random_bytes(4096);

        let mut builder = IncrementalSignature::<8>::new(&table);
        builder.update_bytes(&payload);
        let stream_sig = builder.finalize();

        let one_shot = classic_from_bytes_table_8(&payload, &table);
        assert_eq!(stream_sig.slots(), one_shot.slots());
    }

    /// Two streams chunked differently must produce identical signatures.
    #[test]
    fn stream_chunking_invariant() {
        let seeds = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);

        let payload = random_bytes(64 * 1024);
        let one_shot = classic_from_bytes_table_8(&payload, &table);

        for &chunk in &[1_usize, 7, 17, 63, 64, 65, 1024, 4096] {
            let mut builder = IncrementalSignature::<8>::new(&table);
            for block in payload.chunks(chunk) {
                builder.update_bytes(block);
            }
            assert_eq!(
                builder.finalize().slots(),
                one_shot.slots(),
                "chunk={chunk}"
            );
        }
    }

    /// The non-K=8 path delegates to the scalar reference. It must match the
    /// per-byte hand-rolled K-min update.
    #[test]
    fn incremental_k4_matches_scalar_reference() {
        let seeds = make_test_seeds_4();
        let table = build_byte_table_from_seeds::<4>(&seeds);

        let payload = random_bytes(4096);

        let mut builder = IncrementalSignature::<4>::new(&table);
        builder.update_bytes(&payload);
        let stream_sig = builder.finalize();

        let mut expected = [u64::MAX; 4];
        for &b in &payload {
            for k in 0..4 {
                let h = crate::hash::mix_word((b as u64) ^ seeds[k]);
                if h < expected[k] {
                    expected[k] = h;
                }
            }
        }
        assert_eq!(stream_sig.slots(), &expected);
    }

    /// `snapshot()` returns the live signature without consuming the builder;
    /// further updates must continue to refine it monotonically.
    #[test]
    fn snapshot_does_not_consume_builder() {
        let seeds = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);

        let mut builder = IncrementalSignature::<8>::new(&table);
        builder.update_bytes(b"first half");
        let snap = builder.snapshot();

        builder.update_bytes(b" + second half");
        let later = builder.snapshot();

        // K-min slots only get smaller as more bytes are observed.
        for k in 0..8 {
            assert!(later.slots[k] <= snap.slots[k]);
        }

        // After all updates, the builder's signature equals what we'd get
        // from a fresh one-shot pass over the concatenated input.
        let final_sig = builder.finalize();
        let one_shot = classic_from_bytes_table_8(b"first half + second half", &table);
        assert_eq!(final_sig.slots(), one_shot.slots());
    }

    #[test]
    fn reset_clears_signature() {
        let seeds = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);

        let mut builder = IncrementalSignature::<8>::new(&table);
        builder.update_bytes(b"poison");
        builder.reset();
        builder.update_bytes(b"abcdef");

        let one_shot = classic_from_bytes_table_8(b"abcdef", &table);
        assert_eq!(builder.finalize().slots(), one_shot.slots());
    }

    /// Merging two shard-signatures matches the signature of the
    /// concatenated input. K-min is associative + commutative under min.
    #[test]
    fn merge_matches_concatenation() {
        let seeds = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);

        let payload = random_bytes(8192);
        let (left, right) = payload.split_at(payload.len() / 2);

        let sig_left = classic_from_bytes_table_8(left, &table);
        let sig_right = classic_from_bytes_table_8(right, &table);

        let mut merged = IncrementalSignature::<8>::from_signature(&table, sig_left);
        merged.merge(&sig_right);

        let one_shot = classic_from_bytes_table_8(&payload, &table);
        assert_eq!(merged.finalize().slots(), one_shot.slots());
    }

    /// `from_signature` round-trips: building a signature, snapshotting it,
    /// resuming with `from_signature`, and feeding more bytes equals the
    /// signature of the concatenated input.
    #[test]
    fn from_signature_resumes_correctly() {
        let seeds = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);

        let mut a = IncrementalSignature::<8>::new(&table);
        a.update_bytes(b"first half");
        let snapshot = a.snapshot();

        let mut b = IncrementalSignature::<8>::from_signature(&table, snapshot);
        b.update_bytes(b" rest");

        let one_shot = classic_from_bytes_table_8(b"first half rest", &table);
        assert_eq!(b.finalize().slots(), one_shot.slots());
    }

    // ----- signature_simd / signature_batch_simd public API tests ----------

    /// Hand-roll a per-byte K-min reference signature so the test
    /// asserts against the per-byte family directly (rather than against
    /// another scalar API that might share the same bug).
    fn reference_signature_kway<const K: usize>(bytes: &[u8], seeds: &[u64; K]) -> [u64; K] {
        let mut out = [u64::MAX; K];
        for &b in bytes {
            for k in 0..K {
                let h = crate::hash::mix_word((b as u64) ^ seeds[k]);
                if h < out[k] {
                    out[k] = h;
                }
            }
        }
        out
    }

    #[test]
    fn signature_simd_k8_matches_scalar_one_shot() {
        let seeds = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);
        let payload = random_bytes(4096);

        let actual = signature_simd::<8>(&payload, &table);
        let expected = reference_signature_kway::<8>(&payload, &seeds);
        assert_eq!(actual.slots(), &expected);
        // Equivalent to classic_from_bytes_table_8 too.
        assert_eq!(
            actual.slots(),
            classic_from_bytes_table_8(&payload, &table).slots()
        );
    }

    #[test]
    fn signature_simd_k16_matches_scalar_reference() {
        let seeds: [u64; 16] = core::array::from_fn(|i| 0xABCD_u64.wrapping_add(i as u64));
        let table = build_byte_table_from_seeds::<16>(&seeds);
        let payload = random_bytes(8192);

        let actual = signature_simd::<16>(&payload, &table);
        let expected = reference_signature_kway::<16>(&payload, &seeds);
        assert_eq!(actual.slots(), &expected);
    }

    #[test]
    fn signature_simd_k32_matches_scalar_reference() {
        let seeds: [u64; 32] = core::array::from_fn(|i| 0x1234_5678_u64.wrapping_add(i as u64));
        let table = build_byte_table_from_seeds::<32>(&seeds);
        let payload = random_bytes(16_384);

        let actual = signature_simd::<32>(&payload, &table);
        let expected = reference_signature_kway::<32>(&payload, &seeds);
        assert_eq!(actual.slots(), &expected);
    }

    #[test]
    fn signature_simd_k64_matches_scalar_reference() {
        let seeds: [u64; 64] = core::array::from_fn(|i| 0xFACE_FEED_u64.wrapping_add(i as u64));
        let table = build_byte_table_from_seeds::<64>(&seeds);
        let payload = random_bytes(4096);

        let actual = signature_simd::<64>(&payload, &table);
        let expected = reference_signature_kway::<64>(&payload, &seeds);
        assert_eq!(actual.slots(), &expected);
    }

    /// Edge-case: the empty slice produces an empty signature (every
    /// slot stays `u64::MAX`, none populated).
    #[test]
    fn signature_simd_empty_input_is_empty() {
        let seeds: [u64; 16] = core::array::from_fn(|i| (i as u64) * 0xDEAD_BEEF);
        let table = build_byte_table_from_seeds::<16>(&seeds);
        let sig = signature_simd::<16>(b"", &table);
        for slot in sig.slots() {
            assert_eq!(*slot, u64::MAX);
        }
        assert!(sig.is_empty());
    }

    #[test]
    fn signature_simd_single_byte_matches_table_row() {
        let seeds: [u64; 16] = core::array::from_fn(|i| (i as u64) * 0xCAFE_F00D);
        let table = build_byte_table_from_seeds::<16>(&seeds);

        for b in [0_u8, 1, 0x42, 0xFF] {
            let sig = signature_simd::<16>(&[b], &table);
            for (k, &seed_k) in seeds.iter().enumerate() {
                let expected = crate::hash::mix_word((b as u64) ^ seed_k);
                assert_eq!(sig.slots()[k], expected, "single-byte b={b:#x} k={k}");
            }
        }
    }

    /// Inputs short enough that no full SIMD lane fires still match the
    /// scalar reference exactly.
    #[test]
    fn signature_simd_short_input_matches_scalar() {
        let seeds: [u64; 32] = core::array::from_fn(|i| 0xBABE_FACE_u64.wrapping_add(i as u64));
        let table = build_byte_table_from_seeds::<32>(&seeds);
        for len in [0_usize, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 33, 63] {
            let payload: Vec<u8> = (0..len)
                .map(|i| (i.wrapping_mul(13) ^ 0xC3) as u8)
                .collect();
            let actual = signature_simd::<32>(&payload, &table);
            let expected = reference_signature_kway::<32>(&payload, &seeds);
            assert_eq!(actual.slots(), &expected, "len={len}");
        }
    }

    /// Very long inputs (multi-MB) still match scalar exactly.
    #[test]
    fn signature_simd_long_input_matches_scalar() {
        let seeds: [u64; 16] = core::array::from_fn(|i| 0x2222_3333_u64.wrapping_add(i as u64));
        let table = build_byte_table_from_seeds::<16>(&seeds);
        let payload = random_bytes(1 << 20); // 1 MiB
        let actual = signature_simd::<16>(&payload, &table);
        let expected = reference_signature_kway::<16>(&payload, &seeds);
        assert_eq!(actual.slots(), &expected);
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn signature_batch_simd_matches_per_slice() {
        let seeds: [u64; 16] = core::array::from_fn(|i| 0x99AA_u64.wrapping_add(i as u64));
        let table = build_byte_table_from_seeds::<16>(&seeds);

        let payloads: Vec<Vec<u8>> = (0..8).map(|i| random_bytes(64 + i * 257)).collect();
        let refs: Vec<&[u8]> = payloads.iter().map(|v| v.as_slice()).collect();
        let mut out = vec![Signature::<16>::new(); refs.len()];
        signature_batch_simd::<16>(&refs, &table, &mut out);

        for (i, payload) in payloads.iter().enumerate() {
            let single = signature_simd::<16>(payload, &table);
            assert_eq!(out[i].slots(), single.slots(), "row {i}");
        }
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn signature_batch_simd_empty_batch_is_noop() {
        let seeds: [u64; 8] = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);
        let refs: Vec<&[u8]> = Vec::new();
        let mut out: Vec<Signature<8>> = Vec::new();
        signature_batch_simd::<8>(&refs, &table, &mut out);
        assert!(out.is_empty());
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    #[should_panic(expected = "must match")]
    fn signature_batch_simd_panics_on_shape_mismatch() {
        let seeds: [u64; 8] = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);
        let refs: Vec<&[u8]> = vec![b"a", b"b"];
        let mut out = vec![Signature::<8>::new(); 1];
        signature_batch_simd::<8>(&refs, &table, &mut out);
    }

    #[test]
    fn try_signature_batch_simd_rejects_shape_mismatch() {
        let seeds: [u64; 8] = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);
        let refs: Vec<&[u8]> = vec![b"alpha", b"beta", b"gamma"];
        let mut out = vec![Signature::<8>::new(); 2];
        let err = try_signature_batch_simd::<8>(&refs, &table, &mut out).unwrap_err();
        assert_eq!(err.byte_slices_len, 3);
        assert_eq!(err.out_len, 2);
    }

    #[test]
    fn try_signature_batch_simd_succeeds_on_matching_shape() {
        let seeds: [u64; 8] = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);
        let refs: Vec<&[u8]> = vec![b"hello", b"world", b"!"];
        let mut out = vec![Signature::<8>::new(); 3];
        let res = try_signature_batch_simd::<8>(&refs, &table, &mut out);
        assert!(res.is_ok());
        for (i, payload) in refs.iter().enumerate() {
            assert_eq!(out[i].slots(), signature_simd::<8>(payload, &table).slots());
        }
    }

    /// `update_bytes_table_kway` matches the per-byte scalar reference
    /// across the K family the bench covers.
    #[test]
    fn update_bytes_table_kway_matches_reference() {
        macro_rules! check_k {
            ($k:literal) => {{
                let seeds: [u64; $k] =
                    core::array::from_fn(|i| 0x9876_5432_u64.wrapping_add(i as u64));
                let table = build_byte_table_from_seeds::<$k>(&seeds);
                let payload = random_bytes(2048);

                let mut sig = Signature::<$k>::new();
                update_bytes_table_kway::<$k>(&mut sig, &table, &payload);
                let expected = reference_signature_kway::<$k>(&payload, &seeds);
                assert_eq!(sig.slots(), &expected, "K={}", $k);
                for k in 0..$k {
                    assert!(sig.populated[k], "K={} slot {k} should be populated", $k);
                }
            }};
        }
        check_k!(8);
        check_k!(16);
        check_k!(32);
        check_k!(64);
    }

    /// `build_byte_table_from_seeds_boxed<K>` produces a bit-exact
    /// match with the by-value [`build_byte_table_from_seeds`] across
    /// every K width covered by the MinHash benches. This guards
    /// audit-R5 finding #156: kernel-adjacent callers can swap the
    /// boxed wrapper in without behavior drift.
    ///
    /// At `K >= 16` the test only calls the boxed form and
    /// cross-checks against the per-byte scalar reference. Calling the
    /// by-value [`build_byte_table_from_seeds`] at K = 128 / K = 256
    /// here would put 256 KiB / 512 KiB of table on the test stack —
    /// the very hazard the boxed wrapper exists to avoid.
    #[test]
    fn build_byte_table_from_seeds_boxed_matches_by_value() {
        // K = 8 (16 KiB table) is at the upper edge of the kernel
        // stack budget but safe in the test harness; compare directly
        // against the by-value form.
        {
            const K: usize = 8;
            let seeds: [u64; K] = core::array::from_fn(|i| 0xABCD_1234_u64.wrapping_add(i as u64));
            let by_value = build_byte_table_from_seeds::<K>(&seeds);
            let boxed = build_byte_table_from_seeds_boxed::<K>(&seeds);
            assert_eq!(by_value, *boxed, "K={K}");
        }

        // K >= 16: build via the boxed wrapper and cross-check each
        // entry against the per-(byte, k) reference family directly.
        // This proves the boxed wrapper writes the *correct* table
        // without ever materialising a large array on the test stack.
        macro_rules! check_boxed_against_reference {
            ($k:literal) => {{
                let seeds: [u64; $k] =
                    core::array::from_fn(|i| 0xABCD_1234_u64.wrapping_add(i as u64));
                let boxed = build_byte_table_from_seeds_boxed::<$k>(&seeds);
                for byte in 0..kernels_gather::TABLE_ROWS {
                    for k in 0..$k {
                        assert_eq!(
                            boxed[byte][k],
                            crate::hash::mix_word((byte as u64) ^ seeds[k]),
                            "K={} byte={byte} k={k}",
                            $k
                        );
                    }
                }
            }};
        }
        check_boxed_against_reference!(16);
        check_boxed_against_reference!(32);
        check_boxed_against_reference!(64);
        check_boxed_against_reference!(128);
        check_boxed_against_reference!(256);
    }

    /// Boxed table fed through `signature_simd<K>` produces the same
    /// MinHash signature as the per-byte scalar reference at K = 128
    /// and K = 256 — the widths the new bench exercises and that
    /// audit-R5 #156 flagged. The table never lives on the test
    /// stack.
    #[test]
    fn signature_simd_boxed_table_matches_reference_k128_k256() {
        macro_rules! check_k {
            ($k:literal) => {{
                let seeds: [u64; $k] =
                    core::array::from_fn(|i| 0xFEED_FACE_u64.wrapping_add(i as u64));
                let table_a = build_byte_table_from_seeds_boxed::<$k>(&seeds);
                let table_b = build_byte_table_from_seeds_boxed::<$k>(&seeds);
                let payload = random_bytes(4096);
                let sig_a = signature_simd::<$k>(&payload, &table_a);
                let sig_b = signature_simd::<$k>(&payload, &table_b);
                assert_eq!(sig_a.slots(), sig_b.slots(), "K={}", $k);

                // Cross-check against the per-byte scalar reference so
                // we know the boxed-table signature also matches the
                // hand-rolled K-min update.
                let expected = reference_signature_kway::<$k>(&payload, &seeds);
                assert_eq!(sig_a.slots(), &expected, "K={}", $k);
            }};
        }
        check_k!(128);
        check_k!(256);
    }

    // ----- _into sibling parity tests (audit-R8 #6b) -----------------------

    /// `signature_simd_into<K>` writes a signature bit-exactly equal to
    /// the by-value [`signature_simd`] across the K family the bench
    /// exercises. This guards audit-R8 #6b: the kernel-safe `_into`
    /// path must not drift in semantics from the legacy by-value form.
    #[test]
    fn signature_simd_into_matches_by_value() {
        macro_rules! check_k {
            ($k:literal) => {{
                let seeds: [u64; $k] =
                    core::array::from_fn(|i| 0xFEED_FACE_u64.wrapping_add(i as u64));
                let table = build_byte_table_from_seeds_boxed::<$k>(&seeds);
                let payload = random_bytes(2048);

                let by_value = signature_simd::<$k>(&payload, &table);

                let mut into = Signature::<$k>::new();
                signature_simd_into::<$k>(&payload, &table, &mut into);

                assert_eq!(into.slots(), by_value.slots(), "K={}", $k);
                assert_eq!(into.populated, by_value.populated, "K={}", $k);
            }};
        }
        check_k!(8);
        check_k!(16);
        check_k!(32);
        check_k!(64);
        check_k!(128);
        check_k!(256);
    }

    /// Empty input is a no-op for `signature_simd_into`: no slots
    /// populated, every slot stays `u64::MAX`. Reusing the buffer
    /// before the call (with stale contents) must still produce an
    /// empty signature — the `_into` form clears `out` on entry.
    #[test]
    fn signature_simd_into_empty_input_resets_buffer() {
        let seeds: [u64; 16] = core::array::from_fn(|i| (i as u64) * 0xDEAD_BEEF);
        let table = build_byte_table_from_seeds::<16>(&seeds);

        // Pre-poison the output buffer to verify the `_into` path
        // actually overwrites it.
        let mut out = signature_simd::<16>(b"poison", &table);
        assert!(!out.is_empty());

        signature_simd_into::<16>(b"", &table, &mut out);
        for slot in out.slots() {
            assert_eq!(*slot, u64::MAX);
        }
        assert!(out.is_empty());
    }

    /// Reusing a single signature buffer across multiple calls
    /// produces independent results — the per-call clear inside
    /// `signature_simd_into` discards prior contents.
    #[test]
    fn signature_simd_into_reuses_buffer_across_calls() {
        let seeds: [u64; 32] = core::array::from_fn(|i| 0xCAFE_F00D_u64.wrapping_add(i as u64));
        let table = build_byte_table_from_seeds::<32>(&seeds);

        let payload_a = random_bytes(1024);
        let payload_b = random_bytes(2048);

        let mut buf = Signature::<32>::new();

        signature_simd_into::<32>(&payload_a, &table, &mut buf);
        let snapshot_a = buf;

        signature_simd_into::<32>(&payload_b, &table, &mut buf);
        let snapshot_b = buf;

        // The second call should equal a fresh by-value run on payload_b
        // — independent of payload_a — because `_into` clears `out` on
        // entry.
        let independent_b = signature_simd::<32>(&payload_b, &table);
        assert_eq!(snapshot_b.slots(), independent_b.slots());

        // Snapshots agree on payload_a too.
        let independent_a = signature_simd::<32>(&payload_a, &table);
        assert_eq!(snapshot_a.slots(), independent_a.slots());
    }

    /// `classic_from_bytes_table_8_into` is bit-exact with the
    /// by-value [`classic_from_bytes_table_8`].
    #[test]
    fn classic_from_bytes_table_8_into_matches_by_value() {
        let seeds = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);
        let payload = b"the quick brown fox jumps over the lazy dog";

        let by_value = classic_from_bytes_table_8(payload, &table);

        let mut into = Signature::<8>::new();
        classic_from_bytes_table_8_into(payload, &table, &mut into);

        assert_eq!(into.slots(), by_value.slots());
        assert_eq!(into.populated, by_value.populated);
    }

    /// `classic_from_bytes_table_8_into` resets a stale buffer when
    /// fed an empty input.
    #[test]
    fn classic_from_bytes_table_8_into_empty_input_resets_buffer() {
        let seeds = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);

        let mut out = classic_from_bytes_table_8(b"poison", &table);
        assert!(!out.is_empty());

        classic_from_bytes_table_8_into(b"", &table, &mut out);
        for slot in out.slots() {
            assert_eq!(*slot, u64::MAX);
        }
        assert!(out.is_empty());
    }

    /// `build_byte_table_from_seeds_into<K>` produces a bit-exact
    /// match with the by-value [`build_byte_table_from_seeds`] across
    /// every documented `K` width. This is the parity guarantee for
    /// audit-R8 #6b: the kernel-safe `_into` wrapper must not drift in
    /// semantics from the legacy by-value form.
    ///
    /// At `K >= 16` the test uses heap-allocated buffers so the test
    /// itself does not reproduce the audit hazard (a 32 KiB-512 KiB
    /// table on the test stack). Two `_into` calls with the same seeds
    /// must produce bit-exact results because the kernel is
    /// deterministic.
    #[test]
    fn build_byte_table_from_seeds_into_matches_by_value() {
        // Stack-safe width (table footprint ≤ 16 KiB): compare the
        // by-value path directly against the `_into` path.
        {
            const K: usize = 8;
            let seeds: [u64; K] = core::array::from_fn(|i| 0x9E37_79B9_u64.wrapping_add(i as u64));
            let by_value = build_byte_table_from_seeds::<K>(&seeds);
            let mut into = [[0_u64; K]; kernels_gather::TABLE_ROWS];
            build_byte_table_from_seeds_into::<K>(&seeds, &mut into);
            assert_eq!(by_value, into, "K={K}");
        }

        // K >= 16: use only heap-allocated buffers so the test does
        // not put a 32 KiB-512 KiB array on the test stack.
        macro_rules! check_k_heap {
            ($k:literal) => {{
                let seeds: [u64; $k] =
                    core::array::from_fn(|i| 0x9E37_79B9_u64.wrapping_add(i as u64));
                // Use the boxed path (already proven heap-free) as the
                // reference and the `_into` form (writing into a
                // heap-allocated boxed buffer) as the candidate.
                let reference = build_byte_table_from_seeds_boxed::<$k>(&seeds);

                use core::mem::MaybeUninit;
                let mut uninit: Box<
                    MaybeUninit<[[u64; $k]; kernels_gather::TABLE_ROWS]>,
                > = Box::new_uninit();
                // SAFETY: `_into` writes every entry; the boxed
                // storage is uninitialised but every byte will be
                // overwritten before `assume_init`.
                let mut candidate = unsafe {
                    core::ptr::write_bytes(
                        uninit.as_mut_ptr().cast::<u64>(),
                        0,
                        $k * kernels_gather::TABLE_ROWS,
                    );
                    uninit.assume_init()
                };
                build_byte_table_from_seeds_into::<$k>(&seeds, &mut candidate);

                assert_eq!(*reference, *candidate, "K={}", $k);
            }};
        }
        check_k_heap!(16);
        check_k_heap!(32);
        check_k_heap!(64);
        check_k_heap!(128);
        check_k_heap!(256);
    }

    /// Trivial K=8 input (empty seeds path is structural; here we
    /// exercise the smallest meaningful K against the smallest input
    /// shape) — the `_into` overload writes a fully populated table
    /// without UB.
    #[test]
    fn build_byte_table_from_seeds_into_handles_minimal_k8() {
        let seeds: [u64; 8] = [0; 8]; // all-zero seeds is a valid corner.
        let mut table = [[0_u64; 8]; kernels_gather::TABLE_ROWS];
        build_byte_table_from_seeds_into::<8>(&seeds, &mut table);
        // Spot-check: row 0 with seed 0 is mix_word(0); row 0xFF with
        // seed 0 is mix_word(0xFF).
        for k in 0..8 {
            assert_eq!(table[0][k], crate::hash::mix_word(0));
            assert_eq!(table[0xFF][k], crate::hash::mix_word(0xFF));
        }
    }
}
