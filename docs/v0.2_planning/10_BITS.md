# `bits` module — bit-level primitive surface

**Status:** spec, 2026-05-02. Phase A1/A2 + B1/B2 of `01_PHASES.md`.

This is the highest-leverage new module in v0.2: it has three orthogonal consumers (posting lists, token streams, succinct DS) and one shared SIMD kernel family.

## Goal & scope

Three primitive families under one module:

1. **§ 4 `popcount`** — bit-count over `&[u64]` and `&[u8]` slices. Foundation for `rank_select` and for `bitmap::roaring` cardinality.
2. **§ 2 `bit_pack` / `bit_unpack`** — pack/unpack arbitrary widths (1-32 bits) of integers into byte streams. Used by token-stream encoding, fingerprint sidecar quantization, fast packed-integer columnar layouts.
3. **§ 3 `streamvbyte`** — Lemire & Kurz variable-byte codec. Used by posting-list payloads, delta-coded integer streams.
4. **§ 5 `rank_select`** — bit-rank/bit-select dictionary. Foundation for wavelet trees, FM-index, succinct sequences. Tier B.

All four share `popcount` as the inner kernel; popcount is built first.

## § 1 Module surface

```
bits/
├── mod.rs               // public API
├── popcount.rs          // §4: scalar + SIMD popcount
├── bit_pack.rs          // §2: pack/unpack at arbitrary widths
├── streamvbyte.rs       // §3: encode + decode
├── rank_select.rs       // §5: rank/select dictionary
└── kernels/
    ├── popcount_avx2.rs    // VPSHUFB nibble-LUT path
    ├── popcount_avx512.rs  // VPOPCNTQ path
    ├── popcount_neon.rs    // VCNT + horizontal add
    ├── streamvbyte_ssse3.rs
    ├── streamvbyte_avx2.rs
    └── streamvbyte_neon.rs
```

Public types:

```rust
pub mod bits {
    pub fn popcount_u64_slice(words: &[u64]) -> u64;
    pub fn popcount_u8_slice(bytes: &[u8]) -> u64;

    pub struct BitPacker<const W: u32>;     // const-width fast path
    pub struct DynamicBitPacker { width: u32 }
    impl BitPacker<W> { /* encode_u32_slice, decode_u32_slice */ }

    pub fn streamvbyte_encode_u32(values: &[u32], control_out: &mut [u8], data_out: &mut [u8]) -> usize;
    pub fn streamvbyte_decode_u32(control: &[u8], data: &[u8], n: usize, out: &mut [u32]) -> usize;

    pub struct RankSelectDict<'a> { /* … */ }
    impl<'a> RankSelectDict<'a> {
        pub fn build(bits: &'a [u64], n_bits: usize) -> Self;
        pub fn rank1(&self, i: usize) -> usize;     // # of 1s strictly before position i
        pub fn select1(&self, k: usize) -> Option<usize>;  // position of (k+1)-th 1
    }
}
```

## § 2 Bit-pack / bit-unpack

### Algorithm

Given values `v[0], v[1], …, v[N-1]` each fitting in `W` bits (`1 ≤ W ≤ 32`), pack them into `ceil(N*W/8)` bytes such that `v[i]` occupies bits `[i*W, (i+1)*W)` of the bit stream. Bits within a byte are little-endian (lowest bit first); bytes are written in order.

**Encode:** straightforward shift-and-OR. SIMD wins are modest because the per-element work is just shifts and an OR; scalar already gets ~2-3 GB/s.

**Decode:** the hot kernel. Two regimes:

1. **W ≤ 8:** decode 16-32 elements per SIMD iteration via VPSHUFB-based bit-extraction. Process one or two 64-bit words per iteration, broadcast, shift each lane by a per-lane amount via VPSRLVQ (AVX2) or VPSLLVD/VPSRLVD lane shifts (AVX2), mask. ~5-8 GB/s on AVX2.
2. **8 < W ≤ 32:** decode 4-8 elements per iteration. Each output u32 spans 1-5 input bytes; do unaligned 64-bit load, shift, mask. AVX-512 VPMULTISHIFTQB (VBMI) gives one-shot extraction for any width up to 64.

### Special cases

- **W ∈ {1, 2, 4, 8, 16, 32}** byte-aligned widths: degenerate to memcpy / byte-cast. Hard-coded fast paths.
- **W = 11, 12** (typical token widths): the canonical hot case for token decode. Specialize.
- **W > 32:** use 64-bit element variant (separate `BitPacker64`).

### API choices

- **Const-generic `BitPacker<const W: u32>`**: gives the compiler the width at monomorphization, allowing perfect specialization. Use this when width is known at the call site (token decoder for a fixed vocabulary).
- **`DynamicBitPacker`**: dispatches over width at runtime via match. Use when the width is a per-image config.
- **Both share** the same SIMD kernels under the hood; only the dispatch differs.

### Hardware acceleration plan

| Backend | Approach | Expected throughput (in-L1) |
|---|---|---|
| Scalar | shift-and-OR loop | ~2-3 GB/s |
| AVX2 (W ≤ 8) | VPSHUFB nibble-shuffle into u8 lanes | ~6-8 GB/s |
| AVX2 (8 < W ≤ 32) | VPSRLVD per-lane shifts; mask | ~4-6 GB/s |
| AVX-512 | VPMULTISHIFTQB for one-shot extraction up to W=8 (and chained for higher W) | ~10-15 GB/s |
| NEON | TBL-based byte permute + shifts | ~4-6 GB/s on M2; ~2-4 on Graviton |
| SVE2 | TBL + variable-length predicated ops | not pursued in v0.2 (deferred per existing convention) |

### Test plan

- Scalar oracle is the reference.
- Property test: for every W ∈ {1..32}, every length N ∈ {0, 1, 7, 8, 33, 1024}, `decode(encode(v)) == v` for random `v` clamped to `W` bits.
- SIMD parity: each backend matches scalar bit-exactly.
- Edge cases: N % SIMD_BLOCK_SIZE != 0 (tail handling); writing into mis-aligned buffers; W spanning byte boundaries weirdly.

### Bench plan

- Per W ∈ {1, 4, 8, 11, 12, 16, 32} for both encode and decode.
- Three sizes: 1 KB (L1), 1 MB (L2), 32 MB (L3+DRAM).
- Reported: throughput in GB/s, ns/element, and ratio vs scalar.

## § 3 Stream-VByte codec

### Algorithm (Lemire & Kurz 2017)

Per group of 4 u32s, emit one **control byte** holding four 2-bit codes — code `cc` means the integer takes `cc+1` little-endian bytes (00→1B, 01→2B, 10→3B, 11→4B). The first 2-bit word is the **least** significant bits of the control byte.

All control bytes go into one stream of `ceil(2N/8)` = `ceil(N/4)` bytes. All data bytes go into a separate stream of `Σ_i len_i` bytes. Total size: `ceil(N/4) + Σ len_i`.

### Decode (the hot kernel)

For each control byte `c`:
1. Look up `lengthTable[c]` (precomputed u8, 256 entries) → number of data bytes `C` consumed (4 ≤ C ≤ 16).
2. Load 16 data bytes `Data` (overshoot OK if buffer padded ≥16B at tail).
3. Look up `shuffleTable[c]` (precomputed [u8; 16], 256 entries) → 4 KiB total.
4. `Data = _mm_shuffle_epi8(Data, shuffle)` — one PSHUFB (SSSE3). Output is 4 u32s.
5. Advance data pointer by `C`.

AVX2 dual-pumps two 128-bit lanes (two control bytes per iteration). NEON uses `vqtbl1q_u8` (TBL) on the same 16-byte lookups.

### Encode

Per group of 4 u32s: compute byte length per int (`max(1, (32 - lzcnt(v) + 7) / 8)` or via a small lookup), pack 4 lengths into 2 bits each → one control byte, write each int's `len_i` low bytes. SIMD encode less impressive than decode; scalar ~1 GB/s is fine for ingest.

### Throughput

Lemire/Kurz Haswell numbers: decode 1.1-4.0 billion u32/s = ~4-16 GB/s of decoded ints (raw byte rate ~1-3 GB/s in, ~4-16 GB/s out). Encode ~1 GB/s.

**No upstream AVX-512 variant.** Lemire hasn't published one. Worth experimenting in v0.3+ but not Phase B work.

### Edge cases

- **N % 4 != 0**: the canonical move is to encode the count `N` out of band (in your container header), pad the final group to 4 elements with `00` codes (1 byte each), and ignore the trailing decoded values past `N` on read. Alternative: scalar tail decoder.
- **No terminating control byte**: format is length-prefixed by `N`; the consumer must know `N`.
- **Buffer padding**: SIMD loop reads 16 data bytes for any control byte. Last group must either have ≥16 bytes of trailing slack OR fall back to scalar for the last group when fewer than 16 valid data bytes remain.
- **Width**: upstream paper is u32 only. Variants exist (`streamvbyte_0124` codes mean 0/1/2/4 bytes; zigzag for signed; `streamvbyte64` Rust crate). v0.2 ships **u32 only**. u64 is added in v0.3 if a consumer asks.

### Hardware acceleration plan

| Backend | Approach | Expected throughput (decode, in-L1) |
|---|---|---|
| Scalar | byte-by-byte | ~250-500 MB/s |
| SSSE3+SSE4.1 | PSHUFB lookup table | ~3-5 GB/s |
| AVX2 | dual-pumped PSHUFB | ~6-10 GB/s |
| AVX-512 | (deferred — VBMI VPERMB experiment, not Phase B) | -- |
| NEON | TBL on 16-byte lookups | ~3-5 GB/s |

### Existing Rust crates

- **`stream-vbyte` v0.4.1** (Marshall Pierce, last release 2023-05): SSE4.1 encode + SSSE3 decode behind nightly `target_feature`; u32 only; scalar fallback on stable; **no AVX2, no NEON**. Stable but unmaintained.
- **`streamvbyte64` v0.2.0** (mccullocht, 2023-07): adds u16/u64 width variants. Small user base.
- **`bazhenov/svbyte`**, **`mpetri/streamvbyte`**: experiments / FFI to C; not active.

### Vendor decision

**Ship our own pure-Rust SIMD implementation in `bits::streamvbyte`.** Rationale:

1. **Dispatch coverage**: we want `dispatch::` runtime backend selection (SSSE3 / AVX2 / NEON / scalar). The existing crates pin to a single backend behind nightly `target_feature`.
2. **NEON parity**: AArch64 deployment is a target (Apple Silicon, Graviton). No existing crate ships NEON.
3. **Test integration**: the property/parity test pattern across the crate (per `PRIMITIVE_CONTRACTS.md`) needs to apply uniformly.
4. **Maintenance**: the upstream Rust crates haven't accepted contributions in 2 years.
5. **Surface compatibility**: the encoded format is interchangeable across implementations (Lemire's spec). Consumers using `stream-vbyte` for IO can read/write our format and vice versa — no lock-in either way.

**Not** vendoring upstream code. The crates are MIT/Apache-licensed and we could vendor cleanly, but pure-Rust reimplementation against Lemire's paper + C reference is straightforward and lets us match our existing dispatch / test conventions exactly.

### Reference implementations

- C: https://github.com/lemire/streamvbyte (canonical)
- Paper: Lemire, Kurz, Rupp, "Stream VByte: Faster Byte-Oriented Integer Compression," IPL 2017. arXiv:1709.08990

### Test plan

- Scalar oracle.
- Round-trip property test: `decode(encode(v)) == v` for all N, random u32s.
- Parity vs scalar on each backend, bit-exact.
- Tail-handling: N mod 4 = {0, 1, 2, 3}.
- 256-entry shuffle table validation: regenerate at build time, hash, compare to known-good.

### Bench plan

- Encode + decode at sizes 256, 1K, 16K, 256K, 4M elements.
- Report throughput in elements/sec AND in bytes/sec on the encoded representation.
- Compare against scalar AND against the `stream-vbyte` crate (where it builds — its target_feature path is nightly-only) as a sanity check.

## § 4 Popcount (foundation kernel)

### Algorithm

For dense `&[u64]` slices: AVX-512 `VPOPCNTQ` is one cycle per 64-bit lane. AVX2 has no native 64-bit popcount; use Mula's nibble-LUT method:

```
shuffle = LUT[16]   // popcount of each 4-bit nibble
result_lane = pshufb(shuffle, low_nibbles) + pshufb(shuffle, high_nibbles)
```

Unrolled over 256-byte blocks, this hits ~5 GB/s. NEON has `VCNT` per byte + horizontal add.

For `&[u8]` slices: same kernel, each u8 contributes its own popcount.

### API

```rust
pub fn popcount_u64_slice(words: &[u64]) -> u64;
pub fn popcount_u8_slice(bytes: &[u8]) -> u64;
```

### Hardware acceleration plan

| Backend | Approach | Throughput |
|---|---|---|
| Scalar | `u64::count_ones()` per word | ~3 GB/s |
| AVX2 | Mula nibble-LUT | ~5 GB/s |
| AVX-512 | VPOPCNTQ | ~30-50 GB/s |
| NEON | VCNT + reduce | ~5-8 GB/s |

The 10x AVX-512 jump is *the* canonical AVX-512 win. This kernel alone justifies dispatch infrastructure being aware of AVX-512 vs AVX2.

### Test plan

- Scalar oracle.
- Parity tests vs scalar, bit-exact.
- Edge cases: empty slice, length 1, length 31 (sub-block tail), aligned vs unaligned.

### Bench plan

- Sizes 1 KB, 1 MB, 256 MB.
- AVX-512 vs AVX2 vs scalar vs NEON.

## § 5 Rank/select dictionary (Phase B — **largest single-piece work in v0.2**)

**Honest budget: 1-2 weeks of focused implementation, not "a few days."** The naive scalar version is short, but a fast rank/select dictionary that's competitive with `sucds` / `sdsl-lite` requires real engineering: the multi-level superblock + block + per-block sample tables, careful tuning of block sizes against the target cache hierarchy, optional RRR (Raman/Raman/Rao) compression for sparse bitvectors, and Vigna's broadword tricks for the scalar inner kernels. This module gates the wavelet-tree and FM-index Tier-D items, so getting the foundation right matters.

### Algorithm overview

Given a `&[u64]` representing `n_bits` bits, build an index supporting:
- `rank1(i)` = number of 1-bits in bits `[0, i)`. Constant-time per query.
- `select1(k)` = position of the (k+1)-th 1-bit. Constant-time per query (with sampling tables).

Two levels of sampling:

1. **Superblock counts** (`u32` per superblock of 4096 bits): cumulative popcount-up-to-here. ~0.1% overhead. Two cache-line bound for any rank query (the array entry + the bits themselves).
2. **Block counts** (`u16` per block of 256 bits, relative to the containing superblock): partial popcount within the superblock. ~0.6% overhead.

For rank: look up superblock count + block count + popcount the partial word. Three array accesses, one popcount, ~10-15 ns warm.

For select: binary search through superblock counts (≤ log levels = ~20 cache misses for 1B-bit bitmaps), then within-block scan using **Vigna's broadword select** (Vigna 2008, *Broadword Implementation of Rank/Select Queries*) at the u64 level. Total ~30-50 ns per select.

### Variants worth knowing

- **Plain bitvector (this spec)**: ~0.7% space overhead, fast rank, fast select. The default.
- **RRR (Raman/Raman/Rao 2007)**: compresses sparse bitvectors near information-theoretic minimum. Useful when bitvector is e.g. 1% dense or 99% dense. ~5-10x slower rank but huge space savings.
- **SDArray (Okanohara/Sadakane 2007)**: stores positions of 1-bits as compressed integer sequence with select sampling. Fastest select, slowest rank.

**v0.2 ships plain bitvector only.** RRR and SDArray are deferred to v0.3+ if a sparse-bitvector consumer asks (the `sucds` crate is the natural dependency in the meantime).

### Reference implementations

- **sdsl-lite** (C++): https://github.com/simongog/sdsl-lite — canonical academic reference. Multiple variants.
- **sucds** (Rust): https://github.com/kampersanda/sucds — pure-Rust succinct DS crate. **Use as parity oracle.** v0.2 implements its own to capture SIMD acceleration opportunity (AVX-512 VPOPCNTQ for rank, parallel scan for select); v0.3 may depend on `sucds` for higher-level structures (wavelet tree, SDArray).

### Vendor decision

**Ship our own plain-bitvector implementation.** Rationale:
- We want SIMD acceleration on the rank/select inner loops (`VPOPCNTQ` + parallel select scan).
- We want `dispatch::` runtime backend selection.
- We want kernel-safety (caller-provided output Vec for build).
- `sucds` is mature but doesn't expose SIMD-accelerated kernels at the rank/select level.

For v0.3 wavelet tree / FM-index work, **probably depend on `sucds`** rather than reimplementing the higher-level structures — those are ~10× our complexity budget and `sucds` is well-tested. The seam is: our `bits::rank_select` for the hot inner loop; `sucds` for the structure on top, if/when we ship one.

### Space-time tradeoff

Block sizes parameterize space vs query latency. For TokenFS scales (millions to tens of millions of bits per index, say 10-100 MB total), 256-bit blocks + 4096-bit superblocks give ~0.7% overhead and warm-cache rank latency ~15 ns. Larger superblocks reduce overhead but add cache pressure on the lookup; smaller blocks reduce within-block scan time but cost more space.

For Postgres / MinIO scales (billions of bits), bumping superblock to 65536 bits keeps the count-array L2-resident and gives ~0.1% overhead, at the cost of an additional cache miss per rank.

### API

```rust
pub struct RankSelectDict<'a> {
    bits: &'a [u64],
    n_bits: usize,
    superblock_counts: Vec<u32>,     // every 4096 bits
    block_counts: Vec<u16>,          // every 256 bits
}

impl<'a> RankSelectDict<'a> {
    pub fn build(bits: &'a [u64], n_bits: usize) -> Self;
    pub fn rank1(&self, i: usize) -> usize;
    pub fn rank0(&self, i: usize) -> usize { i - self.rank1(i) }
    pub fn select1(&self, k: usize) -> Option<usize>;
    pub fn select0(&self, k: usize) -> Option<usize>;
    pub fn memory_bytes(&self) -> usize;
}
```

### Hardware acceleration

- **Rank1**: dominated by superblock + block count lookup (2 cache lines) plus one popcount of partial 64-bit word. Bandwidth-tiny per query (~16 bytes). Latency-bound on the 2 cache misses. SIMD doesn't help individual queries; **batch rank** (rank K positions at once) is a real SIMD win.
- **Select1**: one binary search through superblock counts (log levels, 1 cache miss each) + popcount-driven scan within the 256-bit block. AVX-512 VPOPCNTQ accelerates the in-block scan.

### Reference implementations

- **sdsl-lite** (C++) — the canonical reference. https://github.com/simongog/sdsl-lite
- **sucds** (Rust) — https://github.com/kampersanda/sucds — already-existing pure-Rust succinct DS crate. Vendoring vs depending: probably depend on it for the higher-level structures (wavelet tree, SDArray) in v0.3; for v0.2 we ship a clean, simple `RankSelectDict` ourselves so the SIMD acceleration opportunity is captured.

### Test plan

- Scalar oracle.
- Property test: `rank1(0) == 0`; `rank1(n_bits) == popcount`; `select1(rank1(i)-1) <= i if bits[i]==1`.
- Edge cases: all-zero bitvector, all-one, sparse (1 bit set), dense (1 bit clear).

### Bench plan

- Build cost (one-shot): per million bits.
- Per-query latency: rank, select; warm-cache and cold-cache.
- Batch rank-K and select-K throughput.

## § 6 Open questions

1. **Should `bit_pack` accept SIMD-aligned input/output buffers as a precondition, or alignment-agnostic?** Alignment-agnostic loses ~5-10% but is cleaner API. **Tentative: alignment-agnostic for v0.2; alignment-required variant added in v0.3 if benches show the gap matters.**

2. **AVX-512 VBMI VPERMB for Stream-VByte**: building a 65536-entry shuffle table for two control bytes at once is theoretically faster but the table is 4 MiB and likely cache-unfriendly. **Tentative: skip in v0.2; revisit if AVX-512 becomes the dominant target.**

3. **u11/u12 specializations in `bit_pack`**: hard-code these two widths since they're the canonical token widths. **Tentative: yes — auto-generate via macro or const generics for W ∈ {8, 11, 12, 16}; let other widths fall to the generic path.**

4. **`RankSelectDict` ownership**: should it own the bit slice or borrow? If it borrows, `tokenfs-paper` can mmap a sealed image's bit data and build the index over it. **Tentative: borrow (lifetime parameter); add `RankSelectDictOwned` wrapper if a consumer needs ownership.**

5. **SVE/SVE2 backends**: NEON-equivalent kernels are written; SVE/SVE2 backends could vectorize-and-go on Graviton 3+. **Tentative: defer to v0.3 like other modules.**

## § 7 Environment fitness

Per [`02b_DEPLOYMENT_MATRIX.md`](02b_DEPLOYMENT_MATRIX.md):

| API | Kernel module | FUSE | Userspace | Postgres ext | cgo (Go) | Python (PyO3) |
|---|---|---|---|---|---|---|
| `popcount_u64_slice` / `popcount_u8_slice` | ✅ | ✅ | ✅ | ✅ | ⚠️ wrap as batch | ✅ |
| `BitPacker<W>::encode/decode` | ✅ | ✅ | ✅ | ✅ | ✅ batched | ✅ |
| `streamvbyte_encode_u32` / `_decode_u32` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `RankSelectDict::build` | ⚠️ uses Vec — needs `kernel-alloc` flag | ✅ | ✅ | ⚠️ palloc-aware variant | ✅ | ✅ |
| `RankSelectDict::rank1` / `select1` | ✅ once built | ✅ | ✅ | ✅ | ⚠️ batch wrapper | ✅ |

**Notes:**
- All kernels are stateless or operate on borrowed slices. The 4 KiB Stream-VByte shuffle table and 16 B popcount nibble LUT are static rodata.
- `RankSelectDict::build` is the only allocating API; for kernel use, `tokenfs-algos` should expose a `build_into(bits, n_bits, out_superblocks: &mut Vec<u32>, out_blocks: &mut Vec<u16>)` variant where the caller provides the output Vec (Rust-for-Linux's flag-bearing Vec wrapper).
- Single-element popcount via cgo has ~200 ns overhead per call vs ~few-ns kernel; **always wrap as batch (`popcount_u64_batch(slices: &[&[u64]])`) at the cgo boundary**.
- Rank/select queries are individually sub-µs and fit any consumer. Build cost is one-shot — typically run by the writer/build-pipeline consumer (userspace), not the reader.
