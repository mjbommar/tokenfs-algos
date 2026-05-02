//! Token n-gram inverted index — Sprint 38 (C1 demonstrator).
//!
//! Composes the two primitives that just landed in v0.2:
//!
//! * [`tokenfs_algos::bitmap`] — Roaring containers (`ArrayContainer`,
//!   `BitmapContainer`) with SIMD-accelerated `intersect`. Used to hold
//!   per-n-gram posting lists and to answer Boolean conjunctive queries.
//! * [`tokenfs_algos::bits::streamvbyte`] — Lemire & Kurz Stream-VByte
//!   variable-byte codec for `u32` streams. Used to compress delta-coded
//!   posting lists for on-disk / on-the-wire storage; decode is
//!   PSHUFB / `vqtbl1q_u8`-accelerated.
//!
//! ## What this demonstrates
//!
//! 1. **Build**: synthesise a deterministic 1000-document corpus
//!    (Zipf-ish vocab of 256 tokens × 200 tokens / doc).
//! 2. **Index construction**: for each 2-gram present in the corpus,
//!    accumulate the sorted, deduplicated set of documents that contain
//!    it. Encode each posting list two ways:
//!     - **Stream-VByte** (delta-coded `u32`s): compact wire format.
//!     - **Roaring `ArrayContainer`**: in-memory query form.
//! 3. **Verify**: every Stream-VByte round-trip equals the source posting
//!    list bit-exactly.
//! 4. **Boolean query**: pick a pair of high-frequency 2-grams and run
//!    `Container::intersect` to materialise the conjunction.
//! 5. **Bench**: print build wall time and intersection-query throughput.
//!
//! ## Run
//!
//! ```text
//! cargo run --example inverted_index
//! ```
//!
//! ## Scope
//!
//! This is a **proof-of-composability** demonstrator, not a production
//! index. Position-level postings, on-disk persistence, query rewriting,
//! and BM25-style scoring are all out of scope. The goal is the minimum
//! viable plumbing that exercises both primitives together.

#![allow(missing_docs)]

use std::time::Instant;

use tokenfs_algos::bitmap::{ArrayContainer, Container};
use tokenfs_algos::bits::{
    streamvbyte_control_len, streamvbyte_data_max_len, streamvbyte_decode_u32,
    streamvbyte_encode_u32,
};

/// Number of synthetic documents in the corpus.
const NUM_DOCS: u32 = 1_000;

/// Tokens per document.
const TOKENS_PER_DOC: usize = 200;

/// Vocabulary size (token id range is `0..VOCAB_SIZE`).
///
/// Picked so `(VOCAB_SIZE - 1) * VOCAB_SIZE + (VOCAB_SIZE - 1) <= u16::MAX`,
/// i.e. every 2-gram identifier `a * VOCAB_SIZE + b` fits in a `u16`
/// and slots cleanly into the Roaring 16-bit low-key space.
const VOCAB_SIZE: u32 = 256;

/// PRNG seed — fix for run-to-run determinism.
const RNG_SEED: u64 = 0x0005_EEDC_0FFE_EF38_u64;

/// Number of independent intersection queries to time during the bench.
const NUM_QUERIES: usize = 256;

/// Tiny xorshift64* PRNG. Keeps the example dependency-free.
///
/// State must be non-zero. Wikipedia's reference parameters; period
/// `2^64 - 1`.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Seeds the PRNG; force the state non-zero.
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15_u64
        } else {
            seed
        };
        Self { state }
    }

    /// Returns a uniform `u64`.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// Draws a token id with a Zipf-ish skew so a handful of tokens dominate.
///
/// We approximate Zipf with `floor(VOCAB_SIZE * u^2)` for `u ~ Uniform[0,1)`,
/// which biases mass toward small token ids. Cheap, deterministic, no
/// `f64::powf` and no extra deps.
fn draw_token(rng: &mut Xorshift64) -> u32 {
    // Scale the high 24 bits of a fresh u64 into [0, 2^24) → [0, 1) in
    // double precision; square; multiply by the vocab.
    let u_bits = rng.next_u64() >> 40; // 24 bits
    let u = (u_bits as f64) / ((1_u64 << 24) as f64);
    let scaled = (u * u) * (VOCAB_SIZE as f64);
    let id = scaled as u32;
    id.min(VOCAB_SIZE - 1)
}

/// Encodes a `(token_a, token_b)` 2-gram into a single `u16` key.
///
/// Layout: high byte = `token_a`, low byte = `token_b`. Fits the Roaring
/// 16-bit low-key space exactly when `VOCAB_SIZE == 256`.
#[inline]
fn bigram_key(a: u32, b: u32) -> u16 {
    debug_assert!(a < VOCAB_SIZE && b < VOCAB_SIZE);
    ((a << 8) | b) as u16
}

/// Builds the corpus: a `Vec<Vec<u32>>` of token ids, one inner vector
/// per document.
fn build_corpus(rng: &mut Xorshift64) -> Vec<Vec<u32>> {
    let mut corpus = Vec::with_capacity(NUM_DOCS as usize);
    for _doc in 0..NUM_DOCS {
        let mut doc = Vec::with_capacity(TOKENS_PER_DOC);
        for _ in 0..TOKENS_PER_DOC {
            doc.push(draw_token(rng));
        }
        corpus.push(doc);
    }
    corpus
}

/// Builds the inverted index from a corpus of token streams.
///
/// Returns a `Vec<Option<Vec<u16>>>` indexed by `bigram_key(a, b)` —
/// `Some(sorted_doc_ids)` when at least one document contains the
/// 2-gram, else `None`. Doc ids are unique and ascending so they slot
/// straight into `ArrayContainer::from_sorted` without resorting.
fn build_index(corpus: &[Vec<u32>]) -> Vec<Option<Vec<u16>>> {
    // Use `BTreeSet` per bigram to get sorted, deduplicated doc ids in
    // one pass. The whole table is `VOCAB_SIZE * VOCAB_SIZE = 65 536`
    // entries, most empty when the vocab is skewed.
    use std::collections::BTreeSet;
    let table_len = (VOCAB_SIZE * VOCAB_SIZE) as usize;
    let mut sets: Vec<BTreeSet<u16>> = (0..table_len).map(|_| BTreeSet::new()).collect();

    for (doc_id, doc) in corpus.iter().enumerate() {
        let doc_id_u16 = u16::try_from(doc_id).expect("NUM_DOCS fits in u16");
        // Slide a 2-gram window over each document.
        for window in doc.windows(2) {
            let key = bigram_key(window[0], window[1]) as usize;
            sets[key].insert(doc_id_u16);
        }
    }

    sets.into_iter()
        .map(|s| {
            if s.is_empty() {
                None
            } else {
                Some(s.into_iter().collect())
            }
        })
        .collect()
}

/// Stream-VByte payload pair for one posting list.
///
/// `n` is the element count (length-prefixed by the container; the
/// streams themselves do not encode it). `control` and `data` are sized
/// per [`streamvbyte_control_len`] / [`streamvbyte_data_max_len`].
struct VbyteList {
    n: usize,
    control: Vec<u8>,
    data: Vec<u8>,
}

impl VbyteList {
    /// Total on-the-wire byte cost of this posting list.
    fn byte_len(&self) -> usize {
        self.control.len() + self.data.len()
    }
}

/// Delta-codes a sorted, deduplicated `&[u16]` posting list and encodes
/// the resulting `u32` gap stream with Stream-VByte.
///
/// First element is emitted verbatim; subsequent elements are stored as
/// the gap from the previous element. Small gaps compress to one byte
/// each in the codec — this is the canonical posting-list trick.
fn encode_postings(postings: &[u16]) -> VbyteList {
    let mut deltas = Vec::with_capacity(postings.len());
    let mut prev: u32 = 0;
    for (i, &v) in postings.iter().enumerate() {
        let v = u32::from(v);
        deltas.push(if i == 0 { v } else { v - prev });
        prev = v;
    }
    let n = deltas.len();
    let mut control = vec![0_u8; streamvbyte_control_len(n)];
    let mut data = vec![0_u8; streamvbyte_data_max_len(n)];
    let written = streamvbyte_encode_u32(&deltas, &mut control, &mut data);
    data.truncate(written);
    VbyteList { n, control, data }
}

/// Decodes a Stream-VByte payload and undoes the delta-coding to
/// recover the original sorted `Vec<u16>`.
fn decode_postings(list: &VbyteList) -> Vec<u16> {
    let mut deltas = vec![0_u32; list.n];
    streamvbyte_decode_u32(&list.control, &list.data, list.n, &mut deltas);
    let mut out = Vec::with_capacity(list.n);
    let mut acc: u32 = 0;
    for (i, &d) in deltas.iter().enumerate() {
        acc = if i == 0 { d } else { acc + d };
        out.push(u16::try_from(acc).expect("posting fits in u16 by construction"));
    }
    out
}

/// Picks `count` of the most populous bigrams as query candidates,
/// returned as `(key, cardinality)` pairs sorted by descending
/// cardinality. Deterministic given the same index.
fn pick_hot_bigrams(index: &[Option<Vec<u16>>], count: usize) -> Vec<(u16, usize)> {
    let mut hot: Vec<(u16, usize)> = index
        .iter()
        .enumerate()
        .filter_map(|(idx, slot)| {
            slot.as_ref()
                .map(|p| (u16::try_from(idx).expect("vocab^2 fits in u16"), p.len()))
        })
        .collect();
    hot.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    hot.truncate(count);
    hot
}

fn main() {
    let mut rng = Xorshift64::new(RNG_SEED);

    println!("== Sprint 38 — token n-gram inverted index demonstrator ==");
    println!("corpus           = {NUM_DOCS} docs × {TOKENS_PER_DOC} tokens (vocab = {VOCAB_SIZE})");

    // 1. Build corpus.
    let corpus_t0 = Instant::now();
    let corpus = build_corpus(&mut rng);
    let corpus_elapsed = corpus_t0.elapsed();
    let total_tokens: usize = corpus.iter().map(Vec::len).sum();
    println!(
        "corpus build     = {:>9.3} ms  ({} tokens)",
        corpus_elapsed.as_secs_f64() * 1e3,
        total_tokens
    );

    // 2. Build inverted index — 2-grams over each document.
    let index_t0 = Instant::now();
    let index = build_index(&corpus);
    let index_elapsed = index_t0.elapsed();
    let nonempty: usize = index.iter().filter(|slot| slot.is_some()).count();
    let total_postings: usize = index
        .iter()
        .filter_map(|slot| slot.as_ref().map(Vec::len))
        .sum();
    println!(
        "index build      = {:>9.3} ms  ({nonempty} bigrams, {total_postings} postings)",
        index_elapsed.as_secs_f64() * 1e3
    );

    // 3. Encode each posting list with Stream-VByte. Verify round-trip.
    //    Stream-VByte targets `u32` streams; for our `u16` doc ids the
    //    win is concentrated in dense lists where average gap ≤ 255 →
    //    1 byte / value (vs 2 bytes raw). We report total + dense-only
    //    compression to make the trade-off legible.
    const DENSE_THRESHOLD: usize = 16;
    let svb_t0 = Instant::now();
    let mut vbyte: Vec<Option<VbyteList>> = Vec::with_capacity(index.len());
    let mut svb_bytes_total: usize = 0;
    let mut svb_bytes_dense: usize = 0;
    let mut raw_bytes_dense: usize = 0;
    let mut mismatches: usize = 0;
    for slot in &index {
        match slot {
            None => vbyte.push(None),
            Some(postings) => {
                let encoded = encode_postings(postings);
                svb_bytes_total += encoded.byte_len();
                if postings.len() >= DENSE_THRESHOLD {
                    svb_bytes_dense += encoded.byte_len();
                    raw_bytes_dense += postings.len() * std::mem::size_of::<u16>();
                }
                let decoded = decode_postings(&encoded);
                if &decoded != postings {
                    mismatches += 1;
                }
                vbyte.push(Some(encoded));
            }
        }
    }
    let svb_elapsed = svb_t0.elapsed();
    assert_eq!(mismatches, 0, "stream-vbyte round-trip diverged");
    let raw_bytes_total = total_postings * std::mem::size_of::<u16>();
    println!(
        "vbyte encode     = {:>9.3} ms  ({svb_bytes_total} B vs {raw_bytes_total} B raw u16, {:.2}x overall)",
        svb_elapsed.as_secs_f64() * 1e3,
        raw_bytes_total as f64 / svb_bytes_total.max(1) as f64
    );
    println!(
        "vbyte dense only = {svb_bytes_dense} B vs {raw_bytes_dense} B raw u16 ({:.2}x for cardinality >= {DENSE_THRESHOLD})",
        raw_bytes_dense as f64 / svb_bytes_dense.max(1) as f64
    );

    // 4. Convert each posting list to a Roaring ArrayContainer
    //    (`< ARRAY_MAX_CARDINALITY` = 4096; our doc ids < 1000).
    let roar_t0 = Instant::now();
    let mut containers: Vec<Option<Container>> = Vec::with_capacity(index.len());
    for slot in &index {
        match slot {
            None => containers.push(None),
            Some(postings) => {
                let arr = ArrayContainer::from_sorted(postings.clone());
                containers.push(Some(Container::Array(arr)));
            }
        }
    }
    let roar_elapsed = roar_t0.elapsed();
    println!(
        "roaring build    = {:>9.3} ms",
        roar_elapsed.as_secs_f64() * 1e3
    );

    // 5. Pick the top-K hottest bigrams as query operands and time
    //    pairwise intersections.
    let hot = pick_hot_bigrams(&index, NUM_QUERIES + 1);
    if hot.len() < 2 {
        println!("not enough non-empty bigrams to run intersections; bailing");
        return;
    }

    // Demonstrate one intersection in detail before the bench loop.
    let (key_a, card_a) = hot[0];
    let (key_b, card_b) = hot[1];
    let cont_a = containers[key_a as usize]
        .as_ref()
        .expect("hot bigram has postings");
    let cont_b = containers[key_b as usize]
        .as_ref()
        .expect("hot bigram has postings");
    let example_t0 = Instant::now();
    let example_result = cont_a.intersect(cont_b);
    let example_elapsed = example_t0.elapsed();
    let example_card = match &example_result {
        Container::Array(a) => a.cardinality(),
        Container::Bitmap(b) => b.cardinality(),
        Container::Run(r) => r.cardinality(),
    };
    println!(
        "example query    = bigrams 0x{key_a:04x} ({card_a}) AND 0x{key_b:04x} ({card_b}) → {example_card} docs in {:.3} us",
        example_elapsed.as_secs_f64() * 1e6
    );

    // Bench loop: pair adjacent hot bigrams, sum cardinalities to keep
    // the optimiser honest.
    let bench_t0 = Instant::now();
    let mut card_sum: u64 = 0;
    let mut pairs_run: usize = 0;
    for window in hot.windows(2).take(NUM_QUERIES) {
        let a = &containers[window[0].0 as usize]
            .as_ref()
            .expect("hot bigram has postings");
        let b = &containers[window[1].0 as usize]
            .as_ref()
            .expect("hot bigram has postings");
        card_sum += u64::from(a.intersect_cardinality(b));
        pairs_run += 1;
    }
    let bench_elapsed = bench_t0.elapsed();
    let per_query_us = (bench_elapsed.as_secs_f64() * 1e6) / pairs_run.max(1) as f64;
    println!(
        "intersect bench  = {:>9.3} ms total ({pairs_run} queries, {per_query_us:.3} us / query, sum_card = {card_sum})",
        bench_elapsed.as_secs_f64() * 1e3
    );

    // 6. Sanity: pick a random doc and verify the example intersection
    //    contains only doc ids that mention BOTH bigrams.
    let intersect_doc_ids: Vec<u16> = match example_result {
        // External callers no longer have `pub` access to the raw `data`
        // field; `data()` returns an immutable slice that we clone into a
        // fresh owned `Vec<u16>` for downstream consumption.
        Container::Array(a) => a.data().to_vec(),
        Container::Bitmap(b) => b.to_array(),
        Container::Run(_r) => Vec::new(), // not exercised by this corpus
    };
    let raw_a = index[key_a as usize]
        .as_ref()
        .expect("hot bigram has postings");
    let raw_b = index[key_b as usize]
        .as_ref()
        .expect("hot bigram has postings");
    for doc_id in &intersect_doc_ids {
        assert!(
            raw_a.binary_search(doc_id).is_ok(),
            "intersect leaked a doc not in posting A"
        );
        assert!(
            raw_b.binary_search(doc_id).is_ok(),
            "intersect leaked a doc not in posting B"
        );
    }
    println!(
        "verify           = ok ({} docs in conjunction)",
        intersect_doc_ids.len()
    );
}
