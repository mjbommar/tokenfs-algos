//! Fuzz target: bitmap intersect kernels must agree bit-for-bit with
//! the scalar oracle. Covers both array×array (Schlegel SSE4.2 PCMPESTRM)
//! and bitmap×bitmap AND (AVX-512 / AVX2 / NEON over the 1024-word bitmap).
//!
//! Input layout — the first byte selects the variant:
//! - mode 0: array × array. Bytes 1..3 (LE u16) → cardinality split point.
//!   We split the remaining bytes into two streams of u16 values, sort
//!   each, deduplicate, cap at 4096 (ARRAY_MAX_CARDINALITY), and intersect.
//! - mode 1: bitmap × bitmap. Remaining bytes seed both 1024-word u64
//!   bitmaps via a deterministic split; we then run the AND in three
//!   variants (`_card`, `_nocard`, `_justcard`) and assert they agree
//!   with each other and with the scalar reference.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tokenfs_algos::bitmap::{ArrayContainer, BitmapContainer, Container};

/// Roaring's array container threshold; sorted u16 vec must stay at or
/// below this length.
const ARRAY_MAX: usize = 4096;

/// 1024 u64 = 8 KiB exactly; matches BitmapContainer storage layout.
const BITMAP_WORDS: usize = 1024;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let mode = data[0] & 1;
    let payload = &data[1..];

    if mode == 0 {
        run_array_x_array(payload);
    } else {
        run_bitmap_x_bitmap(payload);
    }
});

fn run_array_x_array(payload: &[u8]) {
    if payload.len() < 2 {
        return;
    }
    let split_raw = u16::from_le_bytes([payload[0], payload[1]]) as usize;
    let body = &payload[2..];
    if body.is_empty() {
        return;
    }

    // Split body into halves for the two arrays. The split is interpreted
    // modulo body length so any payload size produces two non-empty sides
    // (when possible).
    let mid_bytes = if body.len() >= 4 {
        // round to even-byte boundary so the u16 unpack lands cleanly
        let raw = split_raw % body.len();
        raw & !1
    } else {
        0
    };
    let (left_bytes, right_bytes) = body.split_at(mid_bytes);

    let a = build_sorted_unique_u16(left_bytes);
    let b = build_sorted_unique_u16(right_bytes);

    // Build the dispatched containers and run intersect through both the
    // top-level Container API (which picks SSE4.2 when available) and a
    // hand-rolled scalar oracle.
    let arr_a = ArrayContainer::from_sorted(a.clone());
    let arr_b = ArrayContainer::from_sorted(b.clone());

    let dispatched = Container::Array(arr_a.clone()).intersect(&Container::Array(arr_b.clone()));
    let dispatched_card =
        Container::Array(arr_a).intersect_cardinality(&Container::Array(arr_b));

    let oracle: Vec<u16> = scalar_intersect_sorted(&a, &b);

    // The dispatch path may promote to a different container shape if the
    // result is dense; pull the sorted u16 list out of whatever variant
    // came back so the comparison is canonical.
    //
    // External callers (this fuzz harness included) use the read-only
    // `data()` / `runs()` accessors; the raw `data` / `runs` fields are
    // `pub(crate)` and only constructible from inside `tokenfs-algos`.
    let dispatched_list: Vec<u16> = match dispatched {
        Container::Array(arr) => arr.data().to_vec(),
        Container::Bitmap(bm) => bm.to_array(),
        Container::Run(run) => run_to_sorted_u16(run.runs()),
    };

    assert_eq!(
        dispatched_list, oracle,
        "array×array intersect diverged: dispatched={dispatched_list:?} oracle={oracle:?}"
    );
    assert_eq!(
        dispatched_card as usize,
        oracle.len(),
        "array×array intersect_cardinality mismatch: card={dispatched_card} oracle.len={}",
        oracle.len()
    );
}

fn run_bitmap_x_bitmap(payload: &[u8]) {
    // Build two BitmapContainers by slicing the payload in half and
    // hashing each half into the 1024-word storage. Empty payload
    // produces empty bitmaps.
    let mid = payload.len() / 2;
    let (left, right) = payload.split_at(mid);

    let mut a_words = Box::new([0_u64; BITMAP_WORDS]);
    let mut b_words = Box::new([0_u64; BITMAP_WORDS]);
    fill_words_from_bytes(&mut a_words, left);
    fill_words_from_bytes(&mut b_words, right);

    let a = BitmapContainer::from_words(a_words);
    let b = BitmapContainer::from_words(b_words);

    // Three variants: _card (returns card and writes out), _nocard
    // (writes out only), _justcard (returns card only). All three must
    // agree with the scalar oracle.
    let mut out_card = BitmapContainer::empty();
    let card = a.and_into(&b, &mut out_card);

    let mut out_nocard = BitmapContainer::empty();
    a.and_into_nocard(&b, &mut out_nocard);

    let card_only = a.and_cardinality(&b);

    // Scalar oracle: word-wise AND over the two bitmaps.
    let mut oracle_words = [0_u64; BITMAP_WORDS];
    let mut oracle_card = 0_u64;
    for i in 0..BITMAP_WORDS {
        let w = a.words[i] & b.words[i];
        oracle_words[i] = w;
        oracle_card += u64::from(w.count_ones());
    }
    let oracle_card = oracle_card as u32;

    assert_eq!(card, card_only, "bitmap AND _card != _justcard");
    assert_eq!(card, oracle_card, "bitmap AND _card diverged from scalar oracle");
    assert_eq!(
        out_card.words[..],
        oracle_words[..],
        "bitmap AND _card materialised words diverged from scalar oracle"
    );
    assert_eq!(
        out_card.words[..],
        out_nocard.words[..],
        "bitmap AND _nocard materialised words diverged from _card"
    );
}

/// Build a sorted, deduplicated `Vec<u16>` from a byte slice. The list is
/// truncated at 4096 entries to satisfy `ArrayContainer`'s invariant.
fn build_sorted_unique_u16(bytes: &[u8]) -> Vec<u16> {
    let mut values: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    values.sort_unstable();
    values.dedup();
    if values.len() > ARRAY_MAX {
        values.truncate(ARRAY_MAX);
    }
    values
}

/// Sorted-merge intersect oracle for two sorted, deduplicated u16 slices.
fn scalar_intersect_sorted(a: &[u16], b: &[u16]) -> Vec<u16> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0_usize, 0_usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// Materialise (start, length_minus_one) runs into a sorted Vec<u16>.
fn run_to_sorted_u16(runs: &[(u16, u16)]) -> Vec<u16> {
    let mut out = Vec::new();
    for &(start, len_m1) in runs {
        for v in start..=start.saturating_add(len_m1) {
            out.push(v);
            if v == u16::MAX {
                break;
            }
        }
    }
    out
}

/// Copy bytes into a 1024-word bitmap container, packing 8 bytes per word.
fn fill_words_from_bytes(words: &mut [u64; BITMAP_WORDS], bytes: &[u8]) {
    for (i, word) in words.iter_mut().enumerate() {
        let off = i * 8;
        if off >= bytes.len() {
            break;
        }
        let mut buf = [0_u8; 8];
        let take = (bytes.len() - off).min(8);
        buf[..take].copy_from_slice(&bytes[off..off + take]);
        *word = u64::from_le_bytes(buf);
    }
}
