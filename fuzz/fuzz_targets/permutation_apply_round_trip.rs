//! Fuzz target: a random permutation `p` and its inverse must round-trip
//! arbitrary source data: `inverse(p).apply(p.apply(src)) == src`.
//!
//! Generating a uniformly random permutation from raw bytes uses a
//! Fisher-Yates shuffle seeded by an in-line LCG over the input. We then
//! validate the constructed permutation via `try_from_vec` so both
//! constructors get exercised.
//!
//! Input layout:
//! - First 2 bytes (LE u16): n value, capped at 1024.
//! - Remaining bytes: shuffle entropy and source values (4 bytes each
//!   for the `Vec<u32>` source). Short inputs are zero-padded.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tokenfs_algos::permutation::Permutation;

const MAX_N: usize = 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let n_raw = u16::from_le_bytes([data[0], data[1]]) as usize;
    let n = n_raw % (MAX_N + 1);
    let payload = &data[2..];

    // Seed the LCG from the first 8 bytes of payload (or zeros if short).
    let mut seed_bytes = [0_u8; 8];
    for k in 0..8.min(payload.len()) {
        seed_bytes[k] = payload[k];
    }
    let mut rng_state = u64::from_le_bytes(seed_bytes).wrapping_add(0x9e37_79b9_7f4a_7c15);

    // Build the identity 0..n then Fisher-Yates shuffle in-place.
    let mut perm_raw: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let j = ((rng_state >> 33) as usize) % (i + 1);
        perm_raw.swap(i, j);
    }

    // Validate via the checked constructor — confirms that the
    // hand-rolled shuffle produced a valid permutation. Then exercise
    // the unchecked path on the same bytes for a second build (smoke
    // test that both constructors agree).
    let perm = Permutation::try_from_vec(perm_raw.clone())
        .expect("Fisher-Yates should produce a valid permutation");
    let perm_unchecked = Permutation::from_vec_unchecked(perm_raw);
    assert_eq!(perm, perm_unchecked, "checked vs unchecked perm differ");

    // Build a Vec<u32> source from the remaining payload bytes; pad with
    // a deterministic synthesis when the corpus runs short.
    let src_off = 8;
    let mut src = Vec::with_capacity(n);
    for i in 0..n {
        let off = src_off + i * 4;
        let v = if off + 4 <= payload.len() {
            u32::from_le_bytes([
                payload[off],
                payload[off + 1],
                payload[off + 2],
                payload[off + 3],
            ])
        } else {
            (i as u32).wrapping_mul(0x9e37_79b9)
        };
        src.push(v);
    }

    let permuted = perm.apply(&src);
    assert_eq!(permuted.len(), n, "apply should preserve length");

    let inv = perm.inverse();
    assert_eq!(inv.len(), n, "inverse should preserve length");
    let recovered = inv.apply(&permuted);
    assert_eq!(
        recovered, src,
        "inverse(perm).apply(perm.apply(src)) != src (n={n})"
    );

    // Inverse-of-inverse round trip.
    let inv_inv = inv.inverse();
    assert_eq!(inv_inv, perm, "inverse(inverse(p)) != p (n={n})");

    // apply_into form must agree with the allocating apply.
    if n > 0 {
        let mut dst = vec![0_u32; n];
        perm.apply_into(&src, &mut dst);
        assert_eq!(dst, permuted, "apply_into != apply (n={n})");
    } else {
        // Zero-length contract: apply returns an empty Vec without
        // touching src — we just confirm permuted is empty.
        assert!(permuted.is_empty());
    }
});
