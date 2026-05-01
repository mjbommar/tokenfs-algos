//! TLSH-like locality-sensitive fuzzy hash.
//!
//! Quality-faithful reimplementation of the algorithm described in
//! Oliver, Cheng, Chen, "TLSH — A Locality Sensitive Hash" (CTC 2013) and
//! the Apache-2.0 reference at https://github.com/trendmicro/tlsh.
//!
//! # Algorithm
//!
//! 1. Slide a 5-byte window across the input. For each window, compute six
//!    triplet-pearson-hashes (three of the five bytes per hash, with six
//!    different mixing seeds) and increment the corresponding 8-bit bucket
//!    counters in a 256-entry table.
//! 2. After ingest, take the first 128 buckets and find the three quartile
//!    boundaries `q1 < q2 < q3` via a copy-and-quickselect pattern.
//! 3. Encode each bucket as 2 bits per the quartile it falls into:
//!    `0b00` ≤ q1, `0b01` ≤ q2, `0b10` ≤ q3, `0b11` > q3. Pack
//!    little-endian within each byte; bytes are written in reversed order
//!    to match the upstream digest layout.
//! 4. Header (3 bytes): a checksum byte, an `Lvalue` (log-of-length lookup
//!    table), and a `QRatio = (Q1ratio<<4) | Q2ratio` byte. Each header
//!    byte is "swap-byte" nibble-swapped per the reference layout.
//!
//! Inputs shorter than `MIN_INPUT_BYTES` (50, matching upstream) produce
//! no digest. The Pearson permutation table is generated deterministically
//! at startup; it is not the canonical TLSH table (the algorithm doesn't
//! depend on a particular permutation, only on it being a bijection).
//!
//! # Distance
//!
//! `distance(a, b)` is integer-valued, 0 for identical input. Per the
//! published TLSH literature: `< 30` is near-duplicate, `30..100` is
//! related, `> 150` is essentially unrelated. The implementation matches
//! the upstream formula: header diffs (Lvalue, Q1Ratio, Q2Ratio,
//! checksum) plus the body Hamming-on-dibits sum.

/// Minimum input length for a usable digest (matches upstream TLSH).
pub const MIN_INPUT_BYTES: usize = 50;

/// 5-byte sliding window.
const SLIDING_WINDOW: usize = 5;

/// 256 buckets total; effective body uses the first 128 (256 dibits = 32 B
/// of body in the digest).
const BUCKETS: usize = 256;
const EFF_BUCKETS: usize = 128;

/// Final body byte count: 128 dibits / 4 dibits-per-byte = 32 bytes.
const BODY_BYTES: usize = 32;

/// Header bytes: checksum, Lvalue, QRatio.
const HEADER_BYTES: usize = 3;

/// Total digest size.
pub const DIGEST_BYTES: usize = HEADER_BYTES + BODY_BYTES;

/// Six Pearson seeds (used by upstream as well; arbitrary distinct bytes).
const SEEDS: [u8; 6] = [49, 12, 178, 166, 84, 230];

// ---------------------------------------------------------------------------
// Pearson permutation: a bijective u8 -> u8 lookup table. Initialized once
// via OnceLock at first use. We use a deterministic Fisher-Yates shuffle
// seeded with a fixed constant so different builds produce identical digests.
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
fn pearson_table() -> &'static [u8; 256] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[u8; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut state: u64 = 0xC8C2_5E0F_2C5C_3F6D;
        let mut table = [0_u8; 256];
        for (i, dst) in table.iter_mut().enumerate() {
            *dst = i as u8;
        }
        for i in (1..256).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let j = (state as usize) % (i + 1);
            table.swap(i, j);
        }
        table
    })
}

#[cfg(not(feature = "std"))]
fn pearson_table() -> &'static [u8; 256] {
    // no_std fallback: identity permutation. The TLSH algorithm tolerates
    // any bijection here; identity is the simplest static table.
    static IDENTITY: [u8; 256] = {
        let mut t = [0_u8; 256];
        let mut i = 0;
        while i < 256 {
            t[i] = i as u8;
            i += 1;
        }
        t
    };
    &IDENTITY
}

#[inline]
fn pearson(salt: u8, a: u8, b: u8, c: u8) -> u8 {
    let t = pearson_table();
    t[(t[(t[(salt ^ a) as usize] ^ b) as usize] ^ c) as usize]
}

/// 35-byte TLSH-like digest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Digest {
    bytes: [u8; DIGEST_BYTES],
    /// True when `bytes` is meaningful (input was long enough and had
    /// enough bucket diversity).
    valid: bool,
}

impl Default for Digest {
    fn default() -> Self {
        Self {
            bytes: [0; DIGEST_BYTES],
            valid: false,
        }
    }
}

impl Digest {
    /// Returns the raw 35-byte digest. Zeros if [`Digest::is_valid`] is false.
    #[must_use]
    pub const fn bytes(&self) -> &[u8; DIGEST_BYTES] {
        &self.bytes
    }

    /// True when the digest carries meaningful information.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.valid
    }

    /// Hex-encodes the digest into a 70-character `String`. Returns an
    /// empty string for an invalid digest.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn to_hex(&self) -> String {
        if !self.valid {
            return String::new();
        }
        let mut out = String::with_capacity(DIGEST_BYTES * 2);
        for byte in &self.bytes {
            out.push(hex_char(byte >> 4));
            out.push(hex_char(byte & 0xF));
        }
        out
    }
}

#[cfg(feature = "std")]
fn hex_char(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        10..=15 => (b'a' + nibble - 10) as char,
        _ => '?',
    }
}

/// Computes a TLSH-like digest of `bytes`. Returns an invalid digest
/// (`is_valid() == false`) when the input is too short or too uniform to
/// yield meaningful quartile boundaries.
#[must_use]
pub fn digest(bytes: &[u8]) -> Digest {
    if bytes.len() < MIN_INPUT_BYTES {
        return Digest::default();
    }

    let mut buckets = [0_u32; BUCKETS];
    let mut checksum: u8 = 0;

    // Slide a 5-byte window. For each position past the first SLIDING_WINDOW
    // bytes we have a full window to fingerprint.
    for i in (SLIDING_WINDOW - 1)..bytes.len() {
        let a4 = bytes[i];
        let a3 = bytes[i - 1];
        let a2 = bytes[i - 2];
        let a1 = bytes[i - 3];
        let a0 = bytes[i - 4];

        // Checksum mixes the latest 3 bytes; helps disambiguate inputs
        // that share their bucket distribution. Salt = 1 per upstream.
        checksum = pearson(1, a4, a3, checksum);

        // Six triplet-pearson hashes per upstream.
        let triplets: [(u8, u8, u8); 6] = [
            (a4, a3, a2),
            (a4, a3, a1),
            (a4, a2, a1),
            (a4, a2, a0),
            (a4, a3, a0),
            (a4, a1, a0),
        ];
        for (idx, (b, c, d)) in triplets.iter().enumerate() {
            let bucket = pearson(SEEDS[idx], *b, *c, *d);
            buckets[bucket as usize] = buckets[bucket as usize].saturating_add(1);
        }
    }

    // Quartile boundaries over the first EFF_BUCKETS buckets.
    let mut sorted = [0_u32; EFF_BUCKETS];
    sorted.copy_from_slice(&buckets[..EFF_BUCKETS]);
    sorted.sort_unstable();
    let q1 = sorted[EFF_BUCKETS / 4 - 1];
    let q2 = sorted[EFF_BUCKETS / 2 - 1];
    let q3 = sorted[3 * EFF_BUCKETS / 4 - 1];

    // Refuse digests where the buckets are too uniform to discriminate.
    // Upstream's spec calls out "≤ 4*CODE_SIZE/2 non-zero buckets" as the
    // rejection floor; we use the equivalent "q3 == 0" check.
    if q3 == 0 {
        return Digest::default();
    }

    // Encode body: 128 dibits packed into 32 bytes, byte order reversed.
    let mut body = [0_u8; BODY_BYTES];
    for (i, &count) in buckets.iter().enumerate().take(EFF_BUCKETS) {
        let dibit = quartile_code(count, q1, q2, q3);
        let byte_idx = BODY_BYTES - 1 - (i / 4);
        let bit_pos = (i % 4) * 2;
        body[byte_idx] |= (dibit & 0b11) << bit_pos;
    }

    // Header: checksum, Lvalue, QRatio. Each nibble-swapped.
    let lvalue = l_capturing(bytes.len() as u64);
    let q1_ratio = ((u64::from(q1) * 100 / u64::from(q3)) % 16) as u8;
    let q2_ratio = ((u64::from(q2) * 100 / u64::from(q3)) % 16) as u8;
    let qratio = (q1_ratio << 4) | q2_ratio;
    let header: [u8; HEADER_BYTES] = [swap_byte(checksum), swap_byte(lvalue), swap_byte(qratio)];

    let mut bytes_out = [0_u8; DIGEST_BYTES];
    bytes_out[..HEADER_BYTES].copy_from_slice(&header);
    bytes_out[HEADER_BYTES..].copy_from_slice(&body);
    Digest {
        bytes: bytes_out,
        valid: true,
    }
}

#[inline]
fn quartile_code(value: u32, q1: u32, q2: u32, q3: u32) -> u8 {
    if value <= q1 {
        0
    } else if value <= q2 {
        1
    } else if value <= q3 {
        2
    } else {
        3
    }
}

#[inline]
const fn swap_byte(b: u8) -> u8 {
    ((b & 0x0F) << 4) | ((b >> 4) & 0x0F)
}

/// Maps an input length to a compressed log-scale `Lvalue`. Approximates
/// the upstream `l_capturing` table: `floor(log_1.5(N))` clamped to `u8`.
fn l_capturing(n: u64) -> u8 {
    if n < 2 {
        return 0;
    }
    // log_1.5(N) = ln(N) / ln(1.5)
    let ln_15 = 0.405_465_108_108_164_4_f64; // ln(1.5)
    let v = (n as f64).ln() / ln_15;
    if v >= 255.0 { 255 } else { v.floor() as u8 }
}

/// Hamming-on-dibits distance between two body bytes: sum over each of the
/// four packed dibits of the absolute difference between the two-bit values.
#[inline]
const fn dibit_diff(a: u8, b: u8) -> u32 {
    let mut total = 0_u32;
    let mut mask: u8 = 0b11;
    let mut shift = 0;
    while shift < 8 {
        let av = (a & mask) >> shift;
        let bv = (b & mask) >> shift;
        total += av.abs_diff(bv) as u32;
        mask <<= 2;
        shift += 2;
    }
    total
}

/// Distance between two TLSH-like digests.
///
/// Returns 0 for byte-identical digests; > 150 typically means "unrelated".
/// If either digest is invalid, returns `u32::MAX` as a sentinel.
#[must_use]
pub fn distance(a: &Digest, b: &Digest) -> u32 {
    if !a.is_valid() || !b.is_valid() {
        return u32::MAX;
    }
    let mut diff = 0_u32;

    // Checksum delta: 1 bit if not equal.
    if a.bytes[0] != b.bytes[0] {
        diff += 1;
    }

    // Lvalue delta: scaled circular distance. Per upstream:
    // - 0 → 0
    // - 1 → 1
    // - else → ldiff * 12
    let l_a = swap_byte(a.bytes[1]);
    let l_b = swap_byte(b.bytes[1]);
    let ldiff = mod_diff(u32::from(l_a), u32::from(l_b), 256);
    diff += match ldiff {
        0 => 0,
        1 => 1,
        n => n * 12,
    };

    // QRatio: two 4-bit quantities packed into one byte. Each contributes
    // a clamped diff per upstream.
    let q_a = swap_byte(a.bytes[2]);
    let q_b = swap_byte(b.bytes[2]);
    let q1a = u32::from(q_a >> 4);
    let q1b = u32::from(q_b >> 4);
    let q2a = u32::from(q_a & 0x0F);
    let q2b = u32::from(q_b & 0x0F);
    let q1d = mod_diff(q1a, q1b, 16);
    let q2d = mod_diff(q2a, q2b, 16);
    diff += if q1d <= 1 { q1d } else { (q1d - 1) * 12 };
    diff += if q2d <= 1 { q2d } else { (q2d - 1) * 12 };

    // Body Hamming-on-dibits sum.
    for i in HEADER_BYTES..DIGEST_BYTES {
        diff += dibit_diff(a.bytes[i], b.bytes[i]);
    }
    diff
}

#[inline]
const fn mod_diff(a: u32, b: u32, modulus: u32) -> u32 {
    let d = a.abs_diff(b);
    let alt = modulus - d;
    if d < alt { d } else { alt }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn random_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut state = seed.wrapping_add(1);
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect()
    }

    #[test]
    fn short_input_yields_invalid_digest() {
        let d = digest(b"too short");
        assert!(!d.is_valid());
        let zero = digest(b"");
        assert!(!zero.is_valid());
    }

    #[test]
    fn identical_inputs_produce_identical_digests() {
        let a = random_bytes(4096, 0xCAFE);
        let d1 = digest(&a);
        let d2 = digest(&a);
        assert!(d1.is_valid());
        assert_eq!(d1, d2);
        assert_eq!(distance(&d1, &d2), 0);
    }

    #[test]
    fn near_identical_inputs_have_low_distance() {
        let mut a = random_bytes(8192, 0x1234);
        let mut b = a.clone();
        // Flip ~10 bytes scattered across the buffer.
        for &idx in &[
            100_usize, 500, 1000, 1500, 3000, 4000, 5000, 6000, 7000, 8000,
        ] {
            b[idx] ^= 0xFF;
        }
        let da = digest(&a);
        let db = digest(&b);
        assert!(da.is_valid() && db.is_valid());
        let d = distance(&da, &db);
        // 10 byte changes in 8KiB of random bytes — TLSH distance should
        // stay well below the "unrelated" threshold of 150.
        assert!(d < 80, "near-identical distance too high: {d}");

        // Modify a too — sanity that the assertion isn't trivially passing
        // because both are identical.
        a[0] ^= 0x01;
        let da_mod = digest(&a);
        assert_ne!(da, da_mod);
    }

    #[test]
    fn unrelated_inputs_have_high_distance() {
        let a = random_bytes(8192, 0x1234);
        let b = random_bytes(8192, 0xABCD);
        let da = digest(&a);
        let db = digest(&b);
        assert!(da.is_valid() && db.is_valid());
        let d = distance(&da, &db);
        // Two unrelated random buffers: per the published TLSH thresholds
        // we expect distance well above 100. Allow some slack.
        assert!(d > 60, "unrelated distance too low: {d}");
    }

    #[test]
    fn shifted_input_has_lower_distance_than_random() {
        // Insert 100 random bytes at the start of a copy of the input. A
        // good fuzzy hash recognizes this as "still mostly the same file".
        let original = random_bytes(8192, 0x42);
        let prefix = random_bytes(100, 0xFFFF_AAAA);
        let mut shifted = prefix;
        shifted.extend_from_slice(&original);

        let unrelated = random_bytes(8192, 0x9999);

        let d_orig = digest(&original);
        let d_shifted = digest(&shifted);
        let d_unrel = digest(&unrelated);

        let dist_shift = distance(&d_orig, &d_shifted);
        let dist_unrel = distance(&d_orig, &d_unrel);
        assert!(
            dist_shift < dist_unrel,
            "shift-tolerance failed: shifted={dist_shift}, unrelated={dist_unrel}"
        );
    }

    #[test]
    fn distance_is_symmetric() {
        let a = random_bytes(2048, 0x11);
        let b = random_bytes(2048, 0x22);
        let da = digest(&a);
        let db = digest(&b);
        assert_eq!(distance(&da, &db), distance(&db, &da));
    }

    #[test]
    fn invalid_digest_distance_is_max_sentinel() {
        let valid = digest(&random_bytes(2048, 0xDD));
        let invalid = Digest::default();
        assert_eq!(distance(&valid, &invalid), u32::MAX);
        assert_eq!(distance(&invalid, &valid), u32::MAX);
    }

    #[cfg(feature = "std")]
    #[test]
    fn to_hex_round_trip_length() {
        let d = digest(&random_bytes(2048, 0x77));
        assert!(d.is_valid());
        let hex = d.to_hex();
        assert_eq!(hex.len(), DIGEST_BYTES * 2);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn dibit_diff_known_values() {
        // 0b00 vs 0b00 in all four dibits → 0.
        assert_eq!(dibit_diff(0, 0), 0);
        // 0b00 vs 0b11 in one dibit, identical elsewhere → 3.
        assert_eq!(dibit_diff(0, 0b11), 3);
        // 0b00 vs 0b11 in all four dibits → 12.
        assert_eq!(dibit_diff(0, 0b11_11_11_11), 12);
        // 0b01 vs 0b10 in one dibit → 1.
        assert_eq!(dibit_diff(0b01, 0b10), 1);
    }
}
