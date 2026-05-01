//! Fuzz target: chunk::find_boundary (Gear) must not panic on any byte slice
//! against any reasonable ChunkConfig.
//!
//! Invariants:
//! - returned boundary is in 0..=bytes.len();
//! - boundary >= min_size when bytes.len() >= min_size, else == bytes.len();
//! - boundary <= max_size when bytes.len() >= max_size.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tokenfs_algos::chunk::{self, ChunkConfig};

fuzz_target!(|data: &[u8]| {
    if data.len() < 12 {
        return;
    }
    // First 12 bytes parameterize the config; rest is the payload.
    let min_raw = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let avg_raw = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let max_raw = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    let bytes = &data[12..];

    // Clamp to a sensible range so we don't spend cycles on absurd configs.
    let min_size = (min_raw % 8192).max(1);
    let avg_size = ((avg_raw % 32_768).max(min_size + 1)).max(2);
    let max_size = ((max_raw % 131_072).max(avg_size + 1)).max(3);

    let config = ChunkConfig::with_sizes(min_size, avg_size, max_size);
    // Re-read the normalized fields: with_sizes() rounds avg_size up to the
    // next power of two and re-clamps min/max accordingly.
    let norm_min = config.min_size;
    let norm_max = config.max_size;

    let boundary = chunk::find_boundary(bytes, config);

    assert!(boundary <= bytes.len(), "boundary out of range");
    if bytes.len() < norm_min {
        assert_eq!(boundary, bytes.len());
    } else {
        assert!(boundary >= norm_min, "boundary below min_size");
    }
    if bytes.len() >= norm_max {
        assert!(boundary <= norm_max, "boundary above max_size");
    }
});
