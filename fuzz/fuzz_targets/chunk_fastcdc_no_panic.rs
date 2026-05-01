//! Fuzz target: chunk::fastcdc_find_boundary must not panic, and must obey
//! the same min/max invariants as the gear path.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tokenfs_algos::chunk::{self, FastCdcConfig};

fuzz_target!(|data: &[u8]| {
    if data.len() < 12 {
        return;
    }
    let min_raw = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let avg_raw = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let max_raw = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    let bytes = &data[12..];

    let min_size = (min_raw % 8192).max(1);
    let avg_size = ((avg_raw % 32_768).max(min_size + 1)).max(2);
    let max_size = ((max_raw % 131_072).max(avg_size + 1)).max(3);

    let config = FastCdcConfig::with_sizes(min_size, avg_size, max_size);
    // Normalized fields after with_sizes() does its avg-power-of-two rounding.
    let norm_min = config.min_size;
    let norm_max = config.max_size;

    let boundary = chunk::fastcdc_find_boundary(bytes, config);

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
