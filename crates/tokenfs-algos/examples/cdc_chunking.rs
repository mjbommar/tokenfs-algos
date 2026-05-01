//! Summarizes normalized FastCDC chunks for a deterministic byte stream.

use tokenfs_algos::chunk;

fn main() {
    let bytes = (0..256 * 1024)
        .map(|index| ((index * 131) ^ (index >> 3)) as u8)
        .collect::<Vec<_>>();
    let config = chunk::FastCdcConfig::with_sizes(1024, 4096, 16 * 1024);
    let quality = chunk::summarize_chunk_quality(
        chunk::fastcdc_chunks(&bytes, config),
        config.min_size,
        config.max_size,
    );

    println!("bytes={}", bytes.len());
    println!("chunks={}", quality.chunks);
    println!("mean_len={:.1}", quality.mean_len);
    println!("min_len={}", quality.min_len);
    println!("max_len={}", quality.max_len);
}
