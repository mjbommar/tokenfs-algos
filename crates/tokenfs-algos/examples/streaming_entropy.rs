#![allow(missing_docs)]

use tokenfs_algos::{entropy::shannon, histogram::ByteHistogram};

fn main() {
    let bytes = b"tokenfs-algos: deterministic byte-stream analysis";
    let histogram = ByteHistogram::from_block(bytes);
    let h1 = shannon::h1(&histogram);

    println!("bytes: {}", histogram.total());
    println!("h1_bits_per_byte: {h1:.4}");
}
