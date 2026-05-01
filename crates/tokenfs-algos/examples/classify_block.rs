//! Prints the common histogram, fingerprint, and selector contract outputs.

use tokenfs_algos::{fingerprint, histogram, selector};

fn main() {
    let bytes = b"tokenfs-algos classifies byte blocks with stable primitives.\n";
    let histogram = histogram::block(bytes);
    let extent = fingerprint::extent(bytes);
    let hint = selector::hint(bytes);

    println!("bytes={}", bytes.len());
    println!("histogram_total={}", histogram.total());
    println!("h1={:.3}", extent.h1);
    println!("h4={:.3}", extent.h4);
    println!("hint={hint:?}");
}
