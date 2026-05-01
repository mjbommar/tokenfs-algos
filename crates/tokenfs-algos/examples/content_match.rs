//! Matches a query byte distribution against a tiny calibrated catalog.

use tokenfs_algos::distribution::{
    ByteDistribution, ByteDistributionMetric, ByteDistributionReference, nearest_reference,
};

fn main() {
    let zeros = ByteDistribution::from_bytes(&[0; 4096]);
    let text = ByteDistribution::from_bytes(b"fn main() { println!(\"hello tokenfs\"); }\n");
    let query = ByteDistribution::from_bytes(b"tokenfs tokenfs tokenfs\n");
    let references = [
        ByteDistributionReference::new("zeros", "application/x-zero", zeros),
        ByteDistributionReference::new("text", "text/plain", text),
    ];

    let nearest = nearest_reference(&query, &references, ByteDistributionMetric::JensenShannon)
        .expect("reference catalog is non-empty");

    println!("label={}", nearest.label);
    println!("mime={}", nearest.mime_type);
    println!("distance={:.6}", nearest.distance);
}
