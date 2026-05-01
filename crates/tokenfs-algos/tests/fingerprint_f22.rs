#![allow(missing_docs)]

use std::{
    env,
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
};

use tokenfs_algos::{fingerprint, paper};

fn deterministic_block(seed: u64) -> [u8; fingerprint::BLOCK_SIZE] {
    let mut state = seed;
    let mut block = [0_u8; fingerprint::BLOCK_SIZE];
    for byte in &mut block {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        *byte = state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u8;
    }
    block
}

#[test]
fn public_f22_aliases_match_product_api() {
    let block = deterministic_block(0xF22);

    assert_eq!(
        fingerprint::block(&block),
        paper::f22::fingerprint_block(&block)
    );
    assert_eq!(
        fingerprint::kernels::scalar::block(&block),
        paper::f22::scalar::fingerprint_block(&block)
    );
    assert_eq!(
        fingerprint::extent(&block),
        paper::f22::fingerprint_extent(&block)
    );
}

#[test]
fn pinned_scalar_matches_default_for_generated_blocks() {
    for seed in 0..128_u64 {
        let block = deterministic_block(seed ^ 0x0A5A_5F22);
        assert_eq!(
            fingerprint::block(&block),
            fingerprint::kernels::scalar::block(&block),
            "seed={seed}"
        );
    }
}

#[test]
fn calibration_matches_f21_sidecar_h1_when_available() {
    let Some(path) = f22_sidecar_path() else {
        eprintln!("F22 sidecar missing; set TOKENFS_ALGOS_F22_DATA to enable calibration");
        return;
    };

    let file = File::open(&path)
        .unwrap_or_else(|error| panic!("failed to open F22 sidecar `{}`: {error}", path.display()));
    let mut reader = BufReader::new(file);

    let mut magic = [0_u8; 8];
    reader
        .read_exact(&mut magic)
        .expect("failed to read F22 sidecar magic");
    assert_eq!(&magic, b"F22EXTV1");

    let n_extents = read_u32(&mut reader) as usize;
    let target = n_extents.min(500);
    let mut total_diff = 0.0_f64;
    let mut max_diff = 0.0_f32;

    for index in 0..target {
        let _extent_id = read_n(&mut reader, 32);
        let _policy = read_n(&mut reader, 16);
        let n_bytes = read_u32(&mut reader) as usize;
        let f21_h1 = read_f32(&mut reader);
        let _f21_h2 = read_f32(&mut reader);
        let _f21_h3 = read_f32(&mut reader);
        let _f21_top16 = read_f32(&mut reader);
        let _f21_rl = read_f32(&mut reader);
        let _f21_skew = read_f32(&mut reader);
        let payload = read_n(&mut reader, n_bytes);

        let fp = fingerprint::kernels::scalar::extent(&payload);
        let diff = (fp.h1 - f21_h1).abs();
        total_diff += f64::from(diff);
        max_diff = max_diff.max(diff);
        assert!(
            diff < 0.05,
            "extent {index}: tokenfs-algos H1={} vs F21 H1={} diff={diff}",
            fp.h1,
            f21_h1
        );
    }

    eprintln!(
        "F22 sidecar H1 calibration: extents={target}, avg_diff={:.6}, max_diff={:.6}",
        total_diff / target.max(1) as f64,
        max_diff
    );
}

fn f22_sidecar_path() -> Option<PathBuf> {
    env::var_os("TOKENFS_ALGOS_F22_DATA")
        .map(PathBuf::from)
        .filter(|path| path.exists())
        .or_else(|| {
            let path = PathBuf::from("/nas4/data/tokenfs-ubuntu/bench/cow/f22-extent-bytes.bin");
            path.exists().then_some(path)
        })
}

fn read_u32<R: Read>(reader: &mut R) -> u32 {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes).expect("failed to read u32");
    u32::from_le_bytes(bytes)
}

fn read_f32<R: Read>(reader: &mut R) -> f32 {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes).expect("failed to read f32");
    f32::from_le_bytes(bytes)
}

fn read_n<R: Read>(reader: &mut R, n: usize) -> Vec<u8> {
    let mut bytes = vec![0_u8; n];
    reader
        .read_exact(&mut bytes)
        .expect("failed to read byte payload");
    bytes
}
