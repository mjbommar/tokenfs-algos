//! Fuzz target: dispatched run-length transition counter must agree with
//! the pinned scalar reference on every input.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tokenfs_algos::runlength;

fuzz_target!(|data: &[u8]| {
    let dispatched = runlength::transitions(data);
    let scalar = runlength::kernels::scalar::transitions(data);
    assert_eq!(
        dispatched, scalar,
        "dispatched transitions diverged from scalar reference on input \
         of length {}",
        data.len()
    );
});
