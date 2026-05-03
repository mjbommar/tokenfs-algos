use super::super::GROUP;
#[cfg(feature = "userspace")]
use super::super::{streamvbyte_control_len, streamvbyte_data_max_len};

/// Returns the per-value length code for `v`: `0..=3`, where the
/// encoded byte width is `code + 1`.
#[inline]
const fn code_for(v: u32) -> u8 {
    // `v.leading_zeros()` is 32 for v=0, so the formula yields 0
    // and the value occupies 1 byte (the 0x00 byte). Otherwise
    // `(32 - lz + 7) / 8 - 1` ∈ {0, 1, 2, 3}.
    if v < 1 << 8 {
        0
    } else if v < 1 << 16 {
        1
    } else if v < 1 << 24 {
        2
    } else {
        3
    }
}

/// Encodes `values` using the spec's reference algorithm.
///
/// # Panics
///
/// Panics if `control_out` is shorter than
/// [`streamvbyte_control_len`] or `data_out` is shorter than
/// [`streamvbyte_data_max_len`]. Available only with
/// `feature = "userspace"`; kernel-safe callers must use
/// [`encode_u32_unchecked`] (audit-R10 #1 / #216).
#[cfg(feature = "userspace")]
pub fn encode_u32(values: &[u32], control_out: &mut [u8], data_out: &mut [u8]) -> usize {
    let n = values.len();
    assert!(
        control_out.len() >= streamvbyte_control_len(n),
        "control_out too small: {} < {}",
        control_out.len(),
        streamvbyte_control_len(n)
    );
    assert!(
        data_out.len() >= streamvbyte_data_max_len(n),
        "data_out too small: {} < {}",
        data_out.len(),
        streamvbyte_data_max_len(n)
    );
    // SAFETY: asserts above establish the precondition.
    unsafe { encode_u32_unchecked(values, control_out, data_out) }
}

/// Scalar encoder body without bounds-checking asserts.
///
/// # Safety
///
/// Caller must ensure
/// `control_out.len() >= streamvbyte_control_len(values.len())`
/// and
/// `data_out.len() >= streamvbyte_data_max_len(values.len())`.
/// Indexing inside this routine is otherwise identical to
/// [`encode_u32`] but does not include the leading `assert!`
/// guards, which keeps it panic-free for the
/// `try_streamvbyte_encode_u32` path even when
/// `panicking-shape-apis` is disabled.
pub unsafe fn encode_u32_unchecked(
    values: &[u32],
    control_out: &mut [u8],
    data_out: &mut [u8],
) -> usize {
    let n = values.len();
    let mut data_pos = 0_usize;
    let mut ctrl_pos = 0_usize;

    // Full groups of 4 → one control byte each.
    let full_groups = n / GROUP;
    for g in 0..full_groups {
        let base = g * GROUP;
        let v0 = values[base];
        let v1 = values[base + 1];
        let v2 = values[base + 2];
        let v3 = values[base + 3];
        let c0 = code_for(v0);
        let c1 = code_for(v1);
        let c2 = code_for(v2);
        let c3 = code_for(v3);
        control_out[ctrl_pos] = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
        ctrl_pos += 1;

        data_pos += write_value(v0, c0, &mut data_out[data_pos..]);
        data_pos += write_value(v1, c1, &mut data_out[data_pos..]);
        data_pos += write_value(v2, c2, &mut data_out[data_pos..]);
        data_pos += write_value(v3, c3, &mut data_out[data_pos..]);
    }

    // Tail group with N % 4 ∈ {1, 2, 3}: pad missing slots with
    // code 0 (1 data byte = 0x00). The padding bytes still count
    // toward `data_pos` so a round-trip read sees the encoded
    // bytes the encoder wrote.
    let tail = n - full_groups * GROUP;
    if tail > 0 {
        let mut codes = [0_u8; GROUP];
        let mut payload = [0_u32; GROUP];
        for k in 0..tail {
            payload[k] = values[full_groups * GROUP + k];
            codes[k] = code_for(payload[k]);
        }
        control_out[ctrl_pos] = codes[0] | (codes[1] << 2) | (codes[2] << 4) | (codes[3] << 6);
        ctrl_pos += 1;
        for k in 0..GROUP {
            data_pos += write_value(payload[k], codes[k], &mut data_out[data_pos..]);
        }
    }

    let _ = ctrl_pos; // Asserted-via-bounds rather than returned.
    data_pos
}

/// Writes the `code + 1` low-order little-endian bytes of `v` into
/// `dst` and returns that length.
#[inline]
fn write_value(v: u32, code: u8, dst: &mut [u8]) -> usize {
    let len = (code as usize) + 1;
    let bytes = v.to_le_bytes();
    dst[..len].copy_from_slice(&bytes[..len]);
    len
}

/// Decodes `n` values; returns the number of data bytes consumed.
///
/// # Panics
///
/// Panics if `control` has fewer than `streamvbyte_control_len(n)`
/// bytes, `out` is shorter than `n`, or `data` runs out before the
/// implied length is reached. Available only with
/// `feature = "userspace"`; kernel-safe callers must use
/// [`decode_u32_unchecked`] (audit-R10 #1 / #216).
#[cfg(feature = "userspace")]
pub fn decode_u32(control: &[u8], data: &[u8], n: usize, out: &mut [u32]) -> usize {
    let ctrl_needed = streamvbyte_control_len(n);
    assert!(
        control.len() >= ctrl_needed,
        "control too small: {} < {}",
        control.len(),
        ctrl_needed
    );
    assert!(
        out.len() >= n,
        "decode output buffer too small: {} < {}",
        out.len(),
        n
    );
    // SAFETY: asserts above establish the buffer-length
    // preconditions; the implied data length is asserted via
    // the natural `data[data_pos..]` slicing inside the kernel
    // when this panicking entry point is used directly.
    unsafe { decode_u32_unchecked(control, data, n, out) }
}

/// Scalar decoder body without bounds-checking asserts.
///
/// # Safety
///
/// Caller must ensure `control.len() >= streamvbyte_control_len(n)`,
/// `out.len() >= n`, and the data stream is long enough for the
/// implied byte sum encoded in the first `streamvbyte_control_len(n)`
/// control bytes (sum of `code+1` over each 2-bit code; padded
/// slots in the tail group still consume their byte).
///
/// Used by [`super::super::try_streamvbyte_decode_u32`] after
/// pre-validation walks the control stream itself; eliminates
/// the `assert!` panic sites that would otherwise leak through
/// the fallible API surface (audit-R6 finding #162).
pub unsafe fn decode_u32_unchecked(
    control: &[u8],
    data: &[u8],
    n: usize,
    out: &mut [u32],
) -> usize {
    let mut data_pos = 0_usize;
    let mut written = 0_usize;
    let full_groups = n / GROUP;
    for g in 0..full_groups {
        let c = control[g];
        let codes = [c & 0b11, (c >> 2) & 0b11, (c >> 4) & 0b11, (c >> 6) & 0b11];
        for k in 0..GROUP {
            let len = (codes[k] as usize) + 1;
            out[written + k] = read_value(&data[data_pos..], len);
            data_pos += len;
        }
        written += GROUP;
    }

    // Tail: only `n - full_groups * GROUP` outputs requested,
    // even though the control byte covers four codes. We must
    // still advance `data_pos` past every code the encoder wrote
    // to make decoder/encoder offset bookkeeping match.
    let tail = n - full_groups * GROUP;
    if tail > 0 {
        let c = control[full_groups];
        let codes = [c & 0b11, (c >> 2) & 0b11, (c >> 4) & 0b11, (c >> 6) & 0b11];
        for k in 0..GROUP {
            let len = (codes[k] as usize) + 1;
            if k < tail {
                out[written + k] = read_value(&data[data_pos..], len);
            }
            data_pos += len;
        }
    }

    data_pos
}

/// Reads a little-endian integer of `len ∈ 1..=4` bytes from
/// `src` into a `u32`.
#[inline]
fn read_value(src: &[u8], len: usize) -> u32 {
    let mut bytes = [0_u8; 4];
    bytes[..len].copy_from_slice(&src[..len]);
    u32::from_le_bytes(bytes)
}
