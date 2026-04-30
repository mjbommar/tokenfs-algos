pub(crate) fn add_block_direct_u64(block: &[u8], counts: &mut [u64; 256]) {
    for &byte in block {
        counts[byte as usize] += 1;
    }
}

pub(crate) fn add_block_local_u32(block: &[u8], counts: &mut [u64; 256]) {
    for chunk in block.chunks(u32::MAX as usize) {
        let mut local = [0_u32; 256];

        for &byte in chunk {
            local[byte as usize] += 1;
        }

        for (dst, src) in counts.iter_mut().zip(local) {
            *dst += u64::from(src);
        }
    }
}

pub(crate) fn add_block_striped_u32<const LANES: usize>(block: &[u8], counts: &mut [u64; 256]) {
    debug_assert!(LANES > 0);

    for chunk in block.chunks(u32::MAX as usize) {
        let mut stripes = [[0_u32; 256]; LANES];

        for (index, &byte) in chunk.iter().enumerate() {
            stripes[index % LANES][byte as usize] += 1;
        }

        for stripe in stripes {
            for (dst, src) in counts.iter_mut().zip(stripe) {
                *dst += u64::from(src);
            }
        }
    }
}

pub(crate) fn add_block_run_length_u64(block: &[u8], counts: &mut [u64; 256]) {
    let mut index = 0;

    while index < block.len() {
        let byte = block[index];
        let start = index;
        index += 1;

        while index < block.len() && block[index] == byte {
            index += 1;
        }

        counts[byte as usize] += (index - start) as u64;
    }
}

pub(crate) fn add_block_adaptive_prefix<const SAMPLE: usize>(
    block: &[u8],
    counts: &mut [u64; 256],
) {
    match choose_kernel_prefix::<SAMPLE>(block) {
        AdaptiveChoice::LocalU32 => add_block_local_u32(block, counts),
        AdaptiveChoice::Stripe8U32 => add_block_striped_u32::<8>(block, counts),
        AdaptiveChoice::RunLengthU64 => add_block_run_length_u64(block, counts),
    }
}

pub(crate) fn add_block_adaptive_spread_4k(block: &[u8], counts: &mut [u64; 256]) {
    match choose_kernel_spread_4k(block) {
        AdaptiveChoice::LocalU32 => add_block_local_u32(block, counts),
        AdaptiveChoice::Stripe8U32 => add_block_striped_u32::<8>(block, counts),
        AdaptiveChoice::RunLengthU64 => add_block_run_length_u64(block, counts),
    }
}

pub(crate) fn add_block_adaptive_run_sentinel_4k(block: &[u8], counts: &mut [u64; 256]) {
    match choose_run_sentinel_4k(block) {
        AdaptiveChoice::RunLengthU64 => add_block_run_length_u64(block, counts),
        _ => add_block_local_u32(block, counts),
    }
}

pub(crate) fn add_block_adaptive_chunked<const CHUNK: usize>(
    block: &[u8],
    counts: &mut [u64; 256],
) {
    debug_assert!(CHUNK > 0);

    for chunk in block.chunks(CHUNK) {
        add_block_adaptive_prefix::<1024>(chunk, counts);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AdaptiveChoice {
    LocalU32,
    Stripe8U32,
    RunLengthU64,
}

#[derive(Clone, Copy, Debug)]
struct SampleMetrics {
    len: usize,
    distinct: u16,
    top_count: u32,
    adjacent_equal: u32,
    longest_run: u32,
}

fn choose_kernel_prefix<const SAMPLE: usize>(block: &[u8]) -> AdaptiveChoice {
    let sample_len = block.len().min(SAMPLE);
    let metrics = SampleMetrics::from_contiguous(&block[..sample_len]);
    choose_from_metrics(metrics)
}

fn choose_kernel_spread_4k(block: &[u8]) -> AdaptiveChoice {
    if block.len() <= 4096 {
        return choose_from_metrics(SampleMetrics::from_contiguous(block));
    }

    let mut metrics = SampleMetricsBuilder::new();
    let segment_len = 1024;
    let starts = [
        0,
        block.len() / 3,
        (block.len() * 2) / 3,
        block.len().saturating_sub(segment_len),
    ];

    for start in starts {
        let end = (start + segment_len).min(block.len());
        metrics.add_contiguous(&block[start..end]);
    }

    choose_from_metrics(metrics.finish())
}

fn choose_run_sentinel_4k(block: &[u8]) -> AdaptiveChoice {
    let sample_len = block.len().min(4096);
    let metrics = SampleMetrics::from_contiguous(&block[..sample_len]);

    if metrics.len >= 512
        && (usize::try_from(metrics.longest_run).unwrap_or(usize::MAX) >= metrics.len / 2
            || fraction_at_least(metrics.top_count, metrics.len, 15, 16))
    {
        AdaptiveChoice::RunLengthU64
    } else {
        AdaptiveChoice::LocalU32
    }
}

fn choose_from_metrics(metrics: SampleMetrics) -> AdaptiveChoice {
    if metrics.len == 0 {
        return AdaptiveChoice::LocalU32;
    }

    if metrics.len >= 512
        && (usize::try_from(metrics.longest_run).unwrap_or(usize::MAX) >= metrics.len / 2
            || fraction_at_least(metrics.top_count, metrics.len, 9, 10))
    {
        return AdaptiveChoice::RunLengthU64;
    }

    if metrics.distinct <= 8
        || fraction_at_least(metrics.top_count, metrics.len, 1, 4)
        || fraction_at_least(metrics.adjacent_equal, metrics.len.saturating_sub(1), 1, 4)
    {
        return AdaptiveChoice::Stripe8U32;
    }

    AdaptiveChoice::LocalU32
}

fn fraction_at_least(count: u32, total: usize, numerator: u32, denominator: u32) -> bool {
    if total == 0 {
        return false;
    }

    u128::from(count) * u128::from(denominator) >= (total as u128) * u128::from(numerator)
}

impl SampleMetrics {
    fn from_contiguous(bytes: &[u8]) -> Self {
        let mut builder = SampleMetricsBuilder::new();
        builder.add_contiguous(bytes);
        builder.finish()
    }
}

struct SampleMetricsBuilder {
    counts: [u32; 256],
    len: usize,
    adjacent_equal: u32,
    longest_run: u32,
}

impl SampleMetricsBuilder {
    fn new() -> Self {
        Self {
            counts: [0; 256],
            len: 0,
            adjacent_equal: 0,
            longest_run: 0,
        }
    }

    fn add_contiguous(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }

        self.len += bytes.len();

        let mut current_byte = bytes[0];
        let mut current_run = 0_u32;

        for &byte in bytes {
            self.counts[byte as usize] += 1;

            if byte == current_byte {
                current_run += 1;
            } else {
                self.longest_run = self.longest_run.max(current_run);
                current_byte = byte;
                current_run = 1;
            }
        }

        self.longest_run = self.longest_run.max(current_run);

        for pair in bytes.windows(2) {
            if pair[0] == pair[1] {
                self.adjacent_equal += 1;
            }
        }
    }

    fn finish(self) -> SampleMetrics {
        let mut distinct = 0_u16;
        let mut top_count = 0_u32;

        for count in self.counts {
            if count != 0 {
                distinct += 1;
                top_count = top_count.max(count);
            }
        }

        SampleMetrics {
            len: self.len,
            distinct,
            top_count,
            adjacent_equal: self.adjacent_equal,
            longest_run: self.longest_run,
        }
    }
}
