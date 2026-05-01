//! Run-length detection and statistics.
//!
//! This module will hold scalar and SIMD run detection kernels plus summary
//! statistics used by composite fingerprints.

/// Summary of equal-byte runs in a byte slice.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RunLengthSummary {
    /// Total bytes scanned.
    pub total_bytes: usize,
    /// Number of transitions where `bytes[i] != bytes[i - 1]`.
    pub transitions: u64,
    /// Longest equal-byte run.
    pub longest_run: usize,
    /// Number of equal-byte runs with length at least 4.
    pub runs_ge4: u64,
    /// Bytes that belong to equal-byte runs with length at least 4.
    pub bytes_in_runs_ge4: u64,
}

impl RunLengthSummary {
    /// Fraction of bytes that belong to runs with length at least 4.
    #[must_use]
    pub fn run_fraction(self) -> f32 {
        if self.total_bytes == 0 {
            0.0
        } else {
            self.bytes_in_runs_ge4 as f32 / self.total_bytes as f32
        }
    }

    /// Fraction of adjacent byte pairs that are transitions.
    #[must_use]
    pub fn transition_rate(self) -> f32 {
        if self.total_bytes <= 1 {
            0.0
        } else {
            self.transitions as f32 / (self.total_bytes - 1) as f32
        }
    }
}

/// Scalar run-length summary.
#[must_use]
pub fn summarize(bytes: &[u8]) -> RunLengthSummary {
    if bytes.is_empty() {
        return RunLengthSummary::default();
    }

    let mut summary = RunLengthSummary {
        total_bytes: bytes.len(),
        longest_run: 1,
        ..RunLengthSummary::default()
    };
    let mut run_len = 1_usize;

    for index in 1..bytes.len() {
        if bytes[index] == bytes[index - 1] {
            run_len += 1;
        } else {
            flush_run(run_len, &mut summary);
            summary.transitions += 1;
            run_len = 1;
        }
    }
    flush_run(run_len, &mut summary);

    summary
}

fn flush_run(run_len: usize, summary: &mut RunLengthSummary) {
    summary.longest_run = summary.longest_run.max(run_len);
    if run_len >= 4 {
        summary.runs_ge4 += 1;
        summary.bytes_in_runs_ge4 = summary.bytes_in_runs_ge4.saturating_add(run_len as u64);
    }
}

/// Returns the longest equal-byte run.
#[must_use]
pub fn longest_run(bytes: &[u8]) -> usize {
    summarize(bytes).longest_run
}

#[cfg(test)]
mod tests {
    use super::summarize;

    #[test]
    fn summarizes_runs_and_transitions() {
        let summary = summarize(b"aaaabcc");
        assert_eq!(summary.total_bytes, 7);
        assert_eq!(summary.longest_run, 4);
        assert_eq!(summary.runs_ge4, 1);
        assert_eq!(summary.bytes_in_runs_ge4, 4);
        assert_eq!(summary.transitions, 2);
    }
}
