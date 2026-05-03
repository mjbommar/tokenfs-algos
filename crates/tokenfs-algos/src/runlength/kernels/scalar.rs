use crate::runlength::RunLengthSummary;

/// Counts transitions where `bytes[i] != bytes[i - 1]`.
#[must_use]
pub fn transitions(bytes: &[u8]) -> u64 {
    if bytes.len() < 2 {
        return 0;
    }
    let mut count = 0_u64;
    for index in 1..bytes.len() {
        if bytes[index] != bytes[index - 1] {
            count += 1;
        }
    }
    count
}

/// Builds a full [`RunLengthSummary`] in one scalar pass.
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
