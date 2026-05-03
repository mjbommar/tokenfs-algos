//! libmagic-style format sniffer.
//!
//! Identifies common file formats from short anchored magic-byte
//! sequences plus two heuristic fallbacks (text-likeness and entropy).
//! The sniffer is built specifically for storage-layer decisions:
//!
//! - "Is this content already compressed?" — gates whether to invoke a
//!   general-purpose compressor on the payload.
//! - "Is this content already binary?" — gates whether to attempt
//!   MIME-sniffing or text extraction.
//!
//! # Detection strategy
//!
//! Detection is layered, cheapest-first:
//!
//! 1. **Magic-byte DFA.** All magic-byte rules feed a `crate::search::PackedDfa`
//!    that scans the input once. A match becomes a [`Detection`] with
//!    confidence 100. Rules with non-zero anchor offsets (notably
//!    `tar`'s `ustar` magic at offset 257) are post-filtered against
//!    the rule's expected offset; mismatched DFA hits are skipped and
//!    the iterator continues.
//! 2. **Container brand dispatch.** A few magics share a common token
//!    and only differentiate on a secondary brand window — `RIFF`
//!    routes to [`Format::Wav`] vs [`Format::Webp`] based on the
//!    `WAVE`/`WEBP` brand at offset 8, and `ftyp` routes to
//!    [`Format::Mp4`], [`Format::Heic`], or [`Format::Avif`] based on
//!    the brand at offset 8.
//! 3. **Text tail.** When no magic matches, [`crate::byteclass::validate_utf8`]
//!    decides between [`Format::AsciiText`], [`Format::Utf8Text`],
//!    [`Format::Json`], and [`Format::Xml`] using the first
//!    non-whitespace byte of the input.
//! 4. **Entropy fallback.** When neither magic nor text fits,
//!    [`crate::histogram::summary::byte_value_moments`] decides
//!    [`Format::HighEntropy`] (when variance, skewness, and excess
//!    kurtosis match a near-uniform byte distribution — the signature
//!    of compressed/encrypted/encoded content with no surviving magic)
//!    or [`Format::Unknown`].
//!
//! # Allocation requirement
//!
//! Both [`Sniffer`] and the stateless [`detect`] function require the
//! `alloc` (or `std`) feature because `crate::search::PackedDfa`
//! itself stores its state table in a `Vec<u32>`.
//!
//! # Performance
//!
//! - DFA construction takes on the order of milliseconds — once.
//! - Per-input detection is one linear scan plus optional UTF-8 /
//!   moments passes. For payloads where Layer 1 fires, this is bounded
//!   by the offset of the matching magic (usually ≤ a few hundred
//!   bytes — the DFA can early-exit at the first accepted hit).
//!
//! Use [`Sniffer`] when classifying many payloads in a row; the cached
//! DFA is on the order of 100x faster than [`detect`] in repeated use.
//!
//! # Coverage
//!
//! See the source module-level constants for the curated rule table.
//! ~30 formats are covered, spanning the major already-compressed
//! groups (general-purpose compressors, lossy media, archives), the
//! columnar/structured group used by analytics workloads, common
//! executable formats, and a small text classification tail. Brotli
//! has no fixed magic and is intentionally absent.

extern crate alloc;

use alloc::vec::Vec;

use crate::byteclass::validate_utf8;
use crate::histogram::summary::byte_value_moments;
use crate::search::packed_dfa::PackedDfa;

/// Identified format of a byte payload.
///
/// Variants are grouped by storage-layer relevance — see
/// [`Format::is_compressible`] and [`Format::category`] for the
/// canonical roll-ups.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Format {
    // Already-compressed (don't try to compress again)
    /// gzip framed payload (`1f 8b`).
    Gzip,
    /// zstandard framed payload (`28 b5 2f fd`).
    Zstd,
    /// xz framed payload (`fd 37 7a 58 5a 00`).
    Xz,
    /// bzip2 framed payload (`BZh`).
    Bzip2,
    /// lz4 framed payload (`04 22 4d 18`).
    Lz4,
    /// brotli payload — has no fixed magic; reserved for future
    /// heuristic detection. Currently unreachable from [`detect`].
    Brotli,

    // Already-compressed media (most lossy)
    /// JPEG image (`ff d8 ff`).
    Jpeg,
    /// PNG image (`89 50 4e 47 0d 0a 1a 0a`).
    Png,
    /// GIF image (`GIF87a` or `GIF89a`).
    Gif,
    /// WebP image (`RIFF` ... `WEBP` at offsets 0/8).
    Webp,
    /// HEIC image (ISOBMFF `ftypheic`).
    Heic,
    /// AVIF image (ISOBMFF `ftypavif`).
    Avif,
    /// MP3 audio (`ID3` tag or MPEG-1/2 frame sync at offset 0).
    Mp3,
    /// MP4 / MOV container (ISOBMFF `ftyp` family other than HEIC/AVIF).
    Mp4,
    /// WAV audio (`RIFF` ... `WAVE` at offsets 0/8).
    Wav,
    /// Ogg container (`OggS`).
    Ogg,
    /// FLAC audio (`fLaC`).
    Flac,

    // Container/archive
    /// ZIP / JAR / OOXML / ODF / EPUB (`PK\x03\x04` or `PK\x05\x06`).
    Zip,
    /// POSIX tar archive (`ustar` at offset 257).
    Tar,
    /// 7-zip archive (`37 7a bc af 27 1c`).
    SevenZ,

    // Columnar / structured
    /// Apache Parquet (`PAR1` at offset 0).
    Parquet,
    /// Apache Arrow IPC stream/file (`ARROW1\0\0`).
    ArrowIpc,
    /// Feather V1 columnar file (`FEA1`).
    Feather,
    /// SQLite database (`SQLite format 3\0`).
    Sqlite,

    // Documents
    /// PDF document (`%PDF-`).
    Pdf,
    /// PostScript document (`%!PS`).
    PostScript,

    // Executables
    /// ELF executable / shared object / core file.
    Elf,
    /// 32-bit Mach-O executable (big-endian or little-endian magic).
    MachO32,
    /// 64-bit Mach-O executable (big-endian or little-endian magic).
    MachO64,
    /// Windows PE/COFF executable (`MZ` DOS stub).
    Pe,
    /// WebAssembly module (`\0asm` plus version 1).
    Wasm,

    // Text-ish (only if looks-like detection fires)
    /// Valid UTF-8 with non-ASCII bytes.
    Utf8Text,
    /// Strictly ASCII printable plus whitespace.
    AsciiText,
    /// JSON-shaped text (UTF-8 starting with `{` or `[`).
    Json,
    /// XML-shaped text (UTF-8 starting with `<?xml` or a `<` tag).
    Xml,

    // Encrypted / random-looking (high entropy + no magic match)
    /// No magic; byte distribution looks near-uniform (random,
    /// encrypted, or a compressor whose framing was stripped).
    HighEntropy,

    /// No magic; no text or entropy signal that crossed the threshold.
    Unknown,
}

impl Format {
    /// Best-guess MIME type (canonical), if known.
    ///
    /// Returns `None` for [`Format::Unknown`], [`Format::HighEntropy`],
    /// and [`Format::Brotli`] (no canonical MIME for raw brotli).
    #[must_use]
    pub const fn mime(self) -> Option<&'static str> {
        match self {
            Self::Gzip => Some("application/gzip"),
            Self::Zstd => Some("application/zstd"),
            Self::Xz => Some("application/x-xz"),
            Self::Bzip2 => Some("application/x-bzip2"),
            Self::Lz4 => Some("application/x-lz4"),
            Self::Brotli => None,
            Self::Jpeg => Some("image/jpeg"),
            Self::Png => Some("image/png"),
            Self::Gif => Some("image/gif"),
            Self::Webp => Some("image/webp"),
            Self::Heic => Some("image/heic"),
            Self::Avif => Some("image/avif"),
            Self::Mp3 => Some("audio/mpeg"),
            Self::Mp4 => Some("video/mp4"),
            Self::Wav => Some("audio/wav"),
            Self::Ogg => Some("application/ogg"),
            Self::Flac => Some("audio/flac"),
            Self::Zip => Some("application/zip"),
            Self::Tar => Some("application/x-tar"),
            Self::SevenZ => Some("application/x-7z-compressed"),
            Self::Parquet => Some("application/vnd.apache.parquet"),
            Self::ArrowIpc => Some("application/vnd.apache.arrow.file"),
            Self::Feather => Some("application/vnd.apache.arrow.feather"),
            Self::Sqlite => Some("application/vnd.sqlite3"),
            Self::Pdf => Some("application/pdf"),
            Self::PostScript => Some("application/postscript"),
            Self::Elf => Some("application/x-elf"),
            Self::MachO32 | Self::MachO64 => Some("application/x-mach-binary"),
            Self::Pe => Some("application/vnd.microsoft.portable-executable"),
            Self::Wasm => Some("application/wasm"),
            Self::Utf8Text | Self::AsciiText => Some("text/plain"),
            Self::Json => Some("application/json"),
            Self::Xml => Some("application/xml"),
            Self::HighEntropy | Self::Unknown => None,
        }
    }

    /// Hint for the storage layer: should this content be passed to a
    /// general-purpose compressor?
    ///
    /// `false` for already-compressed payloads (gzip, zstd, lossy
    /// media, archives, columnar formats whose pages are individually
    /// compressed) and for [`Format::HighEntropy`] (entropy near the
    /// compression-incompressible band). `true` for text-shaped
    /// content, executables, and [`Format::Unknown`] (default to
    /// "try"). Lossy media is grouped with already-compressed because
    /// re-compression yields negligible gains while burning CPU.
    #[must_use]
    pub const fn is_compressible(self) -> bool {
        match self {
            // Compression containers — re-compressing yields nothing.
            Self::Gzip
            | Self::Zstd
            | Self::Xz
            | Self::Bzip2
            | Self::Lz4
            | Self::Brotli
            // Lossy media — the codec already removed redundancy.
            | Self::Jpeg
            | Self::Png
            | Self::Gif
            | Self::Webp
            | Self::Heic
            | Self::Avif
            | Self::Mp3
            | Self::Mp4
            | Self::Wav
            | Self::Ogg
            | Self::Flac
            // Archive containers — assume the payload is already
            // packed with a per-entry compressor (true for ZIP/7z; not
            // necessarily for tar, but tar is usually the inner layer
            // of a tar.gz/tar.zst pipeline so we conservatively skip).
            | Self::Zip
            | Self::Tar
            | Self::SevenZ
            // Columnar files compress their pages internally.
            | Self::Parquet
            | Self::ArrowIpc
            | Self::Feather
            // Random-looking content has no compressible structure.
            | Self::HighEntropy => false,
            // Everything else benefits from a general-purpose pass.
            Self::Sqlite
            | Self::Pdf
            | Self::PostScript
            | Self::Elf
            | Self::MachO32
            | Self::MachO64
            | Self::Pe
            | Self::Wasm
            | Self::Utf8Text
            | Self::AsciiText
            | Self::Json
            | Self::Xml
            | Self::Unknown => true,
        }
    }

    /// Three-letter category for grouping.
    ///
    /// Returned values:
    /// - `"img"` — still images.
    /// - `"vid"` — video / mixed-media containers.
    /// - `"aud"` — audio.
    /// - `"txt"` — text-shaped content.
    /// - `"bin"` — everything else with a known signature.
    /// - `"unk"` — no signature / no signal (also `HighEntropy`).
    #[must_use]
    pub const fn category(self) -> &'static str {
        match self {
            Self::Jpeg | Self::Png | Self::Gif | Self::Webp | Self::Heic | Self::Avif => "img",
            Self::Mp4 => "vid",
            Self::Mp3 | Self::Wav | Self::Ogg | Self::Flac => "aud",
            Self::Utf8Text | Self::AsciiText | Self::Json | Self::Xml => "txt",
            Self::Gzip
            | Self::Zstd
            | Self::Xz
            | Self::Bzip2
            | Self::Lz4
            | Self::Brotli
            | Self::Zip
            | Self::Tar
            | Self::SevenZ
            | Self::Parquet
            | Self::ArrowIpc
            | Self::Feather
            | Self::Sqlite
            | Self::Pdf
            | Self::PostScript
            | Self::Elf
            | Self::MachO32
            | Self::MachO64
            | Self::Pe
            | Self::Wasm => "bin",
            Self::HighEntropy | Self::Unknown => "unk",
        }
    }
}

/// Result of one format-detection call.
#[derive(Clone, Debug)]
pub struct Detection {
    /// Identified format (or [`Format::Unknown`] when nothing fired).
    pub format: Format,
    /// Byte offset in the input where the matching signature starts.
    ///
    /// Most magics live at offset 0; [`Format::Tar`]'s `ustar` magic
    /// is at offset 257. For heuristic-only detections (text or
    /// entropy fallbacks) this is `0`.
    pub offset: usize,
    /// Length of the matched signature in bytes.
    ///
    /// For heuristic detections this is `0`.
    pub length: usize,
    /// Confidence in `[0, 100]`.
    ///
    /// `100` for an exact magic-byte match, `90` for a magic that
    /// passed a secondary brand check (RIFF/`ftyp` dispatch), `60`
    /// for the text-shape heuristics, `50` for [`Format::HighEntropy`],
    /// and `0` for [`Format::Unknown`].
    pub confidence: u8,
}

impl Detection {
    /// Constructs an [`Format::Unknown`] detection.
    #[must_use]
    pub const fn unknown() -> Self {
        Self {
            format: Format::Unknown,
            offset: 0,
            length: 0,
            confidence: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Magic-byte rule table
// ---------------------------------------------------------------------------

/// One magic-byte rule — a literal byte sequence that must appear at a
/// specific offset, plus the format it identifies and a "scope" enum
/// for how the matched format is finalized (direct, or "brand-pending"
/// for cases like `RIFF` that require a secondary check).
#[derive(Clone, Copy, Debug)]
struct MagicRule {
    /// Literal magic bytes.
    pattern: &'static [u8],
    /// Required start offset in the input.
    offset: usize,
    /// Format mapped to this rule, or a sentinel for brand dispatch.
    target: RuleTarget,
}

#[derive(Clone, Copy, Debug)]
enum RuleTarget {
    /// Direct map: pattern hit at `offset` ⇒ this [`Format`].
    Direct(Format),
    /// `RIFF` was matched at offset 0; resolve `Wav` / `Webp` (or
    /// fall through to [`Format::Unknown`]) by inspecting offset 8.
    Riff,
    /// `ftyp` was matched at offset 4; resolve [`Format::Heic`] /
    /// [`Format::Avif`] / [`Format::Mp4`] from the brand at offset 8.
    Ftyp,
}

/// All curated magic-byte rules. Order is significant only when
/// patterns overlap (the DFA reports first end-position; we then
/// iterate for the first match that satisfies its `offset` constraint).
///
/// References:
/// - <https://en.wikipedia.org/wiki/List_of_file_signatures>
/// - <https://github.com/file/file/tree/master/magic/Magdir>
const MAGIC_RULES: &[MagicRule] = &[
    // ---- Compression containers ----
    MagicRule {
        pattern: b"\x1f\x8b",
        offset: 0,
        target: RuleTarget::Direct(Format::Gzip),
    },
    MagicRule {
        pattern: b"\x28\xb5\x2f\xfd",
        offset: 0,
        target: RuleTarget::Direct(Format::Zstd),
    },
    MagicRule {
        pattern: b"\xfd\x37\x7a\x58\x5a\x00",
        offset: 0,
        target: RuleTarget::Direct(Format::Xz),
    },
    MagicRule {
        pattern: b"BZh",
        offset: 0,
        target: RuleTarget::Direct(Format::Bzip2),
    },
    MagicRule {
        pattern: b"\x04\x22\x4d\x18",
        offset: 0,
        target: RuleTarget::Direct(Format::Lz4),
    },
    // ---- Lossy media (still images & raster containers) ----
    MagicRule {
        pattern: b"\xff\xd8\xff",
        offset: 0,
        target: RuleTarget::Direct(Format::Jpeg),
    },
    MagicRule {
        pattern: b"\x89PNG\r\n\x1a\n",
        offset: 0,
        target: RuleTarget::Direct(Format::Png),
    },
    MagicRule {
        pattern: b"GIF87a",
        offset: 0,
        target: RuleTarget::Direct(Format::Gif),
    },
    MagicRule {
        pattern: b"GIF89a",
        offset: 0,
        target: RuleTarget::Direct(Format::Gif),
    },
    // RIFF + WEBP/WAVE — primary token at offset 0; brand checked
    // separately at offset 8 once the DFA reports the hit.
    MagicRule {
        pattern: b"RIFF",
        offset: 0,
        target: RuleTarget::Riff,
    },
    // ftyp container — primary token is `ftyp` at offset 4 (the
    // preceding 4 bytes are the ISOBMFF box length). Brand resolution
    // looks at offset 8 once the hit fires.
    MagicRule {
        pattern: b"ftyp",
        offset: 4,
        target: RuleTarget::Ftyp,
    },
    // MP3: ID3 tag prefix or raw MPEG audio frame sync.
    MagicRule {
        pattern: b"ID3",
        offset: 0,
        target: RuleTarget::Direct(Format::Mp3),
    },
    // MPEG-1/2 frame sync candidates: 11 bits set + version + layer.
    // We list the four most common (mp3 layer III, all bitrate/sample
    // combos collapse to one of these top bytes after the 0xff sync).
    MagicRule {
        pattern: b"\xff\xfb",
        offset: 0,
        target: RuleTarget::Direct(Format::Mp3),
    },
    MagicRule {
        pattern: b"\xff\xfa",
        offset: 0,
        target: RuleTarget::Direct(Format::Mp3),
    },
    MagicRule {
        pattern: b"\xff\xf3",
        offset: 0,
        target: RuleTarget::Direct(Format::Mp3),
    },
    MagicRule {
        pattern: b"\xff\xf2",
        offset: 0,
        target: RuleTarget::Direct(Format::Mp3),
    },
    MagicRule {
        pattern: b"OggS",
        offset: 0,
        target: RuleTarget::Direct(Format::Ogg),
    },
    MagicRule {
        pattern: b"fLaC",
        offset: 0,
        target: RuleTarget::Direct(Format::Flac),
    },
    // ---- Archives ----
    MagicRule {
        pattern: b"PK\x03\x04",
        offset: 0,
        target: RuleTarget::Direct(Format::Zip),
    },
    MagicRule {
        pattern: b"PK\x05\x06",
        offset: 0,
        target: RuleTarget::Direct(Format::Zip),
    },
    MagicRule {
        pattern: b"ustar",
        offset: 257,
        target: RuleTarget::Direct(Format::Tar),
    },
    MagicRule {
        pattern: b"\x37\x7a\xbc\xaf\x27\x1c",
        offset: 0,
        target: RuleTarget::Direct(Format::SevenZ),
    },
    // ---- Columnar / structured ----
    MagicRule {
        pattern: b"PAR1",
        offset: 0,
        target: RuleTarget::Direct(Format::Parquet),
    },
    MagicRule {
        pattern: b"ARROW1\x00\x00",
        offset: 0,
        target: RuleTarget::Direct(Format::ArrowIpc),
    },
    MagicRule {
        pattern: b"FEA1",
        offset: 0,
        target: RuleTarget::Direct(Format::Feather),
    },
    MagicRule {
        pattern: b"SQLite format 3\x00",
        offset: 0,
        target: RuleTarget::Direct(Format::Sqlite),
    },
    // ---- Documents ----
    MagicRule {
        pattern: b"%PDF-",
        offset: 0,
        target: RuleTarget::Direct(Format::Pdf),
    },
    MagicRule {
        pattern: b"%!PS",
        offset: 0,
        target: RuleTarget::Direct(Format::PostScript),
    },
    // ---- Executables ----
    MagicRule {
        pattern: b"\x7fELF",
        offset: 0,
        target: RuleTarget::Direct(Format::Elf),
    },
    MagicRule {
        pattern: b"\xfe\xed\xfa\xce",
        offset: 0,
        target: RuleTarget::Direct(Format::MachO32),
    },
    MagicRule {
        pattern: b"\xce\xfa\xed\xfe",
        offset: 0,
        target: RuleTarget::Direct(Format::MachO32),
    },
    MagicRule {
        pattern: b"\xfe\xed\xfa\xcf",
        offset: 0,
        target: RuleTarget::Direct(Format::MachO64),
    },
    MagicRule {
        pattern: b"\xcf\xfa\xed\xfe",
        offset: 0,
        target: RuleTarget::Direct(Format::MachO64),
    },
    // PE/COFF: `MZ` DOS stub at offset 0. The full PE check would
    // also require `PE\0\0` at the offset stored at 0x3c..0x40, but
    // the DOS stub alone is rarely a false positive on real inputs.
    MagicRule {
        pattern: b"MZ",
        offset: 0,
        target: RuleTarget::Direct(Format::Pe),
    },
    MagicRule {
        pattern: b"\x00asm\x01\x00\x00\x00",
        offset: 0,
        target: RuleTarget::Direct(Format::Wasm),
    },
];

// Confidence levels assigned to each detection layer.
const CONFIDENCE_MAGIC: u8 = 100;
const CONFIDENCE_BRAND: u8 = 90;
const CONFIDENCE_TEXT: u8 = 60;
const CONFIDENCE_ENTROPY: u8 = 50;

// Heuristic threshold: how many leading bytes we examine when
// determining the first non-whitespace byte for JSON/XML shape.
// `< 1 KiB` is enough to distinguish the four text branches.
const TEXT_PROBE_LEN: usize = 1024;

// ---------------------------------------------------------------------------
// Sniffer
// ---------------------------------------------------------------------------

/// Pre-built detector that caches the magic-byte DFA.
///
/// Building a [`PackedDfa`] over the rule table is on the order of
/// milliseconds; per-input scanning is on the order of microseconds.
/// Use [`Sniffer::new`] once and call [`Sniffer::detect`] for each
/// payload when classifying many inputs.
#[derive(Clone, Debug)]
pub struct Sniffer {
    dfa: PackedDfa,
    rules: &'static [MagicRule],
    /// Maximum required `offset + pattern.len()` over all rules — used
    /// to cap the DFA scan when looking for an offset-anchored rule
    /// (we still must visit every byte for offset-0 rules to fire,
    /// but we can stop the scan early once no remaining rule could
    /// possibly start at any later position).
    ///
    /// Currently informational; the DFA itself stops at first match.
    #[allow(dead_code)]
    max_anchor: usize,
}

impl Default for Sniffer {
    fn default() -> Self {
        Self::new()
    }
}

impl Sniffer {
    /// Builds the magic-byte DFA from the curated rule table.
    ///
    /// Construction is `O(P)` over the total length of the patterns —
    /// effectively constant-time for a fixed rule set.
    #[must_use]
    pub fn new() -> Self {
        let patterns: Vec<&[u8]> = MAGIC_RULES.iter().map(|rule| rule.pattern).collect();
        // Use try_new with expect because MAGIC_RULES is a compile-time
        // constant we control: pattern count and per-pattern length are
        // bounded and well below the u32 limits try_new validates. The
        // panicking PackedDfa::new is gated on userspace (audit-R10 #2)
        // so we route through the always-available fallible primitive.
        let dfa = PackedDfa::try_new(&patterns).expect("MAGIC_RULES is compile-time bounded");
        let max_anchor = MAGIC_RULES
            .iter()
            .map(|rule| rule.offset + rule.pattern.len())
            .max()
            .unwrap_or(0);
        Self {
            dfa,
            rules: MAGIC_RULES,
            max_anchor,
        }
    }

    /// Identifies `bytes`'s format using the cached DFA.
    ///
    /// Empty or short inputs never match a magic rule and fall
    /// through to the text/entropy heuristics; very short inputs
    /// (`< 16` bytes) skip the entropy check because the moment
    /// estimates are noise-dominated below that point.
    #[must_use]
    pub fn detect(&self, bytes: &[u8]) -> Detection {
        if bytes.is_empty() {
            return Detection::unknown();
        }

        // Layer 1: walk DFA hits, accept the first one whose start
        // offset matches the rule's `offset` constraint and whose
        // bytes literally equal the rule's pattern. The byte-equality
        // verification is required because `PackedDfa` collapses
        // bytes beyond the first 32 distinct values into a shared
        // wildcard class, which can route unrelated bytes through the
        // same DFA edges and report spurious hits at the wrong state.
        //
        // Cap the DFA scan window at `max_anchor` (audit-R10 #6): no
        // magic rule's start is past `max(rule.offset)`, so scanning
        // beyond the longest anchor only burns cycles and would let
        // an attacker-supplied multi-GB buffer turn a kernel/FUSE
        // sniffing call into an arbitrary-time DoS.
        let scan_end = self.max_anchor.min(bytes.len());
        let scan = &bytes[..scan_end];
        for (start, pat_idx) in self.dfa.find_iter(scan) {
            let rule = self.rules[pat_idx];
            if start != rule.offset {
                continue;
            }
            let end = start + rule.pattern.len();
            if end > bytes.len() || &bytes[start..end] != rule.pattern {
                continue;
            }
            if let Some(detection) = finalize_rule(&rule, bytes, start) {
                return detection;
            }
            // Rule was a brand-pending dispatch (RIFF/ftyp) and the
            // brand check failed; keep scanning for a later magic.
        }

        // Layer 2: text-shape heuristics on valid UTF-8.
        if let Some(detection) = detect_text(bytes) {
            return detection;
        }

        // Layer 3: entropy fallback.
        detect_entropy(bytes)
    }
}

// ---------------------------------------------------------------------------
// Stateless one-shot
// ---------------------------------------------------------------------------

/// Stateless one-shot detection.
///
/// This builds a fresh [`PackedDfa`] on each call. For repeated use
/// prefer [`Sniffer`], which amortizes the build cost. Requires the
/// `alloc` (or `std`) feature.
#[must_use]
pub fn detect(bytes: &[u8]) -> Detection {
    Sniffer::new().detect(bytes)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Resolves a magic-rule hit into a final [`Detection`]. Returns
/// `None` when the rule was a brand-pending dispatch and the brand
/// did not match any known target (so the scan should continue).
fn finalize_rule(rule: &MagicRule, bytes: &[u8], start: usize) -> Option<Detection> {
    match rule.target {
        RuleTarget::Direct(format) => Some(Detection {
            format,
            offset: start,
            length: rule.pattern.len(),
            confidence: CONFIDENCE_MAGIC,
        }),
        RuleTarget::Riff => {
            // `RIFF` matched at offset 0; brand window is bytes 8..12.
            // We need at least 12 bytes for the brand to exist.
            if bytes.len() < 12 {
                return None;
            }
            let brand = &bytes[8..12];
            let format = match brand {
                b"WEBP" => Format::Webp,
                b"WAVE" => Format::Wav,
                _ => return None,
            };
            Some(Detection {
                format,
                offset: 0,
                length: 12,
                confidence: CONFIDENCE_BRAND,
            })
        }
        RuleTarget::Ftyp => {
            // `ftyp` matched at offset 4; brand window is bytes 8..12.
            if bytes.len() < 12 {
                return None;
            }
            let brand = &bytes[8..12];
            let format = match brand {
                b"heic" | b"heix" | b"hevc" | b"hevx" | b"mif1" | b"msf1" => Format::Heic,
                b"avif" | b"avis" => Format::Avif,
                // Most other ftyp brands (`isom`, `mp42`, `M4A `, `M4V `,
                // `qt  `, `dash`, `iso2`, ...) are video/MP4 family.
                _ => Format::Mp4,
            };
            Some(Detection {
                format,
                offset: 0,
                length: 12,
                confidence: CONFIDENCE_BRAND,
            })
        }
    }
}

/// Text-shape detection: returns `Some(detection)` when the input is
/// valid UTF-8 (or its valid prefix covers the probe window) and
/// matches one of the four text branches.
fn detect_text(bytes: &[u8]) -> Option<Detection> {
    if bytes.is_empty() {
        return None;
    }

    let validation = validate_utf8(bytes);
    // Accept either fully valid UTF-8, or a long-enough valid prefix
    // (a single trailing-byte error in a long file — typical for
    // transports that may have been truncated mid-character).
    let usable = if validation.valid {
        bytes.len()
    } else if validation.valid_up_to >= bytes.len().min(TEXT_PROBE_LEN) {
        validation.valid_up_to
    } else {
        return None;
    };

    let probe = &bytes[..usable.min(TEXT_PROBE_LEN)];

    // First non-whitespace byte determines JSON/XML shape.
    let first_non_ws = probe.iter().find(|&&b| !is_ascii_whitespace(b)).copied()?;

    // Determine ASCII-vs-Utf8 once.
    let mut all_ascii = true;
    let mut has_control = false;
    for &b in probe {
        if b >= 0x80 {
            all_ascii = false;
        }
        if b < 0x20 && !is_ascii_whitespace(b) {
            has_control = true;
        }
        if b == 0x7f {
            has_control = true;
        }
    }
    // Control bytes (other than tab/newline/CR) point away from text.
    if has_control {
        return None;
    }

    let format = match first_non_ws {
        b'{' | b'[' if all_ascii => Format::Json,
        b'{' | b'[' => Format::Json,
        b'<' => Format::Xml,
        _ => {
            if all_ascii {
                Format::AsciiText
            } else {
                Format::Utf8Text
            }
        }
    };

    Some(Detection {
        format,
        offset: 0,
        length: 0,
        confidence: CONFIDENCE_TEXT,
    })
}

#[inline]
const fn is_ascii_whitespace(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

/// Entropy-shape detection: declares [`Format::HighEntropy`] when the
/// byte distribution looks discrete-uniform (variance ≈ 5461.25, mean
/// ≈ 127.5, skewness ≈ 0, excess kurtosis ≈ -1.2). Otherwise returns
/// [`Format::Unknown`].
///
/// Inputs shorter than 16 bytes always fall through to `Unknown` —
/// the moment estimates are too noisy to be useful at that scale.
fn detect_entropy(bytes: &[u8]) -> Detection {
    const MIN_FOR_ENTROPY: usize = 16;
    if bytes.len() < MIN_FOR_ENTROPY {
        return Detection::unknown();
    }

    let m = byte_value_moments(bytes);
    // Theoretical near-uniform target:
    //   mean = 127.5, variance = 5461.25, |skew| ~= 0, kurtosis ~= -1.2
    // The thresholds are intentionally loose — real compressed/
    // encrypted payloads sit close to but not exactly at the discrete
    // uniform.
    let mean_ok = (m.mean - 127.5).abs() < 12.0;
    let var_ok = (m.variance - 5461.25).abs() / 5461.25 < 0.10;
    let skew_ok = m.skewness.abs() < 0.20;
    let kurt_ok = (m.kurtosis - (-1.2)).abs() < 0.30;

    if mean_ok && var_ok && skew_ok && kurt_ok {
        Detection {
            format: Format::HighEntropy,
            offset: 0,
            length: 0,
            confidence: CONFIDENCE_ENTROPY,
        }
    } else {
        Detection::unknown()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // The `vec!` and `format!` macros are not in the no-std prelude;
    // alias them from `alloc` for the alloc-only build (audit-R6 #164).
    use alloc::format;
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;

    fn xorshift_bytes(seed: u64, n: usize) -> Vec<u8> {
        let mut state = seed;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            out.push((state & 0xff) as u8);
        }
        out
    }

    fn fixture_with_magic(magic: &[u8], offset: usize, total: usize) -> Vec<u8> {
        let mut buf = vec![0_u8; total.max(offset + magic.len())];
        buf[offset..offset + magic.len()].copy_from_slice(magic);
        buf
    }

    #[test]
    fn empty_input_is_unknown() {
        let d = detect(&[]);
        assert_eq!(d.format, Format::Unknown);
        assert_eq!(d.confidence, 0);
    }

    #[test]
    fn very_short_input_is_unknown() {
        // Single byte: too short for any magic, too short for entropy,
        // doesn't trip text path either (depends — let's pick a
        // non-text byte).
        let d = detect(&[0xff]);
        assert_eq!(d.format, Format::Unknown);
    }

    #[test]
    fn detect_gzip() {
        let d = detect(&fixture_with_magic(b"\x1f\x8b", 0, 64));
        assert_eq!(d.format, Format::Gzip);
        assert_eq!(d.offset, 0);
        assert_eq!(d.length, 2);
        assert!(d.confidence >= 90);
    }

    #[test]
    fn detect_zstd() {
        let d = detect(&fixture_with_magic(b"\x28\xb5\x2f\xfd", 0, 64));
        assert_eq!(d.format, Format::Zstd);
        assert!(d.confidence >= 90);
    }

    #[test]
    fn detect_xz() {
        let d = detect(&fixture_with_magic(b"\xfd\x37\x7a\x58\x5a\x00", 0, 64));
        assert_eq!(d.format, Format::Xz);
        assert!(d.confidence >= 90);
    }

    #[test]
    fn detect_bzip2() {
        let d = detect(&fixture_with_magic(b"BZh", 0, 64));
        assert_eq!(d.format, Format::Bzip2);
        assert!(d.confidence >= 90);
    }

    #[test]
    fn detect_lz4() {
        let d = detect(&fixture_with_magic(b"\x04\x22\x4d\x18", 0, 64));
        assert_eq!(d.format, Format::Lz4);
        assert!(d.confidence >= 90);
    }

    #[test]
    fn detect_jpeg() {
        let d = detect(&fixture_with_magic(b"\xff\xd8\xff", 0, 64));
        assert_eq!(d.format, Format::Jpeg);
        assert!(d.confidence >= 90);
    }

    #[test]
    fn detect_png() {
        let d = detect(&fixture_with_magic(b"\x89PNG\r\n\x1a\n", 0, 64));
        assert_eq!(d.format, Format::Png);
        assert!(d.confidence >= 90);
    }

    #[test]
    fn detect_gif87() {
        let d = detect(&fixture_with_magic(b"GIF87a", 0, 64));
        assert_eq!(d.format, Format::Gif);
    }

    #[test]
    fn detect_gif89() {
        let d = detect(&fixture_with_magic(b"GIF89a", 0, 64));
        assert_eq!(d.format, Format::Gif);
    }

    #[test]
    fn detect_webp() {
        let mut buf = vec![0_u8; 64];
        buf[..4].copy_from_slice(b"RIFF");
        buf[4..8].copy_from_slice(&32_u32.to_le_bytes());
        buf[8..12].copy_from_slice(b"WEBP");
        let d = detect(&buf);
        assert_eq!(d.format, Format::Webp);
        assert!(d.confidence >= 90);
    }

    #[test]
    fn detect_wav() {
        let mut buf = vec![0_u8; 64];
        buf[..4].copy_from_slice(b"RIFF");
        buf[4..8].copy_from_slice(&32_u32.to_le_bytes());
        buf[8..12].copy_from_slice(b"WAVE");
        let d = detect(&buf);
        assert_eq!(d.format, Format::Wav);
    }

    #[test]
    fn riff_with_unknown_brand_is_not_riff() {
        // Don't claim it's WAV/WebP if the brand window doesn't match.
        let mut buf = vec![0_u8; 64];
        buf[..4].copy_from_slice(b"RIFF");
        buf[8..12].copy_from_slice(b"XXXX");
        let d = detect(&buf);
        assert_ne!(d.format, Format::Webp);
        assert_ne!(d.format, Format::Wav);
    }

    #[test]
    fn detect_heic() {
        let mut buf = vec![0_u8; 64];
        buf[..4].copy_from_slice(&32_u32.to_be_bytes());
        buf[4..8].copy_from_slice(b"ftyp");
        buf[8..12].copy_from_slice(b"heic");
        let d = detect(&buf);
        assert_eq!(d.format, Format::Heic);
    }

    #[test]
    fn detect_avif() {
        let mut buf = vec![0_u8; 64];
        buf[..4].copy_from_slice(&32_u32.to_be_bytes());
        buf[4..8].copy_from_slice(b"ftyp");
        buf[8..12].copy_from_slice(b"avif");
        let d = detect(&buf);
        assert_eq!(d.format, Format::Avif);
    }

    #[test]
    fn detect_mp4() {
        let mut buf = vec![0_u8; 64];
        buf[..4].copy_from_slice(&32_u32.to_be_bytes());
        buf[4..8].copy_from_slice(b"ftyp");
        buf[8..12].copy_from_slice(b"isom");
        let d = detect(&buf);
        assert_eq!(d.format, Format::Mp4);
    }

    #[test]
    fn detect_mp3_id3() {
        let d = detect(&fixture_with_magic(b"ID3\x04\x00", 0, 64));
        assert_eq!(d.format, Format::Mp3);
    }

    #[test]
    fn detect_mp3_sync() {
        let d = detect(&fixture_with_magic(b"\xff\xfb\x90\x00", 0, 64));
        assert_eq!(d.format, Format::Mp3);
    }

    #[test]
    fn detect_ogg() {
        let d = detect(&fixture_with_magic(b"OggS", 0, 64));
        assert_eq!(d.format, Format::Ogg);
    }

    #[test]
    fn detect_flac() {
        let d = detect(&fixture_with_magic(b"fLaC", 0, 64));
        assert_eq!(d.format, Format::Flac);
    }

    #[test]
    fn detect_zip() {
        let d = detect(&fixture_with_magic(b"PK\x03\x04", 0, 64));
        assert_eq!(d.format, Format::Zip);
    }

    #[test]
    fn detect_zip_eocd() {
        let d = detect(&fixture_with_magic(b"PK\x05\x06", 0, 64));
        assert_eq!(d.format, Format::Zip);
    }

    #[test]
    fn detect_tar() {
        // tar's magic lives at offset 257.
        let mut buf = vec![0_u8; 512];
        buf[257..262].copy_from_slice(b"ustar");
        let d = detect(&buf);
        assert_eq!(d.format, Format::Tar);
        assert_eq!(d.offset, 257);
    }

    #[test]
    fn tar_magic_at_wrong_offset_is_not_tar() {
        // `ustar` at offset 0 is NOT a tar — the rule is anchored.
        let d = detect(b"ustar plus other bytes that are not a tar archive header");
        assert_ne!(d.format, Format::Tar);
    }

    #[test]
    fn detect_7z() {
        let d = detect(&fixture_with_magic(b"\x37\x7a\xbc\xaf\x27\x1c", 0, 64));
        assert_eq!(d.format, Format::SevenZ);
    }

    #[test]
    fn detect_parquet() {
        let d = detect(&fixture_with_magic(b"PAR1", 0, 64));
        assert_eq!(d.format, Format::Parquet);
    }

    #[test]
    fn detect_arrow_ipc() {
        let d = detect(&fixture_with_magic(b"ARROW1\x00\x00", 0, 64));
        assert_eq!(d.format, Format::ArrowIpc);
    }

    #[test]
    fn detect_feather() {
        let d = detect(&fixture_with_magic(b"FEA1", 0, 64));
        assert_eq!(d.format, Format::Feather);
    }

    #[test]
    fn detect_sqlite() {
        let d = detect(&fixture_with_magic(b"SQLite format 3\x00", 0, 64));
        assert_eq!(d.format, Format::Sqlite);
    }

    #[test]
    fn detect_pdf() {
        let d = detect(&fixture_with_magic(b"%PDF-1.4", 0, 64));
        assert_eq!(d.format, Format::Pdf);
    }

    #[test]
    fn detect_postscript() {
        let d = detect(&fixture_with_magic(b"%!PS-Adobe", 0, 64));
        assert_eq!(d.format, Format::PostScript);
    }

    #[test]
    fn detect_elf() {
        let d = detect(&fixture_with_magic(b"\x7fELF\x02\x01\x01", 0, 64));
        assert_eq!(d.format, Format::Elf);
    }

    #[test]
    fn detect_macho32_be() {
        let d = detect(&fixture_with_magic(b"\xfe\xed\xfa\xce", 0, 64));
        assert_eq!(d.format, Format::MachO32);
    }

    #[test]
    fn detect_macho32_le() {
        let d = detect(&fixture_with_magic(b"\xce\xfa\xed\xfe", 0, 64));
        assert_eq!(d.format, Format::MachO32);
    }

    #[test]
    fn detect_macho64_be() {
        let d = detect(&fixture_with_magic(b"\xfe\xed\xfa\xcf", 0, 64));
        assert_eq!(d.format, Format::MachO64);
    }

    #[test]
    fn detect_macho64_le() {
        let d = detect(&fixture_with_magic(b"\xcf\xfa\xed\xfe", 0, 64));
        assert_eq!(d.format, Format::MachO64);
    }

    #[test]
    fn detect_pe() {
        let d = detect(&fixture_with_magic(b"MZ\x90\x00", 0, 64));
        assert_eq!(d.format, Format::Pe);
    }

    #[test]
    fn detect_wasm() {
        let d = detect(&fixture_with_magic(b"\x00asm\x01\x00\x00\x00", 0, 64));
        assert_eq!(d.format, Format::Wasm);
    }

    // ---- Text branches ----

    #[test]
    fn detect_lorem_ipsum_text() {
        let d = detect(b"Lorem ipsum dolor sit amet, consectetur adipiscing elit.");
        // Either AsciiText or Utf8Text — we don't pin the exact branch.
        assert!(matches!(d.format, Format::AsciiText | Format::Utf8Text));
    }

    #[test]
    fn detect_ascii_text_explicit() {
        let d = detect(b"hello world this is plain ascii text\n");
        assert_eq!(d.format, Format::AsciiText);
    }

    #[test]
    fn detect_utf8_text_with_high_bit() {
        let d = detect("hellö wörld with accénts".as_bytes());
        assert_eq!(d.format, Format::Utf8Text);
    }

    #[test]
    fn detect_json_object() {
        let d = detect(br#"{"foo": 1, "bar": [1, 2, 3]}"#);
        assert_eq!(d.format, Format::Json);
    }

    #[test]
    fn detect_json_array() {
        let d = detect(b"   [1, 2, 3]   ");
        assert_eq!(d.format, Format::Json);
    }

    #[test]
    fn detect_xml_decl() {
        let d = detect(br#"<?xml version="1.0"?><root/>"#);
        assert_eq!(d.format, Format::Xml);
    }

    #[test]
    fn detect_xml_no_decl() {
        let d = detect(b"<root><child>value</child></root>");
        assert_eq!(d.format, Format::Xml);
    }

    #[test]
    fn detect_high_entropy_random() {
        let d = detect(&xorshift_bytes(0xdead_beef_1234_5678, 4096));
        assert_eq!(d.format, Format::HighEntropy);
    }

    #[test]
    fn detect_unknown_low_entropy() {
        // Constant fill — not a recognised magic, not text, not
        // high-entropy random.
        let bytes = vec![0xab_u8; 4096];
        let d = detect(&bytes);
        assert_eq!(d.format, Format::Unknown);
    }

    /// Audit-R10 #6: the magic-rule scan must be bounded by
    /// `max_anchor`. Verify two ways: (a) the longest tail of arbitrary
    /// bytes does not change the verdict on a short prefix, (b) a
    /// pathological 2 MiB blob without a magic still completes
    /// promptly because Layer 1 only walks `max_anchor` bytes.
    #[test]
    fn detect_caps_layer1_scan_at_max_anchor() {
        // Real PNG header with garbage tail.
        let mut payload = Vec::with_capacity(2 * 1024 * 1024);
        payload.extend_from_slice(b"\x89PNG\r\n\x1a\n");
        payload.extend(xorshift_bytes(0xa5, 2 * 1024 * 1024));
        let detection = Sniffer::new().detect(&payload);
        assert_eq!(
            detection.format,
            Format::Png,
            "PNG magic with 2 MiB tail must still detect PNG",
        );

        // Pathological no-magic blob; with the cap, this only walks
        // `max_anchor` bytes for layer 1 (the layer 2/3 fallbacks are
        // bounded separately and acceptable on attacker-supplied
        // input).
        let blob = xorshift_bytes(0xa5, 2 * 1024 * 1024);
        let detection = Sniffer::new().detect(&blob);
        // Only assert it returns; the format itself depends on the
        // entropy fallback heuristic.
        let _ = detection.format;
    }

    // ---- Sniffer parity ----

    #[test]
    fn sniffer_matches_stateless_detect() {
        let cases: Vec<Vec<u8>> = vec![
            fixture_with_magic(b"\x1f\x8b", 0, 64),
            fixture_with_magic(b"\x89PNG\r\n\x1a\n", 0, 64),
            {
                let mut tar = vec![0; 512];
                tar[257..262].copy_from_slice(b"ustar");
                tar
            },
            br#"{"foo": 1}"#.to_vec(),
            xorshift_bytes(0x42, 4096),
            b"".to_vec(),
            b"hi".to_vec(),
        ];
        let sniffer = Sniffer::new();
        for case in &cases {
            let stateless = detect(case);
            let cached = sniffer.detect(case);
            assert_eq!(
                cached.format,
                stateless.format,
                "format mismatch for case len {}",
                case.len()
            );
            assert_eq!(cached.offset, stateless.offset);
            assert_eq!(cached.length, stateless.length);
            assert_eq!(cached.confidence, stateless.confidence);
        }
    }

    // ---- Format enum surface ----

    #[test]
    fn mime_round_trip() {
        // Spot check a few — we only need a representative each.
        assert_eq!(Format::Gzip.mime(), Some("application/gzip"));
        assert_eq!(Format::Json.mime(), Some("application/json"));
        assert_eq!(Format::Unknown.mime(), None);
        assert_eq!(Format::HighEntropy.mime(), None);
    }

    #[test]
    fn is_compressible_invariants() {
        // Compressed / lossy / archive / columnar / entropy → false.
        for f in [
            Format::Gzip,
            Format::Zstd,
            Format::Jpeg,
            Format::Png,
            Format::Mp4,
            Format::Zip,
            Format::Tar,
            Format::Parquet,
            Format::HighEntropy,
        ] {
            assert!(!f.is_compressible(), "{f:?} should not be compressible");
        }
        // Text / unknown / executable → true.
        for f in [
            Format::AsciiText,
            Format::Utf8Text,
            Format::Json,
            Format::Xml,
            Format::Pdf,
            Format::Elf,
            Format::Wasm,
            Format::Unknown,
        ] {
            assert!(f.is_compressible(), "{f:?} should be compressible");
        }
    }

    #[test]
    fn category_groups() {
        assert_eq!(Format::Jpeg.category(), "img");
        assert_eq!(Format::Mp4.category(), "vid");
        assert_eq!(Format::Mp3.category(), "aud");
        assert_eq!(Format::Json.category(), "txt");
        assert_eq!(Format::Elf.category(), "bin");
        assert_eq!(Format::Unknown.category(), "unk");
        assert_eq!(Format::HighEntropy.category(), "unk");
    }

    #[test]
    fn detection_unknown_constructor() {
        let d = Detection::unknown();
        assert_eq!(d.format, Format::Unknown);
        assert_eq!(d.offset, 0);
        assert_eq!(d.length, 0);
        assert_eq!(d.confidence, 0);
        // Make sure ToString-via-Debug works (sanity for the public
        // Debug derive, no functional assertion needed).
        let _ = format!("{d:?}").to_string();
    }
}
