# AVX-512 hardware availability

Per `lscpu` survey 2026-05-01 across the local fleet (s0..s7). Recorded
here so #38 (AVX-512BW byte-class + UTF-8) and any future AVX-512-class
work has a known-good test surface that doesn't depend on GitHub-hosted
runners (none of which expose AVX-512).

| Host | CPU                                       | Microarch    | AVX-512 stack                                               | Notes |
| ---- | ----------------------------------------- | ------------ | ----------------------------------------------------------- | ----- |
| s0   | Intel Core i9-12900K                      | Alder Lake   | ❌ (hybrid; AVX-512 fused-off)                              | AVX2 + AVX-VNNI only. |
| s1   | Intel Xeon E3-1225 v6                     | Kaby Lake    | ❌                                                          | AVX2 only. |
| s2   | Intel Xeon E3-1225 v5                     | Skylake (desktop) | ❌                                                     | AVX2 only. |
| s3   | Intel Xeon W-2123                         | Skylake-X    | ✅ F · DQ · CD · BW · VL                                    | "Skylake-X core" AVX-512 set; sufficient for byte-class + UTF-8 simdutf. |
| s4   | Intel Core i5-12600K                      | Alder Lake   | ❌ (hybrid; AVX-512 fused-off)                              | AVX2 + AVX-VNNI only. |
| s5   | AMD Ryzen 7 7840HS                        | Zen 4 (mobile) | ✅ F · DQ · CD · BW · VL · IFMA · VBMI · VBMI2 · VNNI · BITALG · VPOPCNTDQ · BF16 | Richest stack; double-pumped AVX-512. |
| s6   | AMD Ryzen 7 7840HS                        | Zen 4 (mobile) | ✅ same as s5                                              | |
| s7   | AMD Ryzen 7 7840HS                        | Zen 4 (mobile) | ✅ same as s5                                              | |

**Implication for #38**: AVX-512BW byte-class classify + UTF-8 validation
can be implemented and bench-tested directly. Recommend **s5/s6/s7** as
the primary CI runners — Zen 4's full BW/VBMI/VBMI2/VNNI/BITALG suite
matches everything Lemire's simdutf AVX-512 path uses. **s3** is a good
secondary "Intel-server-AVX-512" reference for parity.

Self-hosted runner registration is documented in
`.github/workflows/ci-avx512.yml` (separate workflow gated on the
`avx512` self-hosted-runner labels).
