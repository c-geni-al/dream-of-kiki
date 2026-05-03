# Wake-Sleep CL baseline — verify attempt (supersede note)

**Supersedes (status only) :** `wake-sleep-baseline-2026-05-03.md`
(numerical values, run_ids and seed grid in that dump are
unchanged ; this entry adds a verify-attempt outcome).

**Source under verify :** Alfarano et al. 2024, IEEE TNNLS,
arXiv 2401.08623. Bibkey `alfarano2024wakesleep`.
**Verified against :** arXiv 2401.08623v1 PDF (14 pp,
pdfTeX 1.40.25, 8.27 MB), pulled 2026-05-03 via WebFetch and
parsed with `pypdf`.

## Outcome — UNVERIFIED (mismatch)

The placeholder pair documented in the parent dump
(`forgetting_rate = 0.082`, `avg_accuracy = 0.847`,
`task_split = split_fmnist_5tasks`) does not match the paper.
Two distinct mismatches were identified ; both block a direct
replacement and require a maintainer-side decision before any
publication claim is filed.

### Mismatch 1 — benchmark not represented

Alfarano 2024 §4.1 ("Benchmarks") evaluates WSCL on three
class-incremental datasets, none of which is Split-FMNIST :

- Split CIFAR-10 (5 binary tasks, dreaming subset of CIFAR-100).
- Split FG-ImageNet (100 fine-grained animal classes, dreaming
  subset of 100 disjoint ImageNet classes).
- Tiny-ImageNet1/2 (first 100 classes as 5 × 20-class tasks ;
  remaining 100 classes as the dreaming dataset).

The `split_fmnist_5tasks` key in
`kiki_oniric/substrates/wake_sleep_cl_baseline.py` and in the
parent milestone dump therefore points at a benchmark that the
paper does not score, so a value imported from any of Tables 1
to 4 cannot be tagged with that key without changing what is
being claimed.

### Mismatch 2 — scale and identity

Tables 2 (forgetting Class-IL) and 3 (final average accuracy
with std) report percentages, not unit-interval decimals. For
the headline ER-ACE+WSCL ("Ours") cells :

| Benchmark | Buffer | FAA (Table 3) | Forgetting (Table 2) |
|-----------|--------|---------------|----------------------|
| CIFAR-10 | 200 | 71.15 ± 2.15 | 11.78 |
| CIFAR-10 | 500 | 74.18 ± 1.28 | 10.69 |
| Tiny-ImageNet1/2 | 200 | 35.68 ± 1.18 | 28.23 |
| Tiny-ImageNet1/2 | 500 | 41.25 ± 1.75 | 23.29 |
| FG-ImageNet | 200 | 12.51 ± 0.86 | 27.24 |
| FG-ImageNet | 1000 | 20.51 ± 0.56 | 33.53 |

(Buffer-free `Wake+REM` row, Table 3 : 41.58 / 25.68 / 6.27.)

Even after rescaling 0.082 → 8.2 % and 0.847 → 84.7 %, neither
value matches any cell of Table 2 or Table 3 at any buffer size
for any of the three benchmarks.

## Action retained

The numerical values, the `run_id` set
`{60a86e8…, 4b6b475e…, fcd2873d…}`, and the
`split_fmnist_5tasks` task_split key are **kept frozen** in the
parent dump and at the five other PLACEHOLDER sites. Fabricating
post-hoc values to match an undocumented derivation, or
silently re-keying the dump to a different benchmark, would
both violate the synthetic-vs-empirical discipline
(`docs/CLAUDE.md`, `papers/CLAUDE.md` numbers ↔ run_id rule).

The maintainer must choose one of :

1. **Re-key on an Alfarano benchmark.** Replace
   `split_fmnist_5tasks` with e.g. `cifar10_5tasks_buffer500`
   and import `(forgetting_rate, avg_accuracy) = (0.1069,
   0.7418)` from Tables 2-3 ER-ACE+WSCL row. Bumps the FC axis
   (schema change) and invalidates the three current `run_id`s.
2. **Switch comparator anchor.** Keep `split_fmnist_5tasks` and
   point the bibkey at a paper that does score Split-FMNIST.
   Bumps both FC (bibkey change) and rewrites the §7.7 caveat
   in `paper2/results.md` + `paper2-fr/results.md`.

Until that decision lands, the PLACEHOLDER discipline at the
six sites holds.

## Reproducing the verify attempt

The PDF was retrieved twice via WebFetch (`https://arxiv.org/
pdf/2401.08623` and `https://arxiv.org/pdf/2401.08623v1`) and
parsed locally with `pypdf 6.10.2`. Tables 1-4 are extracted in
plain text on pages 7-10 of the PDF. The abstract (page 1) and
§4.1 (page 6) confirm the benchmark list independently of the
table extraction.
