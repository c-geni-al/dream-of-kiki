# Spec Section Cross-Reference Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every `§5.1` citation that points to axiom content (axioms live in `§6.2`). Close GitHub issue #3.

**Architecture:** Targeted string replacements across 5 Markdown / Python files. No code logic changes. Scope is limited to axiom-related `§5.1` references; the ambiguous `§5.1 R3` references in 4 other files are explicitly out of scope (separate follow-up issue).

**Tech Stack:** Markdown, Python docstrings, git, Grep, Edit.

---

## Background

`docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` has this structure (verified by explore agent 2026-04-21):

- §5 Invariants (line 227)
  - §5.1 Family I (Information) — **this is what `§5.1` actually contains**
  - §5.2 Family S (Safety)
  - §5.3 Family K (Compute)
- §6 Axioms (line 313)
  - §6.1 Formal framework
  - §6.2 Axioms DR-0..DR-4 — **this is what `§5.1` should be pointing to**

## File Structure

Files to modify (confirmed line numbers):

| File | Lines to edit | Citations |
|------|---------------|-----------|
| `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` | 392 | 1 |
| `docs/axioms/AXIOMS.md` | 143 | 1 |
| `docs/specs/amendments/2026-04-21-dr2-empirical-falsification.md` | 6, 11, 23, 65, 71, 112 | 6 |
| `tests/conformance/axioms/test_dr2_compositionality_empirical.py` | 5, 15, 42, 45, 266, 278, 286, 291 | 8 |
| `tests/conformance/axioms/test_dr2_prime_canonical_order.py` | 4, 17, 47 | 3 |

**Out of scope** (open follow-up issue instead): `§5.1 R3` references in `CHANGELOG.md:84`, `docs/milestones/pilot-cycle3-sanity-1p5b.md:65,304,327`, `docs/superpowers/plans/2026-04-19-dreamofkiki-cycle3-atomic.md:32,297`, `scripts/pilot_cycle3_sanity.py:49,118`. `R3` is a pivot gate label, not a spec section, and requires human clarification of what it means.

---

### Task 1: Replace all axiom-related §5.1 citations with §6.2

**Files:**
- Modify: `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md:392`
- Modify: `docs/axioms/AXIOMS.md:143`
- Modify: `docs/specs/amendments/2026-04-21-dr2-empirical-falsification.md` (6 occurrences, all axiom-related)
- Modify: `tests/conformance/axioms/test_dr2_compositionality_empirical.py` (8 occurrences)
- Modify: `tests/conformance/axioms/test_dr2_prime_canonical_order.py` (3 occurrences)

- [ ] **Step 1: Verify starting state**

Run:
```bash
cd /Users/electron/Documents/Projets/dream-of-kiki
grep -rn '§5\.1' docs/ tests/conformance/axioms/ | grep -v 'R3' | wc -l
```
Expected: `19` (total axiom-related §5.1 citations before fix).

- [ ] **Step 2: Fix the spec self-reference**

Use Edit on `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`:
- old_string: `implication — DR-2 itself remains an unproven working axiom, §5.1).`
- new_string: `implication — DR-2 itself remains an unproven working axiom, §6.2).`

- [ ] **Step 3: Fix AXIOMS.md downstream copy**

Use Edit on `docs/axioms/AXIOMS.md`:
- old_string: `implication — DR-2 itself remains an unproven working axiom, §5.1).`
- new_string: `implication — DR-2 itself remains an unproven working axiom, §6.2).`

- [ ] **Step 4: Fix the amendment draft (bulk)**

Use Edit with `replace_all=true` on `docs/specs/amendments/2026-04-21-dr2-empirical-falsification.md`:
- old_string: `§5.1`
- new_string: `§6.2`

All 6 occurrences in this file are axiom-related (confirmed by explore agent). Safe to bulk-replace.

- [ ] **Step 5: Fix test_dr2_compositionality_empirical.py (bulk)**

Use Edit with `replace_all=true` on `tests/conformance/axioms/test_dr2_compositionality_empirical.py`:
- old_string: `§5.1`
- new_string: `§6.2`

All 8 occurrences in this file are axiom-related (confirmed by explore agent).

- [ ] **Step 6: Fix test_dr2_prime_canonical_order.py (bulk)**

Use Edit with `replace_all=true` on `tests/conformance/axioms/test_dr2_prime_canonical_order.py`:
- old_string: `§5.1`
- new_string: `§6.2`

All 3 occurrences are axiom-related.

- [ ] **Step 7: Verify no axiom-related §5.1 citations remain**

Run:
```bash
cd /Users/electron/Documents/Projets/dream-of-kiki
grep -rn '§5\.1' docs/ tests/conformance/axioms/ | grep -v 'R3'
```
Expected: empty output.

Run:
```bash
grep -rn '§5\.1' docs/ tests/conformance/axioms/
```
Expected: only `§5.1 R3` occurrences (out of scope for this plan).

- [ ] **Step 8: Run conformance tests**

Run:
```bash
uv run python -m pytest tests/conformance/ --no-cov -q
```
Expected: `45 passed, 12 xfailed` (no change from baseline — we only edited docstrings).

- [ ] **Step 9: Commit**

```bash
git add docs/specs/2026-04-17-dreamofkiki-framework-C-design.md \
        docs/axioms/AXIOMS.md \
        docs/specs/amendments/2026-04-21-dr2-empirical-falsification.md \
        tests/conformance/axioms/test_dr2_compositionality_empirical.py \
        tests/conformance/axioms/test_dr2_prime_canonical_order.py

git commit -m "$(cat <<'EOF'
docs(spec): fix §5.1 → §6.2 axiom cross-refs

Axioms DR-0..DR-4 live at §6.2 of the framework-C spec,
not §5.1. §5 is titled Invariants (families I, S, K).

Fix 19 citations across spec, AXIOMS.md, amendment draft,
and two axiom test files. All citations were axiom-
related; invariant content at §5.1 is correctly named.

Out of scope: §5.1 R3 references in CHANGELOG, milestones,
and scripts — those are pivot gate labels, not spec
sections, and need human clarification (follow-up issue).

Closes #3.
EOF
)"
```

Subject is 42 chars ≤ 50 ✓. Body lines ≤ 72.

### Task 2: Open follow-up issue for §5.1 R3 ambiguity

**Files:** none modified; creates a GitHub issue.

- [ ] **Step 1: Inventory the ambiguous references**

Run:
```bash
cd /Users/electron/Documents/Projets/dream-of-kiki
grep -rn '§5\.1 R3' .
```
Expected:
```
./CHANGELOG.md:84:- ... Pivot-4 branch per spec §5.1 R3 ...
./docs/milestones/pilot-cycle3-sanity-1p5b.md:65:... §5.1 R3
./docs/milestones/pilot-cycle3-sanity-1p5b.md:304:...
./docs/milestones/pilot-cycle3-sanity-1p5b.md:327:...
./docs/superpowers/plans/2026-04-19-dreamofkiki-cycle3-atomic.md:32:cf spec §5.1 R3
./docs/superpowers/plans/2026-04-19-dreamofkiki-cycle3-atomic.md:297:Pivot 4 (§5.1 R3)
./scripts/pilot_cycle3_sanity.py:49:pivot-4 branch per spec §5.1 R3
./scripts/pilot_cycle3_sanity.py:118:...
```

- [ ] **Step 2: Create the issue**

Run:
```bash
gh issue create --repo hypneum-lab/dream-of-kiki \
  --title "Clarify §5.1 R3 pivot-gate references (not a spec section)" \
  --label documentation \
  --body "$(cat <<'EOF'
## Context

Issue #3 fixed axiom-related \`§5.1\` → \`§6.2\` citations in doc5 files. A separate class remains: references to \`§5.1 R3\` which are NOT spec section citations at all.

## Affected files

- \`CHANGELOG.md:84\`
- \`docs/milestones/pilot-cycle3-sanity-1p5b.md\` lines 65, 304, 327
- \`docs/superpowers/plans/2026-04-19-dreamofkiki-cycle3-atomic.md\` lines 32, 297
- \`scripts/pilot_cycle3_sanity.py\` lines 49, 118

## Question

\`R3\` appears to be a reproducibility / pivot-gate label (e.g., "Pivot-4 rule 3") but there is no such labelled rule in the spec. Two options:

1. **Remove the spec reference**: rewrite these as "per internal Pivot-4 rule R3" or link to an internal ADR / milestone doc.
2. **Add the missing rule to the spec**: if R3 is a legitimate reproducibility rule that belongs in \`§5.1\` or \`§8.3\`, add it formally and keep the citations.

## Decision needed

Human clarification required before the text can be rewritten correctly.
EOF
)"
```

Expected output: a URL to the new issue.

- [ ] **Step 3: Record the issue number**

Capture the returned URL / issue number in your commit notes or the response to the user. No code change in this step.

---

## Self-review checklist

- [x] Every file:line pair cites specific Explore evidence.
- [x] Every Edit uses exact old_string → new_string pairs.
- [x] No placeholders.
- [x] Types / method signatures not touched (doc-only change).
- [x] Tests re-run after changes to confirm no regression.
- [x] Out-of-scope items explicitly flagged + deferred to issue.
