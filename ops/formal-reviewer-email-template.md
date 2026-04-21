# Email Template — Formal Reviewer Request (DR-2)

**Variables to fill** : `[NAME]`, `[CONTEXT_HOOK]`, `[DATE]`

---

Subject: Brief formal-proof review request — compositional dream
architecture for AI

Dear [NAME],

[CONTEXT_HOOK — e.g., "We met at X conference" / "I read your paper
on Y" / "Colleague Z suggested I reach out"].

I'm finishing up a research program called **dreamOfkiki** — a
formal framework axiomatizing dream-based knowledge consolidation in
artificial cognitive systems. The framework defines four primitive
operations (replay, downscale, restructure, recombine) as a
monoid-like algebraic structure over cognitive state transitions,
with a compositionality axiom (DR-2) that's central to the
framework's substrate-agnosticism claim (DR-3).

I'm reaching out because I'd value a formal reviewer's eye on the
DR-2 compositionality proof before we submit the paper to **PLOS
Computational Biology** (Paper 1 v0.2 draft, 22 pages, rendered
2026-04-20 — Nature HB retired as primary target on the same date).
The proof sketch is in place (closure + budget additivity +
functional composition + associativity) and now carries an
empirically-motivated precondition excluding `RESTRUCTURE ≺ REPLAY`
permutations (DualVer FC-PATCH, 2026-04-21). I'd value an external
pair of eyes on the case analysis for the remaining non-commutative
op pairs (in particular `recombine ∘ downscale` vs
`downscale ∘ recombine`), and on the DR-2' canonical-order fallback.

**Time estimate** : 2-3 hours of your time.

**In return** : Acknowledgements credit in the paper and in the
project's CONTRIBUTORS.md. **No authorship offered** — per
ICMJE / COPE / PLOS policy, a formal proof review does not cross
the substantive-contribution bar for authorship, and we want to
avoid any gift-authorship exposure. **No quid pro quo** : if
PLOS CB later invites you to formally peer-review this
manuscript, please decline this informal request and respond to
the formal invitation instead (don't double-dip). Paper 1 v0.2 is
already rendered (22 pages) and we are targeting a PLOS CB
submission window over the coming weeks, so your feedback would
land on a near-final manuscript.

**Materials I can share** : draft proof (~3 pages), framework spec
section 6 (axioms DR-0..DR-4), context on the monoid construction.

Would you be available around [DATE — target S6-S8 for circulation
and review] ? Happy to chat via email or a short call beforehand if
that's easier.

Best regards,
Clement Saillant
L'Electron Rare, France
clement@saillant.cc

---

**Footer note** : OSF pre-registration locked at
`10.17605/OSF.IO/Q6JYN` (https://osf.io/q6jyn, minted 2026-04-20) ;
arXiv preprint link to be added post-deposit (Paper 1 v0.2 window).
