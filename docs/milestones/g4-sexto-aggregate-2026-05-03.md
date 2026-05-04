# G4-sexto aggregate verdict

**Date** : 2026-05-03
**Pre-registration** : [docs/osf-prereg-g4-sexto-pilot.md](../osf-prereg-g4-sexto-pilot.md)

## Summary

- H6-A (CIFAR-100, 100-class scale) confirmed : **True**
- H6-B (Tiny-ImageNet, 200-class scale) confirmed : **False** (deferred — Option B locked)
- H6-C (universality conjunction) state : **deferred**
- H6-C confirmed : **False**
- H6-C partial : **False**
- H6-C deferred : **True**
- H5-C → H6-C universality extension (4 benchmarks × 4 substrates) : **False**

## H6-A — CIFAR-100 (n_classes=10 per task, G4SmallCNN)

- mean P_max (mog) : 0.3622
- mean P_max (none) : 0.3580
- Hedges' g (mog vs none) : 0.0570
- Welch t : 0.1966
- Welch p two-sided (alpha = 0.0167) : 0.8450
- fail_to_reject_h0 : True -> H6-A confirmed = True

*Honest reading* : Welch fail-to-reject = absence of evidence at this N for a difference between mog and none — under H6-A specifically, this **is** the predicted positive empirical claim that RECOMBINE adds nothing measurable beyond REPLAY+DOWNSCALE on the small CNN substrate at CIFAR-100 100-class scale.

## H6-B — Tiny-ImageNet (n_classes=20 per task, G4MediumCNN)

DEFERRED (compute Option B locked ; Step 2 will run in a G4-septimo follow-up).

## H6-C — universality conjunction (4 benchmarks × 4 substrates)

State : **deferred**

Deferred : Option B locked at pre-reg ; Tiny-ImageNet step deferred to G4-septimo. The H6-C conjunction is an open empirical question. Under H6-A confirmed, universality is provisionally extended to {FMNIST, CIFAR-10, CIFAR-100} × {3-layer MLP, 5-layer MLP, small CNN}, pending Tiny-IN evidence.

## Verdict — DR-4 evidence

Per pre-reg §6 : EC stays PARTIAL across all outcomes ; FC stays at C-v0.12.0. Under H6-A confirmed (the locked Option B success path), the partial refutation of DR-4 established by G4-ter and universalised by G4-quinto is further extended to CIFAR-100 at 100-class scale. Under the deferred Option B path, the H6-C conjunction is incomplete ; STABLE promotion remains blocked pending Tiny-IN / ImageNet-1k / transformer / hierarchical E-SNN follow-ups (pre-reg §6 row 6 of G4-quinto).
