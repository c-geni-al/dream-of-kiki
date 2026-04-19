"""fMRI harness layer — Studyforrest BOLD loader + RSA alignment.

Phase 2 track (c) of cycle 3. Loads BOLD time-series from the
ds000113 Studyforrest dataset (CC0), produces episode-wise 4-D
numpy arrays aligned to stimulus events, and exposes an HRF
canonical-double-gamma helper used by downstream HMM alignment
(C3.16) + CCA (C3.17).

Reference : docs/interfaces/fmri-schema.yaml (schema contract,
v0.7.0+PARTIAL locked at cycle-3 Phase 1 bump).
"""

__version__ = "0.1.0"
