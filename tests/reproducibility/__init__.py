"""R1 bit-exact reproducibility suite.

Contract R1 (see ``harness/storage/run_registry.py``) mandates that
for a fixed ``(c_version, profile, seed, commit_sha)`` tuple, the
same canonical scenario must produce byte-identical outputs across
runs, machines, and MLX driver versions.

This package exercises each of the 4 canonical real-weight ops
(replay, downscale, restructure, recombine) plus a full-pipeline
chain, hashing the serialized output state with SHA-256 and
comparing against golden hashes stored in
``tests/reproducibility/golden_hashes.json``.

Bootstrap mode : on the first run, unknown test names write the
computed hash with ``"status": "pending_review"`` and pass. The
user promotes ``"accepted"`` manually after inspection.
"""
