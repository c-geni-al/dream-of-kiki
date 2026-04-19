"""Base model registry — cycle 3 real-data lock (pre-cycle-3 lock #2).

Every entry pins ``(repo_id, revision_sha, file_sha256)`` for R1
reproducibility contract. See framework-C spec §8.4 and
``docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md``
§5 (risk R1 mitigation) + §8 (glossary : scale-axis).

SHA values obtained from HuggingFace metadata at 2026-04-19
(``https://huggingface.co/api/models/{repo_id}`` +
``/tree/main``). Run::

    python -c "from harness.real_models.base_model_registry \\
        import verify_all; print(verify_all())"

to validate that the recorded pins still match current HF state.
The verifier is network-free by default (returns local-self-check
booleans only) to keep test runs deterministic per R1 ; a live
HTTP check can be enabled by passing ``live=True``.

Note on model version (cycle-3 fallback) : the spec targets
Qwen3.5 at scale ``{1.5B, 7B, 35B}`` Q4. At 2026-04-19 the
Qwen3.5 series is not yet published on HuggingFace ; the closest
publicly-available MLX Q4-quantized weights are from the Qwen2.5
series (1.5B, 7B, 32B). We therefore pin Qwen2.5 MLX-Q4 as the
scale-slot occupant under registry keys ``qwen3p5-*`` and record
the fallback in ``notes``. When Qwen3.5 MLX-Q4 lands, a new pin
entry + DualVer bump (FC-PATCH or EC-MINOR depending on
behavioural delta) will replace the occupant in-place ; key
stability is preserved for downstream ``get_pin`` callers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SHA40_RE = re.compile(r"^[a-f0-9]{40}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class BaseModelPin:
    """Immutable pin for a base model entry.

    Attributes
    ----------
    name
        Canonical scale-slot key (e.g. ``qwen3p5-1p5b``). Stable
        across Qwen2.5 → Qwen3.5 upgrade.
    scale_params
        Nominal parameter count (pre-quantization). Used for the
        scale-axis in cycle-3 H5 tests.
    repo_id
        HuggingFace ``org/repo`` identifier for the pinned model.
    revision_sha
        40-char HuggingFace revision (git) SHA. Matches regex
        ``^[a-f0-9]{40}$``.
    file_sha256
        SHA-256 (LFS oid) of the main weight file (or the first
        shard of a multi-shard safetensors set). 64-char hex, or
        ``None`` if the HF API did not expose one at pin time.
    quantization
        Quantization scheme (e.g. ``4bit-mlx``, ``q4_K_M-gguf``).
    framework
        Inference framework (e.g. ``mlx-lm``, ``llama.cpp``).
    approx_ram_gb
        Approximate resident RAM for loaded weights (order-of-
        magnitude ; used for scheduling on Studio vs Mac Mini).
    notes
        Free-text provenance notes. See module-level docstring for
        the Qwen2.5-fallback rationale.
    """

    name: str
    scale_params: int
    repo_id: str
    revision_sha: str
    file_sha256: str | None
    quantization: str
    framework: str
    approx_ram_gb: float
    notes: str = ""


# Pin values recorded 2026-04-19 from HuggingFace API.
# Method :
#   curl -s https://huggingface.co/api/models/{repo_id} -> .sha
#   curl -s https://huggingface.co/api/models/{repo_id}/tree/main
#       -> siblings[*].lfs.oid for model*.safetensors
# Re-run verify_all() after any pin update.

REGISTRY: dict[str, BaseModelPin] = {
    "qwen3p5-1p5b": BaseModelPin(
        name="qwen3p5-1p5b",
        scale_params=1_500_000_000,
        repo_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        revision_sha="8b403126fc14f14cfc99bb4cfa72ecbc129ea677",
        file_sha256=(
            "0979f33d1bc58afcf696d13f57977644e7b11a6f0eec3e631d"
            "8e9463d18c0717"
        ),
        quantization="4bit-mlx",
        framework="mlx-lm",
        approx_ram_gb=1.0,
        notes=(
            "Qwen2.5 fallback for Qwen3.5 slot ; single-shard "
            "model.safetensors 868 MB ; license apache-2.0 ; "
            "estimated ~150-200 tok/s on M3 Ultra."
        ),
    ),
    "qwen3p5-7b": BaseModelPin(
        name="qwen3p5-7b",
        scale_params=7_000_000_000,
        repo_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
        revision_sha="c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed",
        file_sha256=(
            "86110f368236b53cf4c2336f991a85703b17bcc60bb75f292b"
            "4002ec0219f071"
        ),
        quantization="4bit-mlx",
        framework="mlx-lm",
        approx_ram_gb=4.5,
        notes=(
            "Qwen2.5 fallback for Qwen3.5 slot ; single-shard "
            "model.safetensors 4.28 GB ; license apache-2.0 ; "
            "estimated ~60-90 tok/s on M3 Ultra."
        ),
    ),
    "qwen3p5-35b": BaseModelPin(
        name="qwen3p5-35b",
        scale_params=32_500_000_000,
        repo_id="mlx-community/Qwen2.5-32B-Instruct-4bit",
        revision_sha="2938092373e5f97b95538884112085364c2da315",
        file_sha256=(
            "3187f89267bdebe362410a3b23c2767d9d0707f4ffbbf7a945"
            "e5cd0abf535a21"
        ),
        quantization="4bit-mlx",
        framework="mlx-lm",
        approx_ram_gb=20.0,
        notes=(
            "Qwen2.5-32B fallback for Qwen3.5-35B slot "
            "(closest available public MLX-Q4 ; ~35B target ≈ "
            "32B actual, -8%). Multi-shard 4x safetensors, "
            "file_sha256 pins shard 1/4 (model-00001-of-00004) ; "
            "shards 2-4 oids : "
            "31547d7a3e6583eea8bcc1b7230680f8132143c43c19cab3e"
            "ab75f6b31d33e33, "
            "d00aec153b1ea6540446c1769ef76004e4ee11855844919f4"
            "8434f0d4f77b33f, "
            "4dae2dd0355ac7ad5440ccd4d5f60a08084cda924c0584"
            "02b979654359150743. "
            "Total 18.4 GB ; license apache-2.0 ; "
            "estimated ~25-40 tok/s on M3 Ultra "
            "(ref : Qwen3.5-35B-A3B Opus distill @ 162 tok/s on "
            "RTX 4090, MLX should undershoot by ~4x at similar "
            "scale per memory)."
        ),
    ),
}


def get_pin(name: str) -> BaseModelPin:
    """Return the :class:`BaseModelPin` for scale-slot ``name``.

    Raises :class:`KeyError` if the slot is not registered ; the
    error message lists available slots to ease discovery.
    """
    if name not in REGISTRY:
        available = sorted(REGISTRY.keys())
        raise KeyError(
            f"no base model pinned for slot {name!r} ; "
            f"available : {available}"
        )
    return REGISTRY[name]


def verify_all(live: bool = False) -> dict[str, bool]:
    """Validate pin self-consistency for every registered slot.

    Local-only checks (always run, network-free) :

    - ``revision_sha`` is a 40-char lowercase hex string ;
    - ``file_sha256``, if present, is a 64-char lowercase hex
      string ;
    - ``scale_params`` is strictly positive ;
    - ``repo_id`` is non-empty and contains exactly one ``/``.

    Live HTTP check (opt-in via ``live=True``) : fetches
    ``https://huggingface.co/api/models/{repo_id}`` and compares
    the returned ``sha`` field against the recorded
    ``revision_sha``. Kept opt-in so test runs stay deterministic
    and offline-friendly per R1.

    Returns a mapping ``name -> bool`` (``True`` = all checks
    passed for that entry).
    """
    results: dict[str, bool] = {}
    for name, pin in REGISTRY.items():
        ok = True
        if not _SHA40_RE.match(pin.revision_sha):
            ok = False
        if pin.file_sha256 is not None and not _SHA256_RE.match(
            pin.file_sha256
        ):
            ok = False
        if pin.scale_params <= 0:
            ok = False
        if not pin.repo_id or pin.repo_id.count("/") != 1:
            ok = False
        if ok and live:
            ok = _verify_live(pin)
        results[name] = ok
    return results


def _verify_live(pin: BaseModelPin) -> bool:
    """Fetch HF API and confirm ``revision_sha`` still matches.

    Imported lazily so the module has no hard network dependency
    at import time. Any network/parse failure returns ``False``
    (conservative — caller must interpret as "needs re-pin").
    """
    try:  # pragma: no cover - network path
        from urllib.request import urlopen
        import json

        url = f"https://huggingface.co/api/models/{pin.repo_id}"
        with urlopen(url, timeout=10) as resp:  # noqa: S310
            payload = json.load(resp)
        return payload.get("sha") == pin.revision_sha
    except Exception:  # pragma: no cover - network path
        return False
