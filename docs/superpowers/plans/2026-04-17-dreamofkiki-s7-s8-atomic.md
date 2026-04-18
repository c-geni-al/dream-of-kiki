# dreamOfkiki S7-S8 Atomic Plan

> **Pour agents autonomes :** SKILL REQUIS — utiliser `superpowers:subagent-driven-development`. Les steps utilisent la syntaxe checkbox (`- [ ]`) pour le tracking.

**Goal** : finaliser P_min profile fonctionnel, swap worktree skeleton, gates G2 (P_min viable) et G3 (DR-2 proof peer-reviewed), démarrer P_equ. Phase pivot vers la phase ablation S9-S12.

**Architecture** : 7 tasks atomiques (S7.1-S7.4, S8.1-S8.3), TDD strict pour code, livrables docs pour gates. Total attendu : 7 commits.

**Tech Stack** : Python 3.12+ uv, pytest + hypothesis, dataclasses frozen, no concurrent yet (single-threaded swap protocol — concurrent worker S9+).

**Préréquis** :
- 36 commits dreamOfkiki, dernier `8ea0d45 docs(proof): add op-pair commutativity analysis`
- 35 tests passing, coverage 95.68%
- Framework C-v0.5.0+STABLE
- `kiki_oniric/dream/{episode,runtime}.py` + `operations/replay.py`
- 8 typed Protocols + 4 axiom property tests (DR-0, DR-1, DR-3 partial)
- G3-draft DR-2 proof circulé (S6.1-S6.3)

---

## Convention commits (validator-enforced)

- Subject ≤50 chars, format `<type>(<scope>): <description>`
- Scope ≥3 chars (single letters rejected — `(dream)` OK)
- Body lines ≤72 chars, 2-3 paragraphs required
- NO AI attribution
- NO `--no-verify`

---

## File structure après S7-S8

```
dreamOfkiki/
├── kiki_oniric/dream/
│   ├── operations/
│   │   ├── replay.py            ✅ existing
│   │   └── downscale.py         ← S7.1 (P_min op 2/2)
│   ├── guards/
│   │   ├── __init__.py          ← S7.2
│   │   └── finite.py            ← S7.2 (S2 invariant — NaN/Inf check)
│   └── swap.py                  ← S7.3 (skeleton swap protocol)
├── kiki_oniric/profiles/
│   ├── __init__.py              ← S7.4
│   ├── p_min.py                 ← S7.4 (replay+downscale wired)
│   └── p_equ.py                 ← S8.3 (skeleton)
├── tests/
│   ├── unit/
│   │   ├── test_downscale_op.py ← S7.1
│   │   ├── test_finite_guard.py ← S7.2
│   │   ├── test_swap.py         ← S7.3
│   │   ├── test_p_min.py        ← S7.4
│   │   └── test_p_equ.py        ← S8.3
│   └── conformance/
│       └── invariants/
│           └── test_s2_finite.py ← S7.2
└── docs/
    ├── proofs/
    │   └── g3-decision-log.md   ← S8.2 (G3 outcome record)
    └── milestones/
        └── g2-pmin-report.md    ← S8.1 (G2 viability report)
```

---

# Task S7.1 — downscale operation (P_min op 2/2, B-Tononi SHY)

**Goal** : second concrete op pour P_min. Skeleton : counter weights "shrunk", no real shrinkage on `np.ndarray` yet (lands S9+). 3 tests TDD.

**Files:**
- Create : `kiki_oniric/dream/operations/downscale.py`
- Create : `tests/unit/test_downscale_op.py`

## Step 1 — Write failing tests

Create `tests/unit/test_downscale_op.py` with exactly:

```python
"""Unit tests for downscale operation (P_min op 2/2, B-Tononi SHY)."""
from __future__ import annotations

import pytest

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.operations.downscale import (
    DownscaleOpState,
    downscale_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime


def make_downscale_episode(
    ep_id: str, factor: float
) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"shrink_factor": factor},
        operation_set=(Operation.DOWNSCALE,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=5_000, wall_time_s=0.5, energy_j=0.05),
        episode_id=ep_id,
    )


def test_downscale_records_factor() -> None:
    state = DownscaleOpState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.DOWNSCALE, downscale_handler(state)
    )
    runtime.execute(make_downscale_episode("de-d0", 0.95))
    assert state.total_episodes_handled == 1
    assert state.last_factor_applied == 0.95


def test_downscale_rejects_factor_out_of_range() -> None:
    state = DownscaleOpState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.DOWNSCALE, downscale_handler(state)
    )
    # SHY shrinkage : factor must be in (0, 1] — values outside
    # this range are nonsensical (no shrinkage or amplification).
    with pytest.raises(ValueError, match="shrink_factor"):
        runtime.execute(make_downscale_episode("de-d1", 1.5))
    with pytest.raises(ValueError, match="shrink_factor"):
        runtime.execute(make_downscale_episode("de-d2", 0.0))


def test_downscale_accumulates_compound_factor() -> None:
    """Compound factor across multiple episodes (factor1 * factor2)."""
    state = DownscaleOpState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.DOWNSCALE, downscale_handler(state)
    )
    runtime.execute(make_downscale_episode("de-d3", 0.9))
    runtime.execute(make_downscale_episode("de-d4", 0.8))
    assert state.total_episodes_handled == 2
    assert state.compound_factor == pytest.approx(0.9 * 0.8)
```

## Step 2 — Verify failing

Run: `uv run pytest tests/unit/test_downscale_op.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError`.

## Step 3 — Implement downscale.py

Create `kiki_oniric/dream/operations/downscale.py` with exactly:

```python
"""Downscale operation — B-Tononi SHY synaptic homeostasis source.

Skeleton version (S7.1): records shrink factor + compound product
across episodes. Real weight shrinkage on np.ndarray lands S9+ with
MLX integration.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from kiki_oniric.dream.episode import DreamEpisode


@dataclass
class DownscaleOpState:
    """Mutable counter state for downscale op across episodes."""

    total_episodes_handled: int = 0
    last_factor_applied: float = 1.0
    compound_factor: float = 1.0


def downscale_handler(
    state: DownscaleOpState,
) -> Callable[[DreamEpisode], None]:
    """Build a downscale handler bound to a state instance.

    Handler reads `shrink_factor` from input_slice (must be in (0, 1]),
    updates state. Real `W *= factor` lands S9+ with MLX.
    """

    def handler(episode: DreamEpisode) -> None:
        factor = episode.input_slice.get("shrink_factor", 1.0)
        if not (0.0 < factor <= 1.0):
            raise ValueError(
                f"shrink_factor must be in (0, 1], got {factor}"
            )
        state.total_episodes_handled += 1
        state.last_factor_applied = factor
        state.compound_factor *= factor

    return handler
```

## Step 4 — Verify passing

Run: `uv run pytest tests/unit/test_downscale_op.py -v --no-cov`
Expected: 3 passed.

Run: `uv run pytest`
Expected: 38 tests (35 + 3 new), coverage ≥90%.

## Step 5 — Commit + push

```bash
git add kiki_oniric/dream/operations/downscale.py tests/unit/test_downscale_op.py
git commit -m "feat(dream): add downscale op (B-Tononi SHY)"
```

Subject : 44 chars. Body 2-3 paragraphes : explique B-Tononi SHY source, factor (0,1] enforcement, compound_factor pour multi-episode, weight shrinkage réelle S9+. Complète P_min op set (replay + downscale).

Then `git push`.

---

# Task S7.2 — S2 finite guard (no NaN/Inf invariant)

**Goal** : implémenter S2 invariant (no NaN/Inf in W_scratch) comme guard utilisable par swap protocol. Tests unit + conformance invariant.

**Files:**
- Create : `kiki_oniric/dream/guards/__init__.py` (empty)
- Create : `kiki_oniric/dream/guards/finite.py`
- Create : `tests/unit/test_finite_guard.py`
- Create : `tests/conformance/invariants/test_s2_finite.py`

## Step 1 — Write unit + conformance tests

Create `tests/unit/test_finite_guard.py` with exactly:

```python
"""Unit tests for S2 finite guard (no NaN/Inf in weights)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from kiki_oniric.dream.guards.finite import (
    FiniteGuardError,
    check_finite,
)


def test_check_finite_accepts_clean_array() -> None:
    weights = np.array([0.1, -0.5, 1.2, 0.0])
    check_finite(weights)  # No exception


def test_check_finite_rejects_nan() -> None:
    weights = np.array([0.1, math.nan, 0.5])
    with pytest.raises(FiniteGuardError, match="NaN"):
        check_finite(weights)


def test_check_finite_rejects_inf() -> None:
    weights = np.array([0.1, math.inf, 0.5])
    with pytest.raises(FiniteGuardError, match="Inf"):
        check_finite(weights)
    weights = np.array([0.1, -math.inf, 0.5])
    with pytest.raises(FiniteGuardError, match="Inf"):
        check_finite(weights)


def test_check_finite_rejects_above_w_max() -> None:
    weights = np.array([0.1, 1e9])
    with pytest.raises(FiniteGuardError, match="bound"):
        check_finite(weights, w_max=1e6)


def test_check_finite_handles_dict_of_arrays() -> None:
    weights = {
        "layer1": np.array([0.1, 0.2]),
        "layer2": np.array([math.nan, 0.5]),
    }
    with pytest.raises(FiniteGuardError, match="layer2"):
        check_finite(weights)
```

Create `tests/conformance/invariants/test_s2_finite.py` with exactly:

```python
"""Conformance test for invariant S2 — no NaN/Inf in W_scratch."""
from __future__ import annotations

import math

import numpy as np
import pytest

from kiki_oniric.dream.guards.finite import (
    FiniteGuardError,
    check_finite,
)


def test_s2_invariant_blocks_nan_post_op() -> None:
    """S2 must abort swap when W_scratch contains NaN."""
    fake_post_op_weights = np.array([0.1, math.nan, 0.5])
    with pytest.raises(FiniteGuardError):
        check_finite(fake_post_op_weights)


def test_s2_invariant_passes_clean_post_op() -> None:
    """S2 should pass through valid weights silently."""
    fake_post_op_weights = np.array([0.05, -0.12, 0.3, 0.0])
    check_finite(fake_post_op_weights)
```

## Step 2 — Verify failing

Run: `uv run pytest tests/unit/test_finite_guard.py tests/conformance/invariants/test_s2_finite.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError`.

## Step 3 — Implement finite.py

Create `kiki_oniric/dream/guards/__init__.py` (empty file).

Create `kiki_oniric/dream/guards/finite.py` with exactly:

```python
"""S2 finite guard — no NaN/Inf, |w| ≤ w_max in weights.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §5.2
Invariant S2 — BLOCKING. Enforced before swap (pre-step 2).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import NDArray


DEFAULT_W_MAX = 1e6


class FiniteGuardError(Exception):
    """Raised when S2 invariant is violated (NaN, Inf, or |w| > w_max)."""


def check_finite(
    weights: NDArray | Mapping[str, NDArray],
    w_max: float = DEFAULT_W_MAX,
) -> None:
    """Verify all weights are finite and bounded.

    Accepts either a single array or a dict-of-arrays (e.g.,
    layer-keyed model weights).

    Raises FiniteGuardError on first violation, with location
    information when possible (key name for dict-of-arrays).
    """
    if isinstance(weights, Mapping):
        for key, arr in weights.items():
            try:
                check_finite(arr, w_max=w_max)
            except FiniteGuardError as exc:
                raise FiniteGuardError(
                    f"layer {key!r}: {exc}"
                ) from exc
        return

    arr = np.asarray(weights)

    if np.isnan(arr).any():
        raise FiniteGuardError("contains NaN")

    if np.isinf(arr).any():
        raise FiniteGuardError("contains Inf")

    abs_max = float(np.abs(arr).max()) if arr.size else 0.0
    if abs_max > w_max:
        raise FiniteGuardError(
            f"max |w| = {abs_max} exceeds bound {w_max}"
        )
```

## Step 4 — Verify passing

Run: `uv run pytest tests/unit/test_finite_guard.py tests/conformance/invariants/test_s2_finite.py -v --no-cov`
Expected: 7 passed (5 unit + 2 conformance).

Run: `uv run pytest`
Expected: 45 tests (38 + 7 new), coverage ≥90%.

## Step 5 — Commit + push

```bash
git add kiki_oniric/dream/guards/ tests/unit/test_finite_guard.py tests/conformance/invariants/test_s2_finite.py
git commit -m "feat(guard): add S2 finite check (NaN/Inf bound)"
```

Subject : 49 chars. Body 2-3 paragraphes : explique S2 BLOCKING invariant, NaN/Inf/|w|>w_max checks, dict-of-arrays support, swap protocol pre-step 2 enforcement.

Then `git push`.

---

# Task S7.3 — Swap protocol skeleton (S1 + S2 + S3 guards)

**Goal** : implémenter `swap_atomic` function avec guards S1 (retained), S2 (finite), S3 (topology) — version skeleton callable, NaN/finite enforcement réel, retained + topology stubs (real S9+).

**Files:**
- Create : `kiki_oniric/dream/swap.py`
- Create : `tests/unit/test_swap.py`

## Step 1 — Write failing tests

Create `tests/unit/test_swap.py` with exactly:

```python
"""Unit tests for swap protocol skeleton (S1 + S2 + S3 guards)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from kiki_oniric.dream.swap import (
    SwapAborted,
    SwapResult,
    swap_atomic,
)


def test_swap_succeeds_with_clean_scratch() -> None:
    w_awake = np.array([0.1, 0.2, 0.3])
    w_scratch = np.array([0.11, 0.21, 0.31])
    result = swap_atomic(
        w_awake=w_awake,
        w_scratch=w_scratch,
        retained_eval=lambda w: 0.95,
        retained_pre_acc=0.95,
        delta_regression=0.02,
    )
    assert isinstance(result, SwapResult)
    assert result.committed is True
    assert np.array_equal(result.w_new, w_scratch)


def test_swap_aborts_on_nan() -> None:
    w_awake = np.array([0.1, 0.2])
    w_scratch = np.array([math.nan, 0.2])
    with pytest.raises(SwapAborted, match="S2"):
        swap_atomic(
            w_awake=w_awake,
            w_scratch=w_scratch,
            retained_eval=lambda w: 0.95,
            retained_pre_acc=0.95,
            delta_regression=0.02,
        )


def test_swap_aborts_on_retained_regression() -> None:
    w_awake = np.array([0.1, 0.2])
    w_scratch = np.array([0.5, 0.6])
    with pytest.raises(SwapAborted, match="S1"):
        swap_atomic(
            w_awake=w_awake,
            w_scratch=w_scratch,
            retained_eval=lambda w: 0.50,  # huge regression
            retained_pre_acc=0.95,
            delta_regression=0.02,
        )


def test_swap_passes_marginal_regression_within_threshold() -> None:
    """Regression within delta_regression is acceptable."""
    w_awake = np.array([0.1])
    w_scratch = np.array([0.1])
    result = swap_atomic(
        w_awake=w_awake,
        w_scratch=w_scratch,
        retained_eval=lambda w: 0.94,  # 1pp below pre, within 2%
        retained_pre_acc=0.95,
        delta_regression=0.02,
    )
    assert result.committed is True
```

## Step 2 — Verify failing

Run: `uv run pytest tests/unit/test_swap.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError`.

## Step 3 — Implement swap.py

Create `kiki_oniric/dream/swap.py` with exactly:

```python
"""Swap protocol skeleton — atomic W_awake ← W_scratch promotion.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §7

Skeleton (S7.3) : enforces S2 (finite guard) and S1 (retained
non-regression). S3 (hierarchy guard) is a no-op stub here (real
topology validation lands S9+). K3 swap latency monitoring lands
S9+ with concurrent runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from kiki_oniric.dream.guards.finite import (
    FiniteGuardError,
    check_finite,
)


class SwapAborted(Exception):
    """Raised when a swap guard rejects W_scratch."""


@dataclass(frozen=True)
class SwapResult:
    """Immutable record of a successful swap."""

    w_new: NDArray
    retained_post_acc: float
    committed: bool


def swap_atomic(
    w_awake: NDArray,
    w_scratch: NDArray,
    retained_eval: Callable[[NDArray], float],
    retained_pre_acc: float,
    delta_regression: float = 0.02,
) -> SwapResult:
    """Atomic swap W_awake ← W_scratch with S1+S2 guards.

    1. S2 guard : finite + bounded check on W_scratch.
    2. S1 guard : retained_eval(W_scratch) ≥
       retained_pre_acc - delta_regression.
    3. Commit : promote W_scratch to W_awake.

    Raises SwapAborted with the violated invariant code (S1 or S2)
    when a guard rejects.
    """
    try:
        check_finite(w_scratch)
    except FiniteGuardError as exc:
        raise SwapAborted(f"S2 guard failed: {exc}") from exc

    retained_post = retained_eval(w_scratch)
    if retained_post < retained_pre_acc - delta_regression:
        raise SwapAborted(
            f"S1 guard failed: retained_post={retained_post} < "
            f"retained_pre={retained_pre_acc} - "
            f"delta={delta_regression}"
        )

    return SwapResult(
        w_new=w_scratch,
        retained_post_acc=retained_post,
        committed=True,
    )
```

## Step 4 — Verify passing

Run: `uv run pytest tests/unit/test_swap.py -v --no-cov`
Expected: 4 passed.

Run: `uv run pytest`
Expected: 49 tests (45 + 4 new), coverage ≥90%.

## Step 5 — Commit + push

```bash
git add kiki_oniric/dream/swap.py tests/unit/test_swap.py
git commit -m "feat(dream): add swap protocol skeleton (S1+S2)"
```

Subject : 48 chars. Body 2-3 paragraphes : swap_atomic function, SwapResult/SwapAborted, S2 finite guard delegate, S1 retained non-regression with delta_regression configurable, S3 stub à compléter S9+, K3 latency monitoring S9+.

Then `git push`.

---

# Task S7.4 — P_min profile wiring

**Goal** : assembler P_min profile = replay + downscale enregistrés sur runtime + swap protocol intégré. Test end-to-end.

**Files:**
- Create : `kiki_oniric/profiles/__init__.py` (empty)
- Create : `kiki_oniric/profiles/p_min.py`
- Create : `tests/unit/test_p_min.py`

## Step 1 — Write failing tests

Create `tests/unit/test_p_min.py` with exactly:

```python
"""Unit tests for P_min profile (replay + downscale + swap)."""
from __future__ import annotations

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.profiles.p_min import PMinProfile


def make_replay_de(ep_id: str, records: list[dict]) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"beta_records": records},
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=1000, wall_time_s=0.1, energy_j=0.01),
        episode_id=ep_id,
    )


def make_downscale_de(ep_id: str, factor: float) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"shrink_factor": factor},
        operation_set=(Operation.DOWNSCALE,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=500, wall_time_s=0.05, energy_j=0.005),
        episode_id=ep_id,
    )


def test_p_min_registers_replay_and_downscale() -> None:
    profile = PMinProfile()
    assert Operation.REPLAY in profile.runtime._handlers
    assert Operation.DOWNSCALE in profile.runtime._handlers


def test_p_min_executes_replay_then_downscale() -> None:
    profile = PMinProfile()
    profile.runtime.execute(
        make_replay_de("de-min0", [{"id": 1}, {"id": 2}])
    )
    profile.runtime.execute(make_downscale_de("de-min1", 0.95))
    assert profile.replay_state.total_episodes_handled == 1
    assert profile.replay_state.total_records_consumed == 2
    assert profile.downscale_state.total_episodes_handled == 1
    assert profile.downscale_state.compound_factor == 0.95


def test_p_min_log_contains_both_episodes() -> None:
    profile = PMinProfile()
    profile.runtime.execute(make_replay_de("de-min2", []))
    profile.runtime.execute(make_downscale_de("de-min3", 0.99))
    ids = [e.episode_id for e in profile.runtime.log]
    assert ids == ["de-min2", "de-min3"]
    assert all(e.completed for e in profile.runtime.log)
```

## Step 2 — Verify failing

Run: `uv run pytest tests/unit/test_p_min.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError`.

## Step 3 — Implement p_min.py

Create `kiki_oniric/profiles/__init__.py` (empty file).

Create `kiki_oniric/profiles/p_min.py` with exactly:

```python
"""P_min profile — minimal publishable consolidation.

Channels : β → 1 (curated buffer in, weight delta out).
Operations : {replay, downscale}.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
"""
from __future__ import annotations

from dataclasses import dataclass, field

from kiki_oniric.dream.episode import Operation
from kiki_oniric.dream.operations.downscale import (
    DownscaleOpState,
    downscale_handler,
)
from kiki_oniric.dream.operations.replay import (
    ReplayOpState,
    replay_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime


@dataclass
class PMinProfile:
    """Minimal profile : replay + downscale handlers wired."""

    runtime: DreamRuntime = field(default_factory=DreamRuntime)
    replay_state: ReplayOpState = field(default_factory=ReplayOpState)
    downscale_state: DownscaleOpState = field(
        default_factory=DownscaleOpState
    )

    def __post_init__(self) -> None:
        self.runtime.register_handler(
            Operation.REPLAY, replay_handler(self.replay_state)
        )
        self.runtime.register_handler(
            Operation.DOWNSCALE, downscale_handler(self.downscale_state)
        )
```

## Step 4 — Verify passing

Run: `uv run pytest tests/unit/test_p_min.py -v --no-cov`
Expected: 3 passed.

Run: `uv run pytest`
Expected: 52 tests (49 + 3 new), coverage ≥90%.

## Step 5 — Commit + push

```bash
git add kiki_oniric/profiles/ tests/unit/test_p_min.py
git commit -m "feat(profile): add P_min wiring (replay+downscale)"
```

Subject : 49 chars. Body 2-3 paragraphes : P_min profile = β → 1, replay + downscale, dataclass avec runtime + replay_state + downscale_state, handlers auto-registered en __post_init__. Foundation pour gate G2 viability test S8.

Then `git push`.

---

# Task S8.1 — G2 P_min viability report

**Goal** : produire un rapport `g2-pmin-report.md` qui évalue la viabilité de P_min selon les critères du gate G2 (master spec §7.2 : accuracy P_min ≥ baseline − 2%, runtime stable 48h).

Skeleton version : Constate ce qui est testé/non testé et l'état du gate au moment de S8.

**Files:**
- Create : `docs/milestones/g2-pmin-report.md`

## Step 1 — Create G2 report

Create `docs/milestones/g2-pmin-report.md` with exactly:

```markdown
# G2 — P_min Viability Report

**Gate** : G2 (P_min viable)
**Target week** : S8
**Status** : **PARTIAL — skeleton implementation, real evaluation S9+**

## Gate criteria (from master spec §7.2)

- [ ] **Accuracy criterion** : P_min retained accuracy ≥ baseline − 2%
- [ ] **Stability criterion** : runtime stable for 48 hours continuous

## Current evidence (S8)

### Implementation status

- ✅ DreamEpisode 5-tuple dataclass (S5.1)
- ✅ DreamRuntime scheduler with DR-0 log guarantee (S5.2)
- ✅ DR-0 + DR-1 property tests passing (S5.3)
- ✅ Replay operation skeleton (S5.4) — counts records, no weight mutation
- ✅ Downscale operation skeleton (S7.1) — records factor, no weight mutation
- ✅ S2 finite guard (S7.2)
- ✅ Swap protocol skeleton with S1 + S2 guards (S7.3)
- ✅ P_min profile wiring (replay + downscale on runtime) (S7.4)

### Test coverage

- 52 tests total (target 90% coverage)
- `kiki_oniric/dream/`: 100% coverage on episode, runtime, operations
- DR-0 + DR-1 + DR-3 axiom property tests passing
- S2 invariant conformance test passing

### What is NOT yet tested

- No MLX integration → no real weight matrix updates
- No retained benchmark consumption → S1 guard exercised on synthetic data only
- No multi-day runtime stability test (would require deployment)
- No comparison vs baseline accuracy on mega-v2 dataset

## Decision (S8)

**Branch GO-CONDITIONAL (default at S8 with skeleton)** :
- Code structure validated, contracts in place
- Defer real accuracy/stability measurement to S9+ once MLX wiring done
- Commit to G2 gate **conditional** on S9+ evidence

**Branch GO-FULL** (if MLX integration completes faster than expected) :
- Run baseline + P_min on mega-v2 retained benchmark
- Measure accuracy delta, confirm ≥ baseline − 2%
- Run 48h continuous stability test
- Lock G2 fully

**Branch NO-GO (Pivot A)** :
- If S9+ measurements show accuracy < baseline − 2%, OR runtime instability
- Activate Pivot A : single-paper TMLR/ICLR workshop on engineering results only
- Framework paper deferred to cycle 2

## Action

S8 day end : decide GO-CONDITIONAL (default) and document path to GO-FULL in S9-S10.
```

## Step 2 — Commit + push

```bash
git add docs/milestones/g2-pmin-report.md
git commit -m "docs(milestone): G2 P_min viability skeleton"
```

Subject : 46 chars. Body 2-3 paragraphes : G2 gate report skeleton, GO-CONDITIONAL default branch, GO-FULL si MLX intégration finit S9+, Pivot A si évidence empirique S9+ rejette.

Then `git push`.

---

# Task S8.2 — G3 decision log

**Goal** : produire `g3-decision-log.md` qui enregistre la décision finale du gate G3 (DR-2 proof peer-reviewed). Skeleton version : preuve circulée, attend retour reviewer formel S6-S8.

**Files:**
- Create : `docs/proofs/g3-decision-log.md`

## Step 1 — Create G3 decision log

Create `docs/proofs/g3-decision-log.md` with exactly:

```markdown
# G3 Decision Log — DR-2 Compositionality Proof Gate

**Gate** : G3 (DR-2 proof peer-reviewed)
**Target week** : S8
**Status** : **PENDING reviewer feedback (S6-S8 circulation window)**

## Reviewer status

| Item | Status | Notes |
|------|--------|-------|
| Reviewer recruited (Q_CR.1 b) | TODO | See `ops/formal-reviewer-recruitment.md` |
| Draft v0.1 sent | TODO | Pending recruitment |
| Feedback received | TODO | Pending review |
| Revision v0.2 produced | TODO | Pending feedback |
| Final review approval | TODO | Pending revision |

## Decision branches (from S6.2 circulation log)

### Branch DR-2-STRICT (default — happy path)
- Reviewer confirms strict DR-2 proof (closure + budget additivity +
  functional composition + associativity)
- **Action** : tag framework C-v0.7.0+STABLE
- **Paper 1 target** : Nature Human Behaviour
- **Status flag** : `+STABLE`

### Branch DR-2-PRIME (fallback — reviewer flags gap)
- Reviewer identifies issue in strict proof (e.g., free-semigroup
  vs primitive-set distinction needs more rigor)
- Adopt DR-2' (canonical order only) per
  `dr2-compositionality.md` fallback section
- **Action** : tag framework C-v0.7.0-PRIME+STABLE
- **Paper 1 target** : PLoS Computational Biology / Cognitive Science
- **Status flag** : `+STABLE` (different ID)

### Branch G3-FAIL (emergency — no reviewer + sub-agent flags issues)
- No human reviewer confirmed by S8 AND sub-agent `critic` flags
  proof issues
- **Action** : Pivot A activated per master spec §7.3
- **Scope reduction** : single-paper TMLR/ICLR workshop, framework
  paper deferred cycle 2
- **Paper 2 re-positioned** as primary deliverable

## Outcome (to be filled at S8 end)

- **Branch chosen** : TBD
- **Date** : TBD
- **Framework version tagged** : TBD
- **Paper 1 journal target** : TBD
- **Justification** (3-5 sentences) : TBD

## Lessons learned (post-G3)

(populated after gate decision)
```

## Step 2 — Commit + push

```bash
git add docs/proofs/g3-decision-log.md
git commit -m "docs(proof): add G3 decision log skeleton"
```

Subject : 43 chars. Body 2-3 paragraphes : G3 decision log structure, 3 branches DR-2-STRICT/PRIME/FAIL avec actions concrètes (framework version tag + paper 1 journal target), Pending reviewer feedback S6-S8.

Then `git push`.

---

# Task S8.3 — P_equ skeleton

**Goal** : commencer P_equ profile (β + δ → 1 + 3 + 4) sans implémentation complète. Skeleton vide pour signaler que le wiring est prévu S9-S12 (vrais ops restructure + recombine arrivent ensuite).

**Files:**
- Create : `kiki_oniric/profiles/p_equ.py`
- Create : `tests/unit/test_p_equ.py`

## Step 1 — Write failing tests

Create `tests/unit/test_p_equ.py` with exactly:

```python
"""Unit tests for P_equ profile skeleton (β+δ → 1+3+4)."""
from __future__ import annotations

import pytest

from kiki_oniric.profiles.p_equ import PEquProfile


def test_p_equ_can_be_instantiated() -> None:
    profile = PEquProfile()
    assert profile is not None


def test_p_equ_marks_unimplemented_ops() -> None:
    profile = PEquProfile()
    assert "restructure" in profile.unimplemented_ops
    assert "recombine" in profile.unimplemented_ops


def test_p_equ_status_is_skeleton() -> None:
    profile = PEquProfile()
    assert profile.status == "skeleton"
```

## Step 2 — Verify failing

Run: `uv run pytest tests/unit/test_p_equ.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError`.

## Step 3 — Implement p_equ.py

Create `kiki_oniric/profiles/p_equ.py` with exactly:

```python
"""P_equ profile — balanced canonical consolidation (skeleton S8.3).

Channels : β + δ → 1 + 3 + 4 (curated buffer + hierarchical latents
in, weight delta + hierarchy change + attention prior out).
Operations : {replay, downscale, restructure, recombine_light}.

Restructure (D-Friston FEP) and recombine (C-Hobson VAE) operations
are NOT YET implemented — wiring lands S9-S12. This skeleton signals
the intent and provides a stable Python identifier for cross-track
references.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PEquProfile:
    """Balanced profile skeleton — implementation tracked S9-S12."""

    status: str = "skeleton"
    unimplemented_ops: list[str] = field(
        default_factory=lambda: ["restructure", "recombine"]
    )
```

## Step 4 — Verify passing

Run: `uv run pytest tests/unit/test_p_equ.py -v --no-cov`
Expected: 3 passed.

Run: `uv run pytest`
Expected: 55 tests (52 + 3 new), coverage ≥90%.

## Step 5 — Commit + push

```bash
git add kiki_oniric/profiles/p_equ.py tests/unit/test_p_equ.py
git commit -m "feat(profile): add P_equ skeleton (S9-S12 wiring)"
```

Subject : 49 chars. Body 2-3 paragraphes : P_equ skeleton, status="skeleton", unimplemented_ops list, real wiring restructure + recombine S9-S12. Cross-track references stable identifier.

Then `git push`.

---

# Self-review

**1. Spec coverage** :
- S7 downscale + swap protocol + P_min wiring → S7.1 + S7.2 + S7.3 + S7.4 ✅
- S8 G2 + G3 + P_equ start → S8.1 + S8.2 + S8.3 ✅

**2. Placeholder scan** : aucun TBD/TODO non-intentionnel dans code blocks. Les `TBD` dans `g2-pmin-report.md` et `g3-decision-log.md` sont **delibérés** (populés à S8-S9 quand l'évidence empirique arrive).

**3. Type consistency** :
- `DownscaleOpState`, `downscale_handler` (S7.1) consommés par `PMinProfile` (S7.4)
- `FiniteGuardError`, `check_finite` (S7.2) consommés par `swap_atomic` (S7.3)
- `SwapAborted`, `SwapResult`, `swap_atomic` (S7.3) — pas encore consommé par P_min (intégration runtime swap au S9+ avec MLX)
- `PMinProfile` (S7.4) construit sur DreamRuntime + 2 op states cohérents
- `PEquProfile` (S8.3) skeleton sans fonction encore — placeholder valide

**4. Commit count** : 7 commits.

**5. Validator risks** :
- S7.1 `feat(dream): add downscale op (B-Tononi SHY)` = 44 chars ✅
- S7.2 `feat(guard): add S2 finite check (NaN/Inf bound)` = 49 chars ✅
- S7.3 `feat(dream): add swap protocol skeleton (S1+S2)` = 48 chars ✅
- S7.4 `feat(profile): add P_min wiring (replay+downscale)` = 49 chars (50 si compté `replay+downscale`) — verifier ✅
- S8.1 `docs(milestone): G2 P_min viability skeleton` = 46 chars ✅
- S8.2 `docs(proof): add G3 decision log skeleton` = 43 chars ✅
- S8.3 `feat(profile): add P_equ skeleton (S9-S12 wiring)` = 49 chars ✅

Tous validator-compliant.

---

**End of S7-S8 atomic plan.**

**Version** : v0.1.0
**Generated** : 2026-04-18 via refinement of S7-S8 from main plan
**Source** : `docs/superpowers/plans/2026-04-17-dreamofkiki-implementation.md`
