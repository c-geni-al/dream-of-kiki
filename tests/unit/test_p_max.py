"""Unit tests for P_max profile skeleton (S16.1)."""
from __future__ import annotations

from kiki_oniric.dream.episode import Operation, OutputChannel
from kiki_oniric.profiles.p_max import PMaxProfile


def test_p_max_can_be_instantiated() -> None:
    profile = PMaxProfile()
    assert profile is not None


def test_p_max_status_is_skeleton() -> None:
    profile = PMaxProfile()
    assert profile.status == "skeleton"
    assert "recombine_full" in profile.unimplemented_ops


def test_p_max_declares_target_ops_and_channels() -> None:
    """Skeleton must declare its target ops/channels metadata for
    DR-4 chain checks even when handlers are not wired."""
    profile = PMaxProfile()
    assert profile.target_ops == {
        Operation.REPLAY,
        Operation.DOWNSCALE,
        Operation.RESTRUCTURE,
        Operation.RECOMBINE,
    }
    assert profile.target_channels_out == {
        OutputChannel.WEIGHT_DELTA,
        OutputChannel.LATENT_SAMPLE,
        OutputChannel.HIERARCHY_CHG,
        OutputChannel.ATTENTION_PRIOR,
    }
