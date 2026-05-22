"""Novelty archive + behavior characteristic unit tests."""
import numpy as np

from agents.novelty import (
    BC_DIM,
    BehaviorCharacteristic,
    NoveltyArchive,
    bc_from_agent_trajectory,
    team_bc,
)


def test_bc_dim_and_shape():
    obs_seq = [np.array([0.1, 0.2, 0.3, 0.4, 5.0, 6.0], dtype=np.float32) for _ in range(10)]
    act_seq = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    bc = bc_from_agent_trajectory(obs_seq, act_seq, n_actions=5)
    assert bc.shape == (BC_DIM,)
    # Mean position should equal the constant we put in (0.3, 0.4).
    np.testing.assert_allclose(bc[:2], [0.3, 0.4], rtol=1e-5)
    # Speed = |[0.1, 0.2]| = sqrt(0.05)
    np.testing.assert_allclose(bc[2], float(np.sqrt(0.05)), rtol=1e-5)
    # Uniform over 5 actions => entropy = log(5)
    np.testing.assert_allclose(bc[3], float(np.log(5)), rtol=1e-5)


def test_bc_empty_trajectory():
    bc = bc_from_agent_trajectory([], [], n_actions=5)
    assert bc.shape == (BC_DIM,)
    np.testing.assert_array_equal(bc, np.zeros(BC_DIM, dtype=np.float32))


def test_team_bc_is_mean():
    a = np.array([1, 2, 3, 4], dtype=np.float32)
    b = np.array([3, 4, 5, 6], dtype=np.float32)
    np.testing.assert_allclose(team_bc({"a": a, "b": b}), [2, 3, 4, 5])


def test_novelty_first_is_zero():
    arch = NoveltyArchive(capacity=10, k=3)
    bc = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    s = arch.score_and_add(bc)
    assert s == 0.0  # empty archive => 0
    assert len(arch) == 1


def test_novelty_grows_with_distance():
    arch = NoveltyArchive(capacity=10, k=1)
    arch.add(np.zeros(4, dtype=np.float32))
    near = arch.score(np.array([0.1, 0, 0, 0], dtype=np.float32))
    far = arch.score(np.array([10.0, 0, 0, 0], dtype=np.float32))
    assert far > near > 0


def test_archive_capacity_evicts_oldest():
    arch = NoveltyArchive(capacity=3, k=1)
    for i in range(5):
        arch.add(np.array([float(i), 0, 0, 0], dtype=np.float32))
    assert len(arch) == 3
    # Newest bc nearest neighbor should be the most-recent eviction-survivor.
    s = arch.score(np.array([4.5, 0, 0, 0], dtype=np.float32))
    np.testing.assert_allclose(s, 0.5, rtol=1e-5)


def test_bc_namespace_matches_functions():
    obs = [np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32)]
    np.testing.assert_array_equal(
        BehaviorCharacteristic.from_agent(obs, [0], n_actions=2),
        bc_from_agent_trajectory(obs, [0], n_actions=2),
    )
