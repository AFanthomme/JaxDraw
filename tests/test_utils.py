import pytest
import jax
from experiments.src.utils import RunningStats

@pytest.mark.parametrize("batch_idx", range(10))
def test_running_stats(batch_idx):
    """Generates the reference rollout for a specific seed."""
    B, T = 64, 10
    n = 10
    key = jax.random.key(777 + batch_idx)

    data = 10*jax.random.normal(key, shape=(n, B, T))

    ref_mean = data.mean(axis=(0,1))
    ref_std = data.std(axis=(0,1))
    ref_min = data.min(axis=(0,1))
    ref_max = data.max(axis=(0,1))
    running_stats = RunningStats(shape=(T,))
    for i in range(n):
        running_stats.update(data[i])
    assert jax.numpy.allclose(running_stats.mean, ref_mean, rtol=1e-5, atol=1e-5)
    assert jax.numpy.allclose(running_stats.var, ref_std**2, rtol=1e-5, atol=1e-5)
    assert jax.numpy.allclose(running_stats.std, ref_std, rtol=1e-5, atol=1e-5)
    assert jax.numpy.allclose(running_stats.min, ref_min, rtol=1e-5, atol=1e-5)
    assert jax.numpy.allclose(running_stats.max, ref_max, rtol=1e-5, atol=1e-5)

