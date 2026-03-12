import jax.numpy as jnp

class RunningStats:
    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...]= (), axis=0):
        """
        Calculates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = jnp.zeros(shape, jnp.float32)
        self.var = jnp.ones(shape, jnp.float32)

        self.min = jnp.inf * jnp.ones(shape, jnp.float32)
        self.max = -jnp.inf * jnp.ones(shape, jnp.float32)
        self.count = epsilon
        self.axis = axis

    @property
    def std(self):
        return jnp.sqrt(self.var)

    def copy(self):
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningStats(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other) -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: jnp.ndarray) -> None:
        batch_min = jnp.min(arr, axis=self.axis)
        batch_max = jnp.max(arr, axis=self.axis)
        self.min = jnp.min(jnp.stack((self.min, batch_min), axis=-1), axis=-1)
        self.max = jnp.max(jnp.stack((self.max, batch_max), axis=-1), axis=-1)
        batch_mean = jnp.mean(arr, axis=self.axis)
        batch_var = jnp.var(arr, axis=self.axis)
        batch_count = arr.shape[self.axis]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: jnp.ndarray, batch_var: jnp.ndarray, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + jnp.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count