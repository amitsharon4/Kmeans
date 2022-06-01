import numpy as np
import mykmeanspp


def k_means_pp(k, N, d, observations):
    """
    Calculates clusters using the kmeanspp algorithm

    Parameters
    ----------
        k : int
            The number of clusters.
        N : int
            The number of observations.
        d : int
            The dimensions of each observation.
        observations : array-like
            An array containing the observations.

    :return:  matrix where matrix[i]==the cluster that observation i belongs to
    """
    np.random.seed(0)                                           # Seed randomness
    init = np.zeros(k, int)
    init[0] = np.random.choice(N)
    distances = np.zeros((k, N), dtype=np.float64)
    distances[0] = np.power(np.subtract(observations, observations[init[0]]), 2).sum(axis = 1, dtype=np.float64)
    for j in range(1, k):
        probabilities = np.min(distances[:j, ], axis = 0)
        probabilities = probabilities / probabilities.sum(dtype=np.float64)
        init[j] = np.random.choice(N, p = probabilities)
        distances[j] = np.power(np.subtract(observations, observations[init[j]]), 2).sum(axis=1, dtype=np.float64)
    myCentroids = [observations.tolist()[i] for i in init]
    return mykmeanspp.kmeans(k, N, d, 300, observations.tolist(), myCentroids)
