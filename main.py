import argparse
import sys

import numpy as np
from sklearn.datasets import make_blobs
import kmeans_pp
import normalized_spectral_clustering
import visualization

epsilon = 0.0001
three_d_max_capacity = [20, 540]
two_d_max_capacity = [20, 550]

"""
Maximum capacity for N was chosen as the maximum amount of points that
this program can process under 5 minutes (tested in --no-Random mode)

Maximum capacity for K was chosen as the optimized number of clusters
for both Kmeans and spectral clustering
"""


def get_input():
    """
    Gets the clustering input from the user

    Returns k, n, random
    -------
        k : int
            The number of clusters.
        n : int
            The number of observations.
        random : boolean
            If true, n and k will be generated randomly.
            If false, the provided n and k will be used.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("k", type=int, help="the number of clusters required")
    parser.add_argument("n", type=int,help="the number of observations in the file")
    parser.add_argument("--r","--random", dest='random', action='store_true')
    parser.add_argument("--no-r","--no-random", dest='random', action='store_false')
    parser.set_defaults(random=True)
    args = parser.parse_args()
    if not args.random:
        if args.k <= 0 or args.n <= 0:
            print("Parameters must be greater then 0")
            sys.exit(1)
        elif args.k >= args.n:
            print("K must be smaller then N")
            sys.exit(1)

    return args.k, args.n, args.random


def create_clusters_text(spectral_array, kmeanspp_array):
    """
    Creates a text file containing observations and their cluster number

    Parameters
    ----------
    spectral_array : array-like
        an array, such that the i-th observation is in the
         spectral_array[i] cluster, according to  spectral array clustering
    kmeanspp_array : array-like
        an array, such that the i-th observation is in
        the kmeanspp_array[i] cluster, according to  kmeanspp_array clustering
    """
    with open('clusters.txt', 'w') as outfile:
        outfile.write(str(spectral_array.max()+1) + '\n')
        for i in range(spectral_array.max()+1):
            outfile.write(str(','.join(map(
                str, np.nonzero(spectral_array == i)[0]))) + '\n')
        for i in range(spectral_array.max()+1):
            outfile.write(str(','.join(map(
                str, np.nonzero(kmeanspp_array == i)[0]))) + '\n')


def get_input_by_dimension(d):
    """
    Restructures k and n according to user input.
    Prints a message about the maximum capacity with the chosen dimension.

    Parameters
    ----------
        d : int
        The chosen dimension of the observation.

    Returns k, n, rnd
    -------
        k : int
            The number of clusters.
        n : int
            The number of observations.
        rnd : boolean
            If true, n and k are generated randomly.
            If false, the provided n and k will be used.
    """
    k, n, rnd = get_input()
    max_k = two_d_max_capacity[0] if d == 2 else three_d_max_capacity[0]
    max_n = two_d_max_capacity[1] if d == 2 else three_d_max_capacity[1]
    print("The maximum capacity of the number of clusters and the number of"
          " observations for {} dimensional points are: {} and {}"
          .format(d, max_k, max_n))
    if rnd:
        k = np.random.randint(max_k // 2, max_k + 1)
        n = np.random.randint(max_n // 2, max_n + 1)
    return k, n, rnd


d = np.random.choice(2) + 2
k, n, rnd = get_input_by_dimension(d)
observations = make_blobs(n_samples=n, n_features=d, centers=k)
original_k = k
if rnd:
    cluster_spectral_affiliation = np.array(
        normalized_spectral_clustering.cluster(observations[0], None)).reshape(n, 1)
    k = cluster_spectral_affiliation.max() + 1  # gets the number of clusters
else:
    cluster_spectral_affiliation = np.array(
        normalized_spectral_clustering.cluster(observations[0], k)).reshape(n, 1)
data = np.column_stack((observations[0], observations[1]))  # makes the observations ready to save to txt
np.savetxt('data.txt', data, fmt=','.join(['%f']*d + ['%i']))
kmeanspp_clusters = np.array(kmeans_pp.k_means_pp(k, n, d, observations[0]))
if kmeanspp_clusters is None:
    print("There was a memory allocation error. Please try again."
          "\nThe Program Will now terminate.")
    sys.exit(1)
create_clusters_text(cluster_spectral_affiliation, kmeanspp_clusters)
visualization.create_pdf(
    observations, cluster_spectral_affiliation.transpose(), kmeanspp_clusters, k, n, original_k)
