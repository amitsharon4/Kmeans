import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from itertools import combinations


def calculate_jaccard(computed_clusters, truth_clusters, k):
    """
    Calculates the Jaccard score for a clustered graph.

    Parameters
    ----------
        computed_clusters : ndarray
            An array such that the i-th observation is in the computed_clusters[i] cluster.
        truth_clusters : ndarray
            An array such that the i-th observation was
            originally generated in the truth_clusters[i] cluster.
        k : int
            The number of clusters used in the calculation.

    :return: The calculated Jaccard measure.
    """
    set1 = set()
    set2 = set()
    for i in range(k):
        indexes_for_1 = np.where(computed_clusters == i)
        set1.update(set(combinations(indexes_for_1[0], 2)))
        indexes_for_2 = np.where(truth_clusters == i)
        set2.update(set(combinations(indexes_for_2[0], 2)))
    return len(set1.intersection(set2)) / len(set1.union(set2))


def create_output_string(k, n, sc_jaccard, kmeans_jaccard, original_k):
    """
    Creates the string output for the pdf file.

    Parameters
    ----------
        k : int
            number of clusters used for calculation
        n : int
            number of observations
        sc_jaccard : float
            jaccard measure for spectral clustering
        kmeans_jaccard : float
            jaccard measure for kmeanspp clustering
        original_k : int
            The number of clusters used for the generation of the observations.

    :return: The formatted string.
    """
    return "The Data was Generates from the values:\n" \
           "n = {}, k = {}\n" \
           "The k that was used for both algorithms was {}\n" \
           "The Jaccard measure for Spectral Clustering: {:.2f}\n" \
           "The Jaccard measure for K-means: {:.2f}".format(n, original_k, k, sc_jaccard, kmeans_jaccard)


def create_graph(observations, spectral_labels, kmenaspp_labels, k, n, original_k):

    """
    Creates the visual graphs for pdf file.

    Parameters
    ----------
        observations : ndarray
            The obervations.
        spectral_labels : ndarray
            An array such that the i-th observation is in the spectral_labels[i] cluster,
            according to the Specrtal Clustering algorithm.
        kmenaspp_labels : ndarray
            An array such that the i-th observation is in the kmenaspp_labels[i] cluster,
            according to the K-means algorithm.
        k : int
            The number of clusters calculated
        n : int
            The number of observations
        original_k : int
            The number of clusters used for the generation of the observations.

    :return: The visualization for the pdf file
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))
    coordinates = observations[0]
    if len(coordinates[0]) == 2:
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
    else:
        ax1 = plt.subplot(221, projection='3d')
        ax2 = plt.subplot(222, projection='3d')
    ax3 = plt.subplot(212)
    ax1.scatter(*zip(*coordinates), c = kmenaspp_labels,  cmap ='gist_rainbow')
    ax2.scatter(*zip(*coordinates), c= spectral_labels,  cmap ='gist_rainbow')
    ax1.set_title("K-means")
    ax2.set_title("Normalized Spectral Clustering")
    ax3.set_axis_off()
    sc_jaccard = calculate_jaccard(spectral_labels[0], observations[1], k)
    kmeans_jaccard = calculate_jaccard(kmenaspp_labels, observations[1], k)
    bottom_text = create_output_string(k, n, sc_jaccard, kmeans_jaccard, original_k)
    ax3.text(0.5, 0.5,
             bottom_text,
             horizontalalignment="center",
             verticalalignment="center",
             wrap=True, fontsize=14,
             color="black")
    return fig


def create_pdf(observations, spectral_results, kmeans_result, k, n, original_k):
    """
    Creates the output file as a pdf with the visualization.
    Parameters
    ----------
        observations : ndarray
            The obervations.
        spectral_results : ndarray
            An array such that the i-th observation is in the spectral_labels[i] cluster,
            according to the Specrtal Clustering algorithm.
        kmeans_result : ndarray
            An array such that the i-th observation is in the kmenaspp_labels[i] cluster,
            according to the K-means algorithm.
        k : int
            The number of clusters calculated
        n : int
            The number of observations
        original_k : int
            The number of clusters used for the generation of the observations.
    """
    pp = PdfPages("clusters.pdf")
    pp.savefig(create_graph(observations, spectral_results, kmeans_result, k, n, original_k))
    pp.close()
    plt.close()
