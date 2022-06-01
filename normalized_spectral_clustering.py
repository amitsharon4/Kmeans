import sys

import numpy as np
import kmeans_pp
import math

epsilon = 0.0001


def create_adjacency_matrix(observations):
    """
    Creates the weighted adjacency matrix.

    Parameters
    ----------
        observations : ndarray
            A matrix whose vectors are the observations.

    Returns
    -------
        w: ndarray
            The calculated matrix.
    """
    n = len(observations)
    res = np.zeros(shape=(n, n))
    for i in range(n):
        res[i] = np.exp(-0.5 * np.linalg.norm(
            np.subtract(observations, observations[i]), axis=1))
    np.fill_diagonal(res, 0)
    return res


def create_diagonal_matrix(w):
    """
    Creates a diagonal matrix D whose i-th element along the diagonal
    equals the sum of the i-th row of w.

    Parameters
    ----------
        w : ndarray
            A weighted adjacency matrix.

    Returns
    -------
        d: ndarray
            The calculated matrix.
    """
    d = np.diag(np.sum(w, axis=1))
    return d


def create_laplacian_matrix(w, d):
    """
    Creates the Normalized Graph Laplacian from W and D
    Parameters
    ----------
        w : ndarray
            A weighted adjacency matrix.
        d : ndarray
            w's digonal degree matrix.
    Returns
    -------
        l : ndarray
            The calculated matrix.
    """
    rooted_d = np.diag(np.power(np.diag(d), -0.5))
    l = np.subtract(np.identity(len(d)), np.dot(np.dot(rooted_d, w), rooted_d))
    return l


def modified_gram_schmidt(a):
    """
    Calculates modified gram schmidt for QR decomposition
    Parameters
    ----------
        a: ndarray
            A matrix.

    Returns q, r
    -------
        q : ndarray
            The calculated orthogonal matrix.
        r : ndarray
            The calculated diagonal matrix.
    """
    n = len(a)
    u = np.copy(a)
    r = np.zeros((n, n))
    q = np.empty((n, n))
    for i in range(n):
        r[i, i] = np.linalg.norm(u[:, i])
        try:
            q[:, i] = np.true_divide(u[:, i], r[i][i])
        except:
            print("There was an error while calculaitng"
                  "the modified gram-schmidt. Please try again."
                  "\nThe Program Will now terminate.")
            sys.exit(1)
        i_column = q[:, i]
        r[i, i + 1:n] = np.einsum('i,ij->j', i_column, u[:, i + 1:n])
        u[:, i + 1:n] -= np.einsum('i,j->ji', r[i, i + 1:n], i_column)
    return q, r


def qr_Iteration(a):
    """
    Decomposes the given matrix into
    a matrix whose diagonal elements approach the eigenvalues,
    and a matrix whose columns approach the eigenvectors.

    Parameters
    ----------
        a: ndarray
            A matrix representing a Normalized Graph Laplacian

    Returns a_bar, q_bar
    -------
        a_bar: ndarray
            The matrix whose diagonal elements approach the eigenvalues
        q_bar : ndarray
            A matrix whose columns approach the eigenvectors.
    """
    n = len(a)
    a_bar = np.copy(a)
    q_bar = np.identity(n)
    for i in range(n):
        q, r = modified_gram_schmidt(a_bar)
        a_bar = np.dot(r, q)
        new_q_bar = np.dot(q_bar, q)
        if (abs(abs(q_bar) - abs(new_q_bar)) < epsilon).all():
            return a_bar, q_bar
        q_bar = new_q_bar
    return a_bar, q_bar


def create_k_and_u(a_bar, q_bar, k=None):
    """
    Calculates the number of clusters using the eigengap heuristic
    and a matrix containing the first k eigen vectors
    (for the sorted corresponding eigen values).

    Parameters
    ----------
    a_bar : ndarray
        A matrix that has the eigenvalues as the diagonal.

    q_bar : ndarray
        A matrix that has the eigenvectors as columns.

    k : int, default = None
       If int, the egiengap heuristic will not be used.
       If None, k will be determined by the egiengap heuristic.


    Returns k, u
    -------
        k : int
            The number of clusters to use.
        u : matrix
            The calculated matrix.
    """
    eigen_values = np.diag(a_bar)
    sorted_eigen_values = np.sort(eigen_values)
    deltas = np.absolute(np.diff(sorted_eigen_values))
    if k is None:
        k = np.argmax(deltas[:math.ceil((len(sorted_eigen_values)) / 2)]) + 1
    indexes = eigen_values.argsort()[:k]
    u = q_bar[:, indexes]
    return k, u


def normalize_matrix(u):
    """
    Normalize the given matrix to have unit length.

    Parameters
    ----------
    u : ndarray
        A matrix.

    Returns
    -------
    T : ndarray
        The normalized matrix.
    """
    denominators = np.linalg.norm(u, axis=1)
    return np.copy(u) / denominators[:, None]


def cluster(observations, given_k=None):
    """
    Calculates clusters for the given observations.

    Parameters
    ----------
    observations : ndarray
        An array containing the observations to cluster.

    given_k : int, default = None
       If int, the number of clusters to use.
       If None, the number of clusters will b determined by the eigengap heuristic.

    Returns
    -------
    t_clusters : ndarray
        An array, such that the i-th observation belongs to array[i] cluster
    """

    w = create_adjacency_matrix(observations)
    d = create_diagonal_matrix(w)
    l = create_laplacian_matrix(w, d)
    a_bar, q_bar = qr_Iteration(l)
    k, u = create_k_and_u(a_bar, q_bar, given_k)
    t = normalize_matrix(u)
    t_clusters = kmeans_pp.k_means_pp(k, len(t), k, t)
    return t_clusters
