from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.base import _get_weights
from sklearn.utils.extmath import weighted_mode

import os
os.sys.path.extend([os.path.abspath('~/Packages/flann-1.7.1-src/src/python/')])
import pyflann
import numpy as np
from random import sample


class FAKNeighborsClassifier(KNeighborsClassifier):
    def __init__(self, n_neighbors=5, weights='uniform'):
        super(FAKNeighborsClassifier, self).__init__(
            n_neighbors=n_neighbors, weights=weights)

    def _fit(self, X):
        self.flann = pyflann.FLANN()
        self.flann.build_index(X)

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        ind, dist = self.flann.nn_index(X, n_neighbors)

        if n_neighbors == 1:
            dist = dist[:, None]
            ind = ind[:, None]

#        if np.any(np.equal(dist[:,1:],dist[:,:-1])):
#            print "Dist Tie"

        if return_distance:
            return dist, ind
        else:
            return ind

    def predict(self, X):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X: array
            A 2-D array representing the test points.

        Returns
        -------
        labels: array
            List of class labels (one for each data sample).
        """
        X = np.atleast_2d(X)

        neigh_dist, neigh_ind = self.kneighbors(X)
        pred_labels = self._y[neigh_ind]

        weights = _get_weights(neigh_dist, self.weights)

        if weights is None:
            mode, _ = smart_mode(pred_labels, axis=1)
        else:
            mode, _ = weighted_mode(pred_labels, weights, axis=1)

        return mode.flatten().astype(np.int)


class FAKPseudoNeighbor(FAKNeighborsClassifier):
    def __init__(self, n_neighbors=5, weights='uniform'):
        super(FAKPseudoNeighbor, self).__init__(
            n_neighbors=n_neighbors, weights=weights)

    def fit(self, X, y, G):
        super(FAKPseudoNeighbor, self).fit(X, y)
        self._G = G

    def kneighbors(self, X, idx=None, n_neighbors=None, return_distance=True):
        if idx is None:
            return super(FAKPseudoNeighbor, self).kneighbors(
                X, n_neighbors, return_distance)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        # Get the graph neighbors
        gneighbors = np.append(self._G[:, idx].nonzero()[0], idx)

        if gneighbors.shape[0] > self._y.shape[0] - n_neighbors:
            print "WOAH " + repr(gneighbors.shape[0]) + " NEIGHBORS!!!"
            print "Giving UP :( Return random stuff"
            return np.zeros(n_neighbors), np.array(sample(
                np.arange(self._y.shape[0]), n_neighbors))

        # Get enough neighbors so taht we are sure we'll have n_neighbors non
        # graph neighbors
        ind, dist = self.flann.nn_index(X, n_neighbors + gneighbors.shape[0])

        # find the ones that aren't graph neighbors
        notGneighbor = np.equal(np.in1d(ind.flat, gneighbors, True),
                                False)[None, :]

        ind = ind[notGneighbor][:n_neighbors]
        dist = dist[notGneighbor][:n_neighbors]

        if n_neighbors == 1:
            dist = dist[:, None]
            ind = ind[:, None]

        if return_distance:
            return dist, ind
        else:
            return ind

    def predict(self, X, idx=None):

        neigh_dist, neigh_ind = self.kneighbors(X, idx)
        pred_labels = self._y[neigh_ind]

        weights = _get_weights(neigh_dist, self.weights)

        if weights is None:
            mode, _ = smart_mode(pred_labels, axis=1)
        else:
            mode, _ = weighted_mode(pred_labels, weights)

        return mode.flatten().astype(np.int)


def smart_mode(a, axis=0):
    """
    Returns an array of the modal (most common) value in the passed array.

    If there is more than one such value, only the first is returned.
    The bin-count for the modal bins is also returned.

    Parameters
    ----------
    a : array_like
        n-dimensional array of which to find mode(s).
    axis : int, optional
        Axis along which to operate. Default is 0, i.e. the first axis.

    Returns
    -------
    vals : ndarray
        Array of modal values.
    counts : ndarray
        Array of counts for each mode.

    Examples
    --------
    >>> a = np.array([[6, 8, 3, 0],
                      [3, 2, 1, 7],
                      [8, 1, 8, 4],
                      [5, 3, 0, 5],
                      [4, 7, 5, 9]])
    >>> from scipy import stats
    >>> stats.mode(a)
    (array([[ 3.,  1.,  0.,  0.]]), array([[ 1.,  1.,  1.,  1.]]))

    To get mode of whole array, specify axis=None:

    >>> stats.mode(a, axis=None)
    (array([ 3.]), array([ 3.]))

    """
#    a, axis = _chk_asarray(a, axis)
    a = np.atleast_2d(a)

    if axis == 0:
        a = a.T

    scores = np.unique(np.ravel(a))       # get ALL unique values
    n_scores = len(scores)
    n, n_obs = a.shape

    scoreDict = dict(zip(scores, np.arange(n_scores)))
    invDict = dict(zip(np.arange(n_scores), scores))

    scorecount = np.zeros((n, n_scores))
    mostfrequent = np.zeros(n)
    for i in np.arange(n):
        for j in np.arange(n_obs):
            k = scoreDict[a[i, j]]
            scorecount[i, k] += 1
            if scorecount[i, k] > scorecount[i, mostfrequent[i]]:
                mostfrequent[i] = k

    return np.array([invDict[m] for m in mostfrequent]), scorecount[
        np.arange(n).astype(int), mostfrequent.astype(int)]
