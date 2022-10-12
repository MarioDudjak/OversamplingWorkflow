import numpy as np
from scipy.spatial import distance as dst
from sklearn.neighbors import NearestNeighbors

from program.Oversampling.Oversampler import Oversampler


class MAGICv4(Oversampler):
    """MAGIC Algorithm

        Creates synthetic instances of minority dataset in binary classification problems.

        Parameters
        ----------
        No parameters - that's the magic.

        Returns
        ----------
        S : array_like(float), shape (n_synthetic_instances, n_features)
            Created Synthetic dataset.
        """

    def __init__(self, version="v1"):
        self.version = version
        pass

    def make_samples(self, X, y, N=100, k=5):
        minorityLabel = super()._get_minority_label(y)
        minorityInstances = X[np.where(y[:] == minorityLabel)]
        size = len(X)
        q = len(minorityInstances[0])

        nbrs = NearestNeighbors(n_neighbors=size, metric='euclidean').fit(X)
        S = []  # Synthetic data set

        # Synthetic instances creation
        for instance in minorityInstances:
            distances, indices = nbrs.kneighbors(instance.reshape(1, -1), return_distance=True)
            neighbourhood_size = 0
            for distance, idx in zip(distances[0][1:], indices[0][1:]):
                if int(y[idx]) == minorityLabel:
                    neighbourhood_size += 1
                else:
                    if self.version == "v1.1":
                        s = np.random.default_rng().normal(size=(1, q))
                        u = np.random.default_rng().random((1, 1))

                        s = distance * u ** (1 / q) / np.sqrt(np.sum(s ** 2, 1, keepdims=True)) * s + instance
                        radius = dst.euclidean(instance, s[0])
                        while radius > distance:
                            s = np.random.default_rng().normal(size=(1, q))
                            u = np.random.default_rng().random((1, 1))

                            s = distance * u ** (1 / q) / np.sqrt(np.sum(s ** 2, 1, keepdims=True)) * s + instance
                            radius = dst.euclidean(instance, s[0])
                        S.append(s[0])

                    elif self.version == "v1.0":
                        if neighbourhood_size == 0:
                            neighbourhood_size = 1
                        for i in range(neighbourhood_size):
                            s = np.random.default_rng().normal(size=(1, q))
                            u = np.random.default_rng().random((1, 1))

                            s = distance * u ** (1 / q) / np.sqrt(np.sum(s ** 2, 1, keepdims=True)) * s + instance
                            radius = dst.euclidean(instance, s[0])
                            while radius > distance:
                                s = np.random.default_rng().normal(size=(1, q))
                                u = np.random.default_rng().random((1, 1))

                                s = distance * u ** (1 / q) / np.sqrt(np.sum(s ** 2, 1, keepdims=True)) * s + instance
                                radius = dst.euclidean(instance, s[0])
                            S.append(s[0])

                    elif self.version == "v2.1":
                        s = np.random.default_rng().normal(size=(1, q))
                        u = np.random.default_rng().random((1, 1))

                        s = distance / 2 * u ** (1 / q) / np.sqrt(np.sum(s ** 2, 1, keepdims=True)) * s + instance
                        radius = dst.euclidean(instance, s[0])
                        while radius > (distance / 2):
                            s = np.random.default_rng().normal(size=(1, q))
                            u = np.random.default_rng().random((1, 1))

                            s = distance / 2 * u ** (1 / q) / np.sqrt(np.sum(s ** 2, 1, keepdims=True)) * s + instance
                            radius = dst.euclidean(instance, s[0])
                        S.append(s[0])

                    elif self.version == "v2.0":
                        if neighbourhood_size == 0:
                            neighbourhood_size = 1
                        for i in range(neighbourhood_size):
                            s = np.random.default_rng().normal(size=(1, q))
                            u = np.random.default_rng().random((1, 1))

                            s = distance / 2 * u ** (1 / q) / np.sqrt(np.sum(s ** 2, 1, keepdims=True)) * s + instance
                            radius = dst.euclidean(instance, s[0])
                            while radius > (distance / 2):
                                s = np.random.default_rng().normal(size=(1, q))
                                u = np.random.default_rng().random((1, 1))

                                s = distance / 2 * u ** (1 / q) / np.sqrt(np.sum(s ** 2, 1, keepdims=True)) * s + instance
                                radius = dst.euclidean(instance, s[0])
                            S.append(s[0])
                    break

        synthetic_y = np.full((len(S)), super()._get_minority_label(y))

        return np.array(S), synthetic_y

    def remove_samples(self, X, y):
        minorityLabel = super()._get_minority_label(y)
        minorityInstances = X[np.where(y[:] == minorityLabel)]
        size = len(X)
        q = len(minorityInstances[0])

        nbrs = NearestNeighbors(n_neighbors=size, metric='euclidean').fit(X)
        S = []  # Synthetic data set
        removed_minority_indices = []
        # Synthetic instances creation
        for instance in minorityInstances:
            distances, indices = nbrs.kneighbors(instance.reshape(1, -1), return_distance=True)
            neighbourhood_size = 0
            for idx in indices[0][1:]:
                if int(y[idx]) != minorityLabel:
                    if neighbourhood_size == 0:
                        removed_minority_indices.append(idx)
                    break
                else:
                    neighbourhood_size += 1

        return removed_minority_indices
