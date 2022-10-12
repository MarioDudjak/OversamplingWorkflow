import numpy as np
from sklearn.neighbors import NearestNeighbors

from program.Oversampling.Oversampler import Oversampler


class MAGICv6(Oversampler):
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
            neighbourhood = []
            for idx in indices[0][1:]:
                if int(y[idx]) != minorityLabel:
                    break
                else:
                    neighbourhood.append(X[idx])

            neighbourhood_size = len(neighbourhood)
            if neighbourhood_size != 0:
                if self.version == "v1.0":
                    neighbourhood.append(instance)
                    for i in range(neighbourhood_size):
                        weights = np.random.rand(neighbourhood_size + 1)
                        convex_combination = [neighbourhood[k] * weights[k] for k in range(neighbourhood_size + 1)]
                        s = np.sum(convex_combination, axis=0) / np.sum(weights)
                        S.append(s)

                if self.version == "v1.1":
                    neighbourhood.append(instance)
                    weights = np.random.rand(neighbourhood_size + 1)
                    convex_combination = [neighbourhood[k] * weights[k] for k in range(neighbourhood_size + 1)]
                    s = np.sum(convex_combination, axis=0) / np.sum(weights)
                    S.append(s)

                if self.version == "v2.0":
                    # Random number of neighbours
                    for i in range(neighbourhood_size):
                        no_rnd_neighbours = np.random.randint(low=1, high=neighbourhood_size + 1)
                        rnd_neigbhours_indices = np.random.choice(range(0, neighbourhood_size),
                                                                  size=no_rnd_neighbours)
                        rnd_neigbhours = [neighbourhood[i] for i in rnd_neigbhours_indices]
                        rnd_neigbhours.append(instance)
                        weights = np.random.rand(no_rnd_neighbours + 1)
                        convex_combination = [rnd_neigbhours[k] * weights[k] for k in range(no_rnd_neighbours + 1)]
                        s = np.sum(convex_combination, axis=0) / np.sum(weights)
                        S.append(s)

                if self.version == "v2.1":
                    # Random number of neighbours
                    no_rnd_neighbours = np.random.randint(low=1, high=neighbourhood_size + 1)
                    rnd_neigbhours_indices = np.random.choice(range(0, neighbourhood_size),
                                                              size=no_rnd_neighbours)
                    rnd_neigbhours = [neighbourhood[i] for i in rnd_neigbhours_indices]
                    rnd_neigbhours.append(instance)
                    weights = np.random.rand(no_rnd_neighbours + 1)
                    convex_combination = [rnd_neigbhours[k] * weights[k] for k in range(no_rnd_neighbours + 1)]
                    s = np.sum(convex_combination, axis=0) / np.sum(weights)
                    S.append(s)

                if self.version == "v3.0":
                    # Bounding box
                    neighbourhood.append(instance)
                    for i in range(neighbourhood_size):
                        s = np.empty(q)
                        for j in range(q):
                            feature_weights = np.random.rand(neighbourhood_size + 1)
                            convex_combination = [neighbourhood[k][j] * feature_weights[k] for k in
                                                  range(neighbourhood_size + 1)]
                            s[j] = np.sum(convex_combination) / np.sum(feature_weights)
                        S.append(s)

                if self.version == "v3.1":
                    # Bounding box
                    neighbourhood.append(instance)
                    s = np.empty(q)
                    for j in range(q):
                        feature_weights = np.random.rand(neighbourhood_size + 1)
                        convex_combination = [neighbourhood[k][j] * feature_weights[k] for k in
                                              range(neighbourhood_size + 1)]
                        s[j] = np.sum(convex_combination) / np.sum(feature_weights)
                    S.append(s)

                if self.version == "v4.0":
                    # Random neighbours and bounding box
                    for i in range(neighbourhood_size):
                        no_rnd_neighbours = np.random.randint(low=1, high=neighbourhood_size + 1)
                        rnd_neigbhours_indices = np.random.choice(range(0, neighbourhood_size), size=no_rnd_neighbours)
                        rnd_neigbhours = [neighbourhood[i] for i in rnd_neigbhours_indices]
                        rnd_neigbhours.append(instance)
                        s = np.empty(q)
                        for j in range(q):
                            feature_weights = np.random.rand(no_rnd_neighbours + 1)
                            convex_combination = [rnd_neigbhours[k][j] * feature_weights[k] for k in
                                                  range(no_rnd_neighbours + 1)]
                            s[j] = np.sum(convex_combination) / np.sum(feature_weights)
                        S.append(s)

                if self.version == "v4.1":
                    # Random neighbours and bounding box
                    no_rnd_neighbours = np.random.randint(low=1, high=neighbourhood_size + 1)
                    rnd_neigbhours_indices = np.random.choice(range(0, neighbourhood_size), size=no_rnd_neighbours)
                    rnd_neigbhours = [neighbourhood[i] for i in rnd_neigbhours_indices]
                    rnd_neigbhours.append(instance)
                    s = np.empty(q)
                    for j in range(q):
                        feature_weights = np.random.rand(no_rnd_neighbours + 1)
                        convex_combination = [rnd_neigbhours[k][j] * feature_weights[k] for k in
                                              range(no_rnd_neighbours + 1)]
                        s[j] = np.sum(convex_combination) / np.sum(feature_weights)
                    S.append(s)

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
        for i, instance in enumerate(X):
            if int(y[i]) == minorityLabel:
                distances, indices = nbrs.kneighbors(instance.reshape(1, -1), return_distance=True)
                neighbourhood_size = 0
                for idx in indices[0][1:]:
                    if int(y[idx]) != minorityLabel:
                        if neighbourhood_size == 0 and (len(removed_minority_indices) + 1) < len(minorityInstances):
                            removed_minority_indices.append(i)
                        break
                    else:
                        neighbourhood_size += 1

        return removed_minority_indices
