import math
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors

from program.Oversampling.Oversampler import Oversampler

class LinearSMOTE(Oversampler):
    """Linear SMOTE Algorithm

        Creates synthetic instances of minority dataset in binary classification problems.

        Parameters
        ----------
        N : int, optional (default=100)
            Number of SMOTE.

        k : int, optional (default=5)
            Number of nearest neighbours to used to construct synthetic samples.

        Returns
        ----------
        S : array_like(float), shape (n_synthetic_instances, n_features)
            Created Synthetic dataset.
        """
    def __init__(self):
        pass

    def make_samples(self, X, y, N, k):

        minorityLabel = super()._get_minority_label(y)
        minorityInstances = X[np.where(y[:] == minorityLabel)]

        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(minorityInstances)
        S = []  # Synthetic data set

        for instance in minorityInstances:
            indices = nbrs.kneighbors(instance.reshape(1, -1), return_distance=False)
            Nk = minorityInstances[indices[0][1:]]  # k-neighbourhood of d

            # Synthetic instances creation
            for i in range(math.floor(N / 100)):
                Xr = random.choice(Nk)  # Randomly select Xr from k-neighbourhood of X
                s = instance + np.random.uniform(0, 1) * (Xr - instance)
                S.append(s)

        synthetic_y = np.full((len(S)), super()._get_minority_label(y))

        return np.array(S), synthetic_y


