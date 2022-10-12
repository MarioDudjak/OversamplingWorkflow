import math
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors

from program.Oversampling.Oversampler import Oversampler


class RandomSelectionSMOTE(Oversampler):
    """RadnomSelectionSMOTE Algorithm

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
        q = len(minorityInstances[0])
        count = math.floor(N / 100) * len(minorityInstances)

        S = []  # Synthetic data set
        for i in range(count):
            random_minority_instance = random.choice(list(minorityInstances))
            indices = nbrs.kneighbors(random_minority_instance.reshape(1, -1), return_distance=False)
            Nk = minorityInstances[indices[0][1:]]  # k-neighbourhood of d

            # Synthetic instances creation
            Xr = random.choice(Nk)  # Randomly select Xr from k-neighbourhood of instance
            s = np.empty(q)
            for j in range(q):
                s[j] = random_minority_instance[j] + np.random.uniform(0, 1) * (Xr[j] - random_minority_instance[j])
            S.append(s)

        synthetic_y = np.full((len(S)), super()._get_minority_label(y))

        return np.array(S), synthetic_y


