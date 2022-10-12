import random
import numpy as np
from sklearn.neighbors import NearestNeighbors

from program.Oversampling.Oversampler import Oversampler


class SMOTE(Oversampler):
    """SMOTE Algorithm

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

    def __init__(self, N, k, version="line"):
        self.N = N
        self.k = k
        self.version=version

    def make_samples(self, X, y, N, k):

        minorityLabel = super()._get_minority_label(y)
        minorityInstances = X[np.where(y[:] == minorityLabel)]
        majorityInstances = X[np.where(y[:] != minorityLabel)]
        IR = len(majorityInstances) / len(minorityInstances)

        if self.k +1 <= len(minorityInstances):
            nbrs = NearestNeighbors(n_neighbors=self.k + 1, metric='euclidean').fit(minorityInstances)
        else:
            nbrs = NearestNeighbors(n_neighbors=len(k), metric='euclidean').fit(minorityInstances)

        q = len(minorityInstances[0])
        S = []  # Synthetic data set
        if self.N == 'IR':
            count = np.int32(round(IR) - 1)
        else:
            count = np.int32(np.floor(self.N / 100))

        for instance in minorityInstances:
            indices = nbrs.kneighbors(instance.reshape(1, -1), return_distance=False)
            Nk = minorityInstances[indices[0][1:]]  # k-neighbourhood of d



            # Synthetic instances creation
            for i in range(count):
                Xr = random.choice(Nk)  # Randomly select Xr from k-neighbourhood of instance
                if self.version == "v1":
                    s = instance + random.uniform(0, 1) * (Xr - instance)
                else:
                    s = instance + random.uniform(0, 1) * (Xr - instance)
                    for j in range(q):
                       s[j] = instance[j] + np.random.uniform(0, 1) * (Xr[j] - instance[j])
                S.append(s)

        synthetic_y = np.full((len(S)), minorityLabel)

        return np.array(S), synthetic_y
