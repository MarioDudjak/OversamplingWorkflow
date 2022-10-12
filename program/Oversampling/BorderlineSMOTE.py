import math
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors

from program.Oversampling.Oversampler import Oversampler

class BorderlineSMOTE(Oversampler):
    """Borderline SMOTE Algorithm

        Creates synthetic instances of minority dataset in binary classification problems.

        Parameters
        ----------
        m : int, optional (default=5)
            Number of nearest neighbours to used to search throughout whole dataset.

        k : int, optional (default=5)
            Number of nearest neighbours to used to construct synthetic samples.

        N : int, optional
            Number of SMOTE.

        Returns
        ----------
        S : array_like(float), shape (n_synthetic_instances, n_features)
            Created Synthetic dataset.
        """

    def __init__(self, N, k):
        self.N = N
        self.k = k
        pass

    def make_samples(self, X, y, N, k):

        m = 2*self.k+1

        minorityLabel = super()._get_minority_label(y)
        minorityInstances = X[np.where(y[:] == minorityLabel)]
        majorityInstances = X[np.where(y[:] != minorityLabel)]
        IR = len(majorityInstances) / len(minorityInstances)

        S = []  # Synthetic data set

        _nbrs = NearestNeighbors(n_neighbors=m + 1, metric='euclidean').fit(X)

        DANGER = []  # Borderline data of the minority class P
        for p in minorityInstances:
            indices = _nbrs.kneighbors(p.reshape(1, -1), return_distance=False)
            mn = 0  # Number of majority examples among the m nearest neighbours of p
            for i in indices[0][1:]:
                if y[i] != minorityLabel:
                    mn = mn + 1
            if m/2 <= mn < m:
                DANGER.append(p)

        if self.k + 1 <= len(minorityInstances):
            _nbrs = NearestNeighbors(n_neighbors=self.k + 1, metric='euclidean').fit(minorityInstances)
        else:
            _nbrs = NearestNeighbors(n_neighbors=len(minorityInstances), metric='euclidean').fit(minorityInstances)

        S = []  # Synthetic data set
        if len(DANGER):
            if self.N == 'IR':
                count = math.floor((len(minorityInstances) * (round(IR) - 1)) / len(DANGER))
            else:
                count = math.floor((len(minorityInstances) * (self.N / 100)) / len(DANGER))
            #count = math.floor(len(minorityInstances) / len(DANGER)) * self.N

            for d in DANGER:
                indices = _nbrs.kneighbors(d.reshape(1, -1), return_distance=False)
                Nk = minorityInstances[indices[0][1:]]  # k-neighbourhood of d
                for j in range(count):
                    dr = random.choice(list(Nk))
                    # Synthetic instances creation
                    new_instance = d + random.uniform(0, 1) * (dr - d)
                    S.append(new_instance)

        synthetic_y = np.full((len(S)), super()._get_minority_label(y))

        return np.array(S), synthetic_y

