import random
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors

from program.Oversampling.Oversampler import Oversampler


class ReverseSMOTE(Oversampler):
    """Das et al., "An Oversampling Technique by Integrating Reverse Nearest Neighbor in SMOTE: Reverse-SMOTE", 2020"""

    def __init__(self,  k):
        self.k = k

    def make_samples(self, X, y, N, k):
        minorityLabel = super()._get_minority_label(y)
        minorityInstances = X[np.where(y[:] == minorityLabel)]

        if self.k + 1 <= len(minorityInstances):
            nbrs = NearestNeighbors(n_neighbors=self.k + 1, metric='euclidean').fit(minorityInstances)
        else:
            nbrs = NearestNeighbors(n_neighbors=len(k), metric='euclidean').fit(minorityInstances)

        POTENT = []  # Borderline data of the minority class P
        Ns = []
        for p in minorityInstances:
            indices = nbrs.kneighbors(p.reshape(1, -1), return_distance=False)
            majn = 0  # Number of majority examples among the m nearest neighbours of p
            minn = 0  # Number of minority examples among the m nearest neighbours of p
            for i in indices[0][1:]:
                if y[i] != minorityLabel:
                    majn = majn + 1
                else:
                    minn = minn + 1
            if majn < minn:
                POTENT.append(p)
                if majn != 0:
                    Ns.append(math.ceil(minn/majn))
                else:
                    Ns.append(minn)

        S = []  # Synthetic data set
        if len(POTENT):
            if self.k + 1 <= len(POTENT):
                _nbrs = NearestNeighbors(n_neighbors=self.k + 1, metric='euclidean').fit(POTENT)
            else:
                _nbrs = NearestNeighbors(n_neighbors=len(POTENT), metric='euclidean').fit(POTENT)

            for idx, d in enumerate(POTENT):
                Nk = []
                for instance in POTENT:

                    distances, indices = _nbrs.kneighbors(instance.reshape(1, -1))
                    dist = self._l2_norm(d, instance)
                    if self.k + 1 <= len(POTENT):
                        if dist != 0 and distances[0][k] > dist:
                            Nk.append(instance)
                    else:
                        if dist != 0 and distances[0][len(POTENT)-1] > dist:
                            Nk.append(instance)

                if len(Nk):
                    for j in range(Ns[idx]):
                        dr = random.choice(list(Nk))
                        # Synthetic instances creation
                        new_instance = d + random.uniform(0, 1) * (dr - d)
                        S.append(new_instance)

        synthetic_y = np.full((len(S)), super()._get_minority_label(y))

        return np.array(S), synthetic_y

    def _l2_norm(self, x1, x2):
        norm = [(x1[i] - x2[i]) ** 2 for i in range(len(x1))]
        return np.sqrt(np.sum(norm))
