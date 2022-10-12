import math
import random

import numpy as np
from sklearn.neighbors import NearestNeighbors

from program.Oversampling.Oversampler import Oversampler

"""
Weighted and Random SMOTE - WaRSMOTE
Use WSMOTE for count calculation for each instance
Use RSMOTE equation for synthetic instance creation.
"""


class WaRSMOTE(Oversampler):

    def __init__(self):
        pass

    def make_samples(self, X, y, N, k):

        minorityLabel = super()._get_minority_label(y)
        minorityInstances = X[np.where(y[:] == minorityLabel)]

        neigbourFinder = NearestNeighbors(n_neighbors=len(minorityInstances), metric='euclidean').fit(minorityInstances)
        neighbourDistances, neighbourIndices = neigbourFinder.kneighbors(minorityInstances)

        neighbourDistances = np.array(neighbourDistances, dtype=np.float64)[:, 1:]
        neighbourIndices = np.array(neighbourIndices, dtype=np.int32)[:, 1:]

        ED = np.sum(neighbourDistances, 1)
        ED_min = np.min(ED)
        ED_max = np.max(ED)
        NED = (ED - ED_min) / (ED_max - ED_min)
        RNED = np.sum(NED) - NED
        Weights = RNED / (np.sum(RNED))
        counts = np.rint((N * len(minorityInstances) / 100) * Weights)

        S = []

        for i in range(len(minorityInstances)):
            count = counts[i]
            instance = minorityInstances[i]

            y1, y2 = random.sample(list(minorityInstances), 2)

            for j in range(int(count)):

                t = y1 + np.random.uniform(0, 1) * (y2 - y1)
                synthetic = instance + np.random.uniform(0, 1) * (t - instance)
                S.append(synthetic)

        synthetic_y = np.full((len(S)), super()._get_minority_label(y))

        return np.array(S), synthetic_y
