import math
import random

import numpy as np
from sklearn.neighbors import NearestNeighbors

from program.Oversampling.Oversampler import Oversampler

"""
Weighted SMOTE - WSMOTE

Implementation of "Weighted-SMOTE: A modification to SMOTE for event classification in
sodium cooled fast reactors"
Manas Ranjan Prusty, T. Jayanthi, K. Velusamy
Progress in Nuclear Energy, Vol. 100, pp. 355-364
"""


class WSMOTE(Oversampler):

    def __init__(self, N, k):
        self.N = N
        self.k = k

    def make_samples(self, X, y, N, k):

        minorityLabel = super()._get_minority_label(y)
        minorityInstances = X[np.where(y[:] == minorityLabel)]
        majorityInstances = X[np.where(y[:] != minorityLabel)]
        IR = len(majorityInstances) / len(minorityInstances)



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

        if self.N == 'IR':
            counts = np.rint(((round(IR) - 1) * len(minorityInstances)) * Weights)
        else:
            counts = np.rint((self.N * len(minorityInstances) / 100) * Weights)

        S = []

        for i in range(len(minorityInstances)):
            count = counts[i]
            instance = minorityInstances[i]

            for j in range(int(count)):
                neighbourIdx = neighbourIndices[i][np.random.randint(0, self.k)]
                neighbourInstance = minorityInstances[neighbourIdx]
                weights = np.random.uniform(0, 1, len(instance))
                synthetic = instance + (neighbourInstance - instance) * weights
                S.append(synthetic)

        synthetic_y = np.full((len(S)), super()._get_minority_label(y))

        return np.array(S), synthetic_y
