import math
import random

import numpy as np
from sklearn.neighbors import NearestNeighbors

from program.Oversampling.Oversampler import Oversampler

"""
SIGMA NEAREST OVERSAMPLING BASED ON CONVEX COMBINATION - SNOCC

Implementation of "Oversampling method for imbalanced classification"
Zhuoyuan Zheng, Yunpeng Cai, Ye Li
Computing and Informatics, Vol. 34, 2016, pp.1017-1037

"""


class SNOCC(Oversampler):

    def __init__(self):
        pass

    def make_samples(self, X, y, N, k):

        minorityLabel = super()._get_minority_label(y)
        minorityInstances = X[np.where(y[:] == minorityLabel)]

        neigbourFinder = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(minorityInstances)
        neighbourDistances, neighbourIndices = neigbourFinder.kneighbors(minorityInstances)

        neighbourDistances = np.array(neighbourDistances, dtype=np.float64)[:, 1:]
        neighbourIndices = np.array(neighbourIndices, dtype=np.int32)[:, 1:]

        neighbourhoodDistanceMeans = np.mean(neighbourDistances, 1)
        sigma = np.mean(neighbourhoodDistanceMeans) + np.std(neighbourhoodDistanceMeans)

        sigmaNeighbours = []

        for instanceNeighbourhoodDistances, instanceNeighbourhoodIndices in zip(neighbourDistances, neighbourIndices):
            instanceSigmaNeighbours = []
            for distance, index in zip(instanceNeighbourhoodDistances, instanceNeighbourhoodIndices):
                if distance < sigma:
                    instanceSigmaNeighbours.append(index)
            sigmaNeighbours.append(instanceSigmaNeighbours)

        S = []

        count = math.floor(N / 100) * len(minorityInstances)
        for i in range(count):

            while True:
                index = np.random.randint(low=0, high=len(minorityInstances))
                sigmaNeighbourCount = len(sigmaNeighbours[index])
                if sigmaNeighbourCount > 1:
                    break

            s1 = minorityInstances[index]
            si2, si3 = random.sample(sigmaNeighbours[index], 2)
            s2 = minorityInstances[si2]
            s3 = minorityInstances[si3]

            weights = np.random.uniform(low=0, high=1, size=3)
            weights = weights / sum(weights)

            o = s1 * weights[0]
            o = o + s2*weights[1]
            o = o + s3*weights[2]

            S.append(o)


        synthetic_y = np.full((len(S)), super()._get_minority_label(y))

        return np.array(S), synthetic_y
