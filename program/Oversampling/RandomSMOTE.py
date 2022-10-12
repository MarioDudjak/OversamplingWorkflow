import math
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors

from program.Oversampling.Oversampler import Oversampler

"""
RANDOM SMOTE

A New Over-Sampling Approach: Random-SMOTE for Learning from Imbalanced Data Sets
Yanjie Dong, Xuehua Wang
Xiong H., Lee W.B. (eds) Knowledge Science, Engineering and Management. 
KSEM 2011. Lecture Notes in Computer Science, vol 7091. Springer, Berlin, Heidelberg

"""

class RandomSMOTE(Oversampler):

    def __init__(self, N):
        self.N = N


    def make_samples(self, X, y, N, k):

        minorityLabel = super()._get_minority_label(y)
        minorityInstances = X[np.where(y[:] == minorityLabel)]
        majorityInstances = X[np.where(y[:] != minorityLabel)]
        IR = len(majorityInstances) / len(minorityInstances)

        S = []  # Synthetic data set
        if self.N == 'IR':
            count = np.int32(round(IR) - 1)
        else:
            count = np.int32(np.floor(self.N / 100))
        for instance in minorityInstances:
            y1, y2 = random.sample(list(minorityInstances), 2)    # Random selection of two examples from minority class
            for j in range(count):
                    t = y1 + np.random.uniform(0, 1) * (y2 - y1)  # Generating temporary examples along the line
                    # between two selected minority examples
                    p = instance + np.random.uniform(0, 1) * (t - instance)     # Generating synthetic minority class
                    # example along the line between each temporary example t and instance x
                    S.append(p)

        synthetic_y = np.full((len(S)), super()._get_minority_label(y))
        return np.array(S), synthetic_y


