import math
import random
import numpy as np

from program.Oversampling.Oversampler import Oversampler


class RandomOversampler(Oversampler):
    """Random oversamling algorithm

        Creates synthetic instances of minority dataset in binary classification problems.

        Parameters
        ----------
        N : int, optional (default=100)
            Number of SMOTE.

        Returns
        ----------
        S : array_like(float), shape (n_synthetic_instances, n_features)
            Created Synthetic dataset.
        """
    def __init__(self, N):
        self.N = N

    def make_samples(self, X, y, N, k):

        minorityLabel = super()._get_minority_label(y)
        minorityInstances = X[np.where(y[:] == minorityLabel)]
        majorityInstances = X[np.where(y[:] != minorityLabel)]
        IR = len(majorityInstances) / len(minorityInstances)
        S = []
        if self.N == 'IR':
            count = np.int32(round(IR) - 1)
        else:
            count = np.int32(np.floor(self.N / 100))

        for i in range(count):
            random_minority_instance = random.choice(list(minorityInstances))
            S.append(random_minority_instance)

        synthetic_y = np.full((len(S)), super()._get_minority_label(y))

        return np.array(S), synthetic_y


