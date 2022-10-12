import numpy as np
from imblearn.over_sampling import SMOTE

from program.Oversampling.Oversampler import Oversampler


class SMOTEImb(Oversampler):

    def __init__(self):
        pass

    def make_samples(self, X, y, N, k):

        minorityLabel = super()._get_minority_label(y)
        minorityInstances = X[np.where(y[:] == minorityLabel)]
        minorityCount = len(minorityInstances)
        majorityCount = len(y) - minorityCount

        IR = minorityCount / majorityCount

        desiredIR = (minorityCount + minorityCount * (N/100)) / majorityCount
        if desiredIR > 1.0:
            desiredIR = 1.0

        smote = SMOTE(sampling_strategy=desiredIR, k_neighbors=k)
        X_smo, y_smo = smote.fit_sample(X, y)
        S = X_smo[len(X):, :]
        synthetic_y = np.full((len(S)), minorityLabel)

        return S, synthetic_y