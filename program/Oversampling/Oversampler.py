import numpy as np

class Oversampler:

    def _get_minority_label(self, y):
        # Define minority label
        (values, counts) = np.unique(y, return_counts=True)
        minority_label = np.argmin(counts)
        return minority_label


    def append(self, X, y, X_synthetic, y_synthetic):
        if (len(X_synthetic)) == 0:
            return X, y
        oversampled_X = np.concatenate([X, X_synthetic], axis=0)
        oversampled_y = np.concatenate([y, y_synthetic], axis=0)

        return oversampled_X, oversampled_y


def calculateParameters(y, points=5):

    parameters = []
    k = 5

    (values, counts) = np.unique(y, return_counts=True)
    minorityLabel = np.argmin(counts)
    majorityLabel = np.argmax(counts)

    minorityCount = counts[minorityLabel]
    majorityCount = counts[majorityLabel]

    IR = np.round(majorityCount / minorityCount)
    firstPoint = 100
    lastPoint = (IR-1) * 100
    step = np.round((lastPoint - firstPoint) / (points-2))

    parameters.append([firstPoint, k])
    for i in range(1, points-2):
        N = np.int32(np.round((firstPoint + i * step) / 100) * 100)
        parameters.append([N, k])
    parameters.append([np.int32(np.round(lastPoint/100) * 100), k])
    parameters = np.array(parameters)
    parameters = np.unique(parameters, axis=0)

    return parameters
