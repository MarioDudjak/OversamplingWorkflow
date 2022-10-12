from program.Oversampling.Oversampler import Oversampler


class NoSmoteOversampler(Oversampler):
    """Not performing any oversampling.

        Creates no instances of minority dataset in binary classification problems.

        Parameters
        ----------


        Returns
        ----------
        S : array_like(float), shape (n_synthetic_instances, n_features)
            Original data.
        """
    def __init__(self):
        pass

    def make_samples(self, X, y, N, k):
        return [], []
