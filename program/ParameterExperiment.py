import arff
from sklearn.model_selection import StratifiedShuffleSplit

from program.DatasetManagement import DatasetConstants
from program.DatasetManagement.DatasetConstants import *
from program.Classification.Classifiers import *
from program.DatasetManagement.Preprocessing import preprocess
from program.Evaluation.Evaluation import calculate_performance_measures
from program.Evaluation.OverallScore import OverallScore
from program.Evaluation.Score import Score
from program.Oversampling.Oversampler import calculateParameters
from program.Oversampling.UtilisedOversamplers import OVERSAMPLERS

ITERATIONS = 30
HOLDOUT_TEST_SIZE = 0.25
RANDOM_STATE = 42

RESULT_DIRECTORY = "parametersWaRSMOTE"

datasets = IMBALANCED_DATASETS
classifiers = CLASSIFIERS
oversamplers = OVERSAMPLERS

datasetIds = datasets.keys()
classifierIds = classifiers.keys()
oversamplerIds = oversamplers.keys()


for datasetId in datasetIds:

    dataset = arff.load(open(DatasetConstants.ROOT + datasets[datasetId], 'r'))
    X, y = preprocess(dataset)
    datasetParameters = calculateParameters(y, points=5)

    print(datasetId, datasetParameters)

    resultAggregator = OverallScore(datasetParameters[:, 0], oversamplerIds, classifierIds, datasetId)

    for parameters in datasetParameters:

        N = parameters[0]
        k = parameters[1]

        splitter = StratifiedShuffleSplit(
            n_splits=ITERATIONS,
            test_size=HOLDOUT_TEST_SIZE,
            random_state=RANDOM_STATE
        )

        for train_index, test_index in splitter.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for oversamplerId in oversamplerIds:

                oversampler = oversamplers[oversamplerId]

                X_train_synthetic, y_train_synthetic = oversampler.make_samples(
                    X_train, y_train, N, k
                )

                X_train_augmented, y_train_augmented = oversampler.append(
                    X_train, y_train, X_train_synthetic, y_train_synthetic
                )

                for classifierId in classifierIds:

                    classifier = classifiers[classifierId]

                    model = classifier.fit(X_train_augmented, y_train_augmented)
                    prediction = classifier.predict(X_test)
                    score = calculate_performance_measures(y_test, prediction)

                    resultAggregator.insert(N, oversamplerId, classifierId, score)

    resultAggregator.save_overall_results(RESULT_DIRECTORY)
    resultAggregator.save_overall_averages(RESULT_DIRECTORY)
    resultAggregator.save_overall_stdDeviations(RESULT_DIRECTORY)
