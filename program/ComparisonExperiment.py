import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from program.Classification.Classifiers import CLASSIFIERS
from program.DatasetManagement import DatasetConstants
from program.DatasetManagement.DatasetConstants import DS_PhD_Processed
from program.DatasetManagement.Preprocessing import preprocess
from program.Evaluation.Evaluation import calculate_performance_measures
from program.Evaluation.OverallScore import OverallScore
from program.Oversampling.UtilisedOversamplers import OVERSAMPLERS

ITERATIONS = 30
HOLDOUT_TEST_SIZE = 0.25
RANDOM_STATE = 42
RESULT_DIRECTORY = "OversamplersComparison"

datasets = DS_PhD_Processed
classifiers = CLASSIFIERS
oversamplers = OVERSAMPLERS

datasetIds = datasets.keys()
classifierIds = classifiers.keys()
oversamplerIds = oversamplers.keys()

resultAggregator = OverallScore(datasetIds, oversamplerIds, classifierIds, "Comparison")

for datasetId in datasetIds:
    print(datasetId)
    #dataset = arff.load(open(DatasetConstants.ROOT + datasets[datasetId], 'r'))
    dataset = pd.read_csv(DatasetConstants.ROOT + datasets[datasetId], header=0, index_col=0)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # X, y = preprocess(dataset)
    # data = dataset.iloc[:, :].values
    # data = np.unique(data, axis=0)
    # X = data[:, :-1]
    # y = data[:, -1]

    X = X.astype(np.float64)
    y = y.astype(np.uint32)

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
                X_train, y_train, 100, 5
            )

            synthetic_amount = len(X_train_synthetic)
            print("Created: " + str(synthetic_amount))

            X_train_augmented, y_train_augmented = oversampler.append(
                X_train, y_train, X_train_synthetic, y_train_synthetic
            )

            for classifierId in classifierIds:
                print(datasetId, oversamplerId, classifierId)

                classifier = classifiers[classifierId]

                model = classifier.fit(X_train_augmented, y_train_augmented)
                prediction = classifier.predict(X_test)
                score = calculate_performance_measures(y_test, prediction, synthetic_amount)
                print(score.getPerformanceMeasures())

                resultAggregator.insert(datasetId, oversamplerId, classifierId, score)

resultAggregator.save_overall_results(RESULT_DIRECTORY)
resultAggregator.save_overall_averages(RESULT_DIRECTORY)
resultAggregator.save_overall_stdDeviations(RESULT_DIRECTORY)
