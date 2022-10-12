import copy
import numpy as np

from program.Evaluation import Score


class OverallScore:

    def __init__(self, datasetIds, oversamplerIds, classifierIds, point):
        self.datasetIds = datasetIds
        self.oversamplerIds = oversamplerIds
        self.classifierIds = classifierIds
        self.point = point

        classifier_results = {classifierId: [] for classifierId in self.classifierIds}
        oversampler_results = {oversamplerId: copy.deepcopy(classifier_results) for oversamplerId in
                               self.oversamplerIds}
        self.overall_results = {datasetId: copy.deepcopy(oversampler_results) for datasetId in self.datasetIds}

    def insert(self, datasetId, oversamplerId, classifierId, score):
        data = self.overall_results[datasetId][oversamplerId][classifierId]
        data.append(score)

    def save_overall_results(self, directoryName):
        """
        Saves the results into the apropriate files in the root folder path for further analysis. Root
        folder is determined by the name given as the parameter.

        :param overall_results: A three level dictionary of overall results. First level
        keys are dataset ids, second level keys are oversampler ids and the third level
        key are classifier ids. Each overall_results[datasetId][oversamplerId][classifierId]
        contains a list of REPETITION_COUNT score objects.

        """
        performance_measures = Score.getPerformanceMeasureNames()
        header = ",".join(performance_measures)
        root_results_path = "../results/" + directoryName + "/"

        for datasetId, datasetResult in self.overall_results.items():
            for oversamplerId, oversamplerResult in datasetResult.items():
                for classifierId, classifierResult in oversamplerResult.items():
                    filename = root_results_path + "_".join([str(datasetId), str(oversamplerId), str(classifierId)])
                    with open(filename + "_" + str(self.point) + ".txt", "w") as file:
                        file.write(header + "\n")
                        for score in classifierResult:
                            file.write(score.toCommaSeparatedString() + "\n")


    def save_overall_averages(self, directoryName):

        root_results_path = "../results/" + directoryName + "/"

        for classifierId in self.classifierIds:
            datasetAverages = []
            for datasetId in self.datasetIds:

                oversamplerAverages = []

                for oversamplerID in self.oversamplerIds:

                    scores = self.overall_results[datasetId][oversamplerID][classifierId]
                    measures = [element.getPerformanceMeasures() for element in scores]
                    average = np.mean(measures, 0)
                    average = np.around(average, decimals=6)
                    oversamplerAverages.append(average)

                datasetAverages.append(oversamplerAverages)

            with open(root_results_path + "_avg_" + classifierId + "_" + str(self.point) + ".txt", "w") as file:

                for i, measure in enumerate(Score.getPerformanceMeasureNames()):
                    file.write(measure + "\n")
                    file.write("Dataset," + ",".join(self.oversamplerIds) + "\n")
                    for datasetAverage, name in zip(datasetAverages, self.datasetIds):
                        file.write(str(name) + ",")
                        for oversamplerAverage in datasetAverage:
                            file.write(str(oversamplerAverage[i]) + ",")
                        file.write("\n")
                file.write("\n")


    def save_overall_stdDeviations(self, directoryName):

        root_results_path = "../results/" + directoryName + "/"

        for classifierId in self.classifierIds:
            datasetStdDeviations = []
            for datasetId in self.datasetIds:
                oversamplerStdDeviations = []
                for oversamplerID in self.oversamplerIds:
                    scores = self.overall_results[datasetId][oversamplerID][classifierId]
                    measures = [element.getPerformanceMeasures() for element in scores]
                    stdDeviation = np.std(measures, 0)
                    oversamplerStdDeviations.append(stdDeviation)

                datasetStdDeviations.append(oversamplerStdDeviations)

            with open(root_results_path + "_std_" + classifierId + "_" + str(self.point) + ".txt", "w") as file:

                for i, measure in enumerate(Score.getPerformanceMeasureNames()):
                    file.write(measure + "\n")
                    file.write("Dataset," + ",".join(self.oversamplerIds) + "\n")
                    for datasetAverage, name in zip(datasetStdDeviations, self.datasetIds):
                        file.write(str(name) + ",")
                        for oversamplerAverage in datasetAverage:
                            file.write(str(oversamplerAverage[i]) + ",")
                        file.write("\n")
                file.write("\n")
