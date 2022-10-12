class Score:
    def __init__(self, f1, precision, recall, accuracy, auc, gmean, N, spec, sens):
        self.f1 = f1
        self.precision = precision
        self.recall = recall
        self.accuracy = accuracy
        self.auc = auc
        self.gmean = gmean
        self.N = N
        self.specificity = spec
        self.sensitivity = sens

    def getPerformanceMeasures(self):
        return [self.f1, self.precision, self.recall, self.accuracy, self.auc, self.gmean, self.N, self.specificity, self.sensitivity]

    def toCommaSeparatedString(self):
        output = ",".join(str(measure) for measure in self.getPerformanceMeasures())
        # str(self.f1) + "," + str(self.precision) + "," + str(self.recall) + "," + str(self.accuracy) + "," + str(self.auc) + "," + str(self.gmean)
        return output


def getPerformanceMeasureNames():
    return ["F1", "Precision", "Recall", "Accuracy", "AUC", "gmean", "N", "Specificity", "Sensitivity"]
