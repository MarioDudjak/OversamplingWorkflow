from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, roc_auc_score, confusion_matrix

from program.Evaluation.Score import Score


def calculate_performance_measures(y_test, prediction, N):
    # Calculating performance measures
    f1 = round(f1_score(y_test, prediction, average="macro"), 4)
    precision = round(precision_score(y_test, prediction, average="macro"), 4)
    recall = round(recall_score(y_test, prediction, average="macro"), 4)
    accuracy = round(accuracy_score(y_test, prediction), 4)
    roc_auc = round(roc_auc_score(y_test, prediction, average='macro'), 4)
    gmean = round(geometric_mean_score(y_test, prediction, average='macro'), 4)
    cm = confusion_matrix(y_test, prediction)
    tn, fp, fn, tp = cm.ravel()
    specificity = round(tn / (tn + fp), 4)
    sensitivity = round(tp / (tp + fn), 4)
    score = Score(f1, precision, recall, accuracy, roc_auc, gmean, N, specificity, sensitivity)
    return score
