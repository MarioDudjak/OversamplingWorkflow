from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

CLASSIFIERS = {
    "NN": KNeighborsClassifier(n_neighbors=1),
    #"LogReg": LogisticRegression(solver='lbfgs'),
    "MLP": MLPClassifier(),
    #"DT": DecisionTreeClassifier(),
    "5NN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(),
    #"GNB": GaussianNB(),
    #"Random forest": RandomForestClassifier()
}
