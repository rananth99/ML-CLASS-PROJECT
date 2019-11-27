import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from svm import SVM, linear_kernel, rbf_kernel
from sklearn.metrics import accuracy_score

np.random.seed(31)


class Adaboost():

    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    # Select samples based on weights
    def weighted_selection(self, X, y, w, size):
        n_samples = int(np.shape(X)[0]*size)
        elements = [i for i in range(np.shape(X)[0])]
        choices = np.random.choice(elements, n_samples, replace=True, p=w)
        choices = set(choices)
        x_train = np.array([X[i] for i in choices])
        y_train = np.array([y[i] for i in choices])
        return x_train, y_train

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []
        self.alphas = []

        # Create classifiers
        for _ in range(self.n_clf):
            clf = SVM(rbf_kernel, 1.5)

            x_train, y_train = self.weighted_selection(X, y, w, 1)
            clf.fit(x_train, y_train)
            predictions = clf.predict(X)
            error = 1 - accuracy_score(predictions, y)
            if(error > 0.5):
                predictions *= -1
            
            # Calculate alpha
            alpha = 0.5 * math.log((1.0 - error) / (error + 1e-10))
            w *= np.exp(-alpha * y * predictions)
            # Normalize weights
            w /= np.sum(w)

            # Save classifier and alpha
            self.clfs.append(clf)
            self.alphas.append(alpha)

    def predict(self, X):
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples,))
        
        # For each classifier => label the samples
        for clf, alpha in zip(self.clfs, self.alphas):
            predictions = clf.predict(X)

            # Add predictions weighted by the classifier's alpha
            y_pred += alpha * predictions

        # Return sign of prediction sum
        y_pred = np.sign(y_pred).flatten()

        return y_pred


if __name__ == "__main__":

    # Dataset
    filename = './cat1.csv'
    file = pd.read_csv(filename, index_col=0)
    l = ['class', 'pred', 'galex_objid', 'sdss_objid', 'extinction_u',
         'extinction_g', 'extinction_r', 'extinction_i', 'extinction_z', 'spectrometric_redshift']
    x = file.drop(l, axis=1)
    x = np.array(x)
    y = np.array(file['class'])
    y = np.array([-1 if i == 0 else 1 for i in y])

    # Metrics
    accuracy = []
    precision0 = []
    precision1 = []
    recall0 = []
    recall1 = []
    f1score0 = []
    f1score1 = []

    # K-Fold cross validation
    from sklearn.model_selection import KFold
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    kf = KFold(n_splits=5, random_state=31, shuffle=False)

    for train_index, test_index in kf.split(x):

        x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        x_test = np.array(x_test)

        # Classifier
        classifier = Adaboost(n_clf=10)

        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)

        report = classification_report(y_test, y_pred, output_dict=True)

        accuracy.append(report['accuracy'])
        precision0.append(report['-1']['precision'])
        precision1.append(report['1']['precision'])
        recall0.append(report['-1']['recall'])
        recall1.append(report['1']['recall'])
        f1score0.append(report['-1']['f1-score'])
        f1score1.append(report['1']['f1-score'])

    accuracy = np.average(accuracy)
    precision0 = np.average(precision0)
    precision1 = np.average(precision1)
    recall0 = np.average(recall0)
    recall1 = np.average(recall1)
    f1score0 = np.average(f1score0)
    f1score1 = np.average(f1score1)

    print("K-fold Validation")
    print("Average accuracy", accuracy)
    print("Class : Precision , Recall , F1-Score")
    print("-1 : ", precision0, recall0, f1score0, sep=',')
    print("1 : ", precision1, recall1, f1score1, sep=',')
