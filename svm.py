import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cvxopt
import cvxopt.solvers
cvxopt.solvers.options['show_progress'] = False

np.random.seed(31)


# Different kernels
def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def gaussian_kernel(x, y, sigma=0.9):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


def rbf_kernel(x, y, gamma=6):
    return np.exp(-np.linalg.norm(x-y)/gamma)


# SVM class
class SVM():

    def __init__(self, kernel, C):
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):

        sample_size, feature_size = X.shape

        K = np.zeros((sample_size, sample_size))
        for i in range(sample_size):
            for j in range(sample_size):
                K[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(sample_size) * -1)
        A = cvxopt.matrix(y, (1, sample_size), 'd')
        b = cvxopt.matrix(0.0)

        tmp1 = np.diag(np.ones(sample_size) * -1)
        tmp2 = np.identity(sample_size)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(sample_size)
        tmp2 = np.ones(sample_size) * self.C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        # print("%d support vectors out of %d points" %
        #   (len(self.a), sample_size))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(feature_size)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def predict(self, X):
        if self.w is not None:
            return np.sign(np.dot(X, self.w) + self.b)
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return np.sign(y_predict + self.b)


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

    # Classifier
    classifier = SVM(rbf_kernel, 1.5)

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
    kf = KFold(n_splits=5, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(x):

        x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        x_test = np.array(x_test)

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
