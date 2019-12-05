import numpy as np
import sys
import copy
import random
#ex2_sharon.py

class Model(object):
    def predict(self, x):
        raise NotImplementedError

    def update(self, x, y, lr):
        raise NotImplementedError


class Perceptron(Model):
    def __init__(self, n, k):
        """
        :param n: dimension of the input
        :param k: dimension of the output
        """
        self.W = np.zeros((k, n))

    def predict(self, x):
        return np.argmax(np.dot(self.W, x))

    def update(self, x, y, lr):
        y_hat = self.predict(x)
        if y != y_hat:
            # update W:
            # enforce in the right class
            self.W[y, :] += lr * x
            # discourage in the wrong class we have predicted
            self.W[y_hat, :] -= lr * x


class SVM(Model):
    def __init__(self, n, k, l):
        """
        :param n: dimension of the input
        :param k: dimension of the output
        :param l: lambda constant (of the svm)
        """
        self.k = k
        self.W = np.zeros((k, n))
        self._lambda = l

    def predict(self, x):
        return np.argmax(np.dot(self.W, x))

    def update(self, x, y, lr):
        y_hat = self.predict(x)
        coef = (1 - lr * self._lambda)
        # we need to update 3 parts - correct Y, predicted Y, and all the rest
        for i in range(self.k):
            new_w = self.W[i, :] * coef
            if i == y:
                new_w += lr * x
            elif i == y_hat:
                new_w -= lr * x
            self.W[i, :] = new_w


class PA(Model):
    def __init__(self, n, k):
        self.W = np.zeros((k, n))

    def predict(self, x):
        return np.argmax(np.dot(self.W, x))

    def update(self, x, y, lr):
        y_hat = self.predict(x)
        if y != y_hat:
            w_y = self.W[y, :]
            w_y_hat = self.W[y_hat, :]
            loss = max(0, 1 - np.dot(w_y, x) + np.dot(w_y_hat, x))
            tau = loss / (2 * (np.linalg.norm(x) ** 2))

            self.W[y, :] += tau * x
            self.W[y_hat] -= tau * x


def measure_accuracy(model, X, Y):
    good = bad = 0.0
    for (x, y) in zip(X, Y):
        y_hat = model.predict(x)
        if y == y_hat:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def train(model, X, Y, epochs, lr, verbose=False):
    indices = list(range(len(X)))
    best_model = model
    best_acc = measure_accuracy(model, X, Y)
    for epoch in range(epochs):
        random.shuffle(indices)
        for i in indices:
            x, y = X[i], Y[i]
            model.update(x, y, lr)

        acc = measure_accuracy(model, X, Y)
        if acc > best_acc:
            best_model = copy.deepcopy(model)
            best_acc = acc
        if verbose:
            print('Epoch: {0}, Accuracy: {1}%'.format(epoch + 1, int(100 * acc)))
    return best_model


SEX2F = {'M': 2, 'F': 4, 'I': 8}


def parse_sample(sample):
    x = np.zeros(len(sample))
    x[0] = SEX2F[sample[0]]
    # all the rest are just floats - parse them as is
    for i in range(1, len(sample)):
        x[i] = np.float(sample[i])
    return x


def read_x_file(x_fname):
    samples = np.genfromtxt(x_fname, dtype='str', delimiter=',')
    X = [parse_sample(s) for s in samples]
    return X


def normalize_X(X):
    # currently does nothing
    normalized = X
    # you may do other normalizing - think about it later.
    return normalized


def read_labeled_data(train_x_fname, train_y_fname):
    X = read_x_file(train_x_fname)
    Y = np.genfromtxt(train_y_fname, dtype='str',delimiter='\n')
    Y=Y.astype(np.float)
    return X, Y.astype(np.int)


def acc2s(acc):
    return '{}%'.format(int(100 * acc))


def run_tests(X,Y):
    k = len(set(Y))
    n = len(X[0])

    perceptron = Perceptron(n, k)
    svm = SVM(n, k, 0.1)
    pa = PA(n, k)

    print('Running Perceptron:')
    best = train(perceptron, X, Y, 50, 0.001, verbose=True)
    print('best:',acc2s(measure_accuracy(best,X,Y)))

    print('Running SVM:')
    best = train(svm, X, Y, 50, 0.001, verbose=True)
    print('best:',acc2s(measure_accuracy(best,X,Y)))

    print('Running PA:')
    best = train(pa, X, Y, 50, None, verbose=True)
    print('best:',acc2s(measure_accuracy(best,X,Y)))

def main(train_x_fname, train_y_fname, test_x_fname):
    X, Y = read_labeled_data(train_x_fname, train_y_fname)


    X = normalize_X(X)
    k = len(set(Y))
    n = len(X[0])

    perceptron = Perceptron(n, k)
    svm = SVM(n, k, 0.1)
    pa = PA(n, k)

    best_perceptron = train(perceptron, X, Y, 50, 0.001)
    best_svm = train(svm, X, Y, 50, 0.001)
    best_pa = train(pa, X, Y, 50, None)

    test_X = read_x_file(test_x_fname)
    for x in test_X:
        pred_perceptron = best_perceptron.predict(x)
        pred_svm = best_svm.predict(x)
        pred_pa = best_pa.predict(x)
        print('perceptron: {0}, svm: {1}, pa: {2}'.format(pred_perceptron, pred_svm, pred_pa))





if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
