import numpy as np
import random
from pdb import set_trace
# h(x) = ax + b
# E(a, b) = (1/2m) * soma(ax + b - y)²
# d/da(E(a, b)) = (1/m) * soma((ax + b - y)x)
# d/db(E(a, b)) = (1/m) * soma(ax + b - y)
class Regressor():
    def __init__(self, step, n_ite):
        self.step = step
        self.n_ite = n_ite
    
    def fit(self, X_train, Y_train):
        np.random.seed(32)
        self.a = np.random.random_sample()
        self.b = np.random.random_sample()
        print(self.mean_square_error(self.a, self.b))
        for i in range(self.n_ite):
            der_a, der_b = self.derivada(X_train, Y_train)
            temp0 = self.a - (self.step * der_a)
            temp1 = self.b - (self.step * der_b)
            self.a = temp0
            self.b = temp1
            print(self.mean_square_error(self.a, self.b))
        return([self.a, self.b])

    def derivada(self, X_train, Y_train):
        temp0 = np.sum(((self.a * X_train + self.b - Y_train) * X_train))
        temp1 = np.sum((self.a * X_train + self.b - Y_train))
        return (temp0/(X_train.size), temp1/(Y_train.size))

    def predict(self, X_test):
        Y_test = self.a * X_test + self.b
        return Y_test

    def mean_square_error(self, X_train, Y_train):
        temp = np.sum((self.a * X_train + self.b - Y_train)**2)
        return(temp/(X_train.size))