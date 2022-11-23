from __future__ import annotations
import numpy as np
import math
from globals import *

def ReLU(x):
    return np.where(x > 0, x, 0)

def der_ReLU(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1/(1 + np.e**(-x))

def sigmoid_der(x):
    s = sigmoid(x)
    return s*(1 - s)

def binary_cross_entropy(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where(y == 0, np.log(1 - t), np.log(t)) * -1

def der_binary_cross_entropy(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where(y == 0, -1/(1 - t), 1/(t)) * -1

def multiclass_cross_entropy(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.log(1 - t + epsilon) * (1 - y) + np.log(t + epsilon) * y) * -1

def der_multiclass_cross_entropy(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((-1 / (1 - t + epsilon)) * (1 - y) + (1/(t + epsilon)) * y) * -1

class DifferentiableFunction():
    def __init__(self, f, d_f, name: str = None):
        self.f = f
        self.d_f = d_f
        self.name = name if name is not None else f.__name__
        self.mode = MODE_TRAIN

    def train(self):
        self.mode = MODE_TRAIN
    def predict(self):
        self.mode = MODE_PREDICT

    def __call__(self, *x):
        return self.f(*x)

    def d(self, *x):
        return self.d_f(*x)

class SoftmaxActivation(DifferentiableFunction):
    def __init__(self):
        self.name = 'Softmax'
        self.mode = MODE_TRAIN
        
    def __call__(self, x) -> np.ndarray:
        exp = np.exp(x)
        sum = np.sum(exp, axis=1, keepdims=True)
        sm = exp / sum
        if self.mode == MODE_TRAIN:
            self.sum = sum
            self.sm = sm
        return sm
    def d(self, x: np.ndarray):
        return np.multiply(self.sm, (1 - self.sm))

class Identity(DifferentiableFunction):
    def __init__(self, *args, **kwargs):
        self.f = lambda x: x
        self.d_f = lambda x: 1