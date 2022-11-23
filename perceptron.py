import math
from typing import List
from activations import DifferentiableFunction, Identity
from globals import *
import numpy as np


class Layer():
    def __init__(self, input_dim: int, output_dim: int, activation: DifferentiableFunction = Identity(), 
            layer_index=0, beta=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        sigma = 1/math.sqrt(output_dim) if layer_index is None or layer_index == 0 else math.sqrt(2)/math.sqrt(output_dim)
        self.W = np.random.normal(0, sigma, (input_dim, output_dim))
        self.B = np.zeros((output_dim, 1))
        self.mode = MODE_TRAIN
        self.prev_g_W = np.zeros_like(self.W)
        self.prev_g_B = np.zeros_like(self.B)
        self.beta = beta

    def train(self):
        self.mode = MODE_TRAIN
        self.activation.train()
    def predict(self):
        self.mode = MODE_PREDICT
        self.activation.predict()
    def __str__(self):
        return f'[{self.activation.name}: ({self.input_dim} x N) -> ({self.output_dim} x N)] '

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Prod = np.dot(self.W.T, X) + self.B
        Act = self.activation(Prod.T).T
        if self.mode == MODE_TRAIN:
            self.X = X
            self.P = Prod
        return Act

    def backward(self, chain_tail: np.ndarray, lr: float):
        d_A = np.multiply(chain_tail, self.activation.d(self.P.T).T)
        d_W = np.dot(d_A, self.X.T).T
        d_B = d_A.sum(axis=1, keepdims=True)
        d_X = np.dot(d_A.T, self.W.T) # новый chain_tail

        g_W = self.prev_g_W * (1 - self.beta) + self.beta * d_W
        g_B = self.prev_g_B * (1 - self.beta) + self.beta * d_B
        
        self.W = self.W - lr * g_W
        self.B = self.B - lr * g_B

        self.prev_g_W = g_W
        self.prev_g_B = g_B

        return d_X.T
        
class BaseNet():
    def save(self, name=None):
        arc = '_'.join([str(l.output_dim) for l in self.layers])
        if name is None:
            name = f'./model_chpt__arc_{arc}.npz'
        elif not name.endswith('.npz'):
            name += '.npz'
        # lws = {f'lw_{i}': l.W for i, l in enumerate(self.layers)}
        # las = {f'la_{i}': l.activation for i, l in enumerate(self.layers)}
        ls = {f'l_{i}': l for i, l in enumerate(self.layers)}
        np.savez(name, **ls)
    
    def load(self, name):
        loaded = np.load(name, allow_pickle=True)
        self.layers = [loaded[f].item() for f in sorted(loaded.files, key=lambda x: int(x.split('_')[-1]))]

class MultilayerPerceptron(BaseNet):
    def __init__(self, input_dim, layers: List[tuple], beta=1):
        self.layers = [Layer((layers[i - 1][0] if i > 0 else input_dim), l[0], l[1], layer_index=i, beta=beta) for i, l in enumerate(layers)]

    def train(self):
        for i in range(len(self.layers)):
            self.layers[i].train()
    def predict(self):
        for i in range(len(self.layers)):
            self.layers[i].predict()

    def __call__(self, X: np.ndarray) -> np.ndarray:
        out = X.T
        for l in self.layers:
            out = l(out)
        return out.T

    def __str__(self):
        return '\n'.join([str(l) for l in self.layers])

    def backward(self, d_err: np.ndarray, lr: float):
        chain_tail = d_err.T
        for l in reversed(self.layers):
            chain_tail = l.backward(chain_tail, lr)