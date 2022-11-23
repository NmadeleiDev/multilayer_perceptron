import numpy as np
from activations import DifferentiableFunction
from globals import *
from perceptron import MultilayerPerceptron


class Metric():
    def __init__(self, f, name: str, reduce_fn = np.mean):
        self.f = f
        self.reduce_fn = reduce_fn
        self.name = name
        self.cached_ts = []
        self.cached_ys = []

    def update(self, t, y):
        self.cached_ts.append(t)
        self.cached_ys.append(y)

    def calc(self):
        t = np.concatenate(self.cached_ts, axis=0)
        y = np.concatenate(self.cached_ys, axis=0)
        return self.reduce_fn(self.f(t, y))

    def __call__(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.reduce_fn(self.f(t, y))

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

class Trainer():
    def __init__(self, err_func: DifferentiableFunction, epochs: int, lr: float, metrics: list = [],
            early_stopping_patience=10, early_stopping_min_delta=0):
        self.err_func = err_func
        self.epochs = epochs
        self.lr = lr
        self.metrics = metrics
        self.use_early_stopping = early_stopping_min_delta >= 0
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

    def validate(self, model: MultilayerPerceptron, X: np.ndarray, y: np.ndarray):
        out = model(X)
        metrics = [m(out.squeeze(), y) for m in self.metrics]
        metrics_str = [f'{str(m)}={v:.3f}' for m, v in zip(self.metrics, metrics)]
        print(f'Validation results: {metrics_str}')

    def fit(self, model: MultilayerPerceptron, X: np.ndarray, y: np.ndarray, X_eval=None, y_eval=None, quiet=False):
        history = {'epoch': [], 'train_loss': [], 'val_loss': []}
        best_epoch_loss = None
        best_epoch_idx = 0

        history.update({f'train_{str(m)}': [] for m in self.metrics})
        history.update({f'val_{str(m)}': [] for m in self.metrics})
        for e in range(self.epochs):
            out = model(X)
            loss = np.mean(self.err_func(out, y))
            d_loss = self.err_func.d(out, y)
            train_metrics = [m(out.squeeze(), y) for m in self.metrics]
            train_metrics_str = [f'{str(m)}={v:.3f}' for m, v in zip(self.metrics, train_metrics)]
            history['epoch'].append(e)
            history['train_loss'].append(loss)
            for m, v in zip(self.metrics, train_metrics):
                history[f'train_{str(m)}'].append(v)
            if X_eval is not None and y_eval is not None:
                model.predict()
                out_eval = model(X_eval)
                model.train()
                val_loss = np.mean(self.err_func(out_eval, y_eval))
                val_metrics = [m(out_eval.squeeze(), y_eval) for m in self.metrics]
                val_metrics_str = [f'{str(m)}={v:.3f}' for m, v in zip(self.metrics, val_metrics)]
                val_log = f'| Val loss={val_loss:.3f}, metrics: {val_metrics_str}'
                history['val_loss'].append(val_loss)
                for m, v in zip(self.metrics, val_metrics):
                    history[f'val_{str(m)}'].append(v)
            else:
                val_log = ''
            if not quiet:
                print(f'Epoch {e}/{self.epochs}: loss={loss:.3f}, metrics: {train_metrics_str} {val_log}')

            if self.use_early_stopping:
                if best_epoch_loss is None or best_epoch_loss - val_loss > self.early_stopping_min_delta:
                    best_epoch_loss = loss
                    best_epoch_idx = e

                if e - best_epoch_idx > self.early_stopping_patience:
                    print('Stopping because val_loss is not decreasing anymore.')
                    break

            model.backward(d_loss, self.lr)

        if len(history['epoch']) > 0:
            print(f'Training finished: loss={loss:.3f}, metrics: {train_metrics_str} {val_log}')
        return history