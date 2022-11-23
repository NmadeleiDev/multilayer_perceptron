import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from activations import DifferentiableFunction, ReLU, SoftmaxActivation, der_ReLU, der_multiclass_cross_entropy, multiclass_cross_entropy, sigmoid, sigmoid_der
from perceptron import MultilayerPerceptron
from trainer import Metric, Trainer

from utils import load_data

np.random.seed(21)
sns.set_theme()

metrics_arr = [Metric(lambda t, y: metrics.accuracy_score(y.astype(bool), t > 0.5), 'Accuracy'),
                Metric(lambda t, y: metrics.recall_score(y.astype(bool), t > 0.5, average='macro', zero_division=0), 'Recall'),
                Metric(lambda t, y: metrics.precision_score(y.astype(bool), t > 0.5, average='macro', zero_division=0), 'Precision')]

def train_model(epochs=600, lr=3e-4, early_stopping_patience=100, early_stopping_min_delta=-1, beta=0.3,
        save_plot=False, model_save_name='untitled', data_path=None, test_split=0.3, quiet=False):

    if test_split <= 0 or test_split >= 1:
        print(f'test_split={test_split} is invalid. It must be between 0 and 1.')
        exit(1)
    X_tr, Y_tr, X_te, Y_te = load_data(path=data_path, test_split=test_split, target_as_onehot=True, scaling_save_path=f'scaling_config_{model_save_name}.npy')

    relu_act = DifferentiableFunction(ReLU, der_ReLU)
    softmax_act = SoftmaxActivation()
    err_fn = DifferentiableFunction(multiclass_cross_entropy, der_multiclass_cross_entropy)

    model = MultilayerPerceptron(X_tr.shape[1], [
        (32, relu_act),
        (16, relu_act),
        (16, relu_act),
        (2, softmax_act),
    ], beta=beta)

    trainer = Trainer(err_fn, epochs, lr, metrics_arr,
        early_stopping_patience=early_stopping_patience, early_stopping_min_delta=early_stopping_min_delta)
    print('Model structure:')
    print(model)

    history = trainer.fit(model, X_tr, Y_tr, X_te, Y_te, quiet=quiet)

    if save_plot:
        loss_cols = [k for k in history.keys() if k.endswith('_loss')]
        metric_cols = [k for k in history.keys() if (not k.endswith('_loss') and k != 'epoch')]

        fig, ax = plt.subplots(2, 1, figsize=(12, 10))
        sns.lineplot(data=pd.melt(pd.DataFrame(history)[['epoch'] + loss_cols], ['epoch']), x='epoch', y='value', hue='variable', ax=ax[0])
        sns.lineplot(data=pd.melt(pd.DataFrame(history)[['epoch'] + metric_cols], ['epoch']), x='epoch', y='value', hue='variable', ax=ax[1])
        plt.savefig('./training_log.png')

    model.save(model_save_name)
    
def predict(data_path: str, model_name: str = 'untitled'):
    try:
        model = MultilayerPerceptron(0, [])
        model.load(model_name + '.npz')
    except Exception:
        print(f"Model file {model_name} not found.")
        exit(1)
    
    conf_path = f'scaling_config_{model_name}.npy'
    try:
        X, Y = load_data(data_path, 0, True, scaling_load_path=conf_path) 
    except Exception:
        print(f"Data file {data_path} or config {conf_path} not found.")
        exit(1)

    err_fn = DifferentiableFunction(multiclass_cross_entropy, der_multiclass_cross_entropy)

    trainer = Trainer(err_fn, 0, 0, metrics_arr)

    trainer.validate(model, X, Y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments description')
    parser.add_argument("mode", type=str, choices=['predict', 'train'])
    parser.add_argument("-n", "--model-name", required=False, default='untitled')
    parser.add_argument("-d", "--data", required=False, default='./data.csv')
    parser.add_argument("-p", "--save-plot", required=False, default=False, action='store_true')
    parser.add_argument("-q", "--quiet", required=False, default=False, action='store_true')
    parser.add_argument("-e", "--epochs", required=False, default=70, type=int)
    parser.add_argument("-r", "--learning-rate", required=False, default=1e-3, type=float)
    parser.add_argument("-s", "--test-split", required=False, default=0.3, type=float)
    parser.add_argument("-b", "--beta", required=False, default=0.5, type=float)
    parser.add_argument('--stop-patience', dest='patience', type=int, help='Specify early stopping patience', default=100)
    parser.add_argument('--stop-min-delta', dest='delta', type=float, help='Specify early stopping minimum delta', default=-1)
    args = parser.parse_args()

    if args.mode == 'train':
        train_model(epochs=args.epochs, lr=args.learning_rate, early_stopping_patience=args.patience,
                        early_stopping_min_delta=args.delta, beta=args.beta, save_plot=args.save_plot, 
                        model_save_name=args.model_name, data_path=args.data, test_split=args.test_split,
                        quiet=args.quiet)
    elif args.mode == 'predict':
        predict(args.data, args.model_name)

