from pruning import preprocess, train, train_plots, test, weigh_pruning, neuron_pruning, prune_weights_using_Sw, plots, noise
from tensorflow.keras.models import load_model
import os
from IPython.core.interactiveshell import InteractiveShell
from contextlib import redirect_stdout, redirect_stderr
import sys
from tensorflow.keras.datasets import mnist

# without noise
log_dir = "logs/testing_without_noise"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

output_log_file = os.path.join(log_dir, "balance_0.5%_output.log")

if os.path.exists(output_log_file):
    open(output_log_file, 'w').close()

with open(output_log_file, 'a') as log_file:
    with redirect_stdout(log_file), redirect_stderr(log_file):
        n_features, n_classes, Y_train_onehot, Y_test_onehot, X_train_final, X_valid, Y_train_final, Y_valid, X_test_norm, X_train_norm = preprocess(mnist, imbalance_factor=0.005)
        model, save_best = train(n_features, n_classes, 'mnist.hdf5')
        train_plots(model, X_train_final, Y_train_final, X_valid, Y_valid, save_best)
        test(model, X_test_norm, Y_test_onehot)
        trained_model = load_model("mnist.hdf5")
        total_no_layers = len(trained_model.layers)
        print('Starting neuron pruning')
        neuron_pruning_scores = neuron_pruning(X_test_norm, Y_test_onehot, total_no_layers, trained_model, 'mnist.hdf5')
        print('Starting weight pruning using nc')
        weight_pruning_scores_nc = prune_weights_using_Sw(X_test_norm, Y_test_onehot, X_train_norm, Y_train_onehot, K, trained_model, 'mnist.hdf5')
        print('Starting weight pruning')
        weight_pruning_scores = weigh_pruning(X_test_norm, Y_test_onehot, K, trained_model, 'mnist.hdf5')
        plots(weight_pruning_scores, neuron_pruning_scores, list(map(lambda x: x[1], weight_pruning_scores_nc)), '0.5%')

print(f"Output saved to {output_log_file}")

# with noisy data
log_dir = "logs/testing_with_noise"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


output_log_file = os.path.join(log_dir, "balance_1%_output.log")

if os.path.exists(output_log_file):
    open(output_log_file, 'w').close()

with open(output_log_file, 'a') as log_file:
    with redirect_stdout(log_file), redirect_stderr(log_file):
        n_features, n_classes, Y_train_onehot, Y_test_onehot, X_train_final, X_valid, Y_train_final, Y_valid, X_test_norm,X_train_norm = preprocess(mnist,imbalance_factor=0.01)
        model,save_best = train(n_features, n_classes, 'mnist.hdf5')
        train_plots(model,X_train_final, Y_train_final, X_valid, Y_valid, save_best)
        X_test_noisy,Y_test_onehot = noise(mnist)
        test(model, X_test_noisy, Y_test_onehot)
        trained_model = load_model("mnist.hdf5")
        total_no_layers = len(trained_model.layers)
        neuron_pruning_scores = neuron_pruning(X_test_noisy, Y_test_onehot, total_no_layers, trained_model, 'mnist.hdf5')
        weight_pruning_scores_nc = prune_weights_using_Sw(X_test_noisy, Y_test_onehot,X_train_norm, Y_train_onehot, K, trained_model, 'mnist.hdf5')
        weight_pruning_scores = weigh_pruning(X_test_noisy, Y_test_onehot, K, trained_model, 'mnist.hdf5')

        plots(weight_pruning_scores,neuron_pruning_scores, list(map(lambda x: x[1], weight_pruning_scores_nc)),'1%')

print(f"Output saved to {output_log_file}")
