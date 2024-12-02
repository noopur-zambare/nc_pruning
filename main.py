from pruning import preprocess, train, train_plots, test, weigh_pruning, neuron_pruning, prune_weights_using_Sw, plots
from tensorflow.keras.models import load_model

n_features, n_classes, Y_train_onehot, Y_test_onehot, X_train_final, X_valid, Y_train_final, Y_valid, X_test_norm,X_train_norm = preprocess(mnist,imbalance_factor=0.01)
model,save_best = train(n_features, n_classes, 'mnist.hdf5')
train_plots(model,X_train_final, Y_train_final, X_valid, Y_valid, save_best)
test(model, X_test_norm, Y_test_onehot)
trained_model = load_model("mnist.hdf5")
weight_pruning_scores = weigh_pruning(X_test_norm, Y_test_onehot, K, trained_model, 'mnist.hdf5')
total_no_layers = len(trained_model.layers)
neuron_pruning_scores = neuron_pruning(X_test_norm, Y_test_onehot, total_no_layers, trained_model, 'mnist.hdf5')
weight_pruning_scores_nc = prune_weights_using_Sw(X_test_norm, Y_test_onehot,X_train_norm, Y_train_onehot, K, trained_model, 'mnist.hdf5')
plots(weight_pruning_scores,neuron_pruning_scores, list(map(lambda x: x[1], weight_pruning_scores_nc)),'1%')