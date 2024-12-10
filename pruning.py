import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist, mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from keras.models import Model
from sklearn.metrics import accuracy_score
from IPython.core.interactiveshell import InteractiveShell
from contextlib import redirect_stdout, redirect_stderr
import sys


# Imbalanced MNIST
def preprocess(dataset, imbalance_factor=0.01):

    ((X_train, Y_train), (X_test, Y_test)) = dataset.load_data()
    X_train_reshaped = X_train.reshape(len(X_train), -1)   
    X_test_reshaped = X_test.reshape(len(X_test), -1)

    X_train_norm = X_train_reshaped / 255            
    X_test_norm = X_test_reshaped / 255

    n_features = X_train_norm.shape[1]
    n_classes = 10

    print('Number of input features (image pixels) : ', n_features)
    print('Number of target classes: ', n_classes)

    Y_train_onehot = to_categorical(Y_train, num_classes=n_classes)
    Y_test_onehot = to_categorical(Y_test, num_classes=n_classes)

    class_1_indices = np.where(Y_train == 1)[0]
    class_2_indices = np.where(Y_train == 2)[0]
    class_3_indices = np.where(Y_train == 3)[0]
    
    class_1_sampled = np.random.choice(class_1_indices, int(len(class_1_indices) * imbalance_factor), replace=False)
    class_2_sampled = np.random.choice(class_2_indices, int(len(class_2_indices) * imbalance_factor), replace=False)
    class_3_sampled = np.random.choice(class_3_indices, int(len(class_3_indices) * imbalance_factor), replace=False)

    other_classes_indices = np.where((Y_train != 1) & (Y_train != 2) & (Y_train != 3))[0]
    selected_indices = np.concatenate([class_1_sampled, class_2_sampled,class_3_sampled, other_classes_indices])

    X_train_imbalance = X_train_norm[selected_indices]
    Y_train_imbalance = Y_train_onehot[selected_indices]

    X_train_final, X_valid, Y_train_final, Y_valid = train_test_split(
        X_train_imbalance, Y_train_imbalance, test_size=0.25, stratify=Y_train_imbalance
    )

    return (n_features, n_classes, Y_train_onehot, Y_test_onehot, X_train_final, X_valid, Y_train_final, Y_valid, X_test_norm, X_train_norm)



def train(n_features, n_classes, model_name):
    model = Sequential()
    model.add(Dense(1000, input_dim = n_features, activation='relu', use_bias=False))

    model.add(Dense(200, activation='relu', use_bias=False))
    model.add(Dense(n_classes, activation='softmax', use_bias=False))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
    # model.summary()
    save_at = str(model_name)
    save_best = ModelCheckpoint (save_at, monitor='val_accuracy', verbose=0, 
                             save_best_only=True, save_weights_only=False, mode='max')
    return (model,save_best)


def train_plots(model,X_train_final, Y_train_final, X_valid, Y_valid, save_best):
    history = model.fit( X_train_final, Y_train_final, 
                        epochs = 10, batch_size = 20, 
                        callbacks=[save_best], verbose=1, 
                        validation_data = (X_valid, Y_valid) )

    plt.figure(figsize=(6, 5))
    plt.plot(history.history['accuracy'], color='r')
    plt.plot(history.history['val_accuracy'], color='b')
    plt.title('Model Accuracy', weight='bold', fontsize=16)
    plt.ylabel('accuracy', weight='bold', fontsize=14)
    plt.xlabel('epoch', weight='bold', fontsize=14)
    plt.ylim(0.5, 1)
    plt.xticks(weight='bold', fontsize=12)
    plt.yticks(weight='bold', fontsize=12)
    plt.legend(['train', 'val'], loc='upper left', prop={'size': 14})
    plt.grid(color = 'y', linewidth='0.5')
    plt.show()




def test(model, X_test_norm, Y_test_onehot, verbose=True):
    # Evaluate test accuracy
    score = model.evaluate(X_test_norm, Y_test_onehot, verbose=0)
    test_accuracy = round((score[1] * 100), 2)
    
    if verbose:
        print('Overall accuracy on the test set: ', test_accuracy, '%')
    
    Y_pred_probs = model.predict(X_test_norm, verbose=0)
    Y_pred_classes = np.argmax(Y_pred_probs, axis=1)
    Y_true_classes = np.argmax(Y_test_onehot, axis=1)
    
    class_accuracies = []
    for cls in np.unique(Y_true_classes):
        cls_mask = (Y_true_classes == cls)
        cls_accuracy = accuracy_score(Y_true_classes[cls_mask], Y_pred_classes[cls_mask])
        class_accuracies.append((cls, cls_accuracy * 100))
    
    if verbose:
        print("\nPer-Class Accuracy:")
        for cls, acc in class_accuracies:
            print(f"Class {cls}: {acc:.2f}%")
    print(class_accuracies)
    return test_accuracy, class_accuracies


# pruning%
K = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

def weigh_pruning(X_test_norm, Y_test_onehot, K, trained_model, model_load):

    total_no_layers = len(trained_model.layers)

    all_weights = {}

    for layer_no in range(total_no_layers - 1):
        layer_weights = (pd.DataFrame(trained_model.layers[layer_no].get_weights()[0]).stack()).to_dict() 
        layer_weights = { (layer_no, k[0], k[1]): v for k, v in layer_weights.items() }
        all_weights.update(layer_weights)
    
    all_weights_sorted = {k: v for k, v in sorted(all_weights.items(), key=lambda item: abs(item[1]))}
    total_no_weights = len(all_weights_sorted) 
    print('all_weights',len(all_weights))
    weight_pruning_scores = []

    for pruning_percent in K:

        new_model = load_model(model_load)
        new_weights = trained_model.get_weights().copy()

        prune_fraction = pruning_percent/100
        number_of_weights_to_be_pruned = int(prune_fraction*total_no_weights)
        weights_to_be_pruned = {k: all_weights_sorted[k] for k in list(all_weights_sorted)[ :  number_of_weights_to_be_pruned]}     

        for k, v in weights_to_be_pruned.items():
            new_weights[k[0]][k[1], k[2]] = 0

        for layer_no in range(total_no_layers - 1) :
            new_layer_weights = new_weights[layer_no].reshape(1, new_weights[layer_no].shape[0], new_weights[layer_no].shape[1])
            new_model.layers[layer_no].set_weights(new_layer_weights)
        
        new_score  = new_model.evaluate(X_test_norm, Y_test_onehot, verbose=0)
        test(new_model, X_test_norm, Y_test_onehot, verbose=True)
        weight_pruning_scores .append(new_score[1])
    return weight_pruning_scores


def neuron_pruning(X_test_norm, Y_test_onehot, total_no_layers, trained_model, model_load):

    all_neurons = {}

    for layer_no in range(total_no_layers - 1):         

        layer_neurons = {}
        layer_neurons_df = pd.DataFrame(trained_model.layers[layer_no].get_weights()[0])

        for i in range(len(layer_neurons_df.columns)):
            layer_neurons.update({ i : np.array( layer_neurons_df.iloc[:,i] ) })    
                                                                    
        layer_neurons = { (layer_no, k): v for k, v in layer_neurons.items() }
        all_neurons.update(layer_neurons)
    
    all_neurons_sorted = {k: v for k, v in sorted(all_neurons.items(), key=lambda item: np.linalg.norm(item[1], ord=2, axis=0))}
    total_no_neurons = len(all_neurons_sorted) 

    neuron_pruning_scores = []

    for pruning_percent in K:

        new_model = load_model(model_load)
        new_weights = trained_model.get_weights().copy()

        prune_fraction = pruning_percent/100
        number_of_neurons_to_be_pruned = int(prune_fraction*total_no_neurons)
        neurons_to_be_pruned = {k: all_neurons_sorted[k] for k in list(all_neurons_sorted)[ : number_of_neurons_to_be_pruned]}     

        for k, v in neurons_to_be_pruned.items():
            new_weights[k[0]][:, k[1]] = 0

        for layer_no in range(total_no_layers - 1) :
            new_layer_weights = new_weights[layer_no].reshape(1, new_weights[layer_no].shape[0], new_weights[layer_no].shape[1])
            new_model.layers[layer_no].set_weights(new_layer_weights)
        
        new_score  = new_model.evaluate(X_test_norm, Y_test_onehot, verbose=0)
        test(new_model, X_test_norm, Y_test_onehot, verbose=True)
        neuron_pruning_scores.append(new_score[1])
    return neuron_pruning_scores

# nc  + std
def compute_gradient_Sw(X_train_norm, Y_train_onehot, trained_model):
    """
    Compute the gradient of the within-class scatter matrix (S_w) for each layer.
    """
    N = X_train_norm.shape[0] 
    output_dim = trained_model.layers[-1].output_shape[1]  
    
    gradient_Sw = {}
    all_weights = {}
    for layer_no, layer in enumerate(trained_model.layers[:-1]):
        if layer_no == 0:  # Only process the first layer
            layer_model = Model(inputs=trained_model.input, outputs=layer.output)
            H_layer_first = layer_model.predict(X_train_norm)
            break

    for layer_no, layer in enumerate(trained_model.layers[:-1]):

        layer_model = Model(inputs=trained_model.input, outputs=layer.output)
        H_layer = layer_model.predict(X_train_norm)
        print(f"Layer {layer_no} activation shape: {H_layer.shape}")

        mu_c = np.zeros((output_dim, H_layer.shape[1]))
        for c in range(output_dim):
            class_samples = H_layer[Y_train_onehot[:, c] == 1]
            mu_c[c] = np.mean(class_samples, axis=0)

        W = layer.get_weights()[0]  
        grad_Sw = np.zeros_like(W)

        for i in range(N):
            for c in range(output_dim):
                if Y_train_onehot[i, c] == 1:
                    if (layer_no==0):
                        h = X_train_norm[i]
                    else:
                 
                        h = H_layer_first[i]
             
                    diff = H_layer[i] - mu_c[c]
                    grad_Sw += 2 * np.outer(h, diff.T) 

        gradient_Sw[layer_no] = grad_Sw
    return gradient_Sw


def prune_weights_using_Sw(X_test_norm, Y_test_onehot, X_train_norm, Y_train_onehot, K, trained_model, model_load):
    """
    Perform weight pruning using the gradient of the within-class scatter matrix (S_w),
    and rank the weights based on Neural Collapse scores.
    """
    gradient_Sw = compute_gradient_Sw(X_train_norm, Y_train_onehot, trained_model)
    total_no_layers = len(trained_model.layers)
    weight_pruning_scores = []

    all_weights = {}
    all_ranking_scores = {}

    for layer_no in range(total_no_layers - 1):
        if layer_no not in gradient_Sw:
            print(f"Warning: No gradient found for layer {layer_no}. Skipping this layer.")
            continue

        grad_Sw = gradient_Sw[layer_no]
        print(f"Layer {layer_no}: Gradient shape: {grad_Sw.shape}")

        weight_importance = np.linalg.norm(grad_Sw, ord=2)
        
        W = trained_model.layers[layer_no].get_weights()[0] 
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                all_weights[(layer_no, i, j)] = W[i, j]
                all_ranking_scores[(layer_no, i, j)] = np.abs(grad_Sw[i, j])

    all_weights_sorted = {k: v for k, v in sorted(all_ranking_scores.items(), key=lambda item: item[1])}

    total_no_weights = len(all_weights_sorted)
    print(f"Total weights to consider: {total_no_weights}")

    for pruning_percent in K:
    
        new_model = load_model(model_load)
        new_weights = new_model.get_weights().copy()

        prune_fraction = pruning_percent / 100
        number_of_weights_to_be_pruned = int(prune_fraction * total_no_weights)
        weights_to_be_pruned = list(all_weights_sorted.keys())[:number_of_weights_to_be_pruned]

        for k in weights_to_be_pruned:
            layer_no, i, j = k
            new_weights[layer_no][i, j] = 0

        for layer_no in range(total_no_layers - 1):
            new_layer_weights = new_weights[layer_no].reshape(1, new_weights[layer_no].shape[0], new_weights[layer_no].shape[1])
            new_model.layers[layer_no].set_weights(new_layer_weights)

        new_score = new_model.evaluate(X_test_norm, Y_test_onehot, verbose=0)
        test(new_model, X_test_norm, Y_test_onehot, verbose=True)
        weight_pruning_scores.append((pruning_percent, new_score[1]))

    return weight_pruning_scores

def plots(standard_w, standard__neuron, nc, imbalance):
    plt.figure(figsize=(8, 4))
    plt.plot(pd.DataFrame(standard_w).set_index(pd.Series(K), drop=True), color='r', marker='o', markersize=3,linewidth=1)
    plt.plot(pd.DataFrame(standard__neuron).set_index(pd.Series(K), drop=True), color='g', marker='s', markersize=3,linewidth=1)
    plt.plot(pd.DataFrame(nc).set_index(pd.Series(K), drop=True), color='b', marker='^', markersize=3,linewidth=1)
    plt.title('Effect of Pruning on Accuracy (Balance = ' + imbalance + ')', fontsize=10)
    plt.ylabel('Accuracy', fontsize=8)
    plt.xlabel('Pruning Percentage', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=8)
    plt.ylim(0, 1)
    plt.legend(['Weigh Pruning', 'Neuron Pruning', 'NC Pruning'], loc='best', prop={'size': 8})
    plt.grid(color='gray', linestyle='-', linewidth=0.3)
    filename = f"testing_with_noise/{imbalance}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def noise(dataset, noise_factor=0.2):
    ((X_train, Y_train), (X_test, Y_test)) = dataset.load_data()
    
    # Reshape the data
    X_train_reshaped = X_train.reshape(len(X_train), -1)   
    X_test_reshaped = X_test.reshape(len(X_test), -1)

    # Normalize the images by dividing by 255
    X_train_norm = X_train_reshaped / 255            
    X_test_norm = X_test_reshaped / 255

    # Add Gaussian noise to the data
    noise_train = np.random.normal(0, noise_factor, X_train_norm.shape)
    noise_test = np.random.normal(0, noise_factor, X_test_norm.shape)
    
    X_train_noisy = X_train_norm + noise_train
    X_test_noisy = X_test_norm + noise_test

    # Ensure pixel values remain between 0 and 1
    X_train_noisy = np.clip(X_train_noisy, 0, 1)
    X_test_noisy = np.clip(X_test_noisy, 0, 1)

    # Get the number of features and classes
    n_features = X_train_noisy.shape[1]
    n_classes = 10
    Y_train_onehot = to_categorical(Y_train, num_classes=n_classes)
    Y_test_onehot = to_categorical(Y_test, num_classes=n_classes)

    return X_test_noisy, Y_test_onehot




