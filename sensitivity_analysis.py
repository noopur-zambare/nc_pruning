import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(1000, activation='relu'),
    Dense(200, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train[:1000], y_train[:1000], epochs=1, batch_size=64, verbose=1)

def f(x):
    return model.predict(x[np.newaxis, ...]).flatten()

def sensitivity(f, x, sigma=0.1):
    f_x = f(x)
    epsilon = np.random.normal(0, sigma, x.shape)
    x_noisy = x + epsilon
    f_x_noisy = f(x_noisy)
    norm_x = np.linalg.norm(x)
    sensitivity_value = np.abs(f_x - f_x_noisy) / norm_x
    return sensitivity_value

x_sample = x_test[0]
sensitivity_value = sensitivity(f, x_sample, sigma=0.1)

print(f"Sensitivity: {sensitivity_value}")
