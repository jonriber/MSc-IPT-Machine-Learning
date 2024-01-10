# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5) #if not outputs AssertionError
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

#%%
# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"
#%%
try:
 # %tensorflow_version only exists in Colab.
 #%tensorflow_version 2.x
 IS_COLAB = True
except Exception:
 IS_COLAB = False
# TensorFlow ≥2.0 is required


assert tf.__version__ >= "2.0"
#%%
if not tf.test.is_gpu_available():
    print("No GPU was detected. CNNs can be very slow without a GPU.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")
        
#%%
# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "cnn"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
#%% Plot images
def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")
def plot_color_image(image):
    plt.imshow(image,interpolation="nearest")
    plt.axis("off")

#%%
#%%
"MNIST FASHION"
#10 classes, 70000 images, 28x28 pixel
#55000 samples – Train
#5000 samples – Validation
#10000 samples - Test
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]


#%%
#normalization
X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

np.shape(X_train)

#new dimension to have [samples, height, width, channels]
#channels = 1 (only gray scale)
#The network is expecting a Tensor
X_train = X_train[..., np.newaxis] # (...) indicates all dimensions
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]
#or using reshape X_train = X_train.reshape((55000, 28, 28, 1))
np.shape(X_train)

#%%
#%%
"BUILDING THE ARCHITECTURE"
#The partial function allows you to "freeze" some
# portion of a function's arguments and/or keyword
# arguments, creating a new function with fewer
# arguments than the original.
#This way, we don't need to repeat the parameters
#In this case keras.layers.Conv2D will have these
# default parameters
from functools import partial
DefaultConv2D = partial(keras.layers.Conv2D,kernel_size=3,activation='relu', padding="SAME")

#check all parameters (including default), for example
conv2d_layer = DefaultConv2D(filters=64,kernel_size=7, input_shape=[28, 28, 1])
config = conv2d_layer.get_config()
print(config)

model = keras.models.Sequential([
    DefaultConv2D(filters=64, kernel_size=7,
    input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=10,
    activation='softmax'),
])
#%%
model.summary()
keras.utils.plot_model(model, "MyCNN.png",
show_shapes=True)

#%%
#%%
"COMPILING, TRAINING and testing THE MODEL"
#Configure the learning process before training the model setting
#the Loss, the Optimizer and the Metrics
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=[X_valid, y_valid])
score = model.evaluate(X_test, y_test)
X_new = X_test[:10] # pretend we have new images
y_prob = model.predict(X_new) #returns probabilities
y_prob
y_predict = np.argmax(y_prob, axis=1)
y_predict[0:10]
y_test[0:10]
plot_image(X_test[1])
#%%
#Learning curves ---
import pandas as pd
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)