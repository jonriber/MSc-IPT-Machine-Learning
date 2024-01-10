# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5) #if not outputs AssertionError
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
import tensorflow as tf
from tensorflow import keras

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
#"MNIST FASHION"
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
