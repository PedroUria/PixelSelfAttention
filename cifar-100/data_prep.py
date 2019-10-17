import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.getcwd() + "/cifar-100-python/"

def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

train_dict = unpickle(DATA_DIR + "train")
test_dict = unpickle(DATA_DIR + "test")
label_names = unpickle(DATA_DIR + "meta")

img = train_dict[b"data"][5]
image = np.zeros((32, 32, 3), dtype=np.uint8)
image[..., 0] = np.reshape(img[:1024], (32, 32))  # Red channel
image[..., 1] = np.reshape(img[1024:2048], (32, 32))  # Green channel
image[..., 2] = np.reshape(img[2048:], (32, 32))  # Blue channel
plt.imsave("sample_image.png", image)
print(label_names[b"fine_label_names"][train_dict[b"fine_labels"][5]])
