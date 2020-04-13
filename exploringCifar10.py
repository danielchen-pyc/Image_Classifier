from keras.datasets import cifar10
import matplotlib.pyplot as plt
import keras.utils as utils
import numpy as np


def reshape_image(input_image_arrays):
    output_array = []
    for image_array in input_image_arrays:
        output_array.append(image_array.reshape(-1))
    return np.asarray(output_array)


(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
first_image = train_images[0]

# plt.imshow(train_images[0])
# plt.show()

labels_array = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_images = train_images.astype('float32')
train_images = train_images / 255.0
test_images = test_images.astype('float32')
test_images = test_images / 255.0

# print(test_images[0])
