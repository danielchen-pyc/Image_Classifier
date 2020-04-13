from keras.datasets import cifar10
import keras.utils as utils
from keras.models import load_model
import numpy as np
import os

(_, _), (test_images, test_labels) = cifar10.load_data()

test_images = test_images.astype('float32') / 255.0
test_labels = utils.to_categorical(test_labels)

labels_array = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Image_Classifier.h5'
model = load_model(path)

# results = model.evaluate(x=test_images, y=test_labels)
#
# print("Test loss: ", results[0])
# print("Test accuracy: ", results[1])

test_image_data = np.asarray([test_images[0]])

prediction = model.predict(x=test_image_data)
max_prediction = np.argmax(prediction[0])

print("Prediction: " + labels_array[max_prediction])
