from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm

from keras.datasets import cifar10
import keras.utils as utils

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

train_labels = utils.to_categorical(train_labels)
test_labels = utils.to_categorical(test_labels)

abels_array = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model = Sequential()

# Add the first convolution to output a feature map
# filters: output 32 features
# kernel_size: 3x3 kernel or filter matrix used to calculate output features
# input_shape: each image is 32x32x3
# activation: relu activation for each of the operations as it produces the best results
# padding: 'same' adds padding to the input image to make sure that the output feature map is the same size as the input
# kernel_constraint: maxnorm normalizes the values in the kernel to make sure that the max value is 3
conv_layer = Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), \
                    activation='relu', padding='same', kernel_constraint=maxnorm(3))
model.add(conv_layer)

# Add the max pool layer to decrease the image size from 32x32 to 16x16
# pool_size: finds the max value in each 2x2 section of the input
max_pool_layer = MaxPooling2D(pool_size=(2, 2))
model.add(max_pool_layer)

# Flatten layer converts a matrix into a 1 dimensional array
model.add(Flatten())

# First dense layer to create the actual prediction network
# units: 512 neurons at this layer, increase for greater accuracy, decrease for faster train speed
# activation: relu because it works so well
# kernel_constraint: see above
dense_layer = Dense(units=512, activation='relu', kernel_constraint=maxnorm(3))
model.add(dense_layer)

# Dropout layer to ignore some neurons during training which improves model reliability
# rate: 0.5 means half neurons dropped
dropout_layer = Dropout(rate=0.5)
model.add(dropout_layer)

# Final dense layer used to produce output for each of the 10 categories
# units: 10 categories so 10 output units
# activation: softmax because we are calculating probabilities for each of the 10 categories (not as clear as 0 or 1)
dense_layer_2 = Dense(units=10, activation='softmax')
model.add(dense_layer_2)


model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_images, y=train_labels, epochs=1, batch_size=32)
model.save(filepath='Image_Classifier.h5')
# model.evaluate()
# model.predict()
