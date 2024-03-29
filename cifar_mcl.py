# -*- coding: utf-8 -*-

""" Convolutional network applied to CIFAR-10 dataset classification task.

References:
    Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.

Links:
    [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

# Data loading and preprocessing
from tflearn.datasets import cifar10
(X, Y), (X_test, Y_test) = cifar10.load_data('F:\\USC\\Research\\2017Summer\\mcl_10\\tf_learn')

# Load in MCL10 as training set
# Width and height of each image.
img_size = 32
num_classes = 10

# number of images in MCL10, which is total image divided by 10
num_per_class = 600

# Windows Path:
imgDatamcl_32 = np.fromfile('F:\\USC\\Research\\2017Summer\\mcl_10\\tf_learn\\MCL10_dat\\imgData_6000.dat', dtype=np.uint8)
labelDatamcl_32 = np.loadtxt('F:\\USC\\Research\\2017Summer\\mcl_10\\tf_learn\\MCL10_dat\\labelData_6000.txt', dtype=np.int64)

# OSX Path:
# imgDatamcl_32 = np.fromfile('/Users/erichsieh/Desktop/USC/2017_Summer/MCL10/tflearn/MCL-10/MCL10_dat/imgData_300.dat', dtype=np.uint8)
# labelDatamcl_32 = np.loadtxt('/Users/erichsieh/Desktop/USC/2017_Summer/MCL10/tflearn/MCL-10/MCL10_dat/labelData_300.txt', dtype=np.int64)

imgDatamcl_32 = imgDatamcl_32.reshape([int((imgDatamcl_32.shape)[0]/(img_size*img_size*3)), img_size, img_size, 3])
imgDatamcl_32 = imgDatamcl_32.astype(np.float64)/255.0

classIndices = np.arange(num_classes)*num_per_class
classIndices = classIndices.astype(np.int64)

images_test = np.zeros([int(round(imgDatamcl_32.shape[0])), imgDatamcl_32.shape[1], imgDatamcl_32.shape[2], imgDatamcl_32.shape[3]])
cls_test = np.zeros([int(round(imgDatamcl_32.shape[0]))])
cls_test = cls_test.astype(np.int32)

num_test = int(round(num_per_class))

classIndices = np.arange(num_classes)*num_per_class
test_indices = np.arange(num_classes)*num_test

for i in range(num_classes):
    images_test[test_indices[i]:test_indices[i]+num_test,:,:,:] = imgDatamcl_32[classIndices[i]:classIndices[i]+num_test,:,:,:]
    cls_test[test_indices[i]:test_indices[i]+num_test] = labelDatamcl_32[classIndices[i]:classIndices[i]+num_test]

X, Y = shuffle(X, Y)

# Get a subset from the cifar10 training set
X_subset = X[0:6000][:][:][:]
Y_subset = Y[0:6000]

# Set testing set as MCL-10
X_test = images_test
Y_test = cls_test

'''
for i in range(0,10):
    # Index of image, used to check whatever image you want
    image_index = i
    plt.title(Y_test[image_index])
    plt.imshow(X_test[image_index])
    plt.show()
exit()
'''

Y_subset = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
'''
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
'''
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X_subset, Y_subset, n_epoch=10, shuffle=True, validation_set=0.1,
          show_metric=True, batch_size=10, run_id='cifar10_cnn')

# Evaluate model
score = model.evaluate(X_test, Y_test)
print('Test accuarcy: %0.4f%%' % (score[0] * 100))
