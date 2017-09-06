
"""
Created on Thu Aug 31 10:03:19 2017

@author: mclserver16
"""

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
from sklearn.feature_extraction import image

# Inputpatch = np.load('F:\\USC\\Research\\2017Summer\\mcl_10\\tf_learn\\pca\\Ye\\batches.npy')
Inputpatch = np.load('/Users/erichsieh/Desktop/USC/2017_Summer/MCL10/tflearn/MCL-10/pca/Ye/batches.npy')

compo = 32
patch_size = 8
original_image_size = 32

pca = PCA(n_components=compo)
X_project = pca.fit_transform(Inputpatch)

#aug & ReLU
X_project_neg = X_project * (-1)
X_project_relu = np.maximum(X_project, 0)
X_project_neg_relu = np.maximum(X_project_neg, 0)
X_project_aug = np.hstack((X_project, X_project_neg))
X_project_aug_relu = np.hstack((X_project_relu, X_project_neg_relu))
#inverse the aug & relu
X_project_inv = X_project_relu - X_project_neg_relu
#inverse PCA
Y = pca.inverse_transform(X_project_inv)

#MSE
mse = mean_squared_error(Inputpatch, Y)
print(mse)

#input a new image and test

X_test=Image.open('/Users/erichsieh/Desktop/USC/2017_Summer/MCL10/tflearn/MCL-10/pca/Ye/dog_002.jpg')
X_test = np.asarray(X_test)

plt.figure()
plt.imshow(X_test, cmap='gray')
plt.show()

#X_noise = X_test + 0.6 * X_test.std() * np.random.random(X_test.shape)
#
#plt.figure()
#plt.imshow(X_noise, cmap='gray')
#plt.show()
#
#X_test_patches = image.extract_patches_2d(X_noise, (8, 8))

X_test_patches = image.extract_patches_2d(X_test, (8, 8))

# X_test_patches_resize = np.reshape(X_test_patches, ((180-patch_size + 1)**2, patch_size**2))
X_test_patches_resize = np.reshape(X_test_patches, ((original_image_size - patch_size + 1)**2, patch_size**2*3))


#print(X_test_patches_resize.shape)
#reconstructed = image.reconstruct_from_patches_2d(patches, (180,180))
#np.testing.assert_array_equal(X_test, reconstructed)


X_project_test = pca.transform(X_test_patches_resize)
Y_test_patches = pca.inverse_transform(X_project_test)
Y_test_patches_resize = np.reshape(Y_test_patches, ((original_image_size - patch_size + 1)**2, patch_size, patch_size, 3))
Y_test = image.reconstruct_from_patches_2d(Y_test_patches_resize, (original_image_size,original_image_size,3))


plt.figure()
plt.imshow(Y_test, cmap='gray')
plt.show()

mse_test = mean_squared_error(X_test[:,:,0], Y_test[:,:,0])
#mse_noise_vs_recons = mean_squared_error(X_noise, X_test)
print('R_channel MSE: ',mse_test)

#PSNR
PIXEL_MAX = 255
PSNR_test = 20 * math.log10(PIXEL_MAX / math.sqrt(mse_test))
#PSNR_noise_vs_recons = 20 * math.log10(PIXEL_MAX / math.sqrt(mse_noise_vs_recons))
print (PSNR_test)
#print PSNR_noise_vs_recons
