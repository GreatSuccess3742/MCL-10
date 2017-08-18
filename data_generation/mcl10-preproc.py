import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
from scipy.ndimage import filters
from scipy.ndimage.interpolation import shift
from scipy.misc import imresize
import os
import random
from PIL import Image
import sys
#======================================= Global vars ============================================#
# Set to your corresponding path
# path = '/Users/erichsieh/Desktop/USC/2017_Summer/MCL10/mcl_10_eric_branch'
path = 'F:\\USC\\Research\\2017Summer\\mcl_10\\tf_learn\\data_generation\\'

sizes = [256]#[32, 64, 128, 256]

classes = os.listdir(path+'MCL-10_'+str(sizes[0])+'X'+str(sizes[0]))

# Maps class label in classes to actual label number in cifar-10
# classNumLabel = [8, 5, 4, 6, 2, 9, 1, 0, 7, 3]
classNumLabel = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

imgData_32 = np.empty((0,32*32*3), np.uint8)
# imgData_64 = np.empty((0,64*64*3), np.uint8)
# imgData_128 = np.empty((0,128*128*3), np.uint8)
# imgData_256 = np.empty((0,256*256*3), np.uint8)
labelData_32 = []
# labelData_64 = []
# labelData_128 = []
# labelData_256 = []
#================================================================================================#
def randomDoubles():
	first = random.randint(0,2)
	second = random.randint(0,2)
	while(first==1 and second==1):
		first = random.randint(0,2)
		second = random.randint(0,2)
	return first, second
#================================================================================================#
def getDatasetList():
	listOfImagePaths = []
	listOfclassImage = []
	for s in sizes:
		for c in classes:
			classPath = path+'/MCL-10_'+str(s)+'X'+str(s)+'/'+c
			listFiles = os.listdir(classPath)
			for f in listFiles:
				listOfImagePaths.append(classPath+'/'+f)
				listOfclassImage.append(c)
	return listOfImagePaths, listOfclassImage
#================================================================================================#
def imageRandomizer(image, whiteBG, i, j):
	randResized = imresize(image, random.uniform(0.8, 1),  interp='bilinear')
	obj_R, obj_C, dummyVar = randResized.shape
	rowOffset = i*(32-obj_R)/2
	colOffset = j*(32-obj_C)/2
	shiftImage = whiteBG.copy()
	shiftImage[int(rowOffset):int(rowOffset+obj_R), int(colOffset):int(colOffset+obj_C), :] = randResized.copy()
	flippedImage = whiteBG.copy()
	flippedImage[int(rowOffset):int(rowOffset+obj_R), int(colOffset):int(colOffset+obj_C), :] = np.flip(randResized, axis=1)
	return shiftImage, flippedImage
#================================================================================================#
def extractObject(image):
	gray = (0.2989*image[:,:,0] + 0.5870*image[:,:,1] + 0.1140*image[:,:,2]).astype(np.uint8)
	medImage = filters.median_filter(gray, size=[3, 3], mode='reflect')
	imgBW = (medImage<230).astype(np.uint8)
	i, j = np.where(imgBW==1)
	# bbBoxTopLeft = [i.min(), j.min()]
	imgObject = image[i.min():i.max()+1,j.min():j.max()+1,:]
	return imgObject
#================================================================================================#
def dataAugment(image, label):
	global imgData_32
	global shiftImage1
	# global imgData_64
	# global imgData_128
	# global imgData_256

	global labelData_32
	# global labelData_64
	# global labelData_128
	# global labelData_256
	#---------translations-----------#
	imgObject = extractObject(image)
	if(imgObject.shape[0]>imgObject.shape[1]):
		imgObject = imresize(imgObject, [28, int(imgObject.shape[1]*28.0/imgObject.shape[0]), 3],  interp='bilinear')
	else:
		imgObject = imresize(imgObject, [int(imgObject.shape[0]*28.0/imgObject.shape[1]), 28, 3],  interp='bilinear')
	# imgObject = imresize(imgObject, 0.125,  interp='bilinear')
	obj_R, obj_C, dummyVar = imgObject.shape
	whiteBG = np.ones((32, 32, 3), dtype=np.uint8)*255

	# i, j = randomDoubles()
	possibleRowOffset = [0, (32-obj_R)/2, (32-obj_R)]
	possibleColOffset = [0, (32-obj_C)/2, (32-obj_C)]

	numTx = 2
	txPos = np.random.choice(np.append(np.arange(0,4),np.arange(5,9)), numTx, replace=False) #Generates two random numbers witout replacement in [1,10]
	# Center position is always present
	rowOffset = [(32-obj_R)/2]
	colOffset = [(32-obj_C)/2]

	for i in txPos:
		rowOffset.append(possibleRowOffset[int(i/3)])
		colOffset.append(possibleColOffset[i%3])
	# fig1 = plt
	txPos = np.append(4, txPos) # 4 is the index number for center which is not a tranlated verus
	for i in range(len(rowOffset)):
		# Translate image by offsets
		# fig1.subplot(3, 3, i*3+j+1)
		shiftImage = whiteBG.copy()

		shiftImage[int(rowOffset[i]):int(rowOffset[i]+obj_R), int(colOffset[i]):int(colOffset[i]+obj_C), :] = imgObject.copy()
		# shiftImage = shift(image, (rowOffset[i], colOffset[j], 0), cval=255)
		# shiftImageTrsp = np.transpose(shiftImage, (2, 0, 1))
		shiftFlat = np.array([shiftImage.flatten()])
		
		# Flip translatted image
		flippedImage = whiteBG.copy()
		flippedImage[int(rowOffset[i]):int(rowOffset[i]+obj_R), int(colOffset[i]):int(colOffset[i]+obj_C), :] = np.flip(imgObject, axis=1)
		# flippedImageTrsp = np.transpose(flippedImage, (2, 0, 1))
		flipFlat = np.array([flippedImage.flatten()])
		
		# # Flip translatted and randomly resized image
		randResized, randResizedFlip = imageRandomizer(imgObject, whiteBG, int(txPos[i]/3), txPos[i]%3)
		
		# # randResizedTrsp = np.transpose(randResized, (2, 0, 1))
		randResizedFlat = np.array([randResized.flatten()])

		# # randResizedFlipTrsp = np.transpose(randResizedFlip, (2, 0, 1))
		randResizedFlipFlat = np.array([randResizedFlip.flatten()])

		imgData_32 = np.append(imgData_32, shiftFlat, axis=0)
		imgData_32 = np.append(imgData_32, flipFlat, axis=0)
		imgData_32 = np.append(imgData_32, randResizedFlat, axis=0)
		imgData_32 = np.append(imgData_32, randResizedFlipFlat, axis=0)

		classIndex = classes.index(label)
		# print label, classIndex, classNumLabel[classIndex]
		labelData_32.append(classNumLabel[classIndex])
		labelData_32.append(classNumLabel[classIndex])
		labelData_32.append(classNumLabel[classIndex])
		labelData_32.append(classNumLabel[classIndex])
#================================================================================================#
def saveData():
	os.chdir(path)
	imgData_32.tofile('GreyBG_300.dat', format='%u')
	labelData32 = np.asarray(labelData_32)
	np.savetxt('GreyBG_label_300.txt', labelData32, fmt='%s')

#================================================================================================#
listOfImagePaths, listOfclassImage = getDatasetList()
print(len(listOfImagePaths))
print(len(listOfclassImage))
exit()

for i in range(len(listOfImagePaths)): #length of your filename list
	image = np.asarray(Image.open(listOfImagePaths[i]))
	dataAugment(image, listOfclassImage[i])
	print ("Data augmentation...", str((i+1)*100.0/len(listOfImagePaths))+"% completed")

print ("Writing data to file...")
saveData()
#================================================================================================#