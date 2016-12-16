import numpy as np
import matplotlib.pyplot as plt
import os, random
from scipy.misc import *
from crop import *
from vars import *


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def loadImageFolders(ROOT):
	'''load image folders and store them in list'''
	folderList = [folder[0] for folder in os.walk(ROOT)]
	folderList.pop(0)
	return str(ROOT)[0:2], folderList

def loadImages(folder, crop=False):
	'''load image in folder, return a tuple of '''
	'''(image as a 3-dimensional numpy array, label as a string)'''
	mylist = []
	for file in os.listdir(folder):
		name = str(folder) + '/' + str(file)
		img = imread(name)
		if len(img.shape) == 3 and img.shape[2] == 3:
			if crop:
				img = cropFace(file)
			img = imresize(img, IMAGE_SIZE)			
			mylist.append(img)
		# else:
		# 	print img.shape
	return mylist

def getTrainAndTestData(folderNames, valProtion=0.1, testProtion=0.1):
	dataset = []
	# mapping = {'C': 0, 'J': 1, 'K': 2}
	mapping = {'CM': 0, 'CF': 1, 'JM': 2, 'JF': 3, 'KM': 4, 'KF': 5}

	for folderName in folderNames:
		label, folders = loadImageFolders(folderName)
		label = mapping[label]
		for folder in folders:
			imgs = loadImages(folder)
			for img in imgs:
				dataset.append((img, label))

	random.shuffle(dataset)
	X = [x for x, y in dataset]
	Y = [y for x, y in dataset]
	X, Y = np.array(X), np.array(Y)

	breakPoint1 = int(len(X) * (1 - (valProtion + testProtion)))
	breakPoint2 = int(len(X) * (1 - testProtion))
	Xtrain, Ytrain = X[: breakPoint1], Y[: breakPoint1]
	Xval, Yval = X[breakPoint1: breakPoint2], Y[breakPoint1: breakPoint2]
	Xtest, Ytest = X[breakPoint2:], Y[breakPoint2:]
	return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest


def getSeperateCluster(folderNames):
	dataset = {'Chinese Male': [], 'Chinese Female': [], 'Japanese Male': [], 'Japanese Female': [], 'Korean Male': [], 'Korean Female': []}
	mapping = {'CM': 'Chinese Male', 'CF': 'Chinese Female', 'JM': 'Japanese Male', 'JF': 'Japanese Female', 'KM': 'Korean Male', 'KF': 'Korean Female'}

	for folderName in folderNames:
		label, folders = loadImageFolders(folderName)
		for folder in folders:
			imgs = loadImages(folder)
			for img in imgs:
				dataset[mapping[label]].append(img)
	return dataset


def getData(folderNames):
	dataset = []
	mapping = {'CM': 0, 'CF': 1, 'JM': 2, 'JF': 3, 'KM': 4, 'KF': 5}
	for folderName in folderNames:
		label, folders = loadImageFolders(folderName)
		label = mapping[label]
		for folder in folders:
			imgs = loadImages(folder)
			for img in imgs:
				dataset.append((img, label))
	return dataset