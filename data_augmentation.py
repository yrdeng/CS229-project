'''
This script helps augment the dataset by flipping each image horizontally and
adding or subtracting a random number.
'''

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.misc import *
import random
import pickle
import cv2


def loadImageFolders(ROOT):
	'''load image folders and store them in list'''
	folderList = [folder[0] for folder in os.walk(ROOT)]
	folderList.pop(0)
	return folderList

def augmentImages(folderNames, minval, maxval):
    '''
    Loop over all the directories and overwrite the image after face detection
    '''
    for folderName in folderNames:
        folders = loadImageFolders(folderName)
        for folder in folders:
            for file in os.listdir(folder):

				name = str(folder) + '/' + str(file)
				img = cv2.imread(name)
				filename, file_extension = os.path.splitext(name)
				try:
					# -- Flip the image horizontally
					flipped = np.fliplr(img)
					cv2.imwrite(filename + '_flipped.png', flipped)
					# -- Add a random number to the image
					added = (random.uniform(minval, maxval)* img).astype(np.uint8)
					brighter = cv2.add(img, added)
					cv2.imwrite(filename + '_brighter.png', brighter)
					# -- Subtract a random number from the image
					subtracted = (random.uniform(minval, maxval)* img).astype(np.uint8)
					darker = cv2.subtract(img, subtracted)
					cv2.imwrite(filename + '_darker.png', darker)
				except:
					print name


folderNames = ['JFemale','JMale', 'CFemale','CMale', 'KFemale','KMale']
minval = 0.2
maxval = 0.3
augmentImages(folderNames, minval, maxval);
