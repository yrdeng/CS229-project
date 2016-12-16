import numpy as np
from scipy.misc import *
from vars import *
from preprocess import *

def predict(W1, b1, W2, b2, X):
	z1 = X.dot(W1) + b1
	a1 = np.maximum(0, z1) # pass through ReLU activation function
	scores = a1.dot(W2) + b2
	y_pred = np.argmax(scores, axis=1)
	return scores, y_pred

W1 = np.load('W1.npy')
b1 = np.load('b1.npy')
W2 = np.load('W2.npy')
b2 = np.load('b2.npy')
mean = np.load('mean.npy')

while True:
	name = raw_input('Please enter RGB image file: ')
	if len(name) == 0:
		break
	img = imread(name)
	if len(img.shape) != 3 or img.shape[2] != 3:
		print "Invalid image type"
	else:
		img = imresize(img, IMAGE_SIZE)
		img = np.array([img])
		img = getGrayscale(img)
		img = np.reshape(img, (img.shape[0], -1))
		img -= mean
		score, label = predict(W1, b1, W2, b2, img)
		mapping = {0:'Chinese Male', 1:'Chinese Female', 2:'Japanese Male',
		3:'Japanese Female', 4:'Korean Male', 5:'Korean Female'}
		print 'The scores of CM / CF / JM / JF / KM / KF are:'
		print score
		print mapping[label[0]]