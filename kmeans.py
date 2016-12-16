import numpy as np
import matplotlib.pyplot as plt
from myParser import getData
from vars import *

# folders = ['CMale', 'CFemale', 'JMale', 'JFemale', 'KMale', 'KFemale']
folders = ['CMale_crop', 'CFemale_crop', 'JMale_crop', 'JFemale_crop', 'KMale_crop', 'KFemale_crop']

def display(img, name):
	# plt.figure(figsize=(4,4))
	# plt.imshow(img.reshape(IMAGE_SIZE).astype('uint8')) # visualize the mean image
	plt.imsave(name, img.reshape(IMAGE_SIZE).astype('uint8'))
	# plt.title(name)
	# plt.show()

def getKmeansCentroids(dataset, maxIters, k):
	# initialize centroids
	centroids = [None] * k
	for i in range(k):
		centroids[i] = np.random.randint(255, size=IMAGE_SIZE)
	# iterate until convergence
	for t in range(maxIters):
		print 'Iteration: %d' % (t + 1)
		lastCentroids = list(centroids)
		clusters = [[] for i in range(k)]
		# step 1
		for x, y in dataset:
			dists = [(np.linalg.norm(x - centroids[i]), i) for i in range(k)]
			closest = min(dists)[1]
			clusters[closest] += [x]		
		# step 2
		for i in range(k):
			centroids[i] = np.mean(clusters[i], axis=0)
		# if np.sum(lastCentroids) == np.sum(centroids):
		# 	break
	return centroids


dataset = getData(folders)
centerImages = getKmeansCentroids(dataset, 50, NUM_CLASS)
for i, img in enumerate(centerImages):
	display(img, 'centroid ' + str(i))

yLabel = []
yCenter = []
for x, y in dataset:
	yLabel.append(y)
	dists = [(np.linalg.norm(x - centerImages[i]), i) for i in range(NUM_CLASS)]
	closest = min(dists)[1]
	yCenter.append(closest)

print yLabel
print yCenter