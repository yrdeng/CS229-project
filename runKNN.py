import numpy as np
import matplotlib.pyplot as plt
from myParser import *
from k_nearest_neighbor import KNearestNeighbor
from vars import *
from preprocess import *


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# folders = ['CMale', 'CFemale', 'JMale', 'JFemale', 'KMale', 'KFemale']
folders = ['CMale_crop', 'CFemale_crop', 'JMale_crop', 'JFemale_crop', 'KMale_crop', 'KFemale_crop']
# X_train, y_train, X_test, y_test = getTrainAndTestData(folders)
X_train, y_train, X_val, y_val, X_test, y_test = getTrainAndTestData(folders)

wholeSet = getSeperateCluster(folders)
for k in wholeSet:
    x = np.array(wholeSet[k])
    x = np.reshape(x, (x.shape[0], -1))
    mean = np.mean(x, axis=0)
    plt.imsave(str(k) + '.jpg', mean.reshape(IMAGE_SIZE).astype('uint8'))

print 'hao le'
# extract image features
# numHistBins = 1024
# X_train = extractFeature(X_train, numHistBins)
# X_val = extractFeature(X_val, numHistBins)
# X_test = extractFeature(X_test, numHistBins)

# X_train = getGrayscale(X_train)
# X_val = getGrayscale(X_val)
# X_test = getGrayscale(X_test)

# As a sanity check, we print out the size of the training and test data.
print 'Training data shape: ', X_train.shape
print 'Training labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape


# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# As a sanity check, print out the shapes of the data
print 'Training data shape after reshape: ', X_train.shape
print 'Validation data shape after reshape: ', X_val.shape
print 'Test data shape after reshape: ', X_test.shape

# Preprocessing: subtract the mean image
# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)
# mean_image = mean_image.astype(np.uint8)

def showMeanImage(img):
    plt.figure(figsize=(4,4))
    plt.imshow(img.reshape(IMAGE_SIZE).astype('uint8')) # visualize the mean image
    # plt.imshow(img.reshape(IMAGE_SIZE), cmap='gray')
    plt.title('Mean face')
    plt.show()

# showMeanImage(mean_image)

# second: subtract the mean image from train and test data
# mean_image = np.reshape(mean_image, (1, mean_image.shape[0]))
# X_train -= mean_image
# X_test -= mean_image
# mean_image_new = np.mean(X_train, axis=0)
# showMeanImage(mean_image_new)


# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
print 'Training data new shape: ', X_train.shape
print 'Validation data new shape: ', X_val.shape
print 'Test data new shape: ', X_test.shape

# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
# classifier = KNearestNeighbor()
# classifier.train(X_train, y_train)


# numK = [8,9,10,11,12,13,14,15,16]
numK = [12]
results = {}
bestValAcc = 0
bestK = None

for num in numK:
    knn = KNearestNeighbor()
    knn.train(X_train, y_train)
    y_train_pred = knn.predict(X_train, k=num)
    y_val_pred = knn.predict(X_val, k=num)
    trainAcc = np.mean(y_train == y_train_pred)
    valAcc = np.mean(y_val == y_val_pred)
    print 'k: %d train accuracy: %.4f val accuracy: %.4f' % (num, trainAcc, valAcc)
    if valAcc > bestValAcc:
        bestValAcc = valAcc
        bestK = num

print 'best validation accuracy achieved: %.4f, with best k : %d' % (bestValAcc, bestK)

# Based on the cross-validation results above, choose the best value for k,   
# retrain the classifier using all the training data, and test it on the test
# data.
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=bestK)

# Compute and display the accuracy
num_test = len(y_test)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)

print y_test
print y_test_pred
