import numpy as np
import matplotlib.pyplot as plt
from myParser import *
from linear_classifier import LinearSVM
from vars import *
from preprocess import *


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# folders = ['CMale', 'CFemale', 'JMale', 'JFemale', 'KMale', 'KFemale']
folders = ['CMale_crop', 'CFemale_crop', 'JMale_crop', 'JFemale_crop', 'KMale_crop', 'KFemale_crop']
X_train, y_train, X_val, y_val, X_test, y_test = getTrainAndTestData(folders)

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
# mean_image = np.mean(X_train, axis=0)
# plt.figure(figsize=(4,4))
# plt.imshow(mean_image.reshape(IMAGE_SIZE).astype('uint8')) # visualize the mean image
# plt.title('Mean face in training set')
# plt.show()

# dataset = getSeperateCluster(folders)
# for k in dataset:
#   x = np.array(dataset[k])
#   mean_image = np.mean(x, axis=0)
#   name = 'Mean face (' + k + ')'
#   plt.imsave(name, mean_image.reshape(IMAGE_SIZE).astype('uint8'))


# second: subtract the mean image from train and test data
# X_train -= mean_image
# X_test -= mean_image

# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
print 'Training data new shape: ', X_train.shape
print 'Validation data new shape: ', X_val.shape
print 'Test data new shape: ', X_test.shape

# In the file linear_classifier.py, implement SGD in the function
# LinearClassifier.train() and then run it with the code below.
# svm = LinearSVM()
# loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=5e3,
#                       num_iters=1000, verbose=True)

# A useful debugging strategy is to plot the loss as a function of
# iteration number:
# plt.plot(loss_hist)
# plt.xlabel('Iteration number')
# plt.ylabel('Loss value')
# plt.show()

# tune hyperparameters
learningRates = [1e-7]
regularization = [5e3]
# iteration = [3000, 4000, 5000, 6000, 7000, 8000]
iteration = [6000]
bestParams = []
bestValAcc = 0
bestSvm = None

for eta in learningRates:
    for r in regularization:
        for t in iteration:
            svm = LinearSVM()
            svm.train(X_train, y_train, learning_rate=eta, reg=r, num_iters=t, verbose=True)
            y_train_pred = svm.predict(X_train)
            y_val_pred = svm.predict(X_val)
            trainAcc = np.mean(y_train == y_train_pred)
            valAcc = np.mean(y_val == y_val_pred)
            print 'iteration: %d train accuracy: %.4f val accuracy: %.4f' % (t, trainAcc, valAcc)
            if valAcc > bestValAcc:
                bestParams = [eta, r, t]
                bestValAcc = valAcc
                bestSvm = svm

print 'best validation accuracy achieved: %.4f' % bestValAcc
print bestParams

# Write the LinearSVM.predict function and evaluate the performance on both the
# training and validation set
y_train_pred = bestSvm.predict(X_train)
print 'training accuracy: %.4f' % np.mean(y_train == y_train_pred)

# Evaluate the svm on test set
y_test_pred = bestSvm.predict(X_test)
print 'test accuracy: %.4f' % np.mean(y_test == y_test_pred)

correntGender = 0
for i in range(len(y_test)):
    if abs(y_test_pred[i] - y_test[i]) % 2 == 0:
        correntGender += 1
print 'Gender accuracy: ', 1.0 * correntGender / len(y_test)

print y_test
print y_test_pred

# Visualize the learned weights for each class.
# Depending on your choice of learning rate and regularization strength, these may
# or may not be nice to look at.
w = bestSvm.W[:-1,:] # strip out the bias
print w.shape
w = w.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2], NUM_CLASS)
w_min, w_max = np.min(w), np.max(w)
classes = ['Chinese Male', 'Chinese Female', 'Japanese Male', 'Japanese Female', 'Korean Male', 'Korean Female']
for i in xrange(NUM_CLASS):
  plt.subplot(2, 5, i + 1)    
  # Rescale the weights to be between 0 and 255
  wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
  plt.imshow(wimg.astype('uint8'))
  plt.axis('off')
  plt.title(classes[i])