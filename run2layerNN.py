import numpy as np
import matplotlib.pyplot as plt
from myParser import *
from vars import *
from neural_net import TwoLayerNet
from vis_utils import visualize_grid
from preprocess import *


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

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


# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# As a sanity check, print out the shapes of the data
print 'Training data shape after reshape: ', X_train.shape

# second: subtract the mean image from train and test data
def showMeanImage(img):
    plt.figure(figsize=(4,4))
    # plt.imshow(img.reshape(IMAGE_SIZE).astype('uint8')) # visualize the mean image
    plt.imshow(img.reshape((IMAGE_SIZE[0], IMAGE_SIZE[1])), cmap='gray')
    plt.title('Mean face')
    plt.show()

mean_image = np.mean(X_train, axis=0)
mean_image = mean_image.astype(np.uint8)
mean_image = np.reshape(mean_image, (1, mean_image.shape[0]))
np.save('mean', mean_image)

# bias = min(min(mean_image))
# mean_image -= bias

# X_train -= mean_image
# X_val -= mean_image
# X_test -= mean_image
mean_image_new = np.mean(X_train, axis=0)
# showMeanImage(mean_image_new)

print 'Training data shape after preprocessing: ', X_train.shape

# input_size = np.prod(IMAGE_SIZE)
input_size = np.prod(X_train[0].shape)


# hidden_size = range(6, 18, 2)
# hidden_size = [i**2 for i in hidden_size]
hidden_size = [64]

# net = TwoLayerNet(input_size, hidden_size, num_classes)
# Train the network
# stats = net.train(X_train, y_train, X_train, y_train,
#             num_iters=2000, batch_size=20,
#             learning_rate=1e-4, learning_rate_decay=0.95,
#             reg=0.5, verbose=True)


# tune hyperparameters
# learningRates = [1.5e-4]
learningRates = [1.25e-4, 1.5e-4, 1.75e-4] # dont 1e-3
# regularization = [0.25]
regularization = [0.25, 0.5]
# iteration = [5500]
iteration = [3500, 4500, 5500] # >2500
# batch = [40, 50, 60, 70, 80]
batch = [50]
bestValAcc = 0
bestNN = None
bestTrain = None
bestParams = []

for hidden in hidden_size:
    for eta in learningRates:
        for r in regularization:
            for t in iteration:
                for b in batch:
                    net = TwoLayerNet(input_size, hidden, NUM_CLASS)
                    train = net.train(X_train, y_train, X_train, y_train,\
                        num_iters=t, batch_size=b, learning_rate=eta, learning_rate_decay=0.95, reg=r, verbose=False)
                    y_train_pred = net.predict(X_train)
                    y_val_pred = net.predict(X_val)
                    trainAcc = np.mean(y_train == y_train_pred)
                    valAcc = np.mean(y_val == y_val_pred)
                    print 'learning rate: %e reg: %.2f iteration: %d batch: %d train accuracy: %.4f val accuracy: %.4f'\
                    % (eta, r, t, b, trainAcc, valAcc)
                    if valAcc > bestValAcc:
                        bestValAcc = valAcc
                        bestNN = net
                        bestTrain = train
                        bestParams = [hidden, eta, r, t, b]

print 'best validation accuracy achieved: %.4f' % bestValAcc
print bestParams
f1 = open('./res.txt', 'w+')
f1.write('best validation accuracy achieved: %.4f' % bestValAcc)
f1.write('\n')
f1.write(str(bestParams))

# Predict on the validation set
acc = (bestNN.predict(X_train) == y_train).mean()
print 'Training set accuracy: ', acc

# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(bestTrain['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
# plt.imsave('loss', bestTrain['loss_history'])

plt.subplot(2, 1, 2)
plt.plot(bestTrain['train_acc_history'], label='train')
plt.plot(bestTrain['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()


# Visualize the weights of the network
def show_net_weights(net):
  W1 = net.params['W1']
  W1 = W1.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2], -1).transpose(3, 0, 1, 2)
  plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
  plt.gca().axis('off')
  plt.show()

show_net_weights(bestNN)

y_test_pred = bestNN.predict(X_test)
test_acc = (y_test_pred == y_test).mean()
print 'Test accuracy: ', test_acc

correntGender = 0
for i in range(len(y_test)):
    if abs(y_test_pred[i] - y_test[i]) % 2 == 0:
        correntGender += 1
print 'Gender accuracy: ', 1.0 * correntGender / len(y_test)

print y_test_pred
print y_test

np.save('W1', net.params['W1'])
np.save('b1', net.params['b1'])
np.save('W2', net.params['W2'])
np.save('b2', net.params['b2'])
