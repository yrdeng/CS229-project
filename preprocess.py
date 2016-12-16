from features import *
import numpy as np

def extractFeature(dataset, numBins):
	feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=numBins)]
	dataset_feats = extract_features(dataset, feature_fns)
	return dataset_feats

def getGrayscale(dataset):
	res = []
	for x in dataset:
		nx = np.dot(x[...,:3], [0.299, 0.587, 0.144])
		res.append(nx)
	return np.array(res)

def getPCA(X):
	cov = np.dot(X.T, X) / X.shape[0]
	U, S, V = np.linalg.svd(cov)
	Xrot = np.dot(X, U)
	Xrot_reduced = np.dot(X, U[:,:100])
	return Xrot_reduced

def getWhitening(X):
	cov = np.dot(X.T, X) / X.shape[0]
	print 'done cov'
	U, S, V = np.linalg.svd(cov)
	print 'done svd'
	Xrot = np.dot(X, U)
	print 'done rot'
	Xwhite = Xrot / np.sqrt(S + 1e-3)
	return Xwhite
