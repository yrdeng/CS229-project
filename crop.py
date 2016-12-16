import numpy as np
import cv2


def cropFace(inputImage):
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	img = cv2.imread(inputImage)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	if len(faces) == 0:
		return None
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		x,y,w,h = face_cascade.detectMultiScale(gray, 1.3, 1)[0]
		img = img[y:y+h,x:x+w, :]
		return img