import os
import numpy as np
import cv2
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'images')

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
x_train = []
y_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith('png') or file.endswith('jpg'):
			# get the path of each image
			path = os.path.join(root, file)
			label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
			#print(label, path)

			# create label as integer for data
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1

			id_ = label_ids[label]
			#print(label_ids)

			# open and convert to gray scale
			pil_image = Image.open(path).convert('L') 
			
			# resize image
			size = (550, 550)
			resized_image = pil_image.resize(size, Image.ANTIALIAS)

			# load image to numpy array
			image_array = np.array(pil_image, "uint8")

			# extract region of interest as data train
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors = 5)
			for (x, y, w, h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_train.append(id_)

#print(x_train)
#print(y_train)

# save label maping to file
with open("label_mapping", "wb") as f:
	pickle.dump(label_ids, f)

# train the model
recognizer.train(x_train, np.array(y_train))

# save the model weight after train
recognizer.save("trained_weight.yml")


