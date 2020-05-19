import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

# create model
recognizer = cv2.face.LBPHFaceRecognizer_create()
# load the weight
recognizer.read("trained_weight.yml")

# load label mapping
labels = {}
with open("label_mapping", "rb") as f:
	or_labels = pickle.load(f)
	labels = {v:k for k,v in or_labels.items()}
print(labels)
cap = cv2.VideoCapture(0)

while (True):
	#capture frame by frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors = 5)
	for (x, y, w, h) in faces:
		# print(x, y, w, h)
		roi_gray = gray[y:y+h, x:x+w]
		#roi_color = frame[y:y+h, x:x+w]
		#img_item = "1.png"
		#cv2.imwrite(img_item, roi_color)

		# use our custom model here
		id_, conf = recognizer.predict(roi_gray)
		
		# change this when have our model
		if conf >= 45:
			print(id_)

			# show the name above the rectange
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

		# draw a rectangle around the face
		color = (255, 0, 0) #BGR
		stroke = 2
		end_x = x + w
		end_y = y + h
		cv2.rectangle(frame, (x, y), (end_x, end_y), color, stroke)


	#display result frame
	cv2.imshow('frame', frame)

	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()      