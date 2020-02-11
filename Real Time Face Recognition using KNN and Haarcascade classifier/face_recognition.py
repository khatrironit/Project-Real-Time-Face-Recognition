# Recognise faces using KNN classification algorithm.

# 1.Load the training data into numpy arrays:
		# X-values are stored in numpy arrays.
		# Y-values we need to assign for each person.
# 2.Read a video stream using opencv.
# 3.Extract faces out of it.
# 4.Use the KNN algorithm to make prediction on the faces(int).
# 5.Map the predicted value to the name of the person.
# 6.Display the predictions on the screen - bounding box and name of the person.

import numpy as np
import cv2
import os
import math


##################### KNN Algorithm ###########################

# claculate the Euclidean distance between two points.
def Euclid_distance(x1,x2):
	return math.sqrt(((x1-x2)**2).sum())

# returns the max frequency label in nearest neighbours
def KNN_classifier(x_test,X,Y,k_size = 10):
	distance = []
	for i in range(X.shape[0]):
		d = Euclid_distance(x_test,X[i])
		distance.append((d,Y[i]))
	distance.sort()
	distance = distance[:k_size]

	distance = np.array(distance)
	k_nearest =  np.unique(distance[:,1],return_counts = True)
	index = k_nearest[1].argmax()
	return k_nearest[0][index]

###############################################################

data_path = './Training_data/'

face_data = [] 			# to load training data
label = []		   		# to assign labels to training data

person_id = 0		    # for assigning labels to a given file
names = {} 				# mapping between person names and labels

# Step 1.Data preparation
for fx in os.listdir(data_path):
	if fx.endswith('.npy'):
		names[person_id] = fx[:-4]				 # mapping peron id to name
		print("loaded=>" + fx)
		
		data_item = np.load(data_path + fx)		 # returns a numpy arrray containing the frame training data in fx
		print(data_item.shape)					 
		face_data.append(data_item)				 # face_data is a list of numpy arrays of frame training data

		# Create labels for the person
		target = person_id * np.ones((data_item.shape[0],))
		person_id+=1
		label.append(target)					 # label is a list containing numpy arrays of person ids



face_dataset = np.concatenate(face_data,axis = 0)	# final training set X
label_dataset = np.concatenate(label,axis = 0)		# final training set Y




# Step 2. Read video stream uing opencv

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


while True:
	ret,frame = cap.read()

	if ret == False	:							# frame not captured
		continue

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray_frame,1.3,5)		#(frame,scaling factor,number of neighbours)

	faces = sorted(faces,key = lambda f:f[2] * f[3])

	for face in faces[-1:]:
		(x,y,w,h) = face

		# Step 3.Extract the region of interest.
		required_region = gray_frame[y:y+h,x:x+w]
		required_region = cv2.resize(required_region,(100,100))

		# step 4.Use KNN algorithm to make prediction.
		output = KNN_classifier(required_region.flatten(),face_dataset,label_dataset)

		# step 5.Map the predicted value to the name of the person.
		output_name = names[output]

		# step 6.Display bounding box and name on the screen.
		cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(0,200,200),2)
		cv2.putText(gray_frame,output_name,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,250,250),2)		#(frame,text,location,font,fontScale,color,width)

	cv2.imshow("Video Frame",gray_frame)

	
	keypress = cv2.waitKey(1) & 0xff
	if(keypress == ord('q')):
		break



cap.release()
cv2.destroyAllWindows()


