# This python script captures images from your webcam video stream.
# Detects and extracts all the faces from the image using Haarcascade classifier.
# Flattens the largest face image and stores it in a numpy array.
# Repeat it for multiple people to generate training data.


import cv2
import numpy as np

# Init webcam
cap = cv2.VideoCapture(0)

# import haarcascade classifier
face_capture =  cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0
face_data = []
dataset_path = './Training_data/'
file_name = input("enter the name of the person whose face is being captured:")
while True:
	#capture frame
	ret,frame = cap.read()

	if ret == False:
		#frame not captured
		continue

	# Convert captured image to grayscale image(occupies less space in memory)
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	
	# Detect faces
	face = face_capture.detectMultiScale(gray_frame,1.3,5)#scaling factor and number of neighbours
	# detectMultiScale returns a tuple containing coordinates height and width of the detected face(x,y,w,h)
	
	#sort the faces based on the area covered
	faces = sorted(face,key = lambda f:f[2]*f[3])
	
	# draw a rectangle around the largest face
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(0,200,200),2)

		#extract the region of interest
		extracted_face = gray_frame[y:y+h,x:x+w]
		extracted_face = cv2.resize(extracted_face,(100,100))


		skip+=1
		if skip%10 == 0:
			#store the data of every 10th frame
			face_data.append(extracted_face)
			print(len(face_data))
		
	#diplay the captured frame
	cv2.imshow("Video Frame",gray_frame)
		

	#stop capturing data when user presses 'q'
	key_press = cv2.waitKey(1) & 0xff
	if key_press == ord('q'):
		break

#convert the face data list into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#save the face data array in file system
np.save(dataset_path + file_name+'.npy',face_data)
print("face data saved at "+dataset_path + file_name+'.npy')
cap.release()
cv2.destroyAllWindows()