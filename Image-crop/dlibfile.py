from collections import OrderedDict
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("right_eye", (36, 42)),
	("left_eye", (42, 48))
])
# construct the argument parser and parse the arguments
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/rizwan/Projects/cataract-eye-crop/shape_predictor_68_face_landmarks.dat")
# load the input image, resize it, and convert it to grayscale
image = cv2.imread("/home/rizwan/Projects/cataract-eye-crop/images/inputs/3333.png")
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
rects = detector(gray, 1)
# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the landmark (x, y)-coordinates to a NumPy array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	# loop over the face parts individually
	for (name, (i, j)) in FACIAL_LANDMARKS_IDXS.items():
		# clone the original image so we can draw on it, then
		# display the name of the face part on the image
		clone = image.copy()
		cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)

		# loop over the subset of facial landmarks, drawing the
		# specific face part
		for (x, y) in shape[i:j]:
			cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
		# extract the ROI of the face region as a separate image
			(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
			roi = image[y:y + h, x:x + w]
			roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
		# show the particular face part
		cv2.imshow("ROI", roi)
		cv2.imshow("Image", clone)
		img_name=name+".png"
		cv2.imwrite('/home/rizwan/Projects/cataract-eye-crop/images/output/'+img_name, roi)
		cv2.waitKey(1000)

	# visualize all facial landmarks with a transparent overlay
	output = face_utils.visualize_facial_landmarks(image, shape)
	cv2.imshow("Image", output)
	cv2.waitKey(0)