#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from imutils import face_utils
import dlib
import cv2





p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


 
image= cv2.imread("Enter path of input image")
# Converting the image to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        

rects = detector(gray, 0)
# For each detected face, find the landmark.
for (i, rect) in enumerate(rects):
     # Make the prediction and transfom it to numpy array
     shape = predictor(gray, rect)
     shape = face_utils.shape_to_np(shape)
    
     # Draw on our image, all the finded cordinate points (x,y) 
     for (x, y) in shape:
         cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
#save image
cv2.imwrite("Enter path where you want to save",image)

# Show the image
cv2.imshow("Output", image)
    







