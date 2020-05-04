#!/usr/bin/env python
# coding: utf-8

# In[2]:


from imutils import face_utils
import imutils
import dlib
import cv2
import numpy as np
from collections import OrderedDict


# In[3]:


FACIAL_LANDMARKS_IDXS = OrderedDict([("mouth", (48, 68)),("right_eyebrow", (17, 22)),("left_eyebrow", (22, 27)),("right_eye", (36, 42)),("left_eye", (42, 48)),("nose", (27, 35)),("jaw", (0, 17))])


# In[4]:


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


# In[5]:


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),(163, 38, 32), (180, 42, 220)]

        # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
            # grab the (x, y)-coordinates associated with the
            # face landmark
            (j, k) = FACIAL_LANDMARKS_IDXS[name]
            pts = shape[j:k]

            # check if are supposed to draw the jawline
            if name == "jaw":
                # since the jawline is a non-enclosed facial region,
                # just draw lines between the (x, y)-coordinates
                for l in range(1, len(pts)):
                    ptA = tuple(pts[l - 1])
                    ptB = tuple(pts[l])
                    cv2.line(overlay, ptA, ptB, colors[i], 2)

                    # otherwise, compute the convex hull of the facial
                    # landmark coordinates points and display it
           # else:
            #    hull = cv2.convexHull(pts)
             #   cv2.drawContours(overlay, [hull], -1, colors[i], -1)

     # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)


     # return the output image
    return output


# In[6]:


p="shape_predictor_68_face_landmarks.dat"


# In[7]:


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


# In[ ]:





# In[8]:



cap = cv2.VideoCapture(0)


# In[9]:


while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
       
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", visualize_facial_landmarks(image,shape))
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




