# Anh Dam
# Junior Independent Study
# Last Modified Date: 27 April 2020
# CS200 - Algorithm Analysis

#import nececssary libraries
import cv2
import numpy as numpy
import matplotlib.pyplot as plt


#Load pre-trained Haar-Cascade class of frontal face and eye 
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

def face_detection(cascade1, cascade2, original_image, scaleFactor = 1.2):
    """
    Detects human faces with an input image and draw a rectangle around
    faces and eyes detected
    :param cascade1: pre-trained cascade classifier of frontal face
    :param cascade2: pre-trained cascade classifier of eyes
    :param original_image: an input image that is wanted to detect human faces
    :param scaleFactor = 1.2: parameter compensates a false perception
                in size where a face can be bigger than others in the picture
    :return: A copy of original image with human faces and eyes detected.
    """
    # make a copy of the original image to avoid changes in the original image
    image = original_image.copy()
    
    #convert the copied image to grayscale since OpenCV works with grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Applying the Haar classifiers to detect faces with detectMultiScale() function
    faces = cascade1.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors = 5)
    """
    detectMultiScale(): A function created by OpenCV to detect objects
    :param gray_image: a grayscale image of the copied image
    :param scaleFactor: parameter compensates a false perception
                in size where a face can be bigger than others in the picture
    :param minNeighbors: a detection algorithm uses a moving window to detect objects
    """
    print('# of faces detected: ', len(faces))

    #draw a rectangle around faces detected
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        # Applying the Haar classifiers to detect eyes in the image
        eyes = cascade2.detectMultiScale(roi_gray)
        #draw a rectangle around eyes detected
        for (x1, y1, w1, h1) in eyes:
            cv2.rectangle(roi_color, (x1, y1), (x1+w1, y1+h1), (255, 255, 255), 2)
        
    return image

#Function converts grayscale image to RGB color image
def RGB_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)