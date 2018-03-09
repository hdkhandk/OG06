
import pandas as pd
from imutils import paths
import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt
import os

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp([-x],dtype=np.float128))

def neural_net(x,y,syn0,syn1):
    for j in range(60000):
        l0 = x
        l1 = nonlin(np.dot(l0,syn0))
        l2 = nonlin(np.dot(l1,syn1))
        l2_error = y - l2
 
        if (j% 10000) == 0:
            print ("Error:", str(np.mean(np.abs(l2_error))))
 
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1.T)

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
        l1_delta = l1_error * nonlin(l1,deriv=True)
    
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)
    return syn0,syn1

#function to convert image to feature vector
def image_to_feature_vector(image,size=(32,32)):
    return cv2.resize(image,size).flatten()

#function to create histogram from image
def extract_color_histogram(image, bins=(8,8,8)):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,bins,[0,180,0,256,0,256])
    if imutils.is_cv2():
        hist = cv2.norm(hist)
    else:
        cv2.normalize(hist,hist)
    return hist.flatten()


rawImages = []
features = []
labels = []
#Image format is set to jpeg
imageformat=".jpg"
#path to class 1 images
path="/Users/aruranvijayaratnam/Desktop/ISIC-images/Benign"
imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
for el in imfilelist:
#        print (el)
        image1 = cv2.imread(el, cv2.IMREAD_COLOR)
        img1 = image_to_feature_vector(image1)
        hist1 = extract_color_histogram(image1)
        rawImages.append(img1)
        features.append(hist1)
        label1 = 0
        labels.append(label1)
#path to class 2 images
path="/Users/aruranvijayaratnam/Desktop/ISIC-images/Malignant"
imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
for el in imfilelist:
#        print (el)
        image2 = cv2.imread(el, cv2.IMREAD_COLOR)
        img2 = image_to_feature_vector(image2)
        hist2 = extract_color_histogram(image2)
        rawImages.append(img2)
        features.append(hist2)
        label2 = 1
        labels.append(label2)
#combine image sets of the two classes
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
np.random.seed(1)

# randomly initialize our weights with mean 0

syn0 = 2*np.random.random((88,512)) - 1
syn1 = 2*np.random.random((88,1)) - 1


a,b= neural_net(features,labels,syn1,syn1)