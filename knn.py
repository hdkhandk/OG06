#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:03:24 2018

@author: aruranvijayaratnam
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
import json
import pandas as pd
from imutils import paths
import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt
import os
import argparse

def image_to_feature_vector(image,size=(32,32)):
    return cv2.resize(image,size).flatten()

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
ap = argparse.ArgumentParser()
ap.add_argument("-k", "--neighbors", type=int, default=1,help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())
imageformat=".jpg"
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
        
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)

(trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)

print("[INFO] evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],n_jobs=args["jobs"])
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

#knn = cv2.ml.KNearest_create()
#knn.train(trainRI,trainRL)
#ret,result,neigbours,dist= knn.find_nearest(testRI,k=2)
#matches = result==testRL
#correct = np.count_nonzero(matches)
#accuracy = correct*100.0/result.size
#print (accuracy)
#pixels = image_to_feature_vector(image)
#hist = extract_color_histogram(image)
#rawImages.append(pixels)
#features.append(hist)
#rawImages = np.array(rawImages)
#features = np.array(features)
