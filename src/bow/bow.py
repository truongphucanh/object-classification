import pandas as pd
import numpy as np
import pickle
import cv2
import glob
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from scipy.cluster.vq import * 
from collections import Counter
from sklearn.model_selection import train_test_split

def computeDesCription(list_image,list_label):
    sift = cv2.xfeatures2d.SIFT_create()
    des_list = []
    for i  in range(0,len(list_image)):
        try:
            img  = cv2.resize(list_image[i],(96,96))
            detector = sift.detect(list_image[i],None)
            kpts, des = sift.compute(list_image[i], detector)
            if(len(des)<40):
                img  = cv2.resize(list_image[i],(145,145))
                detector = sift.detect(img,None)
                kpts, des = sift.compute(list_image[i], detector)
            if(des is not None):
                des_list.append((list_label[i], des[0:50]))
        except:
            break
    return des_list

def SplitData(strDirect, split = 0.2, trainDir, testDir):
    for folder in glob.glob(strDirect):
        drive, path = os.path.splitdrive(folder)
        path, labels = os.path.split(path)
        list_image = []
        list_label = []
        for filename in glob.glob(folder+'/*.*'):
            img = cv2.imread(filename)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            list_image.append(gray_image)
            list_label.append(labels)
        X_train, X_test, y_train, y_test = train_test_split(list_image, list_label, test_size=split, random_state=70)
        for x in range(len(X_train)):
            s = trainDir + y_train[x]
            if not os.path.exists(s):
                os.makedirs(s)
            cv2.imwrite(s + '/' + str(x) + '.jpg', X_train[x])
        for x in range(len(X_test)):
            s = testDir + y_test[x]
            if not os.path.exists(s):
                os.makedirs(s)
            cv2.imwrite(s + '/' + str(x) + '.jpg', X_test[x])

def loadAllImage(strDirect):
    list_image = []
    list_label = []
    for folder in glob.glob(strDirect): #assuming gif
        drive, path = os.path.splitdrive(folder)
        path, labels = os.path.split(path)
        for filename in glob.glob(folder+'/*.*'):
            img = cv2.imread(filename)
            list_image.append(img)
            list_label.append(labels)
    return list_image,list_label

if __name__ == '__main__':
    dataDir = '../../datasets/caltech_101/*'
    trainDir = '../../datasets/train/'
    testDir = '../../datasets/test/'
    SplitData(dataDir, split = 0.2, trainDir, testDir)
    list_image, list_label = loadAllImage(trainDir + '*')
    des_list = computeDesCription(list_image,list_label)
    
    with open('des_list.pkl', 'wb') as fid:
        pickle.dump(des_list, fid)
    list_label = [i[0] for i in des_list]
    with open('list_label.pkl', 'wb') as fid:
        pickle.dump(list_label, fid)
    
    descriptors = des_list[0][1]
    i = 0
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor)) 
    
    with open('descriptors.pkl', 'wb') as fid:
        pickle.dump(descriptors, fid)
    with open('descriptors.pkl', 'rb') as fid:
        descriptors = pickle.load(fid)
    with open('list_label.pkl', 'rb') as fid:
        list_label = pickle.load(fid)
    with open('des_list.pkl', 'rb') as fid:
        des_list = pickle.load(fid)
    
    k = 500
    voc, variance = kmeans(descriptors, k , 1)
    with open('voc.pkl', 'wb') as fid:
        pickle.dump(voc, fid)
    with open('variance.pkl', 'wb') as sid:
        pickle.dump(variance, sid)

    im_features = np.zeros((len(list_label), k)
    for i in range(len(list_label)):
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] += 1
    with open('im_features.pkl', 'wb') as sid:
        pickle.dump(im_features, sid)
        
    dic = np.array([],dtype=int)
    list_label = np.array(list_label)
    for i in range(len(list_label)):
        counter_feature = list (zip (np.where(im_features[i] != 0)[0],im_features[i][np.where(im_features[i] != 0)]))
        counter = Counter(dict(counter_feature))
        dic = np.append(dic,counter)
    with open('dic.pkl', 'wb') as fid:
        pickle.dump(dic, fid)
    with open('list_label.pkl', 'wb') as fid:
        pickle.dump(list_label, fid)
    with open('im_features.pkl', 'rb') as f:
        im_features = pickle.load(f)