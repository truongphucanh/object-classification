
import pandas as pd
import numpy as np
import pickle
import cv2
import glob
import os
import math
import matplotlib.pyplot as plt
from scipy.cluster.vq import * 
from collections import Counter

def loadDataTest(strDirect):
    list_image = []
    list_label = []
    for folder in glob.glob(strDirect):
        drive, path = os.path.splitdrive(folder)
        path, labels = os.path.split(path)
        for filename in glob.glob(folder+'/*.*'):
            img = cv2.imread(filename)
            resize_image = cv2.resize(img, (128, 128)) 
            list_image.append(resize_image)
            list_label.append(labels)
    return list_image,list_label

def LoadBagOfWord(invertIndex, voc, label):
    with open(invertIndex, 'rb') as f:
        invertIndex = pickle.load(f)
    with open(voc, 'rb') as f:
        features = pickle.load(f)
    with open(label,'rb') as f:
            list_label = pickle.load(f)
    return invertIndex, features, list_label

def Knn_Euclidean_distance(inputWord, centroid, lst_label, numberK):
    len_word = centroid.shape[0] -1
    v_List=[]
    i_List=[]
    for i in range(len(list_label[0:])):
        similarity = 0
        for j in range(499):
            similarity = similarity + math.pow(inputWord[j]-centroid[i][j], 2)
        v_List.insert(0 , similarity)
        i_List.insert(0, lst_label[i])
        v_List, i_List = insertionSort(v_List,i_List, numberK = numberK)
    return v_List,i_List

def insertionSort(alist,ilist,numberK = 99):
    for index in range(1,len(alist)):
        currentvalue = alist[index]
        icurrentvalue = ilist[index]
        position = index
        while position>0 and alist[position-1]>currentvalue:
            alist[position]=alist[position-1]
            ilist[position] = ilist[position - 1]
            position = position-1
        alist[position]= currentvalue
        ilist[position]= icurrentvalue
    return alist[0:numberK],ilist[0:numberK]

def getWord(Drtimage):
    img = cv2.imread(Drtimage)
    sift = cv2.xfeatures2d.SIFT_create()
    detector = sift.detect(img,None)
    kpts, des = sift.compute(img, detector)
    x = np.zeros((1, 500))
    words, distance = vq(des,features)
    for w in words:
        x[0][w] += 1
    x = x[0]
    return x

def Result(strDirect,Dict,list_label, rank=1):
    list_result = []
    list_label_text_input = []
    list_value = []
    i = 0
    for folder in glob.glob(strDirect):
        drive, path = os.path.splitdrive(folder)
        path, labels = os.path.split(path)
        print (folder)
        for filename in glob.glob(folder+'/*.*'):
            image = getWord(filename)
            value,image = Knn_Euclidean_distance(image,invertIndex,list_label,rank)
            list_result.append(image)
            list_value.append(value)
            list_label_text_input.append(labels)
            print(i)
            i = i+1
    return list_result,list_label_text_input,list_value

if __name__ == '__main__':
    invertIndex , features, list_label = LoadBagOfWord('dic.pkl', 'voc.pkl', 'list_label.pkl')
    list_result,list_label_text_input,list_value = Result('D:\\\\datasetVuong\\\\Newfolder\\\\accordion\\*',invertIndex,list_label,15)
