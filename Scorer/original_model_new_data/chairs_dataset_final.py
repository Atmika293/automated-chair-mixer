# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:28:42 2020

@author: JANAKI
modified by Samuel Hatin
"""

import cv2
import os
import numpy as np

'''
load chair dataset. Dimension refers to the target dimension of the output image, used to save up memory.
The images are originally 224 x 224.

There are opportunities to improve the dataset by performing image operations to augment the dataset and generating
more negative samples based on the given meshes.
'''

def load(dimension):

    imagesTop = []
    imagesLeftSide = []
    imagesRightSide = []
    imagesLeftBack = []
    imagesLeftFront = []
    imagesRightFront = []
    imagesFront = []
    isPositive = False
    fileids = []
    ls = 0

    for id, folder in enumerate(["final-data/setC/"]):
        isPositive = not isPositive

        image_files = os.listdir(folder)
        length = len(image_files) // 7
        ls += length
        
        fileset = set()
        
        for filename in os.listdir(folder):
            view = filename.split(".")[0]

            file_id = ""
            for char in view:
                if ord(char) < 97 or ord(char) > 122:
                    file_id += char
            fileset.add(file_id)
 
        for file_id in fileset:
            fileids.append(file_id)
            for type_id in range(0, 7):
                typename = ""
                if type_id == 0:
                    typename = "leftfront"
                elif type_id == 1:
                    typename = "front"
                elif type_id == 2:
                    typename = "rightfront"
                elif type_id == 3:
                    typename = "leftside"
                elif type_id == 4:
                    typename = "rightside"
                elif type_id == 5:
                    typename = "leftback"
                elif type_id == 6:
                    typename = "top"
                    
                filename = typename + file_id + ".png"
                print(filename)
                img = cv2.imread(folder+filename)

                img = cv2.resize(img, dsize=(dimension, dimension), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.nan_to_num(img)

                if img is not None:
                    if typename == "leftfront":
                        imagesLeftFront.append(1. - img / 255.)
                    elif typename == "front":
                        imagesFront.append(1. - img / 255.)            
                    elif typename == "rightfront":
                        imagesRightFront.append(1. - img / 255.)
                    elif typename == "leftside":
                        imagesLeftSide.append(1. - img / 255.)
                    elif typename == "rightside":
                        imagesRightSide.append(1. - img / 255.)
                    elif typename == "leftback":
                        imagesLeftBack.append(1. - img / 255.)
                    elif typename == "top":
                        imagesTop.append(1. - img / 255.)             
                                   
        if isPositive:
            y_vec_leftfront = np.ones((length), dtype=np.int)
            y_vec_front = np.ones((length), dtype=np.int)
            y_vec_rightfront = np.ones((length), dtype=np.int)
            y_vec_rightside = np.ones((length), dtype=np.int)
            y_vec_leftback = np.ones((length), dtype=np.int)
            y_vec_leftside = np.ones((length), dtype=np.int)
            y_vec_top = np.ones((length), dtype=np.int)
        else:
            y_vec_leftfront = np.append(y_vec_leftfront, np.zeros((length), dtype=np.int), axis=0 )
            y_vec_front = np.append(y_vec_front, np.zeros((length), dtype=np.int), axis=0 )
            y_vec_rightfront = np.append(y_vec_rightfront, np.zeros((length), dtype=np.int), axis=0 )
            y_vec_rightside = np.append(y_vec_rightside, np.zeros((length), dtype=np.int), axis=0 )
            y_vec_leftback = np.append(y_vec_leftback, np.zeros((length), dtype=np.int), axis=0 )
            y_vec_leftside = np.append(y_vec_leftside, np.zeros((length), dtype=np.int), axis=0 )
            y_vec_top = np.append(y_vec_top, np.zeros((length), dtype=np.int), axis=0 )
  
            
    imagesTop = np.array(imagesTop)
    imagesFront = np.array(imagesFront)
    imagesLeftSide = np.array(imagesLeftSide)
    imagesLeftFront = np.array(imagesLeftFront)
    imagesRightSide = np.array(imagesRightSide)
    imagesRightFront = np.array(imagesRightFront)
    imagesLeftBack = np.array(imagesLeftBack)

    #flatten the images
    imagesTop = np.reshape(imagesTop, (ls, dimension * dimension))
    imagesFront = np.reshape(imagesFront, (ls, dimension * dimension))
    imagesLeftFront = np.reshape(imagesLeftFront, (ls, dimension * dimension))
    imagesRightFront = np.reshape(imagesRightFront, (ls, dimension * dimension))
    imagesLeftSide = np.reshape(imagesLeftSide, (ls, dimension * dimension))
    imagesRightSide = np.reshape(imagesRightSide, (ls, dimension * dimension))
    imagesLeftBack = np.reshape(imagesLeftBack, (ls, dimension * dimension))

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(imagesTop)
    np.random.seed(seed)
    np.random.shuffle(imagesFront)
    np.random.seed(seed)
    np.random.shuffle(imagesLeftSide)
    np.random.seed(seed)
    np.random.shuffle(imagesRightSide)
    np.random.seed(seed)
    np.random.shuffle(imagesLeftFront)
    np.random.seed(seed)
    np.random.shuffle(imagesRightFront)
    np.random.seed(seed)
    np.random.shuffle(imagesLeftBack)

    np.random.seed(seed)
    np.random.shuffle(y_vec_top)
    np.random.seed(seed)
    np.random.shuffle(y_vec_front)
    np.random.seed(seed)
    np.random.shuffle(y_vec_leftside)
    np.random.seed(seed)
    np.random.shuffle(y_vec_rightside)
    np.random.seed(seed)
    np.random.shuffle(y_vec_leftfront)
    np.random.seed(seed)
    np.random.shuffle(y_vec_rightfront)
    np.random.seed(seed)
    np.random.shuffle(y_vec_leftback)
    print("Loading completed")
    return imagesTop, imagesFront, imagesLeftSide, imagesRightSide, imagesLeftFront, imagesRightFront, imagesLeftBack, y_vec_top, y_vec_front, y_vec_leftside, y_vec_rightside, y_vec_leftfront, y_vec_rightfront, y_vec_leftback, fileids


# =============================================================================
# def runtime_load_test():
#     import time
#     start_time = time.time()
#     imagesTop, imagesFront, imagesSide, y_vec_top, y_vec_front, y_vec_side = load(56)
#     print("--- %s min ---" % ((time.time() - start_time) /  60))
#     #print(imagesTop.shape[0])
# 
# =============================================================================
