import os
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import ToTensor


class Dataset:
    def __init__(self, view, mode, dimension, root_dir='chairs-data'):

        self.view = view
        self.mode = mode
        self.dimension = dimension
        self.root_dir = root_dir

        self.imagesLeftFront = []
        self.imagesFront = []
        self.imagesRightFront = []
        self.imagesRightSide = []
        self.imagesLeftBack = []
        self.imagesLeftSide = []
        self.imagesTop = []

        self.y_vec_leftfront = []
        self.y_vec_front = []
        self.y_vec_rightfront = []
        self.y_vec_rightside = []
        self.y_vec_leftback = []
        self.y_vec_leftside = []
        self.y_vec_top = []

        self.dataload(self.root_dir, self.dimension)

        self.i = 0

    def dataload(self, root_dir, dimension):
        isPositive = False
        if self.mode == 'train':
            for folder in ["chairs-data/renders/", "chairs-data/renders_bad/"]:
                isPositive = not isPositive
                length = len(os.listdir(folder)) // 7 #dividing the entire data into 7 parts for the seven views
                for filename in os.listdir(folder):
                    v = int(filename.split(".")[0])
                    v = v % 10
                    img_path = folder + filename
                    if img_path is not None:
                        if v == 0:
                            self.imagesLeftFront.append(img_path)
                        elif v == 1:
                            self.imagesFront.append(img_path)
                        elif v == 2:
                            self.imagesRightFront.append(img_path)
                        elif v == 3:
                            self.imagesRightSide.append(img_path)
                        elif v == 4:
                            self.imagesLeftBack.append(img_path)
                        elif v == 5:
                            self.imagesLeftSide.append(img_path)
                        else:
                            self.imagesTop.append(img_path)

                            # print("Image Appended")
                #after the image paths are appended into the lists, if they are the good chairs, we fill the label lists with value 1
                if isPositive:
                    self.y_vec_leftfront = np.ones((length), dtype=np.int)
                    self.y_vec_front = np.ones((length), dtype=np.int)
                    self.y_vec_rightfront = np.ones((length), dtype=np.int)
                    self.y_vec_rightside = np.ones((length), dtype=np.int)
                    self.y_vec_leftback = np.ones((length), dtype=np.int)
                    self.y_vec_leftside = np.ones((length), dtype=np.int)
                    self.y_vec_top = np.ones((length), dtype=np.int)
                    # print("Yvec visited")

                #if they are bad chairs, we fill the label lists with value 0
                else:
                    self.y_vec_leftfront = np.append(self.y_vec_leftfront, np.zeros((length), dtype=np.int), axis=0)
                    self.y_vec_front = np.append(self.y_vec_front, np.zeros((length), dtype=np.int), axis=0)
                    self.y_vec_rightfront = np.append(self.y_vec_rightfront, np.zeros((length), dtype=np.int), axis=0)
                    self.y_vec_rightside = np.append(self.y_vec_rightside, np.zeros((length), dtype=np.int), axis=0)
                    self.y_vec_leftback = np.append(self.y_vec_leftback, np.zeros((length), dtype=np.int), axis=0)
                    self.y_vec_leftside = np.append(self.y_vec_leftside, np.zeros((length), dtype=np.int), axis=0)
                    self.y_vec_top = np.append(self.y_vec_top, np.zeros((length), dtype=np.int), axis=0)
                    # print("Yvec else visited")

        else:
            print("Testing process")
            folder = 'setA/'
            length = len(os.listdir(folder))
            for filename in os.listdir(folder):
                # v = int(filename.split(".")[0])
                only_alpha = ""
                v = filename.split(".")[0]

                #the files are in the format 'front0.png' so we extract only the alphabets out
                for char in v:
                    if ord(char) >= 97 and ord(char) <= 122:
                        only_alpha += char
                # v = v % 10

                img_path = folder + filename
                if img_path is not None:
                    if only_alpha == 'leftfront':
                        self.imagesLeftFront.append(img_path)
                        # print(img_path)
                    elif only_alpha == 'front':
                        self.imagesFront.append(img_path)
                        # print(img_path)
                    elif only_alpha == 'rightfront':
                        self.imagesRightFront.append(img_path)
                        # print(img_path)
                    elif only_alpha == 'rightside':
                        self.imagesRightSide.append(img_path)
                        # print(img_path)
                    elif only_alpha == 'leftback':
                        self.imagesLeftBack.append(img_path)
                        # print(img_path)
                    elif only_alpha == 'leftside':
                        self.imagesLeftSide.append(img_path)
                        # print(img_path)
                    else:
                        self.imagesTop.append(img_path)
                        # print(img_path)

    #this _getitem_ function will get called when enumerating img,label pairs in the for loop for training
    def __getitem__(self, i):
        imglabel = 0

        if self.mode == 'train':
            if self.view == 'Top':
                img = self.imagesTop[i]
                imglabel = self.y_vec_top[i]
            elif self.view == 'Left Side':
                img = self.imagesLeftSide[i]
                imglabel = self.y_vec_leftside[i]
            elif self.view == 'Front':
                img = self.imagesFront[i]  # img is img path
                imglabel = self.y_vec_front[i]
            elif self.view == 'Right Side':
                img = self.imagesRightSide[i]
                imglabel = self.y_vec_rightside[i]
            elif self.view == 'Left Front':
                img = self.imagesLeftFront[i]
                imglabel = self.y_vec_leftfront[i]
            elif self.view == 'Right Front':
                img = self.imagesRightFront[i]
                imglabel = self.y_vec_rightfront[i]
            else:
                img = self.imagesLeftBack[i]
                imglabel = self.y_vec_leftback[i]

        else: #testing, no labels appended

            if self.view == 'Top':
                img = self.imagesTop[i]
            elif self.view == 'Front':
                img = self.imagesFront[i]
            elif self.view == 'Left Side':
                img = self.imagesLeftSide[i]
            elif self.view == 'Right Side':
                img = self.imagesRightSide[i]
            elif self.view == 'Left Front':
                img = self.imagesLeftFront[i]
            elif self.view == 'Right Front':
                img = self.imagesRightFront[i]
            else:
                img = self.imagesLeftBack[i]

        # print("i iteration done")
        inputs = Image.open(img)
        # if self.mode == 'train':
        # data augmentation may be done
        # print(inputs.type)

        inputs = inputs.convert('L') #converting to grayscale for processing
        inputs = ToTensor()(inputs) #to tensor
        load = (1. - inputs / 255.)
        # print(load, imglabel)

        return (load, imglabel)

    def __len__(self):
        # print("len")
        return len(self.imagesFront)

    # if __name__ == '__main__':
#    traindata = main1('C:/Users/JANAKI/Desktop/GM Project/LeChairs_CMPT464/Scorer/chairs-data')

#    train_data_loader = torch.utils.data.DataLoader(traindata, batch_size=10, shuffle=True, num_workers=0)
#   i, (image, label) = next(enumerate(train_data_loader))
