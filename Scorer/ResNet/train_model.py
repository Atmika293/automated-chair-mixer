from __future__ import print_function
import numpy as np
import argparse
import os
import torch
import torch.nn.parallel
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from dataload import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--nepoch', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--mode', type=str, default='train', help='train or test')
parser.add_argument('--chooseepoch', type=str, default='epoch25.pth', help='type the epoch you want to choose for testing')
opt = parser.parse_args()
print(opt)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        ResNet18 = models.resnet18(pretrained=True) #we take pre-trained resnet18 and add additional layers
        ResNet18.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        struct = list(ResNet18.children())[:-1]
        #modules.append(nn.Conv2d(64, 128, kernel_size=(3,3),stride=(2, 2), padding=(3, 3), bias=False))
        #modules.append(nn.Conv2d(128, 256, kernel_size=(3,3),stride=(2, 2), padding=(3, 3), bias=False))
        #modules.append(nn.Conv2d(256, 512, kernel_size=(3,3),stride=(2, 2), padding=(3, 3), bias=False))

        self.restruct = nn.Sequential(*struct)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 2)
        # self.fc3 = nn.Linear(128, 2)
        self.drop = nn.Dropout(p=0.4)  # dropout function to prevent overfitting

    def forward(self, x):
        res = self.restruct(x)
        res = self.relu(self.fc1(res.view(res.size(0), -1)))
        res = self.fc2(self.drop(res))
        #res = self.fc3(res)
        return res


# ResNet34 modified
# def __init__(self):
#         super(ResNet, self).__init__()
#         ResNet34 = models.resnet34(pretrained=True)
#         ResNet34.conv = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=False)
#
#         struct = list(ResNet34.children())[:-1]
#         self.restruct = nn.Sequential(*struct)
#         self.fc1 = nn.Linear(512, 2)
#         self.relu = nn.ReLU()
#         self.dropout1 = nn.Dropout(p=0.4)
#         self.fc2 = nn.Linear(256, 2)
#
#     def forward(self, x):
#         res = self.restruct(x)
#         res = res.view(res.size(0), -1)
#         res = self.relu(self.fc1(out))
# dropout


# LENET
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # inputShape = (numChannels, imgCols, imgRows)
        # conv layers
        self.conv = nn.Sequential(nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2, stride=2),
                                  nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2, stride=2))

        # fully connected layers
        # in nn.Linear, input shape is 64 * 14 * 14 and output shape is 1024
        self.fc = nn.Sequential(nn.Linear(64 * 14 * 14, 1024), nn.ReLU(), nn.Dropout(0.4), nn.Linear(1024, 2))

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.view(-1, 64 * 14 * 14))

        return x

def main(*argv):
    # fix seed
    view = ["Left Front", "Front", "Right Front", "Right Side", "Left Back", "Left Side", "Top"]
    if opt.mode == 'train':
        for v in view:
            print("View chosen : ", v)
            classifier = ResNet()
            classifier.cuda()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(classifier.parameters(), lr=0.00001, momentum=0.8)
            # train_slice = 0.8
            train_data = Dataset(v, opt.mode, 224, 'chairs-data')
            # valid_data = train_data
            print("length of train data", len(train_data))
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
            # validation_dataloader = torch.utils.data.DataLoader(valid_data, batch_size = 32, shuffle = True, num_workers = 0, sampler = valid_sampler)

            num_batch = len(train_data) / opt.batchSize
            print(num_batch)
            for epoch in range(opt.nepoch):
                print('Epoch %d/%d' % (epoch + 1, opt.nepoch))
                print('Training...')
                classifier.train()
                train_accuracy = 0
                epoch_loss = 0
                # print('entering for loop')
                for i, (inputs, labels) in enumerate(train_dataloader):
                    if torch.cuda.is_available():
                        device = torch.device("cuda")
                        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())
                    else:
                        device = torch.device("cpu")
                    # print("Device is : ", device)

                    optimizer.zero_grad()
                    pred = classifier(inputs)
                    # print(len(inputs), len(labels))
                    loss = criterion(pred, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(labels.data).cpu().sum()
                    print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))
                    train_accuracy += correct.item()

                if (epoch + 1) % 5 == 0:
                    model_path = os.path.join('checkpoint/%s' % v, 'epoch%d.pth' % (epoch + 1))
                    torch.save(classifier.state_dict(), model_path)

                print('Epoch %d: Accuracy: %f' % (epoch, train_accuracy / len(train_data)))

                # loss per epoch
                print('Completed epoch %d - Loss: %.6f' % (epoch + 1, epoch_loss / (i + 1)))

    else:
        print("test")
        # TEST
        scores = []
        for v in view:
            classifier = ResNet()
            classifier.cuda()
            test_data = Dataset(v, opt.mode, 'evaluate-chairs', 224)
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

            # print("test data after getting from dataload",test_data)
            print("view : ", v)
            saved_model = torch.load('checkpoint/%s/%s' % (v, opt.chooseepoch))
            classifier.load_state_dict(saved_model)
            classifier.eval()
            temp = []

            with torch.no_grad():
                for i, (img, labels) in enumerate(test_dataloader):

                    # print("iteration of img in test_dataloader:",img)
                    if torch.cuda.is_available():
                        inputs = Variable(img.float().cuda())

                    pred = classifier(inputs)
                    # print(len(inputs), len(labels))
                    # using softmax to compute the probabilities
                    sm = nn.Softmax(dim=1)
                    prob = sm(pred).squeeze()[1]
                    temp.append(prob.detach().cpu().numpy())

                print(v, temp)
                scores.append(temp)
        print("\n test probabilities", scores)
        finalscores = np.amin(scores, axis=0)
        print("\nThe minimum probabilities are : \n", finalscores * 100)


if __name__ == "__main__":
    main()