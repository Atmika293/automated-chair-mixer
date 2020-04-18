from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import PointsDataset
from model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--softmax', action='store_true', help="use softmax label")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


dataset = PointsDataset(
    root=opt.dataset,
    npoints=opt.num_points,
    softmax=opt.softmax,
    test_models = [197, 41103, 37574, 45075, 40015, 39696, 37658, 36193, 309, 1325, 3244, 36881, 37614, 40403, 42663, 37989, 3223, 41656])

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

test_dataset = PointsDataset(
    root=opt.dataset,
    npoints=opt.num_points,
    softmax=opt.softmax,
    mode='eval',
    test_models = [197, 41103, 37574, 45075, 40015, 39696, 37658, 36193, 309, 1325, 3244, 36881, 37614, 40403, 42663, 37989, 3223, 41656])

testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset))
num_classes=2

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)
if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()
num_batch = len(dataset) / opt.batchSize

if opt.softmax:
    criterion = nn.NLLLoss()
else:
    criterion = nn.BCELoss()


for epoch in range(opt.nepoch):
    train_accuracy = 0
    scheduler.step()
    classifier = classifier.train()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        optimizer.zero_grad()
        pred, trans, trans_feat = classifier(points)
        loss = criterion(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

        train_accuracy += correct.item()

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

    print('Epoch %d: Accuracy: %f'%(epoch, train_accuracy / len(dataset)))

    with open('%s/TrainAccuracy.csv'%opt.outf, 'a') as fp:
        fp.write('%d,%f\n'%(epoch+1, train_accuracy / len(dataset)))

    if epoch%5 == 4:
        classifier = classifier.eval()
        test_accuracy = 0
        for i, data in enumerate(testdataloader, 0):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()

            optimizer.zero_grad()
            pred, _, _ = classifier(points)
            
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()

            test_accuracy += correct.item()

        print('Test Accuracy: %f'%(test_accuracy / len(test_dataset)))

        with open('%s/TestAccuracy.csv'%opt.outf, 'a') as fp:
            fp.write('%d,%f\n'%(epoch+1, test_accuracy / len(test_dataset)))

