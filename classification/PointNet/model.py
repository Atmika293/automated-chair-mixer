from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, inC, outC, relu=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(inC, outC, 1)
        self.bn = nn.BatchNorm2d(outC)

        self.add_relu = relu
        if relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        if self.add_relu:
            x = self.relu(x)

        return x

class FCBlock(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(in_feat, out_feat, 1)
        self.bn = nn.BatchNorm1d(out_feat)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class TNet(nn.Module):
    def __init__(self, k=64, dev='cuda'):
        super(TNet, self).__init__()
        self.point_dim = k
        # Each layer has batchnorm and relu on it
        # conv 3 64
        self.c_block1 = ConvBlock(k, 64)
        # conv 64 128
        self.c_block2 = ConvBlock(64, 128)
        # conv 128 1024
        self.c_block3 = ConvBlock(128, 1024)
        # # max pool
        # self.pool = nn.MaxPool2d((3, 1))
        # fc 1024 512
        self.fc_block1 = FCBlock(1024, 512)
        # fc 512 256
        self.fc_block2 = FCBlock(512, 256)
        # fc 256 k*k (no batchnorm, no relu)
        # add bias
        self.weights = torch.zeros(256, k*k, dtype=torch.float, requires_grad=True).to(torch.device(dev))
        self.bias = torch.eye(k, dtype=torch.float, requires_grad=True).to(torch.device(dev))
        # reshape
    
    def forward(self, x): ##data = b x k x n
        x = torch.unsqueeze(x, dim=3) ## b x k x n x 1

        x = self.c_block1(x) ## b x 64 x n x 1
        x = self.c_block2(x) ## b x 128 x n x 1
        x = self.c_block3(x) ## b x 1024 x n x 1

        x = torch.squeeze(x, dim=3) ## b x 1024 x n
        # x = self.pool(x)  
        x = torch.max(x, dim=2)[0] ## b x 1024

        x = self.fc_block1(x)  ## b x 512
        x = self.fc_block2(x)  ## b x 256

        x = torch.matmul(x, self.weights) ## b x (k x k)
        x = x.view(-1, self.point_dim, self.point_dim) ## b x k x k
        x = torch.add(x, 1, self.bias) ## b x k x k

        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False, dev='cuda'):
        super(PointNetfeat, self).__init__()
        self.global_features = global_feat
        self.add_feature_transform = feature_transform
        # Use TNet to apply transformation on input and multiply the input points with the transformation
        self.input_transform = TNet(k=3, dev=dev)
        # conv 3 64
        self.c_block1 = ConvBlock(3, 64)
        # Use TNet to apply transformation on features and multiply the input features with the transformation 
        #                                                                        (if feature_transform is true)
        if feature_transform:
            self.feature_transform = TNet(k=64, dev=dev)
        # conv 64 128
        self.c_block2 = ConvBlock(64, 128)
        # conv 128 1024 (no relu)
        self.c_block3 = ConvBlock(128, 1024, relu=False)
        # max pool


    def forward(self, x):
        n_pts = x.size()[2] ## x = b x 3 x n

        # You will need these extra outputs:
        # trans = output of applying TNet function to input
        # trans_feat = output of applying TNet function to features (if feature_transform is true)
        trans = self.input_transform(x) ## b x 3 x 3
        x = torch.bmm(trans, x) ## b x 3 x n

        x = torch.unsqueeze(x, dim=3) ## b x 3 x n x 1
        x = self.c_block1(x) ## b x 64 x n x 1
        x = torch.squeeze(x, dim=3) ## b x 64 x n
        pointfeat = x

        trans_feat = None
        if self.add_feature_transform:
            trans_feat = self.feature_transform(x) ## b x 64 x 64           
            x = torch.bmm(trans_feat, x) ## b x 64 x n
            pointfeat = x

        x = torch.unsqueeze(x, dim=3) ## b x 64 x n x 1
        x = self.c_block2(x) ## b x 128 x n x 1
        x = self.c_block3(x) ## b x 1024 x n x 1
        x = torch.squeeze(x, dim=3) ## b x 1024 x n

        x = torch.max(x, dim=2)[0] ## b x 1024

        if self.global_features: # This shows if we're doing classification or segmentation
            return x, trans, trans_feat
        else:
            # x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            # return torch.cat([x, pointfeat], 1), trans, trans_feat
            return pointfeat, trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False, dev='cuda'):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform, dev=dev)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False, dev='cuda'):
        super(PointNetDenseCls, self).__init__()
        # get global features + point features from PointNetfeat
        self.pointfeatures_net = PointNetfeat(global_feat=False, feature_transform=feature_transform, dev=dev)
        # conv 1088 512
        self.c_block1 = ConvBlock(64, 512)
        # conv 512 256
        self.c_block2 = ConvBlock(512, 256)
        # conv 256 128
        self.c_block3 = ConvBlock(256, 128)
        # conv 128 k
        self.c_block4 = ConvBlock(128, k)
        # softmax 
        self.final_softmax = nn.Softmax(dim=2)
    
    def forward(self, x):
        # You will need these extra outputs: 
        # trans = output of applying TNet function to input
        # trans_feat = output of applying TNet function to features (if feature_transform is true)
        # (you can directly get them from PointNetfeat)
        x, trans, trans_feat = self.pointfeatures_net(x) ## b x 1088 x n

        x = torch.unsqueeze(x, dim=3) ## b x 1088 x n x 1
        x = self.c_block1(x) ## b x 512 x n x 1
        x = self.c_block2(x) ## b x 256 x n x 1
        x = self.c_block3(x) ## b x 128 x n x 1
        x = self.c_block4(x) ## b x k x n x 1

        x = torch.squeeze(x, dim=3) ## b x k x n
        x = torch.transpose(x, 1, 2) ## b x n x k
        x = self.final_softmax(x) ## b x n x k

        return x, trans, trans_feat

def feature_transform_regularizer(trans, dev='cuda'):
    # compute |((trans * trans.transpose) - I)|^2
    batch_size = trans.size()[0]
    k = trans.size()[1]
    a_aT = torch.bmm(trans, torch.transpose(trans, 1, 2))
    identity = torch.unsqueeze(torch.eye(k, dtype=torch.float), dim=0).repeat(batch_size, 1, 1).to(torch.device(dev))

    ele_square_dist = torch.pow(a_aT - identity, 2.0)
    euclidean_dist = torch.sum(ele_square_dist, dim=(1, 2))
    loss = torch.mean(euclidean_dist)
    
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500)).cuda()
    trans = TNet(k=3).cuda()
    out = trans(sim_data)
    print('TNet', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500)).cuda()
    trans = TNet(k=64).cuda()
    out = trans(sim_data_64d)
    print('TNet 64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True).cuda()
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False).cuda()
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5).cuda()
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3).cuda()
    out, _, _ = seg(sim_data)
    print('seg', out.size())

    loss_data_1 = Variable(torch.unsqueeze(torch.eye(5, dtype=torch.float), dim=0).repeat(3, 1, 1)).cuda()
    loss_data_2 = Variable(2 * torch.unsqueeze(torch.eye(7, dtype=torch.float), dim=0).repeat(3, 1, 1)).cuda()

    print(loss_data_1)
    print(feature_transform_regularizer(loss_data_1))
    print(loss_data_2)
    print(feature_transform_regularizer(loss_data_2))

