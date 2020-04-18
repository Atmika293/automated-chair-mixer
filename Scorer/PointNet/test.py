'''
python test.py --outf <folder path to where the test results will be saved in results.csv> --dataset <path to .obj files>
'''

import numpy as np

import torch.utils.data as data
import os
import torch

import random
import argparse

from model import PointNetCls

def parse_vertices(filename):
    f = open(filename)
    line = f.readline()
    vertices = []
    while line:
        if line[:2] == "v ":
            index1 = line.find(" ") + 1
            index2 = line.find(" ", index1 + 1)
            index3 = line.find(" ", index2 + 1)
            vertex = [float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1])]
            vertices.append(vertex)

        line = f.readline()
    f.close()

    return np.asarray(vertices, dtype=np.float32)

class TestDataset(data.Dataset):
    def __init__(self, root, npoints=2500):
        self.npoints = npoints

        self.data_files = []

        for filename in os.listdir(root):
            if filename.endswith('.obj'):
                self.data_files.append(os.path.join(root, filename))

    def __getitem__(self, index):
        obj_file = self.data_files[index]

        point_set = parse_vertices(obj_file)

        choice = np.random.choice(len(point_set), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        point_set = torch.from_numpy(point_set)

        return obj_file, point_set

    def __len__(self):
        return len(self.data_files)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='scorer.pth', help='model path')
    parser.add_argument('--outf', type=str, default='.', help='output folder')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    parser.add_argument('--gpu', action='store_true', help="use feature transform")
    opt = parser.parse_args()
    
    outfile = os.path.join(opt.outf, 'results.csv')
    fp = open(outfile, 'w')
    fp.close()

    device='cpu'
    if opt.gpu:
        device = 'cuda'

    test_dataset = TestDataset(root=opt.dataset)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False) 
    
    classifier = PointNetCls(k=2, feature_transform=opt.feature_transform, dev=device)
    classifier.load_state_dict(torch.load(opt.model, map_location=torch.device(device)))
    classifier = classifier.eval()

    for i, data in enumerate(testdataloader, 0):
        obj_file, points = data
        points = points.transpose(2, 1)

        log_pred, _, _ = classifier(points)
        pred = torch.exp(log_pred)

        chair_score = pred.data[:, 1]
        max_score, pred_choice = pred.data.max(1)

        class_label = 'Chair' if pred_choice.item() == 1 else 'Not Chair'

        # print('%s: Class=%d, Score=%f, Plausibility=%f'%(obj_file, pred_choice.item(), max_score.item(), chair_score.item()))
        print('%s: Plausibility=%f'%(obj_file, chair_score.item()))

        with open(outfile, 'a') as fp:
            fp.write('%s: Class=%s, Score=%f, Plausibility=%f\n'%(obj_file, class_label, max_score.item(), chair_score.item()))