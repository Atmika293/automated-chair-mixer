import numpy as np

import torch.utils.data as data
import os
import torch

from enum import Enum

import random

class PartLabel(Enum):
    """Types of parts"""
    BACK = 0
    SEAT = 1
    LEG = 2
    ARMREST = 3

def parse_obb_file(filename):  
    """Extract partnet symh info from obb file"""
    part_count = 0
    labels = []
    f = open(filename)
    line = f.readline()

    if line[:2] == "N ":
        part_count = int(line[2:])

        # parse bounding box floats
        for i in range(0, part_count):
            # bounding_boxes.append(parse_bounding_box_obb(f.readline()))
            line = f.readline()
        
        # parse connectivity
        line = f.readline()
        if line[:2] != "C ":
            return None

        connection_count = int(line[2:])
        for i in range(0, connection_count):
            # connectivity.append(parse_connection_obb(f.readline()))
            line = f.readline()

        line = f.readline()
        # skip symmetry
        while line[:2] != "L ":
            line = f.readline()
    
        # parse labels 
        label_count = int(line[2:])
        if label_count != part_count:
            return None
        for i in range(0, part_count):
            labels.append(int(f.readline()))

    else:
        return None

    f.close()

    return part_count, labels

def parse_vertices(filename):
    f = open(filename)
    line = f.readline()
    parts_vertices = []
    order = []
    while line:
        if line[0:2] == "g ":
            part_index = int(line[2:])
            order.append(part_index)
            vertices = []
            line = f.readline()
            while line and line[0:2] != "g ":
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)
                    vertex = [float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1])]
                    vertices.append(vertex)

                line = f.readline()
            parts_vertices.append(np.asarray(vertices, dtype=np.float32))
        else:
            line = f.readline()
    f.close()

    if len(parts_vertices) == 0:
        print("Failed to find part geometry")

    return order, parts_vertices

class PointsDataset(data.Dataset):
    def __init__(self,
                 root,
                 test_models,
                 softmax=False,
                 npoints=2500,
                 mode='train', 
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.data_augmentation = data_augmentation
        self.mode = mode
        self.softmax = softmax

        obb_path = os.path.join(root, 'obbs')
        obj_path = os.path.join(root, 'models')
        
        eval_models = [str(i) + '.obb' for i in test_models]

        self.data_files = []

        if mode == 'train':
            files = os.listdir(obb_path)
            for filename in files:
                if filename in eval_models:
                    continue
                else:
                    self.data_files.append((os.path.join(obb_path, filename), os.path.join(obj_path, filename[:-3] + "obj")))

        else:
            for filename in eval_models:
                self.data_files.append((os.path.join(obb_path, filename), os.path.join(obj_path, filename[:-3] + "obj")))


    def __getitem__(self, index):
        obb_file, obj_file = self.data_files[index]


        part_count, part_labels = parse_obb_file(obb_file)
        order, parts_vertices = parse_vertices(obj_file)

        if random.random() > 0.5: ##good chair
            point_set = np.concatenate(parts_vertices, axis=0)
            label = 1
        else:
            random.shuffle(parts_vertices)
            chosen_parts = []
            n_points = 0
            count = 0
            while count < part_count - 1:
                chosen_idx = random.randint(0, len(parts_vertices)-1)
                count += 1

                chosen = parts_vertices.pop(chosen_idx)
                n_points += chosen.shape[0]

                chosen_parts.append(chosen)

                if n_points > self.npoints:
                    if random.random() > 0.5:
                        break

            point_set = np.concatenate(chosen_parts, axis=0)
            label = 0

        choice = np.random.choice(len(point_set), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        # point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        # dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        # point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        point_set = torch.from_numpy(point_set)
        if self.softmax:
            target = torch.from_numpy(np.array([label]).astype(np.int64))
        else:
            target = torch.from_numpy(np.array([label]).astype(np.float32))

        return point_set, target

    def __len__(self):
        return len(self.data_files)

if __name__ == '__main__':
    d = PointsDataset(root = 'D:\\CMPT 764\\chairs_dataset', softmax=True, mode='eval', test_models = [197, 41103, 37574, 45075, 40015, 39696, 37658, 36193, 309, 1325, 3244, 36881, 37614, 40403, 42663, 37989, 3223, 41656])
    print(len(d))
    for i in range(10):
        ps, target = d[i]
        print(ps.size(), ps.type(), target.size(), target.type(), target)