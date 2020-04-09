import torch
from torch.utils import data
from scipy.io import loadmat
from enum import Enum
import os
import numpy as np

#https://inareous.github.io/posts/opening-obj-using-py
def parse_part_from_obj(filename, node_part_index):
    f = open(filename)
    line = f.readline()
    part_geometry = None

    while line:
        if line[0:2] == "g ":
            part_index = int(line[2:])
            if part_index == node_part_index:
                line = f.readline()

                vertices = []
                faces = []
                
                while line and line[0:2] != "g ":
                    if line[:2] == "v ":
                        index1 = line.find(" ") + 1
                        index2 = line.find(" ", index1 + 1)
                        index3 = line.find(" ", index2 + 1)
                        vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                        vertices.append(vertex)

                    elif line[:2] == "f ":
                        index1 = line.find(" ") + 1
                        index2 = line.find(" ", index1 + 1)
                        index3 = line.find(" ", index2 + 1)
                        face = (int(line[index1:index2])-1, int(line[index2:index3])-1, int(line[index3:-1])-1)
                        faces.append(face)

                    line = f.readline()
                part_geometry = (vertices, faces)
                break
            else:
                line = f.readline()

        else:
            line = f.readline()
    f.close()

    if part_geometry is None:
        print("Failed to find part geometry")

    return part_geometry

class Tree(object):
    class NodeType(Enum):
        BOX = 0  # box node
        ADJ = 1  # adjacency (adjacent part assembly) node
        SYM = 2  # symmetry (symmetric part grouping) node

    class NodeLabel(Enum):
        BACK = 0
        SEAT = 1
        LEG = 2
        ARMREST = 3

    class Node(object):
        def __init__(self, box=None, left=None, right=None, node_type=None, sym=None, label=None, part_indices=None, mesh_file=None, part_index=-1):
            self.box = box          # box feature vector for a leaf node
            self.sym = sym          # symmetry parameter vector for a symmetry node
            self.left = left        # left child for ADJ or SYM (a symmeter generator)
            self.right = right      # right child
            self.node_type = node_type
            self.label = label
            self.part_indices=part_indices
            self.mesh_file = mesh_file
            self.part_geometry = None
            self.part_index = part_index
            if self.is_leaf():
                self.part_geometry = parse_part_from_obj(mesh_file, part_index)

        def is_leaf(self):
            return self.node_type == Tree.NodeType.BOX and self.box is not None

        def is_adj(self):
            return self.node_type == Tree.NodeType.ADJ

        def is_sym(self):
            return self.node_type == Tree.NodeType.SYM

    def __init__(self, boxes, ops, syms, labels, mesh_path, part_mesh_obj_indices):
        box_list = [b for b in torch.split(boxes, 1, 0)]
        sym_param = [s for s in torch.split(syms, 1, 0)]
        label_list = [l for l in labels[0]]
        box_list.reverse()
        sym_param.reverse()
        label_list.reverse()
        part_mesh_obj_indices = np.flip(part_mesh_obj_indices, axis=0)
        queue = []
        leaf_index = 0
        for id in range(ops.size()[1]):
            if ops[0, id] == Tree.NodeType.BOX.value:
                queue.append(Tree.Node(box=box_list.pop(),
                    node_type=Tree.NodeType.BOX,
                    label=Tree.NodeLabel(label_list.pop().item()),
                    mesh_file=mesh_path,
                    part_indices=part_mesh_obj_indices[leaf_index],
                    part_index=leaf_index))
                leaf_index+=1
            elif ops[0, id] == Tree.NodeType.ADJ.value:
                left_node = queue.pop()
                right_node = queue.pop()
                queue.append(Tree.Node(left=left_node, right=right_node, node_type=Tree.NodeType.ADJ))
            elif ops[0, id] == Tree.NodeType.SYM.value:
                node = queue.pop()
                queue.append(Tree.Node(left=node, sym=sym_param.pop(), node_type=Tree.NodeType.SYM))
        assert len(queue) == 1
        self.root = queue[0]

    

class GRASSDataset(data.Dataset,):
    def __init__(self, dir, models_num=0, transform=None):
        self.dir = dir
        num_examples = len(os.listdir(os.path.join(dir, 'ops')))
        self.transform = transform
        self.trees = []
        for i in range(models_num):
            boxes = torch.from_numpy(loadmat(os.path.join(dir, 'boxes', '%d.mat' % (i+1)))['box']).t().float()
            ops = torch.from_numpy(loadmat(os.path.join(dir, 'ops', '%d.mat' % (i+1)))['op']).int()
            syms = torch.from_numpy(loadmat(os.path.join(dir, 'syms', '%d.mat' % (i+1)))['sym']).t().float()
            labels = torch.from_numpy(loadmat(os.path.join(dir, 'labels', '%d.mat' % (i+1)))['label']).int()
            part_mesh = loadmat(os.path.join(dir, 'part mesh indices', '%d.mat' % (i+1)))
            part_mesh_obj_indices = part_mesh['cell_boxs_correspond_objSerialNumber'][0]
            mesh_name = str(part_mesh['shapename'][0])
            mesh_path = os.path.join(dir, 'models', '%s.obj' % mesh_name)
            tree = Tree(boxes, ops, syms, labels, mesh_path, part_mesh_obj_indices)
            self.trees.append(tree)

    def __getitem__(self, index):
        tree = self.trees[index]
        return tree

    def __len__(self):
        return len(self.trees)