# original code from https://github.com/PeppaZhu/grass

import numpy as np
from draw3dobb import showGenshape
import torch
from grassdata import GRASSDataset
import math
from numpy import linalg

def vrrotvec2mat(rotvector):
    s = math.sin(rotvector[3])
    c = math.cos(rotvector[3])
    t = 1 - c
    x = rotvector[0]
    y = rotvector[1]
    z = rotvector[2]
    #m = torch.FloatTensor([[t*x*x+c, t*x*y-s*z, t*x*z+s*y], [t*x*y+s*z, t*y*y+c, t*y*z-s*x], [t*x*z-s*y, t*y*z+s*x, t*z*z+c]]).cuda()
    m = torch.FloatTensor([[t*x*x+c, t*x*y-s*z, t*x*z+s*y], [t*x*y+s*z, t*y*y+c, t*y*z-s*x], [t*x*z-s*y, t*y*z+s*x, t*z*z+c]])
    return m

def get_box_corners(points):
    """
    returns an 8x3 numpy array of points corresponding to the corners of the box
    """
    points = points.squeeze(0).numpy()
    center = points[0: 3]
    lengths = points[3: 6]
    dir_1 = points[6: 9]
    dir_2 = points[9: ]

    dir_1 = dir_1/linalg.norm(dir_1)
    dir_2 = dir_2/linalg.norm(dir_2)
    dir_3 = np.cross(dir_1, dir_2)
    dir_3 = dir_3/linalg.norm(dir_3)
    cornerpoints = np.zeros([8, 3])

    d1 = 0.5*lengths[0]*dir_1
    d2 = 0.5*lengths[1]*dir_2
    d3 = 0.5*lengths[2]*dir_3

    cornerpoints[0][:] = center - d1 - d2 - d3
    cornerpoints[1][:] = center - d1 + d2 - d3
    cornerpoints[2][:] = center + d1 - d2 - d3
    cornerpoints[3][:] = center + d1 + d2 - d3
    cornerpoints[4][:] = center - d1 - d2 + d3
    cornerpoints[5][:] = center - d1 + d2 + d3
    cornerpoints[6][:] = center + d1 - d2 + d3
    cornerpoints[7][:] = center + d1 + d2 + d3

    return cornerpoints

def decode_structure(root):
    """
    Decode a root code into a tree structure of boxes
    """
    syms = [torch.ones(8).mul(10)]
    stack = [root]
    boxes = []
    while len(stack) > 0:
        node = stack.pop()

        node_type = torch.LongTensor([node.node_type.value]).item()
        if node_type == 1:  # ADJ
            stack.append(node.left)
            stack.append(node.right)
            s = syms.pop()
            syms.append(s)
            syms.append(s)
        if node_type == 2:  # SYM
            stack.append(node.left)
            syms.pop()
            syms.append(node.sym.squeeze(0))
        if node_type == 0:  # BOX
            reBox = node.box
            label = node.label
            reBoxes = [(label, get_box_corners(reBox))]
            s = syms.pop()
            l1 = abs(s[0] + 1)
            l2 = abs(s[0])
            l3 = abs(s[0] - 1)

            sList = torch.split(s, 1, 0)
            bList = torch.split(reBox.data.squeeze(0), 1, 0)

            if l1 < 0.15:
                f1 = torch.cat([sList[1], sList[2], sList[3]])
                f1 = f1/torch.norm(f1)
                f2 = torch.cat([sList[4], sList[5], sList[6]])
                folds = round(1/s[7].item())
                for i in range(folds-1):
                    rotvector = torch.cat([f1, sList[7].mul(2*3.1415).mul(i+1)])
                    rotm = vrrotvec2mat(rotvector)
                    center = torch.cat([bList[0], bList[1], bList[2]])
                    dir0 = torch.cat([bList[3], bList[4], bList[5]])
                    dir1 = torch.cat([bList[6], bList[7], bList[8]])
                    dir2 = torch.cat([bList[9], bList[10], bList[11]])
                    newcenter = rotm.matmul(center.add(-f2)).add(f2)
                    newdir1 = rotm.matmul(dir1)
                    newdir2 = rotm.matmul(dir2)
                    newbox = torch.cat([newcenter, dir0, newdir1, newdir2])
                    reBoxes.append((label, get_box_corners(newbox)))
            if l3 < 0.15:
                trans = torch.cat([sList[1], sList[2], sList[3]])
                trans_end = torch.cat([sList[4], sList[5], sList[6]])
                center = torch.cat([bList[0], bList[1], bList[2]])
                trans_length = math.sqrt(torch.sum(trans**2))
                trans_total = math.sqrt(torch.sum(trans_end.add(-center)**2))
                folds = round(trans_total/trans_length)
                for i in range(folds):
                    center = torch.cat([bList[0], bList[1], bList[2]])
                    dir0 = torch.cat([bList[3], bList[4], bList[5]])
                    dir1 = torch.cat([bList[6], bList[7], bList[8]])
                    dir2 = torch.cat([bList[9], bList[10], bList[11]])
                    newcenter = center.add(trans.mul(i+1))
                    newbox = torch.cat([newcenter, dir0, dir1, dir2])
                    reBoxes.append((label, get_box_corners(newbox)))
            if l2 < 0.15:
                ref_normal = torch.cat([sList[1], sList[2], sList[3]])
                ref_normal = ref_normal/torch.norm(ref_normal)
                ref_point = torch.cat([sList[4], sList[5], sList[6]])
                center = torch.cat([bList[0], bList[1], bList[2]])
                dir0 = torch.cat([bList[3], bList[4], bList[5]])
                dir1 = torch.cat([bList[6], bList[7], bList[8]])
                dir2 = torch.cat([bList[9], bList[10], bList[11]])
                if ref_normal.matmul(ref_point.add(-center)) < 0:
                    ref_normal = -ref_normal
                newcenter = ref_normal.mul(2*abs(torch.sum(ref_point.add(-center).mul(ref_normal)))).add(center)
                if ref_normal.matmul(dir1) < 0:
                    ref_normal = -ref_normal
                dir1 = dir1.add(ref_normal.mul(-2*ref_normal.matmul(dir1)))
                if ref_normal.matmul(dir2) < 0:
                    ref_normal = -ref_normal
                dir2 = dir2.add(ref_normal.mul(-2*ref_normal.matmul(dir2)))
                newbox = torch.cat([newcenter, dir0, dir1, dir2])
                reBoxes.append((label, get_box_corners(newbox)))

            boxes.extend(reBoxes)

    return boxes

if __name__ == "__main__":
    grassdata = GRASSDataset('A:\\764dataset\\Chair',4)
    for i in range(len(grassdata)):
        tree = grassdata[i]
        boxes = decode_structure(tree.root)
        showGenshape(boxes)
