# original code from https://github.com/PeppaZhu/grass

import numpy as np
from draw3dobb import showGenshape, renderMeshFromParts, show_obbs_from_bboxes
import torch
from grassdata import GRASSDataset
import math
from numpy import linalg
from utils import write_to_obj, reindex_faces

from grassdata_new import Part, Mesh, PartLabel, GRASSNewDataset
from aggregator import Aggregator
from mixer import Mixer
from extractor import RandomizedExtractor

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


def get_geo_tree(root):
    stack = [root]
    geo = []
    syms = [torch.ones(8).mul(10)]
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
            geo.append((node.part_geometry[0], reindex_faces(node.part_geometry[0], node.part_geometry[1])))
            reBox = node.box
            label = node.label
            s = syms.pop()
            l1 = abs(s[0] + 1)
            l2 = abs(s[0])
            l3 = abs(s[0] - 1)

            sList = torch.split(s, 1, 0)
            bList = torch.split(reBox.data.squeeze(0), 1, 0)
            vertices = node.part_geometry[0]
            faces = node.part_geometry[1]   # figure out how to reindex the faces
            
            if l1 < 0.15:   #rotation symmetry
                f1 = torch.cat([sList[1], sList[2], sList[3]])
                f1 = f1/torch.norm(f1)
                f2 = torch.cat([sList[4], sList[5], sList[6]])
                folds = round(1/s[7].item())
                for i in range(folds-1):
                    rotvector = torch.cat([f1, sList[7].mul(2*3.1415).mul(i+1)])
                    rotm = vrrotvec2mat(rotvector)
                    new_vertices = rotm.matmul(torch.FloatTensor(vertices).add(-f2)).add(f2).tolist()
                    #reindex faces here
                    geo.append((new_vertices, reindex_faces(new_vertices, faces)))

            if l3 < 0.15: # translation
                trans = torch.cat([sList[1], sList[2], sList[3]])
                trans_end = torch.cat([sList[4], sList[5], sList[6]])
                center = torch.cat([bList[0], bList[1], bList[2]])
                trans_length = math.sqrt(torch.sum(trans**2))
                trans_total = math.sqrt(torch.sum(trans_end.add(-center)**2))
                folds = round(trans_total/trans_length)
                for i in range(folds):
                    center = torch.cat([bList[0], bList[1], bList[2]])
                    new_center = center.add(trans.mul(i+1))
                    pt_vertices = torch.FloatTensor(vertices)
                    new_vertices = pt_vertices.add(center - new_center).tolist()
                    geo.append((new_vertices, reindex_faces(new_vertices, faces)))

            if l2 < 0.15: # mirror?
                ref_normal = torch.cat([sList[1], sList[2], sList[3]])
                ref_normal = ref_normal/torch.norm(ref_normal)
                ref_point = torch.cat([sList[4], sList[5], sList[6]])
                center = torch.cat([bList[0], bList[1], bList[2]])
                
                if ref_normal.matmul(ref_point.add(-center)) < 0:
                    ref_normal = -ref_normal
                pt_vertices = torch.FloatTensor(vertices)
                new_center = ref_normal.mul(2*abs(torch.sum(ref_point.add(-center).mul(ref_normal)))).add(center)
                new_vertices = pt_vertices.add(center - new_center).tolist()
                geo.append((new_vertices, reindex_faces(new_vertices, faces)))



    return geo

# def adjust_bbox(orig_target_bbox, adj_bbox):
#     overlap = [np.maximum(orig_target_bbox[0], adj_bbox[0]), np.minimum(orig_target_bbox[1], adj_bbox[1])]
#     difference = overlap[1] - overlap[0]
#     print(difference)

#     target_bbox = [None]* 2
#     target_bbox[0] = np.where(difference > 0, overlap[0], orig_target_bbox[0])
#     target_bbox[1] = np.where(difference > 0, overlap[1], orig_target_bbox[1])

#     target_bbox[0][1] = orig_target_bbox[0][1]
#     target_bbox[1][1] = orig_target_bbox[1][1]

#     target_corners = np.zeros([8, 3])
#     target_corners[0, :] = target_bbox[0]
#     target_corners[1, :] = np.array([target_bbox[0][0], target_bbox[0][1], target_bbox[1][2]])
#     target_corners[2, :] = np.array([target_bbox[0][0], target_bbox[1][1], target_bbox[0][2]])
#     target_corners[3, :] = np.array([target_bbox[0][0], target_bbox[1][1], target_bbox[1][2]])
#     target_corners[4, :] = np.array([target_bbox[1][0], target_bbox[0][1], target_bbox[0][2]])
#     target_corners[5, :] = np.array([target_bbox[1][0], target_bbox[0][1], target_bbox[1][2]])
#     target_corners[6, :] = np.array([target_bbox[1][0], target_bbox[1][1], target_bbox[0][2]])
#     target_corners[7, :] = target_bbox[1]

#     return target_bbox, target_corners

if __name__ == "__main__":
    # dataset = GRASSNewDataset('D:\\CMPT 764\\chairs_dataset',3)
    agg = Aggregator()
    mixer = Mixer('Chair', 5)
    # extractor = RandomizedExtractor(dataset)

    for i in range(5):
        mixer.reset_target()
        mixer.mix_parts()
        renderMeshFromParts(mixer.get_target_mesh().parts)
        
        
    # for i in range(len(mixer.dataset)):
    #     # mesh = mixer.dataset[i]

    #     # extractor.target = mesh

    #     #display info about mesh and parts
    #     # print(mesh)
    #     # for part in mesh.parts:
    #     #     print(part)

    #     # parts = agg.get_all_parts_by_label(mesh, PartLabel.LEG)

    #     # renderMeshFromParts(parts)

    #     boxes = []
    #     labels = []
    #     # for label in PartLabel:
    #     #     _, bbox = agg.get_super_bounding_box(mesh, label)
    #     #     if bbox is not None:
    #     #         boxes.append(bbox)
    #     #         labels.append(label)

    #     bboxes_dict = mixer.extractor.get_target_labels_with_bounding_box()
    #     print(bboxes_dict)

    #     for label in bboxes_dict:
    #         if label == PartLabel.BACK or label == PartLabel.LEG:
    #             leg_bbox = [bboxes_dict[label][1][0, :], bboxes_dict[label][1][-1, :]] ## min coords, max coords
    #             seat_bbox = [bboxes_dict[PartLabel.SEAT][1][0, :], bboxes_dict[PartLabel.SEAT][1][-1, :]] ## min coords, max coords

    #             target_bbox, target_corners = mixer.adjust_bbox(leg_bbox, seat_bbox)

    #             boxes.append(target_corners)
    #         else:
    #             boxes.append(bboxes_dict[label][1])

    #         labels.append(label)

    #     if len(boxes) > 0:
    #         show_obbs_from_bboxes(labels, boxes)

        
        # #display bounding boxes
        # show_obbs_from_parts(mesh.parts)

        # #shows how to get the parts associated with a label
        # export_parts_to_obj("test.obj", mesh.get_parts_from_label(PartLabel.LEG))

        # #can also export all parts
        # export_parts_to_obj("test2.obj", mesh.parts)

        # the bounding boxes should actually match each part properly too
    # grassdata = GRASSDataset('A:\\764dataset\\Chair',9)
    # for i in range(len(grassdata)):
    #     tree = grassdata[i]
    #     boxes = decode_structure(tree.root)
    #     #showGenshape(boxes)

    #     geometry = get_geo_tree(tree.root)
    #     vertices = []
    #     faces = []
    #     offsets = []
    #     for geo_pair in geometry:
    #         current_face_count = len(vertices)
    #         offsets.extend([current_face_count] * len(geo_pair[1]))
    #         vertices.extend(geo_pair[0])
    #         faces.extend(geo_pair[1])

    #     write_to_obj("test.obj", vertices, faces, offsets)
