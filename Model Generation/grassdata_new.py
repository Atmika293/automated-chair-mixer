import os
from enum import Enum
import numpy as np
import torch
from torch.utils import data
from scipy.io import loadmat
from utils import parse_obb_file, parse_part_from_obj_2, reindex_faces, export_parts_to_obj, render_obj
from draw3dobb import show_obbs_from_parts

class PartLabel(Enum):
    """Types of parts"""
    BACK = 0
    SEAT = 1
    LEG = 2
    ARMREST = 3

class Part(object):
    """Represents a physical part of a mesh, includes connectivity, label and geometric information"""
    def __init__(self, label:PartLabel, vertices, faces, bounding_box):
        self.vertices = vertices
        self.faces = faces
        self.label = label
        self.connectivity = []
        self.bounding_box = bounding_box
        self.render = True
        self.colour = [0.2, 0.2, 0.25] # Default colour for all meshes is grey

    def __str__(self):
        return 'Part of type {} has {} connections,v: {} f: {}'.format(self.label, len(self.connectivity), len(self.vertices), len(self.faces))

class Mesh(object):
    """Collection of parts, maintained in a list"""
    def __init__(self, obb_file_path, obj_file_path):
        self.parts:[Part] = []
        self.original_file:str = obb_file_path
        self.original_model:str = obj_file_path

        part_count, obbs, connectivity, labels = parse_obb_file(obb_file_path)
        order, parts_geometry = parse_part_from_obj_2(obj_file_path)
        for i in range(0, part_count):
            geo_index = order.index(i)
            (vertices, faces) = parts_geometry[geo_index]
            faces = reindex_faces(vertices, faces) # now each part can be exported as obj, to combine just make sure to update the face indices
            label = PartLabel(labels[i])
            obb = obbs[i]
            new_part = Part(label, vertices, faces, obb)
            self.parts.append(new_part)

        # process connectivity and symmetry, probably needs to be looked into
        for connection in connectivity:
            (start, end) = connection
            self.parts[start].connectivity.append(self.parts[end])
            
        self.colour = [0.2, 0.2, 0.25] # Default colour for all meshes is grey
        self.set_mesh_and_parts_colour([0.2, 0.2, 0.25])

    def __str__(self):
        return 'Mesh has {} parts, obb file: {} obj file: {}'.format(len(self.parts), self.original_file, self.original_model)

    def get_parts_from_label(self, label:PartLabel) -> [Part]:
        result = []
        for part in self.parts:
            if part.label == label:
                result.append(part)
        return result

    def set_mesh_and_parts_colour(self, col):
        self.colour = col
        for part in self.parts:
            part.colour=col
  
class GRASSNewDataset(data.Dataset,):
    """Dataset of PartNet meshes with symmetry"""
    def __init__(self, dir:str, models_num:int=-1, model_list=None):
        self.dir = dir
        self.meshes = []
        obb_path = os.path.join(dir, 'obbs')
        obj_path = os.path.join(dir, 'models')
        files = os.listdir(obb_path)
        file_count = len(files)

        if model_list is None:
            if models_num < 0:
                model_list = [files[i] for i in range(file_count)]
            else:
                model_list = [files[i] for i in range(min(models_num, file_count))]
        else:
            model_list = [str(mn) + '.obb' for mn in model_list]


        for filename in model_list:
            file = os.path.join(obb_path, filename)
            obj = os.path.join(obj_path, filename[:-3] + "obj")
            self.meshes.append(Mesh(file, obj))

    def __getitem__(self, index):
        mesh = self.meshes[index]
        return mesh

    def __len__(self):
        return len(self.meshes)
