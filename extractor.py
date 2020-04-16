import random
import open3d as o3d

from grassdata_new import Part, Mesh, PartLabel, GRASSNewDataset
from aggregator import Aggregator

import copy
import numpy as np

from utils import reindex_faces

class RandomizedExtractor(object):
	def __init__(self, dataset):
		self.aggregator = Aggregator()
		self.dataset = dataset
		self.target_index, target = self.__choose_random_sample(dataset)
		self.target = copy.deepcopy(target)
		self.target.set_mesh_and_parts_colour([1, 0, 0])
		self.set_colour_coded_sources(dataset) ############# ADD TO RESET TARGET?? ##############

	def __choose_random_sample(self, dataset):
		data_size = len(dataset)

		sample_idx = random.randint(0, data_size-1)
		sample = dataset[sample_idx]

		return sample_idx, sample

	def reset_target(self, dataset):
		idx = self.target_index
		while idx == self.target_index:
			self.target_index, target = self.__choose_random_sample(dataset)
			
		self.target = copy.deepcopy(target)
		self.set_colour_coded_sources(self.dataset) 

	def get_target_labels_with_bounding_box(self):
		assert(self.target is not None)
		
		bboxes_dict = {}
		for part in self.target.parts:
			if part.label not in bboxes_dict:
				bboxes_dict[part.label] = self.aggregator.get_super_bounding_box(self.target, part.label)

		return bboxes_dict

	def get_target_labels_with_parts(self):
		assert(self.target is not None)
		
		bboxes_dict = {}
		for part in self.target.parts:
			if part.label not in bboxes_dict:
				bboxes_dict[part.label] = self.aggregator.get_all_parts_by_label(self.target, part.label)

		return bboxes_dict

	def get_target_parts_by_label(self, label):
		return self.aggregator.get_all_parts_by_label(self.target, label)

	def make_target_parts_invisible_by_label(self, label):
		assert(self.target is not None)

		for part in self.target.parts:
			if part.label == label:
				part.render = False

	def add_parts_to_target(self, parts):
		assert(self.target is not None)

		self.target.parts.extend(parts)

	def replace_target_parts_by_label(self, label, parts):
		assert(self.target is not None)

		target_parts = []
		for part in self.target.parts:
			if part.label != label:
				target_parts.append(part)

		target_parts.extend(parts)
		self.target.parts = target_parts

	def find_source_part(self, dataset, label, iter_limit=10):
		bbox = None
		idx = self.target_index

		iter_count = 0
		while bbox is None or idx == self.target_index:
			idx, source_mesh = self.__choose_random_sample(dataset)
			parts, bbox = self.aggregator.get_parts_with_bounding_box(source_mesh, label)

			iter_count += 1
			if iter_count == iter_limit:
				break

		return parts, bbox

	def __reindex_faces(part_faces, new_offset=0):
		part_faces_np = np.asarray(part_faces, dtype=np.int32)
		part_faces_np = part_faces_np - np.amin(part_faces_np) + new_offset

		return part_faces_np

	def get_triangle_part_meshes(self, parts):
		triangle_meshes = []

		for part in parts:
			tri_mesh = o3d.geometry.TriangleMesh()
			tri_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(part.vertices, dtype=np.float64))
			tri_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(part.faces, dtype=np.int32))

			triangle_meshes.append(tri_mesh)

		return triangle_meshes

	def smooth_part_meshes(self, parts):
		triangle_meshes = self.get_triangle_part_meshes(parts)

		for part, mesh in zip(parts, triangle_meshes):
			mesh.filter_smooth_laplacian(number_of_iterations=3)

			part.vertices = np.asarray(mesh.vertices, dtype=np.float64).tolist()
			part.faces = np.asarray(mesh.triangles, dtype=np.int32).tolist()

		return parts


	def set_colour_coded_sources(self, dataset):
		data_size = len(dataset)
		cols = [[0.2, 0.2, 0.6],[0.2, 0.7, 0.3],[1, 0.7, 0.2],[0.5, 0.1, 0.5]]
		# List of colours
		# 1: Dark Blue, 2: Green, 3: Yellow, 4: Purple
		i=0

		for idx in range (data_size):
			if idx == self.target_index: # For target
				self.target.set_mesh_and_parts_colour([1,0,0])
				print("Setting colour for target with colour red at idx ", idx)
				continue
			mesh = dataset[idx]
			print("Setting colour for ", idx, " with colour number ", i)
			mesh.set_mesh_and_parts_colour(cols[i])
			i = i + 1

	def get_source_meshes(self):
		data_size = len(self.dataset)
		meshes = []
		i=0

		for idx in range (data_size):
			if idx == self.target_index:
				continue
			meshes.append(self.dataset[idx])
		return meshes




