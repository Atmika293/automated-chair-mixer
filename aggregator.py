import open3d as o3d
import numpy as np

from grassdata_new import Part, Mesh, PartLabel

class Aggregator(object):
	def __init__(self):
		pass

	def get_number_of_parts_by_label(self, mesh, label):
		parts_num = 0
		for part in mesh.parts:
			if part.label == label:
				parts_num += 1

		return parts_num

	def get_all_parts_by_label(self, mesh, label):
		parts = []

		for part in mesh.parts:
			if part.label == label:
				parts.append(part)

		return parts

	def get_super_bounding_box(self, mesh, label):
		parts = self.get_all_parts_by_label(mesh, label)

		if len(parts) > 0:
			boxes = np.zeros([len(parts), 8, 3])

			for idx, part in enumerate(parts):
				boxes[idx, :, :] = part.bounding_box

			min_coords = np.amin(boxes, axis=(0,1), keepdims=False)
			max_coords = np.amax(boxes, axis=(0,1), keepdims=False)

			super_bounding_box = np.zeros([8, 3])

			super_bounding_box[0, :] = min_coords
			super_bounding_box[1, :] = np.array([min_coords[0], min_coords[1], max_coords[2]])
			super_bounding_box[2, :] = np.array([min_coords[0], max_coords[1], min_coords[2]])
			super_bounding_box[3, :] = np.array([min_coords[0], max_coords[1], max_coords[2]])
			super_bounding_box[4, :] = np.array([max_coords[0], min_coords[1], min_coords[2]])
			super_bounding_box[5, :] = np.array([max_coords[0], min_coords[1], max_coords[2]])
			super_bounding_box[6, :] = np.array([max_coords[0], max_coords[1], min_coords[2]])
			super_bounding_box[7, :] = max_coords

			return [len(parts), super_bounding_box]

		return [0, None]

	def get_parts_with_bounding_box(self, mesh, label):
		parts = self.get_all_parts_by_label(mesh, label)

		if len(parts) > 0:
			boxes = np.zeros([len(parts), 8, 3])

			for idx, part in enumerate(parts):
				boxes[idx, :, :] = part.bounding_box

			min_coords = np.amin(boxes, axis=(0,1), keepdims=False)
			max_coords = np.amax(boxes, axis=(0,1), keepdims=False)

			super_bounding_box = np.zeros([8, 3])

			super_bounding_box[0, :] = min_coords
			super_bounding_box[1, :] = np.array([min_coords[0], min_coords[1], max_coords[2]])
			super_bounding_box[2, :] = np.array([min_coords[0], max_coords[1], min_coords[2]])
			super_bounding_box[3, :] = np.array([min_coords[0], max_coords[1], max_coords[2]])
			super_bounding_box[4, :] = np.array([max_coords[0], min_coords[1], min_coords[2]])
			super_bounding_box[5, :] = np.array([max_coords[0], min_coords[1], max_coords[2]])
			super_bounding_box[6, :] = np.array([max_coords[0], max_coords[1], min_coords[2]])
			super_bounding_box[7, :] = max_coords

			return parts, super_bounding_box

		return parts, None




