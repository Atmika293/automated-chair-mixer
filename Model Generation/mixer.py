from grassdata_new import Part, Mesh
from grassdata_new import PartLabel, GRASSNewDataset
from extractor import RandomizedExtractor
from draw3dobb import renderMeshFromParts

import numpy as np
import random
import copy
import open3d as o3d

from utils import restore_vertex_order
from warp import warp_part, get_mesh_from_part, pcd_to_part, convert_to_pcd, move_part, renderMeshFromParts_new

class Mixer(object):
	def __init__(self, dataset_path, model_num, model_list=None):
		self.dataset = GRASSNewDataset(dataset_path, max(2, model_num), model_list)
		self.extractor = RandomizedExtractor(self.dataset)

	def __adjust_bbox(self, orig_target_bbox, adj_bbox, shift_factor=0.0):
		overlap = [np.maximum(orig_target_bbox[0], adj_bbox[0]), np.minimum(orig_target_bbox[1], adj_bbox[1])]
		difference = overlap[1] - overlap[0]

		target_bbox = [None] * 2
		target_bbox[0] = np.where(difference > 0, overlap[0], orig_target_bbox[0])
		target_bbox[1] = np.where(difference > 0, overlap[1], orig_target_bbox[1])

		target_center = (orig_target_bbox[0] + orig_target_bbox[1]) / 2
		adj_center = (adj_bbox[0] + adj_bbox[1]) / 2

		shift = shift_factor * (adj_bbox[1][1] - adj_bbox[0][1])
		if target_center[1] > adj_center[1]:
			shift = - (shift + (target_bbox[0][1] - adj_bbox[1][1]))
		else:
			shift = shift + (adj_bbox[0][1] - target_bbox[1][1])

		## do not change in y-direction
		target_bbox[0][1] = orig_target_bbox[0][1] + shift
		target_bbox[1][1] = orig_target_bbox[1][1] + shift

		target_corners = np.zeros([8, 3])
		target_corners[0, :] = target_bbox[0]
		target_corners[1, :] = np.array([target_bbox[0][0], target_bbox[0][1], target_bbox[1][2]])
		target_corners[2, :] = np.array([target_bbox[0][0], target_bbox[1][1], target_bbox[0][2]])
		target_corners[3, :] = np.array([target_bbox[0][0], target_bbox[1][1], target_bbox[1][2]])
		target_corners[4, :] = np.array([target_bbox[1][0], target_bbox[0][1], target_bbox[0][2]])
		target_corners[5, :] = np.array([target_bbox[1][0], target_bbox[0][1], target_bbox[1][2]])
		target_corners[6, :] = np.array([target_bbox[1][0], target_bbox[1][1], target_bbox[0][2]])
		target_corners[7, :] = target_bbox[1]

		return target_bbox, target_corners

	def __stretch_parts(self, target_parts, adj_parts, adj_above, candidate_size_ratio=0.05, delta_value=0.05, delta_step=0.01):
		target_vertices_list = []
		target_vertices_order = []
		for part in target_parts:
			vertices = np.asarray(part.vertices, dtype=np.float64)
			order = np.argsort(vertices[:, 1])
			target_vertices_list.append(vertices[order, :])
			target_vertices_order.append(order)

		vertices = []
		for part in adj_parts:
			vertices.append(np.asarray(part.vertices, dtype=np.float64))
		adj_vertices = np.concatenate(vertices, axis=0)	

		for idx, target_part_vertices in enumerate(target_vertices_list):
			# print(target_part_vertices.shape)
			# print(adj_vertices.shape)
			candidate_size = int(candidate_size_ratio * target_part_vertices.shape[0])

			if adj_above:
				target_vertices = target_part_vertices[-candidate_size:, :]
			else:
				target_vertices = target_part_vertices[:candidate_size, :]

			#print(target_vertices.shape)
			
			for p in range(target_vertices.shape[0]):
				point = target_vertices[p, :]
				
				delta = delta_value
				while True:
					indices = np.where((adj_vertices[:, 0] < (point[0] + delta)) & (adj_vertices[:, 0] > (point[0] - delta)) \
						& (adj_vertices[:, 2] < (point[2] + delta)) & (adj_vertices[:, 2] > (point[2] - delta)))[0]
					# print(indices.shape)
					candidate_points = adj_vertices[indices, :]
					# print(candidate_points.shape)
					if candidate_points.shape[0] > 0:
						break
					else:
						delta += delta_step
				
				distance = np.sum((candidate_points - point) ** 2, axis=1, keepdims=False)
				target_vertices[p, :] = candidate_points[np.argmin(distance, ), :]

			if adj_above:
				target_part_vertices[-candidate_size:, :] = target_vertices
			else:
				target_part_vertices[:candidate_size, :] = target_vertices

			target_parts[idx].vertices = restore_vertex_order(target_part_vertices, target_vertices_order[idx]).tolist()

		return target_parts

	def resize_parts(self, warped_parts, orig_target_bbox, mod_target_bbox, label):
		source_center = (orig_target_bbox[0] + orig_target_bbox[1]) / 2
		target_center = (mod_target_bbox[0] + mod_target_bbox[1]) / 2

		trans1_matrix = np.eye(4, dtype=np.float64)
		trans1_matrix[:3, -1] = -source_center

		resize_matrix = np.eye(4, dtype=np.float64)
		resize_factor = np.absolute((mod_target_bbox[1] - mod_target_bbox[0]) / (orig_target_bbox[1] - orig_target_bbox[0]))
		resize_matrix[:3, :3] = np.diag(resize_factor)

		trans2_matrix = np.eye(4, dtype=np.float64)
		trans2_matrix[:3, -1] = target_center

		transform_matrix = np.matmul(trans2_matrix, np.matmul(resize_matrix, trans1_matrix))

		for part in warped_parts:
			points = np.asarray(part.vertices, dtype=np.float64)
			homogeneous_points = np.ones([4, points.shape[0]])
			homogeneous_points[:3, :] = points.T
			homogeneous_points = np.matmul(transform_matrix, homogeneous_points)
			homogeneous_points = homogeneous_points / homogeneous_points[-1, :]
			homogeneous_points = homogeneous_points.T
			part.vertices = homogeneous_points[:, :3].tolist()

		self.extractor.replace_target_parts_by_label(label, warped_parts)	

	def replace_part(self, source_parts, source_bbox, target_bbox, label):
		source_center = (source_bbox[0] + source_bbox[1]) / 2
		target_center = (target_bbox[0] + target_bbox[1]) / 2

		trans1_matrix = np.eye(4, dtype=np.float64)
		trans1_matrix[:3, -1] = -source_center

		resize_matrix = np.eye(4, dtype=np.float64)
		resize_factor = np.absolute((target_bbox[1] - target_bbox[0]) / (source_bbox[1] - source_bbox[0]))
		resize_matrix[:3, :3] = np.diag(resize_factor)

		trans2_matrix = np.eye(4, dtype=np.float64)
		trans2_matrix[:3, -1] = target_center

		transform_matrix = np.matmul(trans2_matrix, np.matmul(resize_matrix, trans1_matrix))

		for part in source_parts:
			points = np.asarray(part.vertices, dtype=np.float64)
			homogeneous_points = np.ones([4, points.shape[0]])
			homogeneous_points[:3, :] = points.T
			homogeneous_points = np.matmul(transform_matrix, homogeneous_points)
			homogeneous_points = homogeneous_points / homogeneous_points[-1, :]
			homogeneous_points = homogeneous_points.T
			part.vertices = homogeneous_points[:, :3].tolist()

			points = part.bounding_box
			homogeneous_points = np.ones([4, points.shape[0]])
			homogeneous_points[:3, :] = points.T
			homogeneous_points = np.matmul(transform_matrix, homogeneous_points)
			homogeneous_points = homogeneous_points / homogeneous_points[-1, :]
			homogeneous_points = homogeneous_points.T
			part.bounding_box = homogeneous_points[:, :3]

		self.extractor.make_target_parts_invisible_by_label(label)

		self.extractor.add_parts_to_target(source_parts)

	def join_parts(self):
		target_dict = self.extractor.get_target_labels_with_parts()

		target_parts = []
		for label in target_dict:
			if label == PartLabel.LEG:
				target_parts.extend(self.__stretch_parts(target_dict[label], target_dict[PartLabel.SEAT], True))
			elif label == PartLabel.BACK:
				target_parts.extend(self.__stretch_parts(target_dict[label], target_dict[PartLabel.SEAT], False))
			elif label == PartLabel.ARMREST:
				target_parts.extend(self.__stretch_parts(target_dict[label], target_dict[PartLabel.SEAT], False))
			else:
				target_parts.extend(target_dict[label])

		self.extractor.target.parts = target_parts#self.extractor.smooth_part_meshes(target_parts)

	def mix_parts(self):
		target_bboxes_dict = self.extractor.get_target_labels_with_bounding_box()
		#renderMeshFromParts(self.extractor.target.parts)

		i = 0
		print("Entering mix parts")
		for label in target_bboxes_dict:
		# label = PartLabel.BACK
			parts, bbox = self.extractor.find_source_part(self.dataset, label)
			if bbox is None:
				#'''
				target_parts = self.extractor.get_target_parts_by_label(label)
				for part in target_parts:
					bb = part.bounding_box
					col = part.colour
					idx = target_parts.index(part)
					m = get_mesh_from_part(part)
					pcd = convert_to_pcd(m)
                    
					#o3d.io.write_point_cloud("part"+str(i)+".pcd", pcd)
					#o3d.visualization.draw_geometries([pcd], 'TARGET PART')
					print(i)
					i = i+1
                    
					result = pcd_to_part(pcd, label, bb)
					result.colour = col
					print("target ", result.colour)
					target_parts[idx] = result
					print("Target part taken (as no replacement was found)!")
				self.extractor.replace_target_parts_by_label(label, target_parts)
				#'''
				continue
			
			source_bbox = [bbox[0, :], bbox[-1, :]]
            
			#'''
			# Warp if number of parts with the same label is equal
			if len(parts) == target_bboxes_dict[label][0]:
				target_parts = self.extractor.get_target_parts_by_label(label)
				for s, t, bb in zip(parts, target_parts, bbox):
					col = s.colour
					idx = parts.index(s)
					s = get_mesh_from_part(s)
					t = get_mesh_from_part(t)
					result = warp_part(s, t)
                    
					#o3d.io.write_point_cloud("part"+str(i)+".pcd", result)
					#o3d.visualization.draw_geometries([result], 'WARPED PART')
					print(i)
					i = i+1
					
					result = pcd_to_part(result, label, bb) #bbox
					result.colour = col
					print("warp ", result.colour)
					target_parts[idx] = result
					print("Warped a part!")
					
					#o3d.visualization.draw_geometries([source, target, result])
					# Example on how to use move _part: 
                    # result = move_part(result, seat, axis=1, n=5000)
                
                # 
				#self.extractor.replace_target_parts_by_label(label, target_parts)
				#'''
				if label == PartLabel.LEG or label == PartLabel.BACK or label == PartLabel.ARMREST:  
					orig_target_bbox = [target_bboxes_dict[label][1][0, :], target_bboxes_dict[label][1][-1, :]] ## min coords, max coords
					seat_bbox = [target_bboxes_dict[PartLabel.SEAT][1][0, :], target_bboxes_dict[PartLabel.SEAT][1][-1, :]] ## min coords, max coords
					mod_target_bbox, target_corners = self.__adjust_bbox(orig_target_bbox, seat_bbox)
					target_bboxes_dict[label][1] = target_corners
					
					self.resize_parts(target_parts, orig_target_bbox, mod_target_bbox, label)	

				else:
					self.extractor.replace_target_parts_by_label(label, target_parts)
				#'''
                
			# Else replace using bbox, but convert part to mesh to pcd part
			else:
				#continue
				for part, bb in zip(parts, bbox):
					col = part.colour
					idx = parts.index(part)
					m = get_mesh_from_part(part)
					pcd = convert_to_pcd(m)
                    
					#o3d.io.write_point_cloud("part"+str(i)+".pcd", pcd)
					#o3d.visualization.draw_geometries([pcd], 'REPLACED PART')
					print(i)
					i = i+1
                    
					result = pcd_to_part(pcd, label, bb)
					result.colour = col
					print("replace ", result.colour)
					parts[idx] = result
					print("Replaced a part!")
			## readjusting bbox to adjacent part
			#'''

				## if label == LEG or label == BACK, adjust to SEAT
				if label == PartLabel.LEG or label == PartLabel.BACK:
					orig_target_bbox = [target_bboxes_dict[label][1][0, :], target_bboxes_dict[label][1][-1, :]] ## min coords, max coords
					seat_bbox = [target_bboxes_dict[PartLabel.SEAT][1][0, :], target_bboxes_dict[PartLabel.SEAT][1][-1, :]] ## min coords, max coords
					target_bbox, target_corners = self.__adjust_bbox(orig_target_bbox, seat_bbox)
					target_bboxes_dict[label][1] = target_corners
					
					self.replace_part(parts, source_bbox, target_bbox, label)

				elif label == PartLabel.ARMREST:
					if random.random() > 0.5: ##delete armrests
						self.extractor.make_target_parts_invisible_by_label(label)
					else:
						self.replace_part(parts, source_bbox, [target_bboxes_dict[label][1][0, :], target_bboxes_dict[label][1][-1, :]], label)	

				else:
					target_bbox = [target_bboxes_dict[label][1][0, :], target_bboxes_dict[label][1][-1, :]]
					self.replace_part(parts, source_bbox, target_bbox, label)

			#self.join_parts()

	def get_target_mesh(self):
		return self.extractor.target

	def reset_target(self):
		self.extractor.reset_target(self.dataset)
        
	def show_data_meshes(self):
        # To see all the source meshes
        # And the target in red
		#print(self.extractor.target.colour)
		meshes = self.extractor.get_source_meshes()
		#print(len(meshes))
		renderMeshFromParts(self.extractor.target.parts, 'source.png')
		
		for i in range (len(meshes)):
			renderMeshFromParts(meshes[i].parts, 'source'+str(i)+'.png')