from __future__ import print_function, division
from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
from grassdata import Tree
from views import view
import open3d as o3d

#https://datascience.stackexchange.com/questions/11430/how-to-annotate-labels-in-a-3d-matplotlib-scatter-plot
class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)

def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)

def tryPlot():
    cmap = plt.get_cmap('jet_r')
    fig = plt.figure()
    ax = Axes3D(fig)
    draw(ax, [-0.0152730000000000,-0.113074400000000,0.00867852000000000,0.766616000000000,0.483920000000000,0.0964542000000000,
               8.65505000000000e-06,-0.000113369000000000,0.999997000000000,0.989706000000000,0.143116000000000,7.65900000000000e-06], cmap(float(1)/7))
    draw(ax, [-0.310188000000000,0.188456800000000,0.00978854000000000,0.596362000000000,0.577190000000000,0.141414800000000,
               -0.331254000000000,0.943525000000000,0.00456327000000000,-0.00484978000000000,-0.00653891000000000,0.999967000000000], cmap(float(2)/7))
    draw(ax, [-0.290236000000000,-0.334664000000000,-0.328648000000000,0.322898000000000,0.0585966000000000,0.0347996000000000,
               -0.330345000000000,-0.942455000000000,0.0514932000000000,0.0432524000000000,0.0393726000000000,0.998095000000000], cmap(float(3)/7))
    draw(ax, [-0.289462000000000,-0.334842000000000,0.361558000000000,0.322992000000000,0.0593536000000000,0.0350418000000000,
               0.309240000000000,0.949730000000000,0.0485183000000000,-0.0511885000000000,-0.0343219000000000,0.998099000000000], cmap(float(4)/7))
    draw(ax, [0.281430000000000,-0.306584000000000,0.382928000000000,0.392156000000000,0.0409424000000000,0.0348472000000000,
               0.322342000000000,-0.942987000000000,0.0828920000000000,-0.0248683000000000,0.0791002000000000,0.996556000000000], cmap(float(5)/7))
    draw(ax, [0.281024000000000,-0.306678000000000,-0.366110000000000,0.392456000000000,0.0409366000000000,0.0348446000000000,
               -0.322608000000000,0.942964000000000,0.0821142000000000,0.0256742000000000,-0.0780031000000000,0.996622000000000], cmap(float(6)/7))
    draw(ax, [0.121108800000000,-0.0146729400000000,0.00279166000000000,0.681576000000000,0.601756000000000,0.0959706000000000,
               -0.986967000000000,-0.160173000000000,0.0155341000000000,0.0146809000000000,0.00650174000000000,0.999801000000000], cmap(float(7)/7))
    plt.show()

def draw(ax, cornerpoints, color):
    ax.plot([cornerpoints[0][0], cornerpoints[1][0]], [cornerpoints[0][1], cornerpoints[1][1]],
            [cornerpoints[0][2], cornerpoints[1][2]], c=color)
    ax.plot([cornerpoints[0][0], cornerpoints[2][0]], [cornerpoints[0][1], cornerpoints[2][1]],
            [cornerpoints[0][2], cornerpoints[2][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[3][0]], [cornerpoints[1][1], cornerpoints[3][1]],
            [cornerpoints[1][2], cornerpoints[3][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[3][0]], [cornerpoints[2][1], cornerpoints[3][1]],
            [cornerpoints[2][2], cornerpoints[3][2]], c=color)
    ax.plot([cornerpoints[4][0], cornerpoints[5][0]], [cornerpoints[4][1], cornerpoints[5][1]],
            [cornerpoints[4][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[4][0], cornerpoints[6][0]], [cornerpoints[4][1], cornerpoints[6][1]],
            [cornerpoints[4][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[5][0], cornerpoints[7][0]], [cornerpoints[5][1], cornerpoints[7][1]],
            [cornerpoints[5][2], cornerpoints[7][2]], c=color)
    ax.plot([cornerpoints[6][0], cornerpoints[7][0]], [cornerpoints[6][1], cornerpoints[7][1]],
            [cornerpoints[6][2], cornerpoints[7][2]], c=color)
    ax.plot([cornerpoints[0][0], cornerpoints[4][0]], [cornerpoints[0][1], cornerpoints[4][1]],
            [cornerpoints[0][2], cornerpoints[4][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[5][0]], [cornerpoints[1][1], cornerpoints[5][1]],
            [cornerpoints[1][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[6][0]], [cornerpoints[2][1], cornerpoints[6][1]],
            [cornerpoints[2][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[3][0], cornerpoints[7][0]], [cornerpoints[3][1], cornerpoints[7][1]],
            [cornerpoints[3][2], cornerpoints[7][2]], c=color)

def showGenshape(genshape):
    recover_boxes = genshape

    fig = plt.figure(0)
    cmap = plt.get_cmap('jet_r')
    ax = Axes3D(fig)
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)

    for jj in range(len(recover_boxes)):
        label, cornerpoints = recover_boxes[jj]
        annotate3D(ax, s=Tree.NodeLabel(label), xyz=cornerpoints.mean(axis=0), fontsize=10, xytext=(-3,3), textcoords='offset points', ha='center',va='center') 
        draw(ax, cornerpoints, cmap(float(jj)/len(recover_boxes)))
    plt.show()

def show_obbs_from_parts(parts):
    fig = plt.figure(0)
    cmap = plt.get_cmap('jet_r')
    ax = Axes3D(fig)
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)
    n_boxes = len(parts)
    i = 0
    for part in parts:
        label = part.label
        cornerpoints = part.bounding_box
        annotate3D(ax, s=label, xyz=cornerpoints.mean(axis=0), fontsize=10, xytext=(-3,3), textcoords='offset points', ha='center',va='center') 
        draw(ax, cornerpoints, cmap(float(i)/n_boxes))
        i+=1
    plt.show()

def show_obbs_from_bboxes(labels, bboxes):
    fig = plt.figure(0)
    cmap = plt.get_cmap('jet_r')
    ax = Axes3D(fig)
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)
    n_boxes = len(labels)
    i = 0
    for label, bbox in zip(labels, bboxes):
        annotate3D(ax, s=label, xyz=bbox.mean(axis=0), fontsize=10, xytext=(-3,3), textcoords='offset points', ha='center',va='center') 
        draw(ax, bbox, cmap(float(i)/n_boxes))
        i+=1
    plt.show()

def showGenshape(genshape):
    recover_boxes = genshape

    fig = plt.figure(0)
    cmap = plt.get_cmap('jet_r')
    ax = Axes3D(fig)
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)

    for jj in range(len(recover_boxes)):
        label, cornerpoints = recover_boxes[jj]
        annotate3D(ax, s=Tree.NodeLabel(label), xyz=cornerpoints.mean(axis=0), fontsize=10, xytext=(-3,3), textcoords='offset points', ha='center',va='center') 
        draw(ax, cornerpoints, cmap(float(jj)/len(recover_boxes)))

    plt.show()

def reindex_faces(part_faces, new_offset):
    part_faces_np = np.asarray(part_faces, dtype=np.int32)
    part_faces_np = part_faces_np - np.amin(part_faces_np) + new_offset

    return part_faces_np


def renderMesh(part_meshes):
    vertices = []
    faces = []
    current_min_idx = 0
    # for part_vertices, part_faces, min_vertex_idx, max_vertex_idx in part_meshes:
    #     vertices.append(np.asarray(part_vertices, dtype=np.float64))
    #     faces.append(modify_faces(part_faces, min_vertex_idx, current_min_idx))
    #     current_min_idx += len(part_vertices)

    for part_mesh in part_meshes:
        part_vertices = part_mesh[0]
        part_faces = part_mesh[1]
        min_vertex_idx = part_mesh[2]
        max_vertex_idx = part_mesh[3]

        v_points = np.asarray(part_vertices, dtype=np.float64)
        vertices.append(v_points)

        part_faces_np = modify_faces(part_faces, min_vertex_idx, current_min_idx)
        faces.append(part_faces_np)
        current_min_idx += len(part_vertices)

        if len(part_mesh) > 4:
            symm_op = part_mesh[4]
            # print(symm_op)

            if symm_op == 'rotate':
                op_matrices = part_mesh[5]
                rot_center = part_mesh[6]
                 
                v_points_t = v_points.T - rot_center

                for rot_mat in op_matrices:
                    v_points_rot = np.matmul(rot_mat, v_points_t) + rot_center
                    vertices.append(v_points_rot.T)

                    part_faces_np = modify_faces(part_faces, min_vertex_idx, current_min_idx)
                    faces.append(part_faces_np)
                    current_min_idx += len(part_vertices)

            elif symm_op == 'translate':
                op_vectors = part_mesh[5]

                for trans_vec in op_vectors:
                    v_points_trans = v_points.T + trans_vec
                    vertices.append(v_points_trans.T)

                    part_faces_np = modify_faces(part_faces, min_vertex_idx, current_min_idx)
                    faces.append(part_faces_np)
                    current_min_idx += len(part_vertices)

            elif symm_op == 'reflect':
                ref_normal = part_mesh[5]
                ref_point = part_mesh[6]

                v_points_t = ref_point - v_points.T
                v_points_reflect = (ref_normal * (2 * np.absolute(np.repeat(np.sum(v_points_t * ref_normal, axis=0, keepdims=True), ref_normal.shape[0], axis=0)))) + v_points.T
                vertices.append(v_points_reflect.T)

                part_faces_np = modify_faces(part_faces, min_vertex_idx, current_min_idx)
                faces.append(part_faces_np)
                current_min_idx += len(part_vertices)

    if len(vertices) > 0 and len(faces) > 0:
        vertices_np = np.concatenate(vertices, axis=0)
        faces_np = np.concatenate(faces, axis=0)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
        mesh.triangles = o3d.utility.Vector3iVector(faces_np)

        mesh.compute_vertex_normals()

        o3d.visualization.draw_geometries([mesh])

        print('Done')

# Change made here
'''
def renderMeshFromParts(parts):
    vertex_list = []
    faces_list = []
    current_offset = 0
    for part in parts:
        if part.render:
            vertex_list.append(np.asarray(part.vertices, dtype=np.float64))
            if part.faces!=None:
                faces_list.append(reindex_faces(part.faces, current_offset))
            current_offset += len(part.vertices)
    
    if part.faces!=None:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.concatenate(vertex_list, axis=0))
        mesh.triangles = o3d.utility.Vector3iVector(np.concatenate(faces_list, axis=0))
        mesh.compute_vertex_normals()
    else:
        mesh = o3d.geometry.PointCloud()
        mesh.points = o3d.utility.Vector3dVector(np.concatenate(np.asarray(vertex_list), axis=0))
        mesh.estimate_normals()

    #o3d.io.write_point_cloud("final.pcd", mesh)
    o3d.visualization.draw_geometries([mesh])
'''
def renderMeshFromParts(parts, filename = 'screenshot.png'):
    vertex_list = []
    faces_list = []
    current_offset = 0
    if parts[0].faces!=None:
        mesh = o3d.geometry.TriangleMesh()
        full = o3d.geometry.TriangleMesh()
    else:
        pcd = o3d.geometry.PointCloud()
        full = o3d.geometry.PointCloud()
        
    for part in parts:
        if part.render:
            if part.faces!=None:
                vertex_list = (np.asarray(part.vertices, dtype=np.float64))
                faces_list = (reindex_faces(part.faces, current_offset))
                #current_offset += len(part.vertices)
                mesh.vertices = o3d.utility.Vector3dVector(vertex_list)
                mesh.triangles = o3d.utility.Vector3iVector(faces_list)
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color(part.colour)
                full = full + mesh
                #mesh = None
                mesh = o3d.geometry.TriangleMesh()
            else:
                vertex_list = (np.asarray(part.vertices, dtype=np.float64))
                pcd.points = o3d.utility.Vector3dVector(np.asarray(vertex_list))
                pcd.estimate_normals()
                pcd.paint_uniform_color(part.colour)
                full = full + pcd
                #pcd = None
                pcd = o3d.geometry.PointCloud()
                
    #o3d.io.write_point_cloud("final.pcd", full)
    #o3d.visualization.draw_geometries([full])
    view(full, filename)
    
