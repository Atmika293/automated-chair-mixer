import copy
import numpy as np
import open3d as o3d
from probreg import cpd
from probreg import l2dist_regs
import math
#import pclpy
#from pclpy import pcl
import time
from grassdata_new import Part, PartLabel
from draw3dobb import reindex_faces

def renderMeshFromParts_new(parts):
    vertex_list = []
    seat = None
    pcd = o3d.geometry.PointCloud()
    full = o3d.geometry.PointCloud()
    
    # getting the seat
    for part in parts:
        if part.render:
            if part.label == PartLabel.SEAT:
                seat = part_to_pcd(part)
            
    
    for part in parts:
        if part.render:
            if part.label == PartLabel.BACK or part.label == PartLabel.ARMREST:
                print("Moving back or arm")
                bb=part.bounding_box
                lab = part.label
                col = part.colour
                part = part_to_pcd(part)
                part = move_part(part, seat, axis=1, n=1000, dir='down')
                part = pcd_to_part(part, lab, bb) 
                part.colour = col
                
            if part.label == PartLabel.LEG:
                print("Moving leg")
                bb=part.bounding_box
                lab = part.label
                col = part.colour
                part = part_to_pcd(part)
                part = move_part(part, seat, axis=1, n=1000, dir='up')
                part = pcd_to_part(part, lab, bb)
                part.colour = col
            
            vertex_list = (np.asarray(part.vertices, dtype=np.float64))
            pcd.points = o3d.utility.Vector3dVector(np.asarray(vertex_list))
            pcd.estimate_normals()
            pcd.paint_uniform_color(part.colour)
            full = full + pcd
            #pcd = None
            pcd = o3d.geometry.PointCloud()
        

    o3d.io.write_point_cloud("final.ply", full)
    o3d.visualization.draw_geometries([full])


def move_part(mesh, target, axis=1, n=1000, dir='down'):
    '''
    Here mesh is source, target is the mesh to move towards
    Axis is which axis to move it in (x y or z)
    n is which value distance to take, around 1000 is on an average a good value
    '''
    dist = mesh.compute_point_cloud_distance(target)
    dist = np.sort(dist)
    if n>np.shape(dist)[0]:
        n = np.shape(dist)[0]-10
    print(dist[0], dist[n])
    op = np.asarray(mesh.points)
    op_axis = op[:,axis]
    if dir =='down':
        op_axis = op_axis - dist[n]
    elif dir =='up':
        op_axis = op_axis + dist[n]
    op[:,axis] = op_axis
    mesh.points = o3d.utility.Vector3dVector(op)
    
    return mesh

def part_to_pcd(part):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(part.vertices))
    pcd.estimate_normals()
    return pcd
        

def pcd_to_part(pcd, label, bb):
    vertices = list(np.asarray(pcd.points))
    faces = None
    part = Part(label, vertices, faces, bb)
    return part

def get_mesh_from_part(part):
    vertex_list = []
    faces_list = []
    current_offset = 0
    if part.render:
        vertex_list.append(np.asarray(part.vertices, dtype=np.float64))
        faces_list.append(reindex_faces(part.faces, current_offset))
        current_offset += len(part.vertices)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.concatenate(vertex_list, axis=0))
    mesh.triangles = o3d.utility.Vector3iVector(np.concatenate(faces_list, axis=0))

    mesh.compute_vertex_normals()

    return mesh

def convert_to_pcd(source_mesh):
    # Convert mesh to point cloud
    source = o3d.geometry.PointCloud()
    
    source = source_mesh.sample_points_uniformly(2*int(1e5))
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
  
    return source


def affine_cpd(source, target, threshold=0.5):
    # Affine CPD using probreg
    #### NOTE:
    #### Increase threshold value to increase processing speed
    #### Results in more downsampled point cloud
    
    source = source.voxel_down_sample(voxel_size=threshold) 
    target = target.voxel_down_sample(voxel_size=threshold)
    
    print("Starting CPD")
    #o3d.visualization.draw_geometries([source, target])
    tf_param, _, _ = cpd.registration_cpd(source, target, 'affine')
    #tf_param = l2dist_regs.registration_svr(source, target, 'nonrigid')
    print("Done!")
    
    return tf_param

'''
def upsample(mesh):
    #start_time = time.time()
    # Loading to pcl to upsample
    
    # To load from numpy array of points
    print("In upsample")
    pc = np.asarray(mesh.points)
    point_cloud = pclpy.pcl.PointCloud.PointXYZ(pc)
    print("Input pcd size: ", point_cloud.size())
    
    mls = pcl.surface.MovingLeastSquaresOMP.PointXYZ_PointXYZ()
    tree = pcl.search.KdTree.PointXYZ()
    
    mls.setComputeNormals(True)
    mls.setInputCloud(point_cloud)
    mls.setSearchMethod(tree)
    mls.setSearchRadius(0.1)
    mls.setPolynomialFit(True)
    #mls.setNumberOfThreads(12)
    
    # Upsampling method
    mls.setUpsamplingMethod(pclpy.pcl.surface.MovingLeastSquares.PointXYZ_PointXYZ.SAMPLE_LOCAL_PLANE)
    mls.setUpsamplingRadius(0.06) #0.06
    mls.setUpsamplingStepSize(0.055) #0.05
    
    print("Now upsampling")
    output = pcl.PointCloud.PointXYZ()
    mls.process(output)
    print("Output pcd size: ", output.size())
    
    #pcl.io.savePCDFile('new.pcd', output)
    points = output.xyz
    #print(np.shape(points))

    return points
'''

def warp_part(s, t):
    '''
    Function to warp source part to target part
    '''
    
    # Convert source (chosen part) to pcd
    source = convert_to_pcd(s)
    source.paint_uniform_color([1, 0, 0])
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #o3d.visualization.draw_geometries([source])
    
    # Convert target (standard chair part) to pcd
    target = convert_to_pcd(t)
    target.paint_uniform_color([0, 1, 0])
    
    # Apply transform
    tf_param = affine_cpd(source, target, threshold=0.04)
    result = copy.deepcopy(source)
    result.points = tf_param.transform(result.points)

    # Draw result in blue
    result.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([source, target, result])
    #o3d.visualization.draw_geometries([result])    
    return result




if __name__ == "__main__":  
    # Read mesh file
    s = o3d.io.read_triangle_mesh("182_part0.obj")
    t = o3d.io.read_triangle_mesh("1095_part0.obj")
    
    source = convert_to_pcd(s)
    target = convert_to_pcd(t)
    
    # Paint the source as red and target as green
    source.paint_uniform_color([1, 0, 0])
    target.paint_uniform_color([0, 1, 0])
    
    #o3d.io.write_point_cloud("source.pcd", source)
    #o3d.io.write_point_cloud("target.pcd", target)
    
    o3d.visualization.draw_geometries([source], 'SOURCE')
    o3d.visualization.draw_geometries([target], 'TARGET')
    
    tf_param = affine_cpd(source, target, threshold=0.04)
    
    result = copy.deepcopy(source)
    result.points = tf_param.transform(result.points)
    
    # Draw result in blue
    result.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([source, target, result], 'COMPARISON')
    o3d.visualization.draw_geometries([result], 'FINAL RESULT')
    '''
    up_points = upsample(result)
    new_mesh = o3d.geometry.PointCloud()
    new_mesh.points = o3d.utility.Vector3dVector(up_points) 
    new_mesh.paint_uniform_color([0.7, 0.3, 0.8])
    o3d.visualization.draw_geometries([new_mesh])
    
    ########################
    o3d.io.write_point_cloud("final_result.pcd", new_mesh)
    '''