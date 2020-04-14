import copy
import pclpy
import math
from pclpy import pcl
import numpy as np
import open3d as o3d
from probreg import cpd
from probreg import l2dist_regs
from grassdata_new import Part
from draw3dobb import reindex_faces

def mesh_to_part(mesh, label, bb):
    vertices = list(mesh.vertices)
    faces = list(mesh.triangles)
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


#### CHANGES HERE

def convert_to_pcd(source_mesh):
    # Convert mesh to point cloud
    source = o3d.geometry.PointCloud()
    #n = np.asarray(source_mesh.vertices)
    #n = np.shape(n)[0]
    #print(n)
    
    source = source_mesh.sample_points_uniformly(int(1e5))
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

def upsample(mesh):
    #start_time = time.time()
    # Loading to pcl to upsample
    
    # To load from numpy array of points
    pc = np.asarray(mesh.points)
    point_cloud = pclpy.pcl.PointCloud.PointXYZ(pc)
    '''    
    # To save and reload
    # Takes more time
    o3d.io.write_point_cloud("result.pcd", mesh)
    point_cloud=pclpy.pcl.PointCloud.PointXYZ()
    pcl.io.loadPCDFile('result.pcd',point_cloud)
    '''
    
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
    
    #mls.setUpsamplingMethod(pclpy.pcl.surface.MovingLeastSquares.PointXYZ_PointXYZ.RANDOM_UNIFORM_DENSITY)
    #mls.setPointDensity(500)
    
    #mls.setUpsamplingMethod(pclpy.pcl.surface.MovingLeastSquares.PointXYZ_PointXYZ.VOXEL_GRID_DILATION)
    #mls.setDilationIterations(1000)
    
    #mls.setUpsamplingMethod(pclpy.pcl.surface.MovingLeastSquares.PointXYZ_PointXYZ.DISTINCT_CLOUD)
    
    #print("--- %s seconds ---" % (time.time() - start_time))
    
    print("Now upsampling")
    output = pcl.PointCloud.PointXYZ()
    mls.process(output)
    print("Output pcd size: ", output.size())
    points = output.xyz
    #print(np.shape(points))

    return points

def apply_transform_on_mesh(mesh, tf_param):
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    mesh.paint_uniform_color([1, 0, 0])
    
    # splitting up the point cloud affine transform
    tf_b = np.asarray(tf_param.b)
    #tf_b = np.transpose(tf_b)
    #tf_b = tf_b *-1.0
    #tf_b[:,2] = tf_b[:,2]*-1.0
    #tf_b[2,:] = tf_b[2,:]*-1.0   
    #print(tf_b)
    
    tf_t = np.asarray(tf_param.t)
    #tf_t[2] = tf_t[2]*-1.0
    print(tf_b)
    print(tf_t)
    
    tf = np.eye(4)
    tf[0:3, 0:3] = tf_b
    tf[0:3, 3] = tf_t
    #print(tf)
    
    mesh.transform(tf)
    #o3d.visualization.draw_geometries([mesh]) 
    #o3d.io.write_triangle_mesh("inv.obj", mesh)
    '''
    v = np.asarray(mesh.vertices)
    print(v)
    temp = copy.deepcopy(v[:,0])
    v[:,0] = v[:,1]
    v[:,1] = temp
    print(v)
    mesh.vertices = o3d.utility.Vector3dVector(v)
    
    # Rotate by 180
    R = np.eye(3)
    rad = (math.pi)/2
    R[0,0] = R[1,1] = math.cos(rad)
    R[1,0] = math.sin(rad)
    R[0,1] = -R[1,0]
    mesh.rotate(R)
    '''
    
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh.normalize_normals()
    mesh.paint_uniform_color([0, 0, 1])
    #o3d.visualization.draw_geometries([mesh]) 
    return mesh

def warp_part(s, t):
    '''
    Function to warp source part to target part
    '''
    #o3d.visualization.draw_geometries([s])
    #o3d.visualization.draw_geometries([t])
    # Convert source (chosen part) to pcd
    source = convert_to_pcd(s)
    source.paint_uniform_color([1, 0, 0])
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #o3d.visualization.draw_geometries([source])
    
    # Convert target (standard chair part) to pcd
    target = convert_to_pcd(t)
    target.paint_uniform_color([0, 1, 0])
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Apply transform
    tf_param = affine_cpd(source, target, threshold=0.04)
    
    
    result = copy.deepcopy(source)
    result.points = tf_param.transform(result.points)

    # Draw result in blue
    result.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([source, target, result])
    #o3d.visualization.draw_geometries([result])  
    '''
    return result
    '''
    return tf_param, source, target

def move_part(mesh, target, axis=1, n=1000):
    '''
    Here mesh is source, target is the mesh to move towards
    Axis is which axis to move it in (x y or z)
    n is which value distance to take, around 1000 is on an average a good value
    '''
    # We need point clouds to calculate distance to move
    dist = convert_to_pcd(mesh).compute_point_cloud_distance(convert_to_pcd(target))
    dist = np.sort(dist)
    #print(dist[0], dist[n])
    translate_vec = np.zeros(3)
    translate_vec[axis] = -dist[n]
    mesh = mesh.translate(translate_vec)
    return mesh



if __name__ == "__main__":  
    # Read mesh file
    s = o3d.io.read_triangle_mesh("182_part0.obj")
    t = o3d.io.read_triangle_mesh("1095_part0.obj")
    
    print(s)
    
    source = convert_to_pcd(s)
    target = convert_to_pcd(t)
    
    # Paint the source as red and target as green
    s.paint_uniform_color([1, 0, 0])
    t.paint_uniform_color([0, 1, 0])
    
    #o3d.io.write_point_cloud("source.pcd", source)
    #o3d.io.write_point_cloud("target.pcd", target)
    
    tf_param, source, target = warp_part(s, t)
    
    result = copy.deepcopy(s)
    result = apply_transform_on_mesh(result, tf_param)
    
    # Draw result in blue
    result.paint_uniform_color([0, 0, 1])
    result.compute_triangle_normals()
    result.compute_vertex_normals()
    o3d.visualization.draw_geometries([source, target, result])
    #o3d.visualization.draw_geometries([result])
    '''
    ########################
    o3d.io.write_triangle_mesh("final_result.obj", result)
    '''
    
    
    # For reference
    # in warping, red is source, green is target, blue is result