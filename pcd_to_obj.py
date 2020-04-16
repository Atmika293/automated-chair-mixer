import numpy as np
import open3d as o3d

def pcd_to_obj_conv(pcd):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = pcd.points
    tri = np.ones((10,3))
    tri[:,1] = 2
    tri[:,2] = 3
    mesh.triangles = o3d.utility.Vector3iVector(tri)
    return mesh


if __name__ == "__main__":  
    s = o3d.io.read_point_cloud('final.pcd')
    #s.paint_uniform_color([0.5, 0.1, 0.5])
    print(s)
    o3d.visualization.draw_geometries([s])
    
    mesh = pcd_to_obj_conv(s)
    #o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh('pc_to_trimesh.obj', mesh)
    
    # To test that the point cloud actually works if we take just vertices
    test = o3d.io.read_triangle_mesh('pc_to_trimesh.obj')
    pcd = o3d.geometry.PointCloud()
    pcd.points = test.vertices
    o3d.visualization.draw_geometries([pcd])