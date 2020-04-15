from utils import export_parts_to_obj
import numpy as np
import os
import open3d as o3d

if __name__ == '__main__':
    dir = 'A:\\764dataset\\geomproject'
    obb_path = os.path.join(dir, 'ply files')
    obj_path = os.path.join(dir, 'ply_obj')
    files = os.listdir(obb_path)
    file_count = len(files)

    for i in range(0, file_count):

        ply = os.path.join(obb_path, files[i])
        obj = os.path.join(obj_path, files[i][:-3] + "obj")
        mtl = os.path.join(obj_path, files[i][:-3] + "mtl")
        pcd = o3d.io.read_point_cloud(ply)
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
        mesh = mesh.simplify_quadric_decimation(30000)
        o3d.io.write_triangle_mesh(obj, mesh)
        os.remove(mtl)
        print("Completed %d" % i)
        # TODO: add code to write the obj and delete the mtl file for the regular renders
        