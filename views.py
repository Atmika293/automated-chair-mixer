import numpy as np
import open3d as o3d
import math
import copy
import matplotlib.pyplot as plt

def view(pcd, filename, zoomfactor = 10):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1,1,1])#([0.2,0.2,0.25])
    # White: ([1,1,1])
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()    
    ctr.scale(zoomfactor)
    vis.run()   
    vis.capture_screen_image(filename)
    vis.destroy_window()

if __name__ == "__main__": 
    # FRONT
    f = o3d.io.read_point_cloud('final1.ply')
    view(f, 'front.png')
    #o3d.visualization.draw_geometries([f])    
    
    l = copy.deepcopy(f)
    t = copy.deepcopy(f)
    
    # LEFT
    # Rotate by 90
    R = np.eye(3)
    rad = (math.pi)/2
    R[2,2] = R[0,0] = math.cos(rad)
    R[0,2] = math.sin(rad)
    R[2,0] = -R[0,2]
    l.rotate(R)
    view(l, 'left.png')
    #o3d.visualization.draw_geometries([l])
    
    
    # TOP
    # Rotate by 90
    R = np.eye(3)
    rad = (math.pi)/2
    R[2,2] = R[1,1] = math.cos(rad)
    R[2,1] = math.sin(rad)
    R[1,2] = -R[2,1]
    t.rotate(R)
    view(t, 'top.png')
    #o3d.visualization.draw_geometries([t])
