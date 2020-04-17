import numpy as np
import open3d as o3d
import math
import copy
import matplotlib.pyplot as plt

def view(pcd, filename, zoomfactor = 10):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.2,0.2,0.2])
    # White: ([1,1,1])
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()    
    ctr.scale(zoomfactor)
    vis.run()   
    vis.capture_screen_image(filename)
    vis.destroy_window()
    

def view_lenet(pcd, filename, zoomfactor = 10):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1,1,1])
    # White: ([1,1,1])
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()    
    ctr.scale(zoomfactor)
    vis.run()   
    vis.capture_screen_image(filename)
    vis.destroy_window()
    
        

if __name__ == "__main__": 
    
    # Change the addresses and the values for i
    for i in [3,4,8,9,10]:
        # FRONT
        f = o3d.io.read_point_cloud('SETB/SETB_Final/final'+str(i)+'.ply')
        f.paint_uniform_color([0.4,0.4,0.4])  
        # Change this to ([0.6,0.6,0.6]) when using view 
        # Change this to ([0.4,0.4,0.4]) when using view_lenet
        view_lenet(f, 'SETB/SETB_Final/front'+str(i)+'.png')
        #o3d.visualization.draw_geometries([f])    
        
        l = copy.deepcopy(f)
        t = copy.deepcopy(f)
        
        # LEFT
        # Rotate by 90
        R = np.eye(3)
        rad = -(math.pi)/2
        R[2,2] = R[0,0] = math.cos(rad)
        R[0,2] = math.sin(rad)
        R[2,0] = -R[0,2]
        l.rotate(R)
        view_lenet(l, 'SETB/SETB_Final/left'+str(i)+'.png')
        #o3d.visualization.draw_geometries([l])
        
        
        # TOP
        # Rotate by 90
        R = np.eye(3)
        rad = (math.pi)/2
        R[2,2] = R[1,1] = math.cos(rad)
        R[2,1] = math.sin(rad)
        R[1,2] = -R[2,1]
        t.rotate(R)
        view_lenet(t, 'SETB/SETB_Final/top'+str(i)+'.png')
        #o3d.visualization.draw_geometries([t])
    
    
    # For diagonal views
    
    # Change the addresses and the values for i
    for i in [3,4,8,9,10]:
        # FRONT
        f = o3d.io.read_point_cloud('SETB/SETB_Final/final'+str(i)+'.ply')
        f.paint_uniform_color([0.6,0.6,0.6]) 
        
        b_l = copy.deepcopy(f)
        b_r = copy.deepcopy(f)
        t_l = copy.deepcopy(f)
        t_r = copy.deepcopy(f)
        
        # BOTTOM LEFT
        R = np.eye(3)
        rad = (math.pi)/4
        R[2,2] = R[0,0] = math.cos(rad)
        R[0,2] = math.sin(rad)
        R[2,0] = -R[0,2]
        b_l.rotate(R)
        
        R = np.eye(3)
        rad = (math.pi)/15
        R[2,2] = R[1,1] = math.cos(rad)
        R[2,1] = math.sin(rad)
        R[1,2] = -R[2,1]
        b_l.rotate(R)
        
        view(b_l, 'SETB/SETB_Final/bottom_left'+str(i)+'.png')
        
               
        # BOTTOM RIGHT
        R = np.eye(3)
        rad = -(math.pi)/4
        R[2,2] = R[0,0] = math.cos(rad)
        R[0,2] = math.sin(rad)
        R[2,0] = -R[0,2]
        b_r.rotate(R)
        
        R = np.eye(3)
        rad = (math.pi)/15
        R[2,2] = R[1,1] = math.cos(rad)
        R[2,1] = math.sin(rad)
        R[1,2] = -R[2,1]
        b_r.rotate(R)
        
        view(b_r, 'SETB/SETB_Final/bottom_right'+str(i)+'.png')
        
        
        
        # TOP LEFT
        R = np.eye(3)
        rad = (math.pi)/2 + (math.pi)/4
        R[2,2] = R[0,0] = math.cos(rad)
        R[0,2] = math.sin(rad)
        R[2,0] = -R[0,2]
        t_l.rotate(R)
        
        R = np.eye(3)
        rad = (math.pi)/15
        R[2,2] = R[1,1] = math.cos(rad)
        R[2,1] = math.sin(rad)
        R[1,2] = -R[2,1]
        t_l.rotate(R)
        
        view(t_l, 'SETB/SETB_Final/bottom_left'+str(i)+'.png')
        
               
        # TOP RIGHT
        R = np.eye(3)
        rad = -((math.pi)/2 + (math.pi)/4)
        R[2,2] = R[0,0] = math.cos(rad)
        R[0,2] = math.sin(rad)
        R[2,0] = -R[0,2]
        t_r.rotate(R)
        
        R = np.eye(3)
        rad = (math.pi)/15
        R[2,2] = R[1,1] = math.cos(rad)
        R[2,1] = math.sin(rad)
        R[1,2] = -R[2,1]
        t_r.rotate(R)
        
        view(t_r, 'SETB/SETB_Final/bottom_right'+str(i)+'.png')