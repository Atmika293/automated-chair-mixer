import numpy as np
import open3d as o3d
import math
import copy
import matplotlib.pyplot as plt

def view(pcd, filename, zoomfactor = 10, model_type=1):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    if model_type==1:
        opt.background_color = np.asarray([0.2,0.2,0.2])# Resnet
    elif model_type==2:
        opt.background_color = np.asarray([1,1,1])      # Lenet
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
    
    # 1 for resnet
    # 2 for lenet
    type_of_model = 1
    models = [3,4,8,9,10]
    folder = 'SETB/SETB_Final/'
    
    # Change the addresses and the values for i
    for i in models:
        # FRONT
        f = o3d.io.read_point_cloud(folder+'final'+str(i)+'.ply')
        if type_of_model==1:
            f.paint_uniform_color([0.6,0.6,0.6])  
        else:
            f.paint_uniform_color([0.4,0.4,0.4])
        view(f, folder+'front'+str(i)+'.png', model_type=type_of_model)
        #o3d.visualization.draw_geometries([f])    
        
        r = copy.deepcopy(f)
        t = copy.deepcopy(f)
        
        # RIGHT
        # Rotate by 90
        R = np.eye(3)
        rad = (math.pi)/2
        R[2,2] = R[0,0] = math.cos(rad)
        R[0,2] = math.sin(rad)
        R[2,0] = -R[0,2]
        r.rotate(R)
        view(r, folder+'rightside'+str(i)+'.png', model_type=type_of_model)
        #o3d.visualization.draw_geometries([l])
        
        
        # TOP
        # Rotate by 90
        R = np.eye(3)
        rad = (math.pi)/2
        R[2,2] = R[1,1] = math.cos(rad)
        R[2,1] = math.sin(rad)
        R[1,2] = -R[2,1]
        t.rotate(R)
        view(t, folder+'top'+str(i)+'.png', model_type=type_of_model)
        #o3d.visualization.draw_geometries([t])
    
    # For diagonal views
    if type_of_model==1:
        for i in models:
            # FRONT
            f = o3d.io.read_point_cloud(folder+'final'+str(i)+'.ply')
            f.paint_uniform_color([0.6,0.6,0.6]) 
            
            b_l = copy.deepcopy(f)
            b_r = copy.deepcopy(f)
            l = copy.deepcopy(f)
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
            
            view(b_l, folder+'rightfront'+str(i)+'.png')
            
                   
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
            
            view(b_r, folder+'leftfront'+str(i)+'.png')
            
            
            
            # TOP LEFT
            # Rotate by 90
            R = np.eye(3)
            rad = -(math.pi)/2
            R[2,2] = R[0,0] = math.cos(rad)
            R[0,2] = math.sin(rad)
            R[2,0] = -R[0,2]
            l.rotate(R)
            view(l, folder+'leftside'+str(i)+'.png', type_of_model)
            
                   
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
            
            view(t_r, folder+'leftback'+str(i)+'.png')