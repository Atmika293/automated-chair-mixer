from grassdata_new import GRASSNewDataset
from utils import export_parts_to_obj
from grassdata_new import PartLabel, Mesh, Part
import numpy as np
import os

if __name__ == '__main__':
    dir = 'A:\\764dataset\\Chair'
    obb_path = os.path.join(dir, 'obbs')
    obj_path = os.path.join(dir, 'models')
    files = os.listdir(obb_path)
    file_count = len(files)
    model_count = 6201
    for i in range(0, model_count):
        file = os.path.join(obb_path, files[i])
        obj = os.path.join(obj_path, files[i][:-3] + "obj")
        mesh = Mesh(file, obj)

        new_parts = []

        if np.random.random_sample() < 0.60:
            # 60% of the time include the back
            new_parts.extend(mesh.get_parts_from_label(PartLabel.BACK))

        if np.random.random_sample() < 0.60:
            # 60% of the time include the seat
            new_parts.extend(mesh.get_parts_from_label(PartLabel.SEAT))

        if np.random.random_sample() < 0.60:
            # 60% of the time include at least one arm rest
            armrests = mesh.get_parts_from_label(PartLabel.ARMREST)
            if(len(armrests) > 1):
                if np.random.random_sample() < 0.5:
                    new_parts.append(armrests[0])
                else:
                    new_parts.extend(armrests[1:])
            else:
                new_parts.extend(armrests)
            
        # include some of the legs but not all of them
        legs = mesh.get_parts_from_label(PartLabel.LEG)
        if np.random.random_sample() < 0.33 and len(legs) >= 1: # 33% of the time include only one leg
            new_parts.append(legs[0])
        elif np.random.random_sample() < 0.66 and len(legs) >= 2:
            new_parts.append(legs[0])
            new_parts.append(legs[1])
        elif len(legs) >= 3:
            new_parts.append(legs[0])
            new_parts.append(legs[1])
            new_parts.append(legs[2])
        
        # make sure there is at least something in the file
        if len(new_parts) == 0: 
            new_parts.extend(mesh.get_parts_from_label(PartLabel.SEAT))

        filename = mesh.original_model.split('\\')[-1]

        filename = "A:\\764dataset\\Chair\\models_bad\\" + filename

        export_parts_to_obj(filename, new_parts)

        del mesh