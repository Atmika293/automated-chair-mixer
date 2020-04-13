import bpy
import pathlib
#original code from  https://blender.stackexchange.com/questions/80938/batch-rendering-of-10-000-images-from-10-000-obj-files

obj_root = pathlib.Path('A:/764dataset/Chair/models')
pi = 3.14159

bpy.ops.object.select_all(action='DESELECT')
render = bpy.context.scene.render

for obj_fname in obj_root.glob('*.obj'):
    bpy.ops.import_scene.obj(filepath=str(obj_fname))
    cur_index = 0

    for i in range(0, 8):

        if i == 4 or i == 5:
            continue

        for ob in bpy.context.selected_objects:
            ob.rotation_euler = ( pi /2 ,0, i * 3.14159 / 4)

        render.filepath = 'A:/764dataset/Chair/renders/%s_%d' % (obj_fname.stem, cur_index)
        bpy.ops.render.render(write_still=True)
        cur_index +=1

    meshes_to_remove = []
    for ob in bpy.context.selected_objects:
        meshes_to_remove.append(ob.data)

    bpy.ops.object.delete()

    # Remove the meshes from memory too
    for mesh in meshes_to_remove:
        bpy.data.meshes.remove(mesh)