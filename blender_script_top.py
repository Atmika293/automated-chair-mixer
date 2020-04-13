import bpy
import pathlib
#original code from  https://blender.stackexchange.com/questions/80938/batch-rendering-of-10-000-images-from-10-000-obj-files

obj_root = pathlib.Path('A:/764dataset/Chair/models')
pi = 3.14159

bpy.ops.object.select_all(action='DESELECT')
render = bpy.context.scene.render

for obj_fname in obj_root.glob('*.obj'):
    bpy.ops.import_scene.obj(filepath=str(obj_fname))

    for i in range(0, 5):
        for ob in bpy.context.selected_objects:
            if i == 4:
                ob.rotation_euler = ( pi /2 , 0, 0)
            else:
                ob.rotation_euler = ( 0 , i * 3.14159 / 2, 0)
        render.filepath = 'A:/764dataset/Chair/renders_ortho/%s_%d' % (obj_fname.stem, i)
        bpy.ops.render.render(write_still=True)

    meshes_to_remove = []
    for ob in bpy.context.selected_objects:
        meshes_to_remove.append(ob.data)

    bpy.ops.object.delete()

    # Remove the meshes from memory too
    for mesh in meshes_to_remove:
        bpy.data.meshes.remove(mesh)