def write_to_obj(filename, vertices, faces):
    f = open(filename,"w+")

    for vert in vertices:
        f.write("v %f %f %f\n" % (vert[0], vert[1], vert[2]))

    for face in faces:
        f.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))

    f.close()

# TODO: implement renderer like Wallace specified, using depth only with raytracing
def render_obj(filename, vertices, faces, angles, width, height):
    #compute bounding sphere
    #foreach angle compute position on bounding sphere, get normal vector and build plane parallel to sphere
    # get size of image and cast rays onto the triangles for each point
    # save image_suffix.png
    return None