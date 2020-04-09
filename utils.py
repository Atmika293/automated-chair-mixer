def write_to_obj(filename, vertices, faces, face_offsets=None):
    f = open(filename,"w+")

    for vert in vertices:
        f.write("v %f %f %f\n" % (vert[0], vert[1], vert[2]))

    for i in range(0, len(faces)):

        face = faces[i]

        if face_offsets is not None:
            face_offset = face_offsets[i]
        else:
            face_offset = 0

        f.write("f %d %d %d\n" % (face[0] + 1 + face_offset, face[1] + 1 + face_offset, face[2] + 1 + face_offset))
        

    f.close()

# TODO: implement renderer like Wallace specified, using depth only with raytracing
def render_obj(filename, vertices, faces, angles, width, height):
    #compute bounding sphere
    #foreach angle compute position on bounding sphere, get normal vector and build plane parallel to sphere
    # get size of image and cast rays onto the triangles for each point
    # save image_suffix.png
    return None

def reindex_faces(vertices, faces):
    offset = 99999999999999
    for face in faces:
        index_max = min(face)
        if index_max < offset : offset = index_max
    
    #print(offset)
    #print(len(vertices))
    
    new_faces = []
    for face in faces:
        new_faces.append((face[0] - offset, face[1] - offset, face[2] - offset))
    return new_faces