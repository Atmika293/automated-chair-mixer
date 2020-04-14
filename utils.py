import numpy as np
from numpy import linalg
from PIL import Image

def restore_vertex_order(sorted_vertices, sorted_order):
    unsorted_vertices = np.zeros_like(sorted_vertices)

    for i in range(sorted_order.shape[0]):
        unsorted_vertices[sorted_order[i], :] = sorted_vertices[i, :]\

    return unsorted_vertices

#https://inareous.github.io/posts/opening-obj-using-py
def parse_part_from_obj_2(filename):
    f = open(filename)
    line = f.readline()
    parts_geometry = []
    order = []
    while line:
        if line[0:2] == "g ":
            part_index = int(line[2:])
            order.append(part_index)
            vertices = []
            faces = []
            line = f.readline()
            while line and line[0:2] != "g ":
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)
                    vertex = [float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1])]
                    vertices.append(vertex)

                elif line[:2] == "f ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)
                    face = [int(line[index1:index2])-1, int(line[index2:index3])-1, int(line[index3:-1])-1]
                    faces.append(face)

                line = f.readline()
            parts_geometry.append((vertices, faces))
        else:
            line = f.readline()
    f.close()

    if len(parts_geometry) == 0:
        print("Failed to find part geometry")

    return order, parts_geometry

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

def reindex_faces(vertices, faces):
    offset = 99999999999999
    for face in faces:
        index_max = min(face)
        if index_max < offset : offset = index_max
    
    #print(offset)
    #print(len(vertices))

    new_faces = []
    for face in faces:
        new_faces.append([face[0] - offset, face[1] - offset, face[2] - offset])
    return new_faces

def get_box_corners(points):
    """returns an 8x3 numpy array of points corresponding to the corners of the box"""
    center = points[0: 3]
    dir_3 = points[3: 6]
    dir_1 = points[6: 9]
    dir_2 = points[9: 12]
    lengths = np.array([points[13], points[14], points[12]]) # swap to match grassdata, verify that it is correct

    dir_1 = dir_1 / linalg.norm(dir_1)
    dir_2 = dir_2 / linalg.norm(dir_2)
    #dir_3 = np.cross(dir_1, dir_2)
    dir_3 = dir_3 / linalg.norm(dir_3)
    cornerpoints = np.zeros([8, 3])

    d1 = 0.5 * lengths[0] * dir_1
    d2 = 0.5 * lengths[1] * dir_2
    d3 = 0.5 * lengths[2] * dir_3

    cornerpoints[0][:] = center - d1 - d2 - d3
    cornerpoints[1][:] = center - d1 + d2 - d3
    cornerpoints[2][:] = center + d1 - d2 - d3
    cornerpoints[3][:] = center + d1 + d2 - d3
    cornerpoints[4][:] = center - d1 - d2 + d3
    cornerpoints[5][:] = center - d1 + d2 + d3
    cornerpoints[6][:] = center + d1 - d2 + d3
    cornerpoints[7][:] = center + d1 + d2 + d3

    return cornerpoints

def parse_bounding_box_obb(line:str):
    """Read bounding box info from obb file"""
    float_strs = line.split(' ')
    points = []
    for fstr in float_strs:
        points.append(float(fstr))
    corners = get_box_corners(np.array(points))
    return corners

def parse_connection_obb(line:str):
    """Parse connectivity info from obb file"""
    index = line.find(" ") + 1
    first_part = int(line[:index])
    second_part = int(line[index:])
    return (first_part, second_part)

def parse_obb_file(filename):  
    """Extract partnet symh info from obb file"""
    part_count = 0
    bounding_boxes = []
    connectivity = []
    #symmetry = []
    labels = []
    f = open(filename)
    line = f.readline()

    if line[:2] == "N ":
        part_count = int(line[2:])

        # parse bounding box floats
        for i in range(0, part_count):
            bounding_boxes.append(parse_bounding_box_obb(f.readline()))
        
        # parse connectivity
        line = f.readline()
        if line[:2] != "C ":
            return None

        connection_count = int(line[2:])
        for i in range(0, connection_count):
            connectivity.append(parse_connection_obb(f.readline()))

        line = f.readline()
        # skip symmetry
        while line[:2] != "L ":
            line = f.readline()
    
        # parse labels 
        label_count = int(line[2:])
        if label_count != part_count:
            return None
        for i in range(0, part_count):
            labels.append(int(f.readline()))

    else:
        return None

    f.close()

    return part_count, bounding_boxes, connectivity, labels

def export_parts_to_obj(filename, parts):
    """Exports a list of parts to obj file. Assumes faces have been reindexed at 0"""
    vertices = []
    faces = []
    offsets = []
    for part in parts:
        current_vert_count = len(vertices)
        offsets.extend([current_vert_count] * len(part.faces))
        vertices.extend(part.vertices)
        faces.extend(part.faces)

    write_to_obj(filename, vertices, faces, offsets)

def raycast_triangles(ray_origin, ray_dir, vertices, faces, normals):

    for i in range(0, len(faces)):
        face = faces[i]
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        pvec = normals[i]
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        det = np.dot(v0v1, pvec)
        if det < -0.001: continue
        inv_det = 1 / det
        tvec = ray_origin - v0
        u = np.dot(tvec, pvec) * inv_det
        if u < 0 or u > 1: continue
        qvec = np.cross(tvec, v0v1)
        v = np.dot(ray_dir, qvec) * inv_det
        if v < 0 or u + v > 1: continue

        return np.dot(v0v2, qvec) * inv_det

    return 1.0

# TODO: implement renderer like Wallace specified, using depth only with raytracing
def render_obj(filename, vertices, faces, angles, width):

    np_verts = np.array(vertices)
    np_f_normals = np.zeros((len(faces), 3))

    for i in range(0, len(faces)):
        face = faces[i]
        v1 = np_verts[face[1]] - np_verts[face[0]]
        v2 = np_verts[face[2]] - np_verts[face[1]]
        v3 = np.cross(v1, v2)
        np_f_normals[i] = v3 / np.sqrt(np.sum(v3**2))

    #compute bounding sphere
    center = np.mean(np_verts, axis=0)
    c_to_p = np_verts - center
    distances = np.sqrt(np.sum(c_to_p**2, axis=1))
    radius = np.max(distances, axis=0) * 1.5
    
    #foreach angle compute position on bounding sphere, get normal vector and build plane parallel to sphere
    img_count = 0
    for angle in angles:
        (theta, phi) = angle # inclination, azimuth
        # this also serves as the normal to the view plane
        view_vector = radius * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        n_view_vector = view_vector / np.sqrt(np.sum(view_vector**2))
        # tangent to the sphere and inside the plane
        tangent = np.array([np.cos(phi - np.pi), np.sin(phi - np.pi), 0])
        tangent /= np.sqrt(np.sum(tangent**2))
        binormal = np.cross(n_view_vector, tangent)
        binormal /= np.sqrt(np.sum(binormal**2))
        upper_left = center + view_vector + (0.5 * radius) * tangent + (0.5 * radius) * binormal
        image = np.zeros((width, width))
        for y in range(0, width):
            for x in range(0, width):
                ray_origin = upper_left - tangent * ((radius * x + 0.5) / width) - binormal * ((radius * y + 0.5) / width)
                ray_dir = -n_view_vector
                image[y,x] = raycast_triangles(ray_origin, ray_dir, np_verts, faces, np_f_normals)
        
        im = Image.fromarray((255 *image).astype(np.uint8))
        
        img_name = filename + "_" + str(img_count) + ".png"
        im.save(img_name)
        img_count += 1

    return

