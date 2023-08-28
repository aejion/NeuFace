import trimesh
import numpy as np
import json, os

### todo: optimize
def find_dot2(face, vertice, final_face, r, circle):
    f = vertice[face]
    (a, b, c) = f.shape
    for i in range(a):
        ff = f[i] - circle
        ff = np.multiply(ff, ff).sum(axis=1)
        pos = np.argmin(ff, axis=0)
        x1, y1, z1 = f[i][pos]
        p1 = face[i][pos]
        x2, y2, z2 = f[i][(pos + 1) % 3]
        p2 = face[i][(pos + 1) % 3]
        x3, y3, z3 = f[i][(pos + 2) % 3]
        p3 = face[i][(pos + 2) % 3]

        lx = x1
        wx = x2
        while (1):
            tx = (lx + wx) / 2
            ty = (tx - x1) / (x2 - x1) * (y2 - y1) + y1
            tz = (tx - x1) / (x2 - x1) * (z2 - z1) + z1
            dis = (tx - circle[0]) ** 2 + (ty - circle[1]) ** 2 + (tz - circle[2]) ** 2
            if (abs(dis - r) <= 1e-6):
                break
            if dis > r:
                wx = tx
            else:
                lx = tx
        vertice = np.concatenate([vertice, [[tx, ty, tz]]], axis=0)
        tpos1 = vertice.shape[0] - 1

        lx = x1
        wx = x3
        while (1):
            tx2 = (lx + wx) / 2
            ty2 = (tx2 - x1) / (x3 - x1) * (y3 - y1) + y1
            tz2 = (tx2 - x1) / (x3 - x1) * (z3 - z1) + z1
            dis = (tx2 - circle[0]) ** 2 + (ty2 - circle[1]) ** 2 + (tz2 - circle[2]) ** 2
            if (abs(dis - r) <= 1e-6):
                break
            if dis > r:
                wx = tx2
            else:
                lx = tx2
        vertice = np.concatenate([vertice, [[tx2, ty2, tz2]]], axis=0)
        tpos2 = vertice.shape[0] - 1
        final_face = np.concatenate([final_face, [[p1, tpos2, tpos1]]], axis=0)
    return vertice, final_face


def find_dot(face, vertice, final_face, r, circle):
    f = vertice[face]
    (a, b, c) = f.shape
    for i in range(a):
        ff = f[i] - circle
        ff = np.multiply(ff, ff).sum(axis=1)

        pos = np.argmax(ff, axis=0)
        x1, y1, z1 = f[i][pos]
        p1 = face[i][pos]
        x2, y2, z2 = f[i][(pos + 1) % 3]
        p2 = face[i][(pos + 1) % 3]
        x3, y3, z3 = f[i][(pos + 2) % 3]
        p3 = face[i][(pos + 2) % 3]

        wx = x1
        lx = x2
        while (1):
            tx = (lx + wx) / 2
            ty = (tx - x1) / (x2 - x1) * (y2 - y1) + y1
            tz = (tx - x1) / (x2 - x1) * (z2 - z1) + z1
            dis = (tx - circle[0]) ** 2 + (ty - circle[1]) ** 2 + (tz - circle[2]) ** 2
            if (abs(dis - r) <= 1e-6):
                break
            if dis > r:
                wx = tx
            else:
                lx = tx
        vertice = np.concatenate([vertice, [[tx, ty, tz]]], axis=0)
        tpos1 = vertice.shape[0] - 1

        wx = x1
        lx = x3
        while (1):
            tx2 = (lx + wx) / 2
            ty2 = (tx2 - x1) / (x3 - x1) * (y3 - y1) + y1
            tz2 = (tx2 - x1) / (x3 - x1) * (z3 - z1) + z1
            dis = (tx2 - circle[0]) ** 2 + (ty2 - circle[1]) ** 2 + (tz2 - circle[2]) ** 2
            if (abs(dis - r) <= 1e-6):
                break
            if dis > r:
                wx = tx2
            else:
                lx = tx2

        vertice = np.concatenate([vertice, [[tx2, ty2, tz2]]], axis=0)
        tpos2 = vertice.shape[0] - 1

        final_face = np.concatenate([final_face, [[p2, tpos2, tpos1], [p2, p3, tpos2]]], axis=0)
    return vertice, final_face

"""
input：
    vertices： nx3 points cloud
    mesh：     mx3 faces
    circle：   center of the circle
    r：        radius
output：
    new_vertice： new points after cutting
    new_mesh：    new faces after cutting
"""
def cut(vertices, mesh, circle, r):
    r = r ** 2
    v = vertices - circle
    v = np.multiply(v, v).sum(axis=1)
    v = v <= r

    face_condition = v[mesh]
    face_condition = face_condition.sum(axis=1)
    face3 = mesh[face_condition >= 3]

    face2 = mesh[face_condition == 2]
    new_vertice, myfaces = find_dot(face2, vertices, face3, r, circle)

    face1 = mesh[face_condition == 1]
    new_vertice, new_mesh = find_dot2(face1, new_vertice, myfaces, r, circle)

    return new_vertice, new_mesh
    
expression = ['neutral', 'smile', 'mouth_stretch', 'anger', 'jaw_left', 'jaw_right', 'jaw_forward', 'mouth_left', 'mouth_right', 'dimpler', 'chin_raiser',
                  'lip_puckerer', 'lip_funneler', 'sadness', 'lip_roll', 'grin', 'cheek_blowing', 'eye_closed', 'brow_raiser', 'brow_lower']

with open('./Rt_scale_dict.json') as f:
    Rt_scale_dict = json.load(f)

id_index = 1
exp_index = 2

land_dir = os.path.join('landmark_indices.npz') # from facescape
nose_index = np.load(land_dir)['v10'][30]

tu_mesh_path = '' # tu model path
high_mesh_path = '' # origin mesh path (high quality)
output_mesh_path = '' # output mesh path

mview_mesh = trimesh.load(tu_mesh_path, process=False, maintain_order=True)
# nose tip's position
point = mview_mesh.vertices[nose_index]

mview_mesh = trimesh.load(high_mesh_path, process=False, maintain_order=True)

scale = Rt_scale_dict['%d' % id_index]['%d' % exp_index][0]
Rt = np.array(Rt_scale_dict['%d' % id_index]['%d' % exp_index][1])
mview_mesh.vertices *= scale
mview_mesh.vertices = np.tensordot(Rt[:3,:3], mview_mesh.vertices.T, 1).T + Rt[:3, 3]
mview_mesh.vertices -= point
mview_mesh.vertices[:, 2] += 40
mview_mesh.vertices /= 100

v, f = cut(mview_mesh.vertices, mview_mesh.faces, [0, 0, 0], 1)

mesh = trimesh.Trimesh(vertices=v, faces=f)
mesh.export(output_mesh_path)