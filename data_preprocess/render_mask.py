import cv2, json, os, trimesh, math
import numpy as np
import renderer
from tqdm import tqdm
import cv2 as cv

origin_path = r'' # origin image path
mask_path = r'./mask' # output mask path
mask_image_path = r'./image' # output image path
ply_path = r'' # cut mesh path (high quality)
tu_mesh_path = '' # tu mesh path
id_idx = 1  # identity index
exp_idx = 2  # expression index
# read Rt scale
with open("Rt_scale_dict.json", 'r') as f: # from facescape
    Rt_scale_dict = json.load(f)

land_dir = os.path.join('landmark_indices.npz') # from facescape
landmark = np.load(land_dir)['v10'][30]

expression = ['neutral', 'smile', 'mouth_stretch', 'anger', 'jaw_left', 'jaw_right', 'jaw_forward', 'mouth_left', 'mouth_right', 'dimpler', 'chin_raiser',
              'lip_puckerer', 'lip_funneler', 'sadness', 'lip_roll', 'grin', 'cheek_blowing', 'eye_closed', 'brow_raiser', 'brow_lower']

for i in tqdm(range(1, 2)):
    for j in range(1, 2):
        exp_idx = j
        id_idx = i
        i_path = os.path.join(origin_path, str(i), '{0}_{1}'.format(j, expression[j-1]))
        p_path = os.path.join(ply_path, str(i), '{0}_{1}.obj'.format(j, expression[j-1]))

        image_path = os.listdir(i_path)
        with open(os.path.join(i_path, f"params.json"), 'r') as f:
            meta = json.load(f)

        scale = Rt_scale_dict['%d' % id_idx]['%d' % exp_idx][0]
        Rt = np.array(Rt_scale_dict['%d' % id_idx]['%d' % exp_idx][1])

        mview_mesh = trimesh.load(p_path, process=False, maintain_order=True)
        tu_mesh_ = os.path.join(tu_mesh_path, '{0}/models_reg/{1}_{2}.obj'.format(i, j, expression[j-1])) # tu model path
        tu_mesh = trimesh.load(tu_mesh_, process=False, maintain_order=True)
        point = tu_mesh.vertices[landmark]

        for z in image_path:
            if not z.endswith('.jpg'): continue
            num = z[:-4]
            print(num)

            image = os.path.join(i_path, z)
            image = cv.imread(image)
            height, width, _ = image.shape
            K = meta[str(num) + '_K']

            pose = np.array(meta[str(num) + '_Rt'])[:3, :4]
            R_cv2gl = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            pose = R_cv2gl.dot(pose)
            cam_pose = np.eye(4)
            cam_pose[:3, :3] = pose[:3, :3].T
            cam_pose[:3, 3] = -pose[:3, :3].T.dot(pose[:, 3])

            rays_o = cam_pose[:3, 3]
            rays_o *= scale
            rays_o = np.dot(Rt[:3, :3], rays_o.T).T + Rt[:3, 3]

            rays_o -= point
            rays_o[2] += 40
            rays_o /= 100

            r = cam_pose[:3, :3]
            r = np.dot(Rt[:3, :3], r)

            cam_pose[:3, 3] = rays_o
            cam_pose[:3, :3] = r

            _, image_full = renderer.render_cvcam(mview_mesh, K, None, cam_pose, rend_size=(height, width),
                                                  light_trans=np.zeros((3, 1)),
                                                  flat_shading=True)
            
            image_full[image_full < 255] = 1
            image_full[image_full == 255] = 0

            image = np.multiply(image, image_full).astype(np.uint8)
            m_path = os.path.join(mask_image_path, str(i))
            if not os.path.exists(m_path):
                os.mkdir(m_path)
            m_path = os.path.join(mask_image_path, str(i), '{0}_{1}'.format(j, expression[j - 1]))
            if not os.path.exists(m_path):
                os.mkdir(m_path)
            mask_i2_path = os.path.join(m_path, str(num) + '.jpg')
            cv.imwrite(mask_i2_path, image)
            
            image_full[image_full == 1] = 255
            m_path = os.path.join(mask_path, str(i))
            if not os.path.exists(m_path):
                os.mkdir(m_path)
            m_path = os.path.join(mask_path, str(i), '{0}_{1}'.format(j, expression[j - 1]))
            if not os.path.exists(m_path):
                os.mkdir(m_path)
            
            mask_i_path = os.path.join(m_path, str(num) + '.jpg')
            cv.imwrite(mask_i_path, image_full)
