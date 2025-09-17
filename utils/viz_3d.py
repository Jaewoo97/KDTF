import numpy as np
import torch
import pyrender
from pyrender import OffscreenRenderer, PerspectiveCamera, PointLight
import trimesh       
import sys
import matplotlib.pyplot as plt
from matplotlib import cm as cmx
from matplotlib import colors
import copy


sys.path.append('/mnt/jaewoo4tb/textraj/')
from bev.cfg import bev_settings
from bev.post_parser import *    

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

jet = plt.get_cmap('twilight')
cNorm  = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

LIGHT_POSE_1 = np.array([
    [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 0],
    [1.0, 0.0,           0.0,           0.0],
    [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 3.0],
    [0.0,  0.0,           0.0,          1.0]
])
LIGHT_POSE_2 = np.array([
    [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 5],
    [1.0, 0.0,           0.0,           5],
    [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 3.0],
    [0.0,  0.0,           0.0,          1.0]
])
LIGHT_POSE_3 = np.array([
    [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, -5],
    [1.0, 0.0,           0.0,           5],
    [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 3.0],
    [0.0,  0.0,           0.0,          1.0]
])
LIGHT_POSE_4 = np.array([
    [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 5],
    [1.0, 0.0,           0.0,           -5],
    [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 3.0],
    [0.0,  0.0,           0.0,          1.0]
])
LIGHT_POSE_5 = np.array([
    [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, -5],
    [1.0, 0.0,           0.0,           -5],
    [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 3.0],
    [0.0,  0.0,           0.0,          1.0]
])
LIGHT_POSES = [LIGHT_POSE_1, LIGHT_POSE_2, LIGHT_POSE_3, LIGHT_POSE_4, LIGHT_POSE_5]


class Viz3D:    
    def __init__(self, scene_data):
        self.scene_data = scene_data

    def create_thick_line_prism(self, start, end, thickness=0.1):
        # Compute direction vector and perpendicular vector
        direction = np.array(end) - np.array(start)
        length = np.linalg.norm(direction)
        direction /= length  # Normalize direction
        perpendicular1 = np.array([-direction[1], direction[0], 0])  # Perpendicular vector in the XY plane
        perpendicular2 = np.array([direction[1], -direction[0], 0])  # Perpendicular vector in the XY plane
        
        # Define the half-thickness offset
        offset = thickness / 2
        
        # Vertices of the rectangular prism
        vertices = np.array([
            start + perpendicular1 * offset,  # Corner 1
            start - perpendicular1 * offset,  # Corner 2
            end - perpendicular1 * offset,    # Corner 3
            end + perpendicular1 * offset,     # Corner 4
            start + perpendicular2 * offset,  # Corner 5
            start - perpendicular2 * offset,  # Corner 6
            end - perpendicular2 * offset,    # Corner 7
            end + perpendicular2 * offset     # Corner 8
        ])
        
        # Define the faces of the rectangular prism
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Side faces
            [0, 1, 3], [0, 3, 2],  # Top face
            [4, 5, 6], [4, 6, 7],  # Bottom face
            [0, 1, 5], [0, 5, 4],  # Side faces
            [2, 3, 7], [2, 7, 6]   # Side faces
        ])
        
        # Create and return the Trimesh object
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    def create_grid_lines_mesh(self, size=10, divisions=10, robot_pos=None):   
        # Initialize arrays for vertices and indices
        vertices = []
        faces = []
        start_x = int(robot_pos[0] / 10) * 10
        start_y = int(robot_pos[1] / 10) * 10
        # Create lines parallel to the X-axis
        for i in range(divisions + 1):
            # y = i * (size / divisions) + start_y
            y = i * (size / divisions) - size / 2 + start_y
            if abs(robot_pos[1] - (y + start_y))<5.01:
                start = [robot_pos[0] - 5, y + start_y, 0]
                end = [robot_pos[0] + 5, y + start_y, 0]
                prism = self.create_thick_line_prism(start, end, 0.05)
                start_index = len(vertices)
                vertices.extend(prism.vertices)
                # Adjust faces indices
                for face in prism.faces:
                    faces.append([index + start_index for index in face])
        
        # Create lines parallel to the Y-axis
        for i in range(divisions + 1):
            # x = i * (size / divisions) + start_x
            x = i * (size / divisions) - size / 2 + start_x
            if abs(robot_pos[0] - (x + start_x))<5.01:
                start = [x + start_x, robot_pos[1] - 5, 0]
                end = [x + start_x, robot_pos[1] + 5, 0]
                prism = self.create_thick_line_prism(start, end, 0.05)
                start_index = len(vertices)
                vertices.extend(prism.vertices)
                # Adjust faces indices
                for face in prism.faces:
                    faces.append([index + start_index for index in face])
                    
        vertices = np.array(vertices)
        faces = np.array(faces)
        grid_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        grid_mesh.visual.vertex_colors[:, 3] = 255
        grid_mesh.visual.vertex_colors[:, :3] = [0,0,0]
        return grid_mesh

    def create_arrow(self, start, end, shaft_radius=0.05, shaft_resolution=20, head_radius=0.1, head_length=0.2):
        vector = end - start
        if vector.shape[0] == 2: vector = np.append(vector, 0)
        else: vector = vector.squeeze()
        if torch.is_tensor(vector): vector = vector.cpu().numpy()
        length = np.linalg.norm(vector)

        # Create the shaft of the arrow
        shaft = trimesh.creation.cylinder(radius=shaft_radius, height=length - head_length, sections=shaft_resolution) 
        
        # Create the arrowhead
        head = trimesh.creation.cone(radius=head_radius, height=head_length, sections=shaft_resolution)
        
        # Translate the arrowhead to the end of the shaft
        head.apply_translation([0, 0, length - head_length / 2])

        # Combine shaft and head
        arrow = trimesh.util.concatenate([shaft, head])

        # Rotate the arrow to align with the vector
        axis = np.array([0, 0, 1])  # Default arrow direction along z-axis
        if not np.allclose(vector, axis):  # Check if the vector is not aligned with z-axis
            rot_vector = np.cross(axis, vector)
            rot_vector = rot_vector / np.linalg.norm(rot_vector)
            angle = np.arccos(np.dot(axis, vector) / length)
            rotation = trimesh.transformations.rotation_matrix(angle, rot_vector)
            arrow.apply_transform(rotation)
        
        # Translate arrow to the start position
        start_ = start
        start_[2] = 0
        translation = trimesh.transformations.translation_matrix(start_)
        translation_matrix = np.eye(4)  # Start with an identity matrix
        translation_matrix[:2, 3] = translation[:2,2]
        arrow.apply_transform(translation_matrix)

        return arrow

    def create_sphere(self, position, color_factor):
        material = pyrender.MetallicRoughnessMaterial( metallicFactor=0.0, roughnessFactor=0.5, baseColorFactor=color_factor)
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
        sphere_mesh = pyrender.Mesh.from_trimesh(sphere, material=material)
        translation_matrix = np.array([
            [1.0, 0.0, 0.0, position[0]],  # x-axis translation
            [0.0, 1.0, 0.0, position[1]],  # y-axis translation
            [0.0, 0.0, 1.0, position[2]],  # z-axis translation
            [0.0, 0.0, 0.0, 1.0]
        ])
        return sphere_mesh, translation_matrix
        
    def plot_3d(self):
        scene = pyrender.Scene(ambient_light=np.array([0.1, 0.1, 0.1, 0.1]))
        offscreen_r = OffscreenRenderer(viewport_width=640*2, viewport_height=480*2)
        triangles, verts_tran = [], None
        draw_person_idx = 0
        default_cfg = bev_settings()
        smpl_parser = SMPLA_parser(default_cfg.smpl_path, default_cfg.smil_path)
        DEFAULT_BETAS = np.array([[0.40479207,  0.17771187,  0.25489786,  0.22483926, -0.06253424, 0.03777225, -0.05464039,  0.0384823 ,  0.0689365 , -0.04577354, 0.10536828]])
        T_TOTAL = self.scene_data['positions'].shape[1]
        t_0_frame = 8
        for frame_idx in range(T_TOTAL):
            fig = plt.figure()
            ax = fig.subplots()
            for i in range(25):
                for j, position in enumerate(self.scene_data['positions'][i]):
                    if self.scene_data['padding_mask'][i, frame_idx]: continue
                    color1 = [1.0, 0.0, 0.0, 1.0] # red
                    color2 = [0.0, 1.0, 0.0, 1.0] # green
                    if j < t_0_frame:
                        sphere_mesh, trans = self.create_sphere(torch.cat((position, torch.Tensor([0]))), color1)
                    else:
                        sphere_mesh, trans = self.create_sphere(torch.cat((position, torch.Tensor([0]))), color2)
                    scene.add(sphere_mesh, pose=trans)
                # for position in self.scene_data['y'][i]:
                #     if self.scene_data['padding_mask'][i, frame_idx]: continue
                #     color = [0.0, 0.0, 1.0, 1.0] # blue
                #     position = self.scene_data['positions'][i, t_0_frame] + position
                #     sphere_mesh, trans = self.create_sphere(torch.cat((position, torch.Tensor([0]))), color)
                #     scene.add(sphere_mesh, pose=trans)
                
                # print(f"{i}: pose mask {self.scene_data['y_pose_mask'][i, 0]}, padding mask {self.scene_data['padding_mask'][i, 8]}")
                if self.scene_data['y_pose_mask'][i, frame_idx]: continue
                if self.scene_data['padding_mask'][i, frame_idx]: continue
                smpl_theta = self.scene_data['y_pose'][i, frame_idx]
                draw_person_idx += 1
                verts_tran, joints, face = smpl_parser(DEFAULT_BETAS, smpl_theta)            
                verts_tran[...,:3] = verts_tran[...,:3] + torch.tensor([self.scene_data['positions'][i, frame_idx, 0], self.scene_data['positions'][i, frame_idx, 1], -torch.min(verts_tran[:, :, 2])]).unsqueeze(0).unsqueeze(0)
                # else:
                #     verts_tran_ = person[draw_frame_idx][person_id]['pose']['verts']
                #     verts_tran_[...,:2] = verts_tran_[...,:2] - torch.tensor(person[draw_frame_idx][person_id]['global_position'][:2]).unsqueeze(0).unsqueeze(0)
                #     verts_tran = torch.cat((verts_tran, verts_tran_), dim=0)
                triangles.append(face)
                # Add humans
                body_meshes = []
                m = trimesh.Trimesh(vertices=verts_tran[0], faces=triangles[0])
                m.visual.vertex_colors[:, 3] = 255
                colors = np.asarray(scalarMap.to_rgba(draw_person_idx / verts_tran.shape[0])[:3]) * 255
                colors = np.clip(colors * 1.5, 0, 255)  # Increase brightness by 50%, clamp to [0, 255]
                m.visual.vertex_colors[:, :3] = colors.astype(np.uint8)        
                body_meshes.append(m)
                body_mesh = pyrender.Mesh.from_trimesh(body_meshes, smooth=False)
                body_node = pyrender.Node(mesh=body_mesh, name='body')
                scene.add_node(body_node)

            # Add floor
            floor = trimesh.creation.box(extents=np.array([10, 10, 0.02]),
                                transform=np.array([[1.0, 0.0, 0.0, 0],
                                                    [0.0, 1.0, 0.0, 0],
                                                    [0.0, 0.0, 1.0, -0.05],
                                                    [0.0, 0.0, 0.0, 1.0],
                                                    ]),)
            floor.visual.vertex_colors = [0.3, 0.3, 0.3]
            floor_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(floor), name='floor')
            scene.add_node(floor_node)
            
            # Add other stuff
            point_l = PointLight(color=np.ones(3), intensity=40.0)
            for light_pose in LIGHT_POSES:
                light_pose_ = copy.deepcopy(light_pose)
                light_pose_[0, 3] = light_pose_[0, 3] + 0
                light_pose_[1, 3] = light_pose_[1, 3] + 0
                _ = scene.add(point_l, pose=light_pose_)
            cam = PerspectiveCamera(yfov=(np.pi / 3))
            cam_pose = np.array([
                [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 5.5 + 0],
                [1.0, 0.0,           0.0,           0.0 + 0],
                [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 5],
                [0.0,  0.0,           0.0,          1.0]
            ])
            cam_node = scene.add(cam, pose=cam_pose)
            axis_mesh = pyrender.Mesh.from_trimesh(trimesh.creation.axis(), smooth=False)
            robot_position = np.array([
                [1.0, 0.0, 0.0, 0],  # x-axis translation
                [0.0, 1.0, 0.0, 0],  # y-axis translation
                [0.0, 0.0, 1.0, 0],  # z-axis translation
                [0.0, 0.0, 0.0, 1.0]
            ])
            
            # Add grid on floor
            # TODO: need to manage disappearing lines
            grid_lines_mesh = self.create_grid_lines_mesh(size=20, divisions=20, robot_pos=[0, 0, 0])
            scene.add(pyrender.Mesh.from_trimesh(grid_lines_mesh))
            scene.add(axis_mesh, pose=robot_position)
            color, depth = offscreen_r.render(scene)
            ax.imshow(color)
            ax.axis('off')
            # print(draw_person_idx)
            # plt.show()
            fig.savefig(f"/mnt/jaewoo4tb/textraj/temp/test_pose_zero_{frame_idx}.jpg", dpi=300)
 

if __name__ == "__main__":
    target_scene = torch.load("/mnt/jaewoo4tb/textraj/preprocessed_2nd/v1_fps_2_5_frame_20/bytes-cafe-2019-02-07_0_agents_0_to_1726/0.pt")
    viz3d = Viz3D(target_scene)
    viz3d.plot_3d()
    print('done')