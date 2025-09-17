import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import trimesh
import cv2
import pyrender
from pyrender import OffscreenRenderer, PerspectiveCamera, PointLight
from matplotlib import cm as cmx
from matplotlib import colors
import copy
import random
from tqdm import tqdm 
import os
import sys

sys.path.append('/mnt/jaewoo4tb/textraj/')
from bev.cfg import bev_settings
from bev.post_parser import *

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


class VizWhole:
    def __init__(self, scene_name, agent_dir, interaction_dir, final_preprocessed_dir, initial_frame):
        self.img_root = f"/mnt/jaewoo4tb/t2p/jrdb/train_dataset/images/image_stitched/{scene_name}/"
        self.agent_processed = torch.load(agent_dir)
        self.interaction_processed = torch.load(interaction_dir)
        self.preprocessed_final = torch.load(final_preprocessed_dir)
        self.initial_frame = initial_frame
        
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

    def plot_bev_position(self, focus_frame, ax, cluster_cmap):
        for agent_id, value in self.agent_processed[focus_frame].items():
            pos = value['global_position'][:2]
            cluster_id = value['cluster_id']
            if cluster_id not in cluster_cmap:
                r = lambda: random.randint(0,255)
                cluster_cmap[cluster_id] = '#%02X%02X%02X' % (r(),r(),r())
            agent_name = f'Agent: {agent_id}'
            ax.scatter(pos[0], pos[1], color=cluster_cmap[cluster_id], s=10)
            ax.text(pos[0], pos[1] - 0.5, agent_name, fontsize=5)

        for _, value in self.agent_processed[focus_frame].items():
            center = value['robot_pos']
            break
        
        ax.scatter(center[0], center[1], color='black', s=10)

        max_v = 0
        mask = self.preprocessed_final["agent_mask"]
        positions = self.preprocessed_final["agent_position"]

        x = positions[:, :, 0][~mask]
        y = positions[:, :, 1][~mask]
        if x.numel() == 0 or y.numel() == 0:
            return
        
        s = ax.scatter(x, y, c=list(torch.nonzero(~mask)[:, 1]), cmap='inferno', s=1.0)
        max_t = max([torch.max(torch.abs(x)), torch.max(torch.abs(y))])
        if max_v < max_t:
            max_v = max_t

        max_v += 0.5

        max_v = 15
        ax.set_xlim(-max_v + center[0], max_v + center[1])
        ax.set_ylim(-max_v + center[0], max_v + center[1])
        ax.set_aspect('equal')
        return s

    def plot_3d_human(self, focus_frame, ax):
        scene = pyrender.Scene(ambient_light=np.array([0.1, 0.1, 0.1, 0.1]))
        offscreen_r = OffscreenRenderer(viewport_width=640*2, viewport_height=480*2)
        triangles, verts_tran = [], None
        draw_person_idx = 0
        assert focus_frame in self.agent_processed.keys()
        for person_id in self.agent_processed[focus_frame].keys():
            if self.agent_processed[focus_frame][person_id]['pose'] == None: continue
            draw_person_idx += 1
            verts_tran = self.agent_processed[focus_frame][person_id]['pose']['verts']
            verts_tran[...,:3] = verts_tran[...,:3] + torch.tensor(self.agent_processed[focus_frame][person_id]['global_position'][:3]).unsqueeze(0).unsqueeze(0)
            # else:
            #     verts_tran_ = person[draw_frame_idx][person_id]['pose']['verts']
            #     verts_tran_[...,:2] = verts_tran_[...,:2] - torch.tensor(person[draw_frame_idx][person_id]['global_position'][:2]).unsqueeze(0).unsqueeze(0)
            #     verts_tran = torch.cat((verts_tran, verts_tran_), dim=0)
            triangles.append(self.agent_processed[focus_frame][person_id]['pose']['smpl_face'])
            
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
            # arrow = create_arrow(person[draw_frame_idx][person_id]['global_position'][:3], person[draw_frame_idx][person_id]['global_position'][:3] + \
            #     (person[draw_frame_idx][person_id]['pose']['b_ori'] / torch.norm(person[draw_frame_idx][person_id]['pose']['b_ori'])).cpu().numpy())
            # arrow_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(arrow), name='arrow')
            # scene.add_node(arrow_node)
            # Create a sphere using trimesh
            material = pyrender.MetallicRoughnessMaterial( metallicFactor=0.0, roughnessFactor=0.5, baseColorFactor=[1.0, 0.0, 0.0, 1.0])
            sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
            sphere_mesh_start = pyrender.Mesh.from_trimesh(sphere, material=material)
            translation_matrix_start = np.array([
                [1.0, 0.0, 0.0, self.agent_processed[focus_frame][person_id]['global_position'][0]],  # x-axis translation
                [0.0, 1.0, 0.0, self.agent_processed[focus_frame][person_id]['global_position'][1]],  # y-axis translation
                [0.0, 0.0, 1.0, self.agent_processed[focus_frame][person_id]['global_position'][2]],  # z-axis translation
                [0.0, 0.0, 0.0, 1.0]
            ])
            scene.add(sphere_mesh_start, pose = translation_matrix_start)
            material2 = pyrender.MetallicRoughnessMaterial( metallicFactor=0.0, roughnessFactor=0.5, baseColorFactor=[1.0, 1.0, 0.0, 1.0])
            sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
            sphere_mesh_end = pyrender.Mesh.from_trimesh(sphere, material=material2)
            self.agent_processed[focus_frame][person_id]['pose']['b_ori'] = self.agent_processed[focus_frame][person_id]['pose']['b_ori'].cpu()
            translation_matrix_end = np.array([
                [1.0, 0.0, 0.0, self.agent_processed[focus_frame][person_id]['global_position'][0] + (self.agent_processed[focus_frame][person_id]['pose']['b_ori'] / torch.norm(self.agent_processed[focus_frame][person_id]['pose']['b_ori']))[0,0]],  # x-axis translation
                [0.0, 1.0, 0.0, self.agent_processed[focus_frame][person_id]['global_position'][1] + (self.agent_processed[focus_frame][person_id]['pose']['b_ori'] / torch.norm(self.agent_processed[focus_frame][person_id]['pose']['b_ori']))[0,1]],  # y-axis translation
                [0.0, 0.0, 1.0, self.agent_processed[focus_frame][person_id]['global_position'][2] + (self.agent_processed[focus_frame][person_id]['pose']['b_ori'] / torch.norm(self.agent_processed[focus_frame][person_id]['pose']['b_ori']))[0,2]],  # z-axis translation
                [0.0, 0.0, 0.0, 1.0]
            ])
            scene.add(sphere_mesh_end, pose = translation_matrix_end)
            
        # Add floor
        floor = trimesh.creation.box(extents=np.array([10, 10, 0.02]),
                            transform=np.array([[1.0, 0.0, 0.0, self.agent_processed[focus_frame][person_id]['robot_pos'][0]],
                                                [0.0, 1.0, 0.0, self.agent_processed[focus_frame][person_id]['robot_pos'][1]],
                                                [0.0, 0.0, 1.0, -0.05],
                                                [0.0, 0.0, 0.0, 1.0],
                                                ]),
                            )
        floor.visual.vertex_colors = [0.3, 0.3, 0.3]
        floor_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(floor), name='floor')
        scene.add_node(floor_node)
        
        # Add other stuff
        point_l = PointLight(color=np.ones(3), intensity=40.0)
        for light_pose in LIGHT_POSES:
            light_pose_ = copy.deepcopy(light_pose)
            light_pose_[0, 3] = light_pose_[0, 3] + self.agent_processed[focus_frame][person_id]['robot_pos'][0]
            light_pose_[1, 3] = light_pose_[1, 3] + self.agent_processed[focus_frame][person_id]['robot_pos'][1]
            _ = scene.add(point_l, pose=light_pose_)
        cam = PerspectiveCamera(yfov=(np.pi / 3))
        cam_pose = np.array([
            [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 5.5+self.agent_processed[focus_frame][person_id]['robot_pos'][0]],
            [1.0, 0.0,           0.0,           0.0+self.agent_processed[focus_frame][person_id]['robot_pos'][1]],
            [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 5],
            [0.0,  0.0,           0.0,          1.0]
        ])
        cam_node = scene.add(cam, pose=cam_pose)
        axis_mesh = pyrender.Mesh.from_trimesh(trimesh.creation.axis(), smooth=False)
        robot_position = np.array([
            [1.0, 0.0, 0.0, self.agent_processed[focus_frame][person_id]['robot_pos'][0]],  # x-axis translation
            [0.0, 1.0, 0.0, self.agent_processed[focus_frame][person_id]['robot_pos'][1]],  # y-axis translation
            [0.0, 0.0, 1.0, 0],  # z-axis translation
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Add grid on floor
        # TODO: need to manage disappearing lines
        grid_lines_mesh = self.create_grid_lines_mesh(size=20, divisions=20, robot_pos=self.agent_processed[focus_frame][person_id]['robot_pos'])
        scene.add(pyrender.Mesh.from_trimesh(grid_lines_mesh))
        scene.add(axis_mesh, pose=robot_position)
        color, depth = offscreen_r.render(scene)
        ax.imshow(color)
        ax.axis('off')

    def plot_3d_human_new(self, focus_frame, ax):
        scene = pyrender.Scene(ambient_light=np.array([0.1, 0.1, 0.1, 0.1]))
        offscreen_r = OffscreenRenderer(viewport_width=640*2, viewport_height=480*2)
        triangles, verts_tran = [], None
        draw_person_idx = 0
        default_cfg = bev_settings()
        smpl_parser = SMPLA_parser(default_cfg.smpl_path, default_cfg.smil_path)
        DEFAULT_BETAS = np.array([[0.40479207,  0.17771187,  0.25489786,  0.22483926, -0.06253424, 0.03777225, -0.05464039,  0.0384823 ,  0.0689365 , -0.04577354, 0.10536828]])

        for i in range(25):
            if self.preprocessed_final['y_pose_mask'][i, 0] == True:
                continue
            
            smpl_theta = self.preprocessed_final['y_pose'][i, 0]
            draw_person_idx += 1
            verts, joints, face = smpl_parser(DEFAULT_BETAS, smpl_theta)
            verts_tran = verts
            verts_tran[...,:3] = verts_tran[...,:3] + torch.tensor([self.preprocessed_final['positions'][i, 8, :][0], self.preprocessed_final['positions'][i, 8, :][1], -torch.min(verts_tran[:, 2])]).unsqueeze(0).unsqueeze(0)
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
            
        person_id = 1 # just arbitrary value
        # Add floor
        floor = trimesh.creation.box(extents=np.array([10, 10, 0.02]),
                            transform=np.array([[1.0, 0.0, 0.0, self.agent_processed[focus_frame][person_id]['robot_pos'][0]],
                                                [0.0, 1.0, 0.0, self.agent_processed[focus_frame][person_id]['robot_pos'][1]],
                                                [0.0, 0.0, 1.0, -0.05],
                                                [0.0, 0.0, 0.0, 1.0],
                                                ]),
                            )
        floor.visual.vertex_colors = [0.3, 0.3, 0.3]
        floor_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(floor), name='floor')
        scene.add_node(floor_node)
        
        # Add other stuff
        point_l = PointLight(color=np.ones(3), intensity=40.0)
        for light_pose in LIGHT_POSES:
            light_pose_ = copy.deepcopy(light_pose)
            light_pose_[0, 3] = light_pose_[0, 3] + self.agent_processed[focus_frame][person_id]['robot_pos'][0]
            light_pose_[1, 3] = light_pose_[1, 3] + self.agent_processed[focus_frame][person_id]['robot_pos'][1]
            _ = scene.add(point_l, pose=light_pose_)
        cam = PerspectiveCamera(yfov=(np.pi / 3))
        cam_pose = np.array([
            [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 5.5+self.agent_processed[focus_frame][person_id]['robot_pos'][0]],
            [1.0, 0.0,           0.0,           0.0+self.agent_processed[focus_frame][person_id]['robot_pos'][1]],
            [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 5],
            [0.0,  0.0,           0.0,          1.0]
        ])
        cam_node = scene.add(cam, pose=cam_pose)
        axis_mesh = pyrender.Mesh.from_trimesh(trimesh.creation.axis(), smooth=False)
        robot_position = np.array([
            [1.0, 0.0, 0.0, self.agent_processed[focus_frame][person_id]['robot_pos'][0]],  # x-axis translation
            [0.0, 1.0, 0.0, self.agent_processed[focus_frame][person_id]['robot_pos'][1]],  # y-axis translation
            [0.0, 0.0, 1.0, 0],  # z-axis translation
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Add grid on floor
        # TODO: need to manage disappearing lines
        grid_lines_mesh = self.create_grid_lines_mesh(size=20, divisions=20, robot_pos=self.agent_processed[focus_frame][person_id]['robot_pos'])
        scene.add(pyrender.Mesh.from_trimesh(grid_lines_mesh))
        scene.add(axis_mesh, pose=robot_position)
        color, depth = offscreen_r.render(scene)
        ax.imshow(color)
        ax.axis('off')
 
    def plot_text_annot(self, focus_frame, ax):
        x_position = 0.1
        y_position = 0.1
        line_spacing = 0.05
        fontsize = 7
        for agent_id in self.agent_processed[focus_frame].keys():
            agent_descript = self.agent_processed[focus_frame][agent_id]['description']
            agent_sentence = f'Agent {agent_id}: {agent_descript}'
            y_position += line_spacing
            ax.text(x_position, y_position, agent_sentence, fontsize=fontsize, ha='left', va='top', transform=ax.transAxes)

        y_position += line_spacing # spacing

        for cluster_id in self.interaction_processed[focus_frame]['cluster']:
            cluster_descript = self.agent_processed[focus_frame][agent_id]['description']
            cluster_sentence = f'Cluster {cluster_id}: {cluster_descript}'
            y_position += line_spacing
            ax.text(x_position, y_position, cluster_sentence, fontsize=fontsize, ha='left', va='top', transform=ax.transAxes)

        y_position += line_spacing # spacing

        for dual_id, dual_descript in self.interaction_processed[focus_frame]['dual_interaction']:
            dual_sentence = f'Two agent {dual_id}: {dual_descript}'
            y_position += line_spacing
            ax.text(x_position, y_position, dual_sentence, fontsize=fontsize, ha='left', va='top', transform=ax.transAxes)
        
        y_position += line_spacing # spacing

        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, y_position)
        ax.axis('off')

    def plot(self, focus_frame, save_dir):
        fig, axs = plt.subplots(4, 1, figsize=(7, 15))
        
        img = cv2.imread(self.img_root + str(focus_frame).zfill(6) + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[0].imshow(img)
        axs[0].axis('off')
        
        scatter = self.plot_bev_position(10, axs[1], {})
        self.plot_3d_human(10, axs[2])
        self.plot_text_annot(10, axs[3])
        plt.show()
        plt.colorbar(scatter, ax=axs[1])
        plt.savefig(save_dir, dpi=1000)

if __name__ == "__main__":
    viz = VizWhole('gates-to-clark-2019-02-28_1', \
                   '/mnt/jaewoo4tb/textraj/preprocessed_1st/v1/bytes-cafe-2019-02-07_0_agents_0_to_1726.pt', \
                    '/mnt/jaewoo4tb/textraj/preprocessed_1st/v1/bytes-cafe-2019-02-07_0_interactions_0_to_1726.pt', \
                    '/mnt/jaewoo4tb/textraj/preprocessed_2nd/v1_fps_2_5_frame_20/bytes-cafe-2019-02-07_0_agents_0_to_1726/2.pt', 5)
    
    fig = plt.figure()
    ax = fig.add_subplot()
    viz.plot_3d_human_new(50, ax)
    plt.show()
    plt.savefig("test_pose_no_rotate.jpg", dpi=1000)