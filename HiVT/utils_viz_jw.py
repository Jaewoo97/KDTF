import numpy as np
import os, sys
# sys.path.append('/mnt/jaewoo4tb/textraj')
import torch
from tqdm import tqdm
import copy
from utils.utils import *
from utils.plot_2d_3d import *
from parser_model.captions import Caption3DPoseEngine

from matplotlib import cm as cmx
from matplotlib import colors
from matplotlib.image import imsave
jet = plt.get_cmap('twilight')
cNorm  = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)


class VizScene:
    def __init__(self, x_traj, y_traj, rotate_mat, padding_mask, y_traj_pred, verts, faces, pose_mask, robot_position, initial_frame, save_root, save_name, scene_name, y_hat_best_baseline=None, y_hat_best_baseline2=None, historical_steps=8):
        self.historical_steps = historical_steps
        self.x_traj = x_traj
        self.y_traj = y_traj
        self.rotate_mat = rotate_mat
        self.padding_mask = padding_mask
        self.y_traj_pred = y_traj_pred
        self.y_traj_pred_baseline = y_hat_best_baseline
        self.y_traj_pred_baseline2 = y_hat_best_baseline2
        self.verts = verts
        self.faces = faces
        self.pose_mask = pose_mask
        self.robot_position = robot_position[0, self.historical_steps]
        self.initial_frame = initial_frame
        self.save_root = save_root
        self.save_name = save_name
        self.scene_name = scene_name
        self.traj_offset = 0
        self.y_traj_pred_baseline_rotated = False
        self.y_traj_pred_baseline_rotated2 = False
        self.y_traj_pred_rotated = False


    def draw_frames(self, frames, PRINT_AGENT=None, draw_pose=True, draw_past_traj=True, draw_future_traj=True):
        for frame in frames:
            self.draw_single_frame(frame, PRINT_AGENT, draw_pose=draw_pose, draw_past_traj=draw_past_traj, draw_future_traj=draw_future_traj)


    def __draw_sphere(self, scene, position, color=[1.0, 0.0, 0.0, 1.0]):
        if torch.norm(position-self.robot_position, p=2, dim=-1)>9:
            return None
        material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, roughnessFactor=0.5, baseColorFactor=color)
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
        sphere_mesh = pyrender.Mesh.from_trimesh(sphere, material=material)
        position = position.cpu()
        translation_matrix = np.array([
            [1.0, 0.0, 0.0, position[0]],  # x-axis translation
            [0.0, 1.0, 0.0, position[1]],  # y-axis translation
            [0.0, 0.0, 1.0, self.traj_offset],  # z-axis translation
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene.add(sphere_mesh, pose = translation_matrix)


    def __draw_human(self, scene, position, verts, faces, colors):
        verts_tran = verts.cpu()
        verts_tran[...,:2] = verts_tran[...,:2] + position.cpu()
        verts_tran[..., 2] -= torch.min(verts_tran[..., 2])
        triangles = faces.cpu()
        body_meshes = []
        m = trimesh.Trimesh(vertices=verts_tran, faces=triangles)
        m.visual.vertex_colors[:, 3] = 255

        
        m.visual.vertex_colors[:, :3] = colors.astype(np.uint8)        
        body_mesh = pyrender.Mesh.from_trimesh(m, smooth=False)
        body_node = pyrender.Node(mesh=body_mesh, name='body')
        scene.add_node(body_node)


    def __draw_line(self, scene, start, end, color=[0.0, 0.0, 1.0, 1.0], width=0.01):
        # if torch.norm(start, p=2, dim=-1)>9:
        #     return None
        # Calculate the direction vector and length of the line
        start = start.cpu()
        end = end.cpu()
        start_3d = np.array([start[0], start[1], self.traj_offset])
        end_3d = np.array([end[0], end[1], self.traj_offset])
        
        # Calculate the direction vector and length of the line in 3D
        direction = end_3d - start_3d
        length = np.linalg.norm(direction)
        
        # Create a cylinder to represent the line, with specified width and length
        cylinder = trimesh.creation.cylinder(radius=width, height=length, sections=10)
        
        # Orient the cylinder along the line direction
        cylinder.apply_translation([0, 0, length / 2])  # Center the cylinder
        cylinder.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], direction))

        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = start_3d

        # Apply the translation matrix to the cylinder
        cylinder.apply_transform(translation_matrix)

        # Create material for the line
        material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, roughnessFactor=0.5, baseColorFactor=color)
        
        # Convert Trimesh cylinder to a pyrender Mesh
        line_render_mesh = pyrender.Mesh.from_trimesh(cylinder, material=material)
        
        # Add the cylinder to the scene
        scene.add(line_render_mesh)


    def __draw_others(self, scene):
        robot_position = self.robot_position.cpu()
        # Add floor            
        floor = trimesh.creation.box(extents=np.array([10, 10, 0.02]),
                            transform=np.array([[1.0, 0.0, 0.0, robot_position[0]],
                                                [0.0, 1.0, 0.0, robot_position[1]],
                                                [0.0, 0.0, 1.0, -0.1],
                                                [0.0, 0.0, 0.0, 1.0],
                                                ]),
                            )
        floor.visual.vertex_colors = [0.8, 0.8, 0.8]
        floor_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(floor), name='floor')
        # scene.add_node(floor_node)
        
        # Add other stuff
        point_l = PointLight(color=np.ones(3), intensity=40.0)
        for light_pose in LIGHT_POSES:
            light_pose_ = copy.deepcopy(light_pose)
            light_pose_[0, 3] = light_pose_[0, 3] + robot_position[0]
            _ = scene.add(point_l, pose=light_pose_)
            light_pose_[1, 3] = light_pose_[1, 3] + robot_position[1]
        cam = PerspectiveCamera(yfov=(np.pi / 3))
        cam_pose = np.array([
            [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 2.0+robot_position[0]],
            [1.0, 0.0,           0.0,           -3+robot_position[1]],
            [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 6.0],
            [0.0,  0.0,           0.0,          1.0]
        ])
        # cam_pose = np.array([
        #     [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 4+robot_position[0]],
        #     [1.0, 0.0,           0.0,           4.5+robot_position[1]],
        #     [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 4.0],
        #     [0.0,  0.0,           0.0,          1.0]
        # ])
        
        cam_pose = self.rotate_camera(cam_pose, -np.pi*0.5)
        # cam_pose = self.rotate_camera(cam_pose, 0)
        
        cam_node = scene.add(cam, pose=cam_pose)
        axis_mesh = pyrender.Mesh.from_trimesh(trimesh.creation.axis(), smooth=False)
        scene.add(axis_mesh)
        robot_scene = np.array([
            [1.0, 0.0, 0.0, robot_position[0]],  # x-axis translation
            [0.0, 1.0, 0.0, robot_position[1]],  # y-axis translation
            [0.0, 0.0, 1.0, 0],  # z-axis translation
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Add grid on floor
        # TODO: need to manage disappearing lines
        grid_lines_mesh = create_grid_lines_mesh(size=30, divisions=30, robot_pos=robot_position)
        scene.add(pyrender.Mesh.from_trimesh(grid_lines_mesh))

        # scene.add(axis_mesh, pose=robot_scene)

    def rotate_camera(self, cam_pose, theta):
        # Define a +90 degree rotation matrix around the z-axis (3x3 rotation part)
        rotation_z = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])

        # Extract the original rotation part (upper-left 3x3) and translation part
        original_rotation = cam_pose[:3, :3]
        translation = cam_pose[:3, 3]

        # Apply the rotation to the orientation only
        new_rotation = rotation_z @ original_rotation

        # Construct the new cam_pose with updated rotation and original position
        rotated_cam_pose = np.eye(4)
        rotated_cam_pose[:3, :3] = new_rotation
        rotated_cam_pose[:3, 3] = translation
        return rotated_cam_pose

    def save_meta(self, captions, interactions, ids):
        f = open(f"{self.save_root}/{self.save_name}.txt", "w")
        f.write(f"{self.scene_name} - {self.initial_frame}\n")

        N = self.x_traj.shape[0]

        for idx in range(N):
            human_color = np.asarray(scalarMap.to_rgba(idx / N)[:3]) * 255
            human_color = np.clip(human_color * 1.5, 0, 255) 
            caption = captions[idx]
            position = self.x_traj[idx, self.historical_steps-1]
            d = f"{idx} - id: {ids[idx]} - caption: {caption} - color: {human_color} - position: {position}\n"
            f.write(d)

        if interactions is not None:
            f.write("====dual interactions====\n")
            for cluster, description in interactions:
                d = f"ids: {cluster} - description: {description}\n"
                f.write(d)
            
            f.close()


    def draw_single_frame(self, now_frame, PRINT_AGENTS=None, draw_pose=True, draw_past_traj=True, draw_future_traj=True):
        scene = pyrender.Scene(ambient_light=np.array([0.1, 0.1, 0.1, 0.1]))
        offscreen_r = OffscreenRenderer(viewport_width=640*8, viewport_height=480*8)
        
        copy_pad_mask = self.padding_mask
        copy_pad_mask[:, :now_frame] = True
        
        N = self.x_traj.shape[0]
        in_F = self.historical_steps
        # in_F = self.x_traj.shape[1]
        out_F = self.y_traj.shape[1]

        if not self.y_traj_pred_rotated:
            inv_rot_mat = self.rotate_mat.transpose(1, 2)
            self.y_traj_pred = self.y_traj_pred @ inv_rot_mat
            self.y_traj_pred = self.y_traj_pred + self.x_traj[:, -1].unsqueeze(1)
            self.y_traj_pred_rotated = True
        if not self.y_traj_pred_baseline_rotated and self.y_traj_pred_baseline is not None:
            inv_rot_mat = self.rotate_mat.transpose(1, 2)
            self.y_traj_pred_baseline = self.y_traj_pred_baseline @ inv_rot_mat
            self.y_traj_pred_baseline = self.y_traj_pred_baseline + self.x_traj[:, -1].unsqueeze(1)
            self.y_traj_pred_baseline_rotated = True
        if not self.y_traj_pred_baseline_rotated2 and self.y_traj_pred_baseline2 is not None:
            inv_rot_mat = self.rotate_mat.transpose(1, 2)
            self.y_traj_pred_baseline2 = self.y_traj_pred_baseline2 @ inv_rot_mat
            self.y_traj_pred_baseline2 = self.y_traj_pred_baseline2 + self.x_traj[:, -1].unsqueeze(1)
            self.y_traj_pred_baseline_rotated2 = True
        
        for idx in range(N):
            if self.x_traj[idx, -1, 0].item() == 0. and self.x_traj[idx, -1, 1].item() == 0.: 
                continue
            
            # DRAW HUMANS
            if draw_pose and self.pose_mask[idx][self.historical_steps-1] == False and self.padding_mask[idx, self.historical_steps-1] == False:
                # human_color = np.asarray(scalarMap.to_rgba(idx / N)[:3]) * 255
                # human_color = np.clip(human_color * 1.5, 0, 255)  # Increase brightness by 50%, clamp to [0, 255]
                if PRINT_AGENTS is None:
                    human_color = np.asarray(scalarMap.to_rgba(idx / N)[:3]) * 255
                    human_color = np.clip(human_color * 1.5, 0, 255)  # Increase brightness by 50%, clamp to [0, 255]
                else:
                    if idx not in PRINT_AGENTS:
                        human_color = np.array([200., 200., 200.])
                    else:
                        # human_color = np.array([255., 158., 158.])
                        human_color = np.asarray(scalarMap.to_rgba(idx / N)[:3]) * 255
                        human_color = np.clip(human_color * 1.5, 0, 255)  # Increase brightness by 50%, clamp to [0, 255]
                        
                # if torch.norm(self.x_traj[idx, now_frame, :], p=2, dim=-1)>9:
                #     continue
                self.__draw_human(scene, self.x_traj[idx, self.historical_steps-1], self.verts[idx, self.historical_steps-1], self.faces[idx, self.historical_steps-1], human_color)
                
            if PRINT_AGENTS is not None:
                if idx not in PRINT_AGENTS: ####################################################################################################################
                    continue
                
            # DRAW SPHERES
            # draw x trajectory
            if draw_past_traj:    
                for t in range(in_F):
                    if copy_pad_mask[idx, t] == False:
                    # if self.pose_mask[idx][now_frame] == False and self.padding_mask[idx, t] == False:
                        if self.x_traj[idx, t, 0] == 0 and self.x_traj[idx, t, 1] == 0:
                            sdfljksdf=1
                        self.__draw_sphere(scene, self.x_traj[idx, t], [1, 0.07, 0.07, 1.0])
                
            # draw y trajectory
            if draw_future_traj:    
                for t in range(out_F):
                    if copy_pad_mask[idx, in_F + t] == False:
                        self.__draw_sphere(scene, self.y_traj[idx, t], [0.12, 0.85, 0.12, 1.0])
                
            # draw y prediction
            if draw_future_traj:    
                for t in range(out_F):
                    if self.padding_mask[idx, in_F + t] == False:
                    # if self.padding_mask[idx, in_F + t] == False and self.padding_mask[idx, now_frame] == False:
                        # self.__draw_sphere(scene, self.y_traj_pred[idx, t], [0.0, 1, 1, 1.0])
                        self.__draw_sphere(scene, self.y_traj_pred[idx, t], [0.0, 0.12, 0.38, 1.0])
                        
            # draw y prediction baseline
            if draw_future_traj and self.y_traj_pred_baseline is not None:
                for t in range(out_F):
                    if self.padding_mask[idx, in_F + t] == False:
                    # if self.padding_mask[idx, in_F + t] == False and self.padding_mask[idx, now_frame] == False:
                        self.__draw_sphere(scene, self.y_traj_pred_baseline[idx, t], [0.85, 0.85, 0, 1.0])
                    
            # draw y prediction baseline2    # TRAJ POSE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if draw_future_traj and self.y_traj_pred_baseline2 is not None:
                for t in range(out_F):
                    if self.padding_mask[idx, in_F + t] == False:
                    # if self.padding_mask[idx, in_F + t] == False and self.padding_mask[idx, now_frame] == False:
                        self.__draw_sphere(scene, self.y_traj_pred_baseline2[idx, t], [0.0, 1, 1, 1.0])

            # DRAW LINES
            # x
            prev = None
            if draw_past_traj:
                for t in range(in_F):
                    # if self.pose_mask[idx][now_frame] == False and self.padding_mask[idx, now_frame] == False:
                    # if self.pose_mask[idx][t] == False and copy_pad_mask[idx, t] == False:
                    if copy_pad_mask[idx, t] == False:
                        if torch.all(self.x_traj[idx, t].cpu() == torch.Tensor([0., 0.])):
                            sdf = 1
                        if prev is not None:    
                            self.__draw_line(scene, prev, self.x_traj[idx, t], [1, 0.1, 0.1, 0.5])
                        prev = self.x_traj[idx, t]

            prev = self.x_traj[idx, -1]
            if draw_future_traj:
                for t in range(out_F):
                    if copy_pad_mask[idx, in_F + t] == False:
                        if prev is not None:
                            self.__draw_line(scene, prev, self.y_traj[idx, t], [0.2, 0.75, 0.2, 0.5])
                        prev = self.y_traj[idx, t]

                prev = self.x_traj[idx, -1]
                for t in range(out_F):
                    # if self.padding_mask[idx, in_F + t] == False and self.padding_mask[idx, now_frame] == False:
                    if self.padding_mask[idx, in_F + t] == False:
                        if prev is not None:
                            self.__draw_line(scene, prev, self.y_traj_pred[idx, t], [0.0, 0.12, 0.38, 0.5])
                            # self.__draw_line(scene, prev, self.y_traj_pred[idx, t], [0.0, 1, 1, 1.0])
                        prev = self.y_traj_pred[idx, t]
                
                if self.y_traj_pred_baseline is not None:
                    prev = self.x_traj[idx, -1]
                    for t in range(out_F):
                        if self.padding_mask[idx, in_F + t] == False:
                        # if self.padding_mask[idx, in_F + t] == False and self.padding_mask[idx, now_frame] == False:
                            if prev is not None:
                                self.__draw_line(scene, prev, self.y_traj_pred_baseline[idx, t], [1.0, 1, 0, 0.5])
                            prev = self.y_traj_pred_baseline[idx, t]
                
                if self.y_traj_pred_baseline2 is not None:
                    prev = self.x_traj[idx, -1]
                    for t in range(out_F):
                        if self.padding_mask[idx, in_F + t] == False:
                        # if self.padding_mask[idx, in_F + t] == False and self.padding_mask[idx, now_frame] == False:
                            if prev is not None:
                                self.__draw_line(scene, prev, self.y_traj_pred_baseline2[idx, t], [0.0, 1, 1, 0.5])
                            prev = self.y_traj_pred_baseline2[idx, t]

        self.__draw_others(scene)

        color, depth = offscreen_r.render(scene)
        # print(f'Number of agents in frame {draw_frame_idx} is {person_count}')
        if self.save_root != None:
            imsave(f"{self.save_root}/{self.save_name}-{now_frame}.jpeg", color, dpi=1100)