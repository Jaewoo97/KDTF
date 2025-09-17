import numpy as np
from scipy.spatial.transform import Rotation as R_
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import time
from PIL import Image
from io import BytesIO
import sys
import tqdm
sys.path.append('/mnt/jaewoo4tb/textraj')
import pandas as pd
import csv
from bev.model import BEV
from bev.cfg import bev_settings
from bev.post_parser import *
from romp.vis_human.pyrenderer import Py3DR
from utils.utils import *
from io import StringIO
# from transformers import CLIPProcessor, CLIPModel
import copy
from utils.pllava_utils import *
from utils.pllava_eval_utils import conv_templates
import torchvision
from decord import VideoReader, cpu

def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta), np.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
rotate_x = np.radians(90)
rotate_y = np.radians(90)
rotate_z = np.radians(90)
rotate_z_ = np.radians(72)
RX = Rx(rotate_x)
RZ = Rz(rotate_z)
RY = Ry(rotate_y)
RZ_ = Rz(rotate_z_)
R = RZ
# BOX_SIZE = [120, 120]
MARGIN = [200, 160]
ORIGINAL_IMG_SIZE = [576, 720]  # h, w
# R = RX * RZ
RESOLUTION = 672
camera_fov_horizontal = 42.5
CAMERA_FOV = {1: [(72-camera_fov_horizontal, 72+camera_fov_horizontal), (72-camera_fov_horizontal, 72+camera_fov_horizontal)],
              2: [(144-camera_fov_horizontal, 144+camera_fov_horizontal), (144-camera_fov_horizontal, 144+camera_fov_horizontal)],
              3: [(216-camera_fov_horizontal, 216+camera_fov_horizontal), (216-camera_fov_horizontal, 216+camera_fov_horizontal)],
              4: [(0, camera_fov_horizontal), (320.5, 360)],
              5: [(288-camera_fov_horizontal, 288+camera_fov_horizontal), (288-camera_fov_horizontal, 288+camera_fov_horizontal)]
    }
CAPTION_DISTANCE_THRESHOLD = 10
POSE_DISTANCE_THRESHOLD = 8
FILTER_2D_BBOX_THRESHOLD = 33000
 
def concat_images_horizontally(img1, img2):
    # Get the height and width of both images
    img1 = np.array(img1)
    img2 = np.array(img2)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Determine the maximum height
    max_height = max(h1, h2)
    
    # Create black images to pad the smaller image
    if h1 < max_height:
        pad1 = np.zeros((max_height - h1, w1, 3), dtype=np.uint8)  # Assuming color images
        img1 = np.vstack((img1, pad1))
    if h2 < max_height:
        pad2 = np.zeros((max_height - h2, w2, 3), dtype=np.uint8)  # Assuming color images
        img2 = np.vstack((img2, pad2))
    
    # Concatenate the images horizontally
    result = np.concatenate((img1, img2), axis=1)
    
    return result


def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn

def rotate_coordinates(local_coords, theta):
    """
    Rotates the local coordinates of a point based on the robot's rotation around the z-axis.

    :param local_coords: tuple (x', y'), the local coordinates of the person
    :param theta: float, rotation angle in degrees
    :return: tuple (x, y), the global coordinates after rotation
    """
    # Convert theta from degrees to radians
    theta_rad = np.radians(theta)

    # Rotation matrix for counterclockwise rotation around the z-axis
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])

    # Convert local coordinates to a numpy array
    local_coords_array = np.array(local_coords)

    # Calculate global coordinates after rotation
    new_local_coords = np.dot(rotation_matrix, local_coords_array)

    return new_local_coords

class Agent:
    def __init__(self, id):
        self.id = id
        self.type = None
        self.local_pos = {}
        self.global_pos = {}
        self.bbox = {} # frame_num: [cam_id, top left x, top left y, w, h]
        self.caption = None
        self.rot_z = {}
        self.pose = None        
        self.visible = None
        self.robot_ori = None
        self.robot_pos = None
        self.frames = {}
        self.frames_original = {}
        self.pixel_pos = {}

class Socials:
    def __init__(self):
        self.clusters = {}  # key: cluster ID, value: (caption |str|, list of agent IDs |int|)
        self.interactions = {} # key: (agent_id, agent_id), value: |str| of interaction description. Order of agent_id doesn't matter
        
class Caption3DPoseEngine:
    def __init__(self, conv_mode, num_frames):
        self.HORIZONTAL_MARGIN = 100
        self.CROP_IMG_DIR = "/mnt/jaewoo4tb/textraj/temp/crop_imgs"
        self.caption_count = 0
        self.agents = None
        
        # Load video captioning model (pllava)
        self.vid_model, self.vid_processor = load_pllava('/mnt/jaewoo4tb/nayoung_ws/zeropose/VFRL/PLLaVA/MODELS/pllava-13b', num_frames=num_frames, use_lora=True, weight_dir='/mnt/jaewoo4tb/nayoung_ws/zeropose/VFRL/PLLaVA/MODELS/pllava-13b', lora_alpha=4, pooling_shape=(num_frames, 12, 12))
        self.vid_model = self.vid_model.to('cuda:0')
        self.vid_model = self.vid_model.eval()
        
        # Load 3D pose regressor
        default_cfg = bev_settings()
        self.pose_model = BEV(default_cfg)
        self.smpl_parser = SMPLA_parser(default_cfg.smpl_path, default_cfg.smil_path)

        self.conv = conv_templates[conv_mode].copy()
        prompt_1 = 'The input consists of camera view from a mobile robot. In the images, people are standing still or walking, alone or in a group. Your job is to describe the behavior of the person marked with red bounding box. What is the person doing? In one full sentence. Concisely, less than 20 words. Do not talk about clothing of person.'
        prompt_2 = 'The input consists of camera view from a mobile robot. In the images, people are standing still or walking, alone or in a group. Is there any obstacle in the way of person with red bounding box? In one full sentence. Concisely, less than 20 words.'
        prompt_3 = 'The input consists of camera view from a mobile robot. In the images, people are standing still or walking, alone or in a group. What will the person in red bounding box do in the future? In one full sentence. Concisely, less than 20 words.'
        self.prompt = {1: prompt_1, 2: prompt_2, 3: prompt_3}
        # self.prompt = 'The input consists of camera view from a mobile robot. In the images, people are standing still or walking, alone or in a group. Your job is to describe the behavior of the person marked with red bounding box. Is the person walking or standing? If the legs are moving, then the person is walking. What is the person doing? In one sentence, concise, less than 20 words. Do not talk about clothing of person.'
        print('Caption3DPoseEngine initialized.')

    def get_bbox_pts(self, pos, sceneIdx):
        if sceneIdx == 0:   # eth
            pt1, pt2 = (pos[0]-10, pos[1]), (pos[0]+10, pos[1]+40)
        elif sceneIdx == 1: # hotel
            pt1, pt2 = (pos[0]-40, pos[1]), (pos[0]+40, pos[1]+50)
        elif sceneIdx == 2: # zara1
            pt1, pt2 = (pos[0]-25, pos[1]-80), (pos[0]+25, pos[1])
        elif sceneIdx == 3: # zara2
            pt1, pt2 = (pos[0]-25, pos[1]), (pos[0]+25, pos[1]+80)
        elif sceneIdx == 4: # students
            pt1, pt2 = (pos[0]-25, pos[1]-80), (pos[0]+25, pos[1])
        
        return pt1, pt2
    
    def _make_question_single(self, agent):
        x, y = agent["local_pos"]
        q = "The position of people is {x}m, {y}m. Describe this ".format(x=x, y=y)
        print(q)


    def _crop_img_single(self, img, bbox, draw_bbox=True, hori_margin=None):
        h, w, c = img.shape
        if draw_bbox:
            cv2.rectangle(img, bbox[:2], [bbox[0] + bbox[2], bbox[1] + bbox[3]], (0, 0, 255), 2)

        big_img = np.hstack([img, img, img]) # hstack to handle bbox that overflows the boundary, may need to modify images for smooth stitching
        if hori_margin is None:
            x_start = bbox[0] + w - self.HORIZONTAL_MARGIN
            x_end = x_start + bbox[2] + 2 * self.HORIZONTAL_MARGIN
        else:
            x_start = bbox[0] + w - hori_margin
            x_end = x_start + bbox[2] + 2 * hori_margin
        
        return big_img[:, x_start:x_end]
    
    def _crop_img_dot(self, img, bbox, sceneIdx):
        # bbox: [cam #, top left x, top left y, w, h]
        h, w, c = img.shape
        
        original_img = copy.deepcopy(img)
        # Draw bounding box
        cv2.rectangle(img, (int(bbox[1]), int(bbox[2])), (int(bbox[1])+int(bbox[3]), int(bbox[2])+int(bbox[4])), (0, 0, 255), 2)

        # method 2
        x_start = int(bbox[1]) - MARGIN[0]
        x_end = x_start + int(bbox[3]) + MARGIN[0] + MARGIN[0]
        y_start = int(bbox[2]) - MARGIN[1]
        y_end = y_start + int(bbox[4]) + MARGIN[1] + MARGIN[1]
        x_start, y_start = max(0, x_start), max(0, y_start)
        x_end, y_end = min(x_end, int(w-1)), min(y_end, int(h-1))
        
        cropped_img = img[int(y_start):int(y_end), int(x_start):int(x_end)]
        cropped_original = original_img[int(bbox[2]):int(bbox[2]+bbox[4]), int(bbox[1]):int(bbox[1]+bbox[3])]
        return cropped_img, cropped_original, img
    
    def _crop_img_arrow(self, img, pos, pos_prev, sceneIdx):
        # Using cv2
        pos = [float(i) for i in pos]
        if pos_prev is not None: pos_prev = [float(i) for i in pos_prev]
        img_original = copy.deepcopy(img)
        h, w, c = img.shape
        box_size = [130, 130]
        box_size_small = [75, 75]
        
        world_coords_homogeneous = np.array([pos[0], pos[1], 1]).reshape(3, 1)

        # Apply the homography matrix to get the image coordinates in homogeneous form
        H_inv = np.linalg.inv(self.H)
        image_coords_homogeneous = np.dot(H_inv, world_coords_homogeneous)

        # Convert back from homogeneous to image coordinates
        u, v = image_coords_homogeneous[0] / image_coords_homogeneous[2], image_coords_homogeneous[1] / image_coords_homogeneous[2]
        
        if pos_prev is not None:
            world_coords_homogeneous_prev = np.array([pos_prev[0], pos_prev[1], 1]).reshape(3, 1)
            image_coords_homogeneous_prev = np.dot(H_inv, world_coords_homogeneous_prev)
            u_prev, v_prev = image_coords_homogeneous_prev[0] / image_coords_homogeneous_prev[2], image_coords_homogeneous_prev[1] / image_coords_homogeneous_prev[2]
            
        if sceneIdx >1:
        # if sceneIdx != 1: 
            pos = np.array((u[0], v[0]))
            if pos_prev is not None: pos_prev = np.array((u_prev[0], v_prev[0]))
        else: 
            pos = np.array((v[0], u[0]))
            if pos_prev is not None: pos_prev = np.array((v_prev[0], u_prev[0]))
        assert pos[0] < 720 and pos[1] < 576
        if pos_prev is not None: assert pos_prev[0] < 720 and pos_prev[1] < 576
        
        if pos_prev is None or np.linalg.norm(pos_prev-pos, ord=2, axis=-1) < 2:
            pos = (int(round(pos[0])), int(round(pos[1])))
            cv2.circle(img, pos, 7, (0, 0, 255), -1, cv2.LINE_AA)
        else:
            pos = (int(round(pos[0])), int(round(pos[1])))
            pos_prev = (int(round(pos_prev[0])), int(round(pos_prev[1])))
            cv2.arrowedLine(img, pos_prev, pos, (0, 0, 255), 2, tipLength=0.3)

        x_start = pos[0] - box_size[0]
        x_end = x_start + 2 * box_size[0]
        y_start = pos[1] - box_size[1]
        y_end = y_start + 2 * box_size[1]
        
        x_start, y_start = max(0, x_start), max(0, y_start)
        x_end, y_end = min(x_end, w), min(y_end, h)
        
        x_start_small = pos[0] - box_size_small[0]
        x_end_small = x_start_small + 2 * box_size_small[0]
        y_start_small = pos[1] - box_size_small[1]
        y_end_small = y_start_small + 2 * box_size_small[1]
        
        x_start_small, y_start_small = max(0, x_start_small), max(0, y_start_small)
        x_end_small, y_end_small = min(x_end_small, w), min(y_end_small, h)
        
        img_cropped = img[int(y_start):int(y_end), int(x_start):int(x_end)]
        img_cropped_small = img_original[int(y_start_small):int(y_end_small), int(x_start_small):int(x_end_small)]
        return img_cropped, img, img_cropped_small
    
    def _cluster_agents(self, thr=1.7):
        # simply group all two agents those close enough
        clusters = []
        for center_agent in self.agents:
            for neighbor_agent in self.agents:
                if center_agent == neighbor_agent:
                    continue
                
                if np.linalg.norm(center_agent.local_pos - neighbor_agent.local_pos) < thr:
                    pair = sorted([center_agent, neighbor_agent], key=lambda a:a.id)
                    if pair not in clusters:
                        clusters.append(pair)

        return clusters


    def _crop_img_dual(self, img, pair, draw_bbox=True):
        h, w, c = img.shape
        bbox0 = pair[0].bbox
        bbox1 = pair[1].bbox

        if draw_bbox:
            cv2.rectangle(img, bbox0[:2], [bbox0[0] + bbox0[2], bbox0[1] + bbox0[3]], (0, 0, 255), 2)
            cv2.rectangle(img, bbox1[:2], [bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]], (0, 0, 255), 2)
        
        big_img = np.hstack([img, img, img]) # hstack to handle bbox that overflows the boundary, may need to modify images for smooth stitching

        if bbox0[0] < bbox1[0]:
            a = bbox0[0]
            da = bbox0[2]
            b = bbox1[0]
            db = bbox1[2]
        else:
            b = bbox0[0]
            db = bbox0[2]
            a = bbox1[0]
            da = bbox1[2]
        
        if (2 * b + db - 2 * a - da - w) < 0:
            x_start = a - self.HORIZONTAL_MARGIN + w
            x_end = b + db + self.HORIZONTAL_MARGIN + w
        else:
            x_start = b - self.HORIZONTAL_MARGIN + w
            x_end = a + da + w + self.HORIZONTAL_MARGIN + w
        
        return big_img[:, x_start:x_end]

    def load_files(self, annot3d_dir, annot2d_dir, img_dir, ego_dir):
        print('Initializing annotations. . .')
        self.annot3d_dir = annot3d_dir
        self.annot2d_dir = annot2d_dir
        self.img_dir = img_dir
        self.ego_dir = ego_dir        
        self.caption_count = 0

    def preprocess_frame(self, target_frame, sceneIdx, use_frame):
        ''' Process 3D, 2D bbox annotation (3Dbbox reference for agents in this scene)'''
        # Process robot pose
        ego_traj = open(os.path.join(self.ego_dir, f'{target_frame}.txt'), 'r')
        ego_traj_ = ego_traj.readlines()[0]
        ego_traj.close()
        ego_traj_ = np.array(ego_traj_.split(',')).astype(np.float64).reshape(4,4)
        ego_pos = ego_traj_[:2, -1]
        forward_vector = matrix_to_forward_vector(ego_traj_[:3, :3])
        ego_yaw = extract_yaw_angle_from_matrix(ego_traj_)
        ego_yaw = ego_yaw * 180 / np.pi
        # print(f'frame: {target_frame}, robot_pos: {ego_pos}, robot_yaw: {ego_yaw}')
        # Process 3d bbox annotation
        annot3d = open(os.path.join(self.annot3d_dir, f'{target_frame}.txt'), 'r')
        annot3d_list = annot3d.readlines()
        annot3d.close()
        annot3d = {}
        for annot_3d_ in annot3d_list:
            annot_3d = annot_3d_.split(' ')
            for i_ in range(2,9): annot_3d[i_] = float(annot_3d[i_])
            angle = angle_around_z_axis(ego_traj_[:2, 3], (annot_3d[5], annot_3d[6]), forward_vector)
            
            if 'Pedestrian' in annot_3d[0]:
                agent_id = int(annot_3d[1].split(':')[1])
                if angle < 0: angle += 360
                assert angle > 0 and angle < 360
                annot3d[agent_id] = [annot_3d[5], annot_3d[6], annot_3d[8], angle]     # x, y, rot_z, angle around robot
                
        annot2d = open(os.path.join(self.annot2d_dir, f'{target_frame}.txt'), 'r')
        annot2d_list = annot2d.readlines()
        annot2d.close()
        annot2d = {}
        cam_bbox_ = {1:{}, 2:{}, 3:{}, 4:{}, 5:{}}
        for annot2d_j in annot2d_list:
            annot2d_ = annot2d_j.split(' ')
            if 'Pedestrian' not in annot2d_[1].split(':')[0]: continue
            annot2d_[2], annot2d_[3], annot2d_[4], annot2d_[5], annot2d_[6] = int(annot2d_[2]), float(annot2d_[3]), float(annot2d_[4]), float(annot2d_[5]), float(annot2d_[6])
            agent_id_temp = int(annot2d_[1].split(':')[1])
            if agent_id_temp not in annot3d.keys(): continue    # if no 3d annotation, skip this agent
            if (annot3d[agent_id_temp][3] > CAMERA_FOV[int(annot2d_[2])][0][0] and annot3d[agent_id_temp][3] < CAMERA_FOV[int(annot2d_[2])][0][1]) \
                or (annot3d[agent_id_temp][3] > CAMERA_FOV[int(annot2d_[2])][1][0] and annot3d[agent_id_temp][3] < CAMERA_FOV[int(annot2d_[2])][1][1]): # Filter error 2d annotations
                    if agent_id_temp not in annot2d.keys():
                        cam_bbox_[annot2d_[2]][agent_id_temp] = [annot2d_[3], annot2d_[4], annot2d_[5], annot2d_[6]]
                        annot2d[agent_id_temp] = [annot2d_[2], annot2d_[3], annot2d_[4], annot2d_[5], annot2d_[6]]  # cam #, top left x, top left y, w, h
                    else: # Choose annotation that is closer to the center of image
                        center_annot_org = annot2d[agent_id_temp][1] + annot2d[agent_id_temp][3]/2
                        center_annot_new = annot2d_[3] + annot2d_[5]/2
                        if abs(center_annot_new-960) < abs(center_annot_org-960): # New annotation is closer to the center of image
                            annot2d[agent_id_temp] = [annot2d_[2], annot2d_[3], annot2d_[4], annot2d_[5], annot2d_[6]]  # cam #, top left x, top left y, w, h
                            cam_bbox_[annot2d_[2]][agent_id_temp] = [annot2d_[3], annot2d_[4], annot2d_[5], annot2d_[6]]
        deleted_agents = []
        occluded_boxes = []
        small_bbox = []
        for cam_bbox_idx in range(1,6):
            boxes = [(agent_id_temp, cam_bbox_[cam_bbox_idx][agent_id_temp][0], cam_bbox_[cam_bbox_idx][agent_id_temp][1], cam_bbox_[cam_bbox_idx][agent_id_temp][2], cam_bbox_[cam_bbox_idx][agent_id_temp][3]) for agent_id_temp in cam_bbox_[cam_bbox_idx].keys()]

            # Track if a box is occluded
            occluded = [False] * len(boxes)
            small_bbox_ = [False] * len(boxes)

            # Iterate over all pairs of boxes
            for i in range(len(boxes)):
                if occluded[i]:
                    continue  # Skip if this box is already marked as occluded
                    
                for j in range(i + 1, len(boxes)):
                    if occluded[j]:
                        continue  # Skip if this box is already marked as occluded

                    # Convert both boxes to (xmin, ymin, xmax, ymax) format
                    box1_corners = box_to_corners(boxes[i])
                    box2_corners = box_to_corners(boxes[j])

                    # Compute IoU between the two boxes
                    iou = compute_iou(box1_corners, box2_corners)

                    # Check which box is smaller
                    box1_area = (box1_corners[2] - box1_corners[0]) * (box1_corners[3] - box1_corners[1])
                    box2_area = (box2_corners[2] - box2_corners[0]) * (box2_corners[3] - box2_corners[1])

                    if iou > 0.9:
                        # Mark the smaller box as occluded
                        if box1_area < box2_area:
                            occluded[i] = True
                        else:
                            occluded[j] = True
            
            # Filter too small bounding boxes
            for i in range(len(boxes)):
                if occluded[i]:
                    continue  # Skip if this box is already marked as occluded
                if boxes[i][3] * boxes[i][4] < FILTER_2D_BBOX_THRESHOLD:    # Not occluded, but small bbox (save only 3d location)
                    # occluded[i] = True
                    small_bbox_[i] = True

            # Collect all non-occluded boxes / small bbox
            for i in range(len(boxes)):
                if occluded[i]:
                    occluded_boxes.append(boxes[i][0])      # Agent id of occluded 2dbboxes
                if small_bbox_[i]:
                    small_bbox.append(boxes[i][0])      # Agent id of not occluded but small bbox
                    
        annot2d_copy = copy.deepcopy(annot2d)
        for agent_id_temp in annot2d_copy.keys():
            if agent_id_temp in occluded_boxes:
                del annot2d[agent_id_temp]
        imgs = []
        # /ssd4tb/sit_new/Cafe_street_1-002/cam_img/3/data_rgb/175.png
        for img_idx in range(1,6):
            imgs.append(cv2.imread(self.img_dir + f'/{img_idx}/data_rgb/' + str(target_frame) + ".png"))
        if self.agents is None: self.agents = {}
        # for annot_3d_ in annot3d_list:
        #     annot_3d = annot_3d_.split(' ')
        #     if annot_3d[0] == 'Pedestrian':
        #         agent_id = int(annot_3d[1].split(':')[1])
        for agent_id in annot3d.keys():
            if int(agent_id) not in annot2d.keys():     # If occluded, continue
                continue    
            if agent_id not in self.agents.keys():
                temp = Agent(int(agent_id))
                temp.type = 'pedestrian'
                temp.global_pos[target_frame] = np.array((annot3d[agent_id][0], annot3d[agent_id][1], 0)).astype(np.float64)
                temp.bbox[target_frame] = annot2d[agent_id]
                temp.rot_z[target_frame] = annot3d[agent_id][2]
                temp.robot_pos = ego_pos
                temp.robot_ori = ego_yaw
                if agent_id not in small_bbox and np.linalg.norm(ego_pos-temp.global_pos[target_frame][:2], ord=2, axis=-1) < CAPTION_DISTANCE_THRESHOLD and np.linalg.norm(ego_pos-temp.global_pos[target_frame][:2], ord=2, axis=-1) > 1.0:
                    cropped_img, cropped_img_original, _ = self._crop_img_dot(copy.deepcopy(imgs[temp.bbox[target_frame][0]-1]), temp.bbox[target_frame], sceneIdx)
                    if 0 in cropped_img_original.shape:
                        continue
                    temp.frames[target_frame] = cropped_img
                    temp.frames_original[target_frame] = cropped_img_original
                    self.agents[int(agent_id)] = temp
            else:
                self.agents[int(agent_id)].global_pos[target_frame] = np.array((annot3d[agent_id][0], annot3d[agent_id][1], 0)).astype(np.float64)
                self.agents[int(agent_id)].rot_z[target_frame] = annot3d[agent_id][2]
                self.agents[int(agent_id)].bbox[target_frame] = annot2d[agent_id]
                self.agents[int(agent_id)].robot_pos = ego_pos
                self.agents[int(agent_id)].robot_ori = ego_yaw
                if agent_id not in small_bbox and np.linalg.norm(ego_pos-self.agents[int(agent_id)].global_pos[target_frame][:2], ord=2, axis=-1) < CAPTION_DISTANCE_THRESHOLD and np.linalg.norm(ego_pos-self.agents[int(agent_id)].global_pos[target_frame][:2], ord=2, axis=-1) > 1.0:
                    cropped_img, cropped_img_original, _ = self._crop_img_dot(copy.deepcopy(imgs[self.agents[int(agent_id)].bbox[target_frame][0]-1]), self.agents[int(agent_id)].bbox[target_frame], sceneIdx)
                    if 0 in cropped_img_original.shape:
                        continue
                    self.agents[int(agent_id)].frames[target_frame] = cropped_img
                    self.agents[int(agent_id)].frames_original[target_frame] = cropped_img_original
                agents_copy = copy.deepcopy(self.agents)
                for frame_key in self.agents[int(agent_id)].frames.keys():
                    if frame_key < target_frame - use_frame:
                        del agents_copy[int(agent_id)].frames[frame_key]
                        del agents_copy[int(agent_id)].frames_original[frame_key]
                self.agents = agents_copy
        

    def caption_multiframe(self, frame_idx, sceneIdx, numFrames2use, cap_frame_save_dir, use_prompt=1):
        for agent_id in self.agents.keys():
            agent = self.agents[agent_id]
            agent_vid = []
            # agent_vid = np.zeros((0, BOX_SIZE[0]*2, BOX_SIZE[1]*2, 3))
            num_false = 0
            frames2repeat = 0
            for frame_net in range(numFrames2use,-1,-1):
                if frame_idx - frame_net in self.agents[agent_id].frames.keys():
                    agent_vid.append(self.agents[agent_id].frames[frame_idx - frame_net])
                    if frames2repeat != 0:
                        for _ in range(frames2repeat): agent_vid.append(agent_vid[-1])
                        frames2repeat = 0
                else:
                    num_false += 1
                    if len(agent_vid) > 0:
                        agent_vid.append(agent_vid[-1]) # if not present in this frame, copy previous frame 
                    else:
                        frames2repeat += 1
            if num_false > numFrames2use//2: continue
                    
            agent_vid = np.array(pad_to_largest(agent_vid))
            vid, msg = load_video(agent_vid, num_segments=agent_vid.shape[0], num_frames=agent_vid.shape[0], return_msg=False, resolution=RESOLUTION)
            prompt = self.prompt[use_prompt]
            self.conv.user_query(prompt, is_mm=True)
            llm_response, conv = pllava_answer(conv=self.conv, model=self.vid_model, processor=self.vid_processor, do_sample=True, img_list=vid, max_new_tokens=50, print_res=False, prompt_override=prompt)
            add_line_to_txt(f'{cap_frame_save_dir}/_captions.txt', f'Frame {frame_idx} / caption #: {self.caption_count}, {llm_response}')
            if self.caption_count < 40:
                for image_idx, image_ in enumerate(vid):
                    image_.save(f'{cap_frame_save_dir}/scene{sceneIdx}_{self.caption_count}_agent{agent_id}_frame{frame_idx}_iter{image_idx}.png')
            # caption = caption[caption.find(next(filter(str.isalpha, caption))):]
            self.caption_count += 1
            agent.caption = llm_response

    def caption_dual(self):
        pairs = self._cluster_agents()
        self.pairs = []
        for pair in pairs:
            temp_img = self.img.copy()
            pair_img = self._crop_img_dual(temp_img, pair)
            # img_name = self.CROP_IMG_DIR + self.target_frame + "_" + str(pair[0].id) + "_" + str(pair[1].id) +".jpg"
            # cv2.imwrite(img_name, pair_img)
            img = Image.fromarray(pair_img)
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            buffer.seek(0)
            jpeg_image = Image.open(buffer)
            caption = self.caption_model.get_caption(jpeg_image, "[INST] <image>\nIs there are active interaction between two people indicated by red bounding box? Explain the type of interaction in 30 words. [/INST]")
            self.pairs.append({"pair": pair, "caption": caption})        
    
    def regress_3dpose(self, frame_idx):
        for agent_id in self.agents.keys():
            agent = self.agents[agent_id]
            if frame_idx not in agent.global_pos.keys(): continue
            if np.linalg.norm(agent.global_pos[frame_idx][:2]-agent.robot_pos, ord=2, axis=-1)<POSE_DISTANCE_THRESHOLD and np.linalg.norm(agent.global_pos[frame_idx][:2]-agent.robot_pos, ord=2, axis=-1)>1.5:
                if frame_idx not in agent.frames_original.keys(): continue
                # crop_img = self._crop_img_single(temp_img, bbox, draw_bbox=False, hori_margin=35)
                try:
                    pose = self.pose_model.forward_parse(agent.frames_original[frame_idx])
                except:
                    pose = None
                if pose is not None:
                    if pose['verts'].shape[0] > 1: 
                        choose_agent = np.argmin(np.abs(pose['cam'][:,0]))
                        pose['smpl_thetas'] = np.expand_dims(pose['smpl_thetas'][choose_agent], 0)
                        pose['smpl_betas'] = np.expand_dims(pose['smpl_betas'][choose_agent], 0)
                    pose['smpl_thetas'] = rotate_root(pose['smpl_thetas'], -90, 'x')
                    verts, joints, face = self.smpl_parser(pose['smpl_betas'], pose['smpl_thetas'])
                    b_ori = get_b_ori(joints)
                    b_ori_theta = torch.atan2(b_ori[0, 1], b_ori[0, 0]) * (180/np.pi)
                    rot_z = (agent.rot_z[frame_idx] * (180/np.pi))
                    # rot_z = (agent.rot_z * (180/np.pi)) - 180
                    robot_ori = agent.robot_ori
                    # observation_angle = agent.observation_angle * (180/np.pi)
                    pose['smpl_thetas'] = rotate_root(pose['smpl_thetas'], -b_ori_theta - rot_z + 180, 'z')   # 이게 맞을듯?
                    # pose['smpl_thetas'] = rotate_root(pose['smpl_thetas'], -b_ori_theta - rot_z, 'z')
                    verts, joints, face = self.smpl_parser(pose['smpl_betas'], pose['smpl_thetas'])
                    b_ori = get_b_ori(joints)
                    pose.update({'verts': verts, 'joints': joints, 'smpl_face':face, 'b_ori':b_ori})
                    agent.pose = pose
                    agent.global_pos[frame_idx] = agent.global_pos[frame_idx] - [0, 0, joints[:,:,2].min().cpu().numpy()]

def local_to_global_coordinates(x_local, y_local, yaw_angle):
    """
    Transforms the local (x, y) coordinates of the robot to global coordinates
    based on the yaw angle (rotation around the z-axis).
    
    Parameters:
    - x_local, y_local: Coordinates in the robot's local frame.
    - yaw_angle: Yaw angle in radians (rotation of the robot in the global frame).
    
    Returns:
    - x_global, y_global: Transformed coordinates in the global frame.
    """
    # Create the 2D rotation matrix based on the yaw angle
    yaw_angle = yaw_angle * np.pi / 180
    rotation_matrix = np.array([
        [np.cos(yaw_angle), -np.sin(yaw_angle)],
        [np.sin(yaw_angle), np.cos(yaw_angle)]
    ])
    
    # Local coordinates as a vector
    local_coords = np.array([x_local, y_local])
    
    # Apply the rotation matrix to get global coordinates
    global_coords = rotation_matrix @ local_coords
    
    return global_coords[0], global_coords[1]

def get_b_ori(joints):
    x_axis = joints[:, 2, :] - joints[:, 1, :]
    z_axis = joints[:, 0, :] - joints[:, 12, :]
    # x_axis[:, -1] = 0
    # z_axis = torch.cuda.FloatTensor([[0, 0, 1]], device='cuda').repeat(x_axis.shape[0], 1)
    y_axis = torch.cross(x_axis.cuda(), z_axis.cuda(), dim=-1)
    b_ori = y_axis[:, :3]  # body forward dir of GAMMA is y axis
    return b_ori

def rotate_smpl_thetas_toLidar(smpl_thetas):
    smpl_thetas[0][0] = smpl_thetas[0][0]-(np.pi/2)
    return smpl_thetas

def rotate_smpl_thetas_rotateZ(smpl_thetas, b_ori, rot_z, robot_ori):
    robot_ori = robot_ori / 180.0
    b_ori = torch.atan2(b_ori[0, 1], b_ori[0, 0])
    smpl_thetas[0][1] = smpl_thetas[0][1]-np.pi
    return smpl_thetas

def rotate_root(pose, rotation_angle_degrees, axis):
    # Convert the angle to radians
    if torch.is_tensor(rotation_angle_degrees): 
        if rotation_angle_degrees.device.type=='cuda': rotation_angle_degrees = rotation_angle_degrees.cpu()
    pose = pose.squeeze()
    rotation_angle_radians = np.radians(rotation_angle_degrees)
    rotation_matrix = R_.from_euler(axis, rotation_angle_radians).as_matrix()
    root_rotation_matrix = R_.from_rotvec(pose[:3]).as_matrix()
    new_root_rotation_matrix = rotation_matrix @ root_rotation_matrix
    new_root_rotation_vector = R_.from_matrix(new_root_rotation_matrix).as_rotvec()
    new_pose = np.copy(pose)
    new_pose[:3] = new_root_rotation_vector
    return np.expand_dims(new_pose, 0)

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def load_video(video, num_segments=8, return_msg=False, num_frames=4, resolution=336):
    video = video[:, :, :, ::-1]
    transforms = torchvision.transforms.Resize(size=resolution)
    # vr = VideoReader(video, ctx=cpu(0), num_threads=1)
    
    # frame_indices = get_index(num_frames, num_segments)
    frame_indices = np.linspace(0, num_frames-1, num_frames)
    images_group = []
    for frame_index in frame_indices:
        frame = video[int(frame_index)]
        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
        img = Image.fromarray(frame)
        images_group.append(transforms(img))
    if return_msg:
        raise Exception('Not implemented yet')
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return images_group, msg
    else:
        return images_group, None

def matrix_to_forward_vector(rotation_matrix):
    """
    Extracts the forward vector from the 3x3 rotation matrix.
    In a Z-up system, the forward vector in the local frame is along the x-axis (1, 0, 0).
    """
    # Forward vector in the local frame (1, 0, 0)
    local_forward = np.array([1, 0, 0])
    
    # Rotate the local forward vector to the world frame using the rotation matrix
    world_forward = rotation_matrix @ local_forward
    
    return world_forward

def angle_around_z_axis(robot_position, object_position, forward_vector):
    """
    Calculates the angle around the z-axis between the robot's forward vector
    and the position of an object relative to the robot.
    """
    # Calculate the relative position of the object
    relative_position = np.array(object_position) - np.array(robot_position)
    
    # Project the forward vector and the relative position onto the xy-plane
    forward_xy = forward_vector[:2]  # Forward vector in the xy-plane
    relative_xy = relative_position[:2]  # Object relative position in the xy-plane
    
    # Normalize the vectors to avoid scaling effects
    forward_xy /= np.linalg.norm(forward_xy)
    relative_xy /= np.linalg.norm(relative_xy)
    
    # Calculate the angle between the forward vector and the relative position
    angle = np.arctan2(relative_xy[1], relative_xy[0]) - np.arctan2(forward_xy[1], forward_xy[0])
    
    # Normalize the angle to the range [-pi, pi]
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    
    return np.degrees(angle)

def extract_yaw_angle_from_matrix(rotation_matrix):
    """
    Extracts the yaw angle (rotation around the z-axis) from a 3x3 rotation matrix.
    """
    # The yaw angle is derived from the elements of the rotation matrix
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    
    return yaw

def box_to_corners(box):
    """
    Convert bounding box from (top_left_x, top_left_y, width, height) 
    to (xmin, ymin, xmax, ymax).
    """
    xmin = box[1]
    ymin = box[2]
    xmax = box[1] + box[3]  # top_left_x + width
    ymax = box[2] + box[4]  # top_left_y + height
    return [xmin, ymin, xmax, ymax]

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    Each box is represented by (xmin, ymin, xmax, ymax).
    """

    # Determine the (x, y) coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute the area of intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the union area
    union_area = box1_area + box2_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area
    return iou

def is_occlusion(box1, box2, threshold=0.5):
    """
    Determine if one bounding box is occluding the other based on IoU.
    If the IoU of the smaller bounding box is greater than a threshold, it is considered occluded.
    """

    # Convert both boxes to (xmin, ymin, xmax, ymax) format
    box1_corners = box_to_corners(box1)
    box2_corners = box_to_corners(box2)

    # Compute the area of both boxes
    box1_area = (box1_corners[2] - box1_corners[0]) * (box1_corners[3] - box1_corners[1])
    box2_area = (box2_corners[2] - box2_corners[0]) * (box2_corners[3] - box2_corners[1])

    # Identify the smaller box
    if box1_area < box2_area:
        smaller_box = box1_corners
        larger_box = box2_corners
    else:
        smaller_box = box2_corners
        larger_box = box1_corners

    # Compute IoU between the two boxes
    iou = compute_iou(smaller_box, larger_box)

    # Check if the IoU is above the occlusion threshold
    return iou > threshold, iou

def add_line_to_txt(filename, text_line):
    # Open the file in append mode, this will create the file if it doesn't exist
    with open(filename, 'a') as file:
        # Add the line of text to the file
        file.write(text_line + '\n')

def pad_to_largest(images):
    """
    Pads a list of images (numpy arrays) to match the size of the largest image in the list.
    The images are centered during padding.
    
    Args:
    - images: List of numpy arrays representing images. Each array has shape (height, width, channels).
    
    Returns:
    - padded_images: List of padded numpy arrays where each array matches the size of the largest image.
    """
    # Find the largest height and width among all images
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)

    padded_images = []

    for image in images:
        height, width, channels = image.shape

        # Compute padding amounts
        pad_top = (max_height - height) // 2
        pad_bottom = max_height - height - pad_top
        pad_left = (max_width - width) // 2
        pad_right = max_width - width - pad_left

        # Pad the image (constant padding with 0, you can change the constant as needed)
        padded_image = np.pad(
            image, 
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
            mode='constant', 
            constant_values=0
        )

        padded_images.append(padded_image)

    return padded_images

if __name__ == "__main__":
    target_frame = 20
    target_label_2d = "/ssd4tb/jaewoo/t2p/jrdb/train_dataset/labels/labels_2d_stitched/bytes-cafe-2019-02-07_0.json"
    target_label_3d = "/ssd4tb/jaewoo/t2p/jrdb/train_dataset/labels/labels_3d/bytes-cafe-2019-02-07_0.json"
    img_dir = "/ssd4tb/jaewoo/t2p/jrdb/train_dataset/images/image_stitched/bytes-cafe-2019-02-07_0/"
    test = Caption3DPoseEngine()
    test.load_files(target_label_2d, target_label_3d, img_dir)
    test.preprocess_frame(target_frame) 

    test.caption_single()
    test.caption_dual()

    for agent in test.agents:  
        print(str(agent.id), agent.caption)
    
    for pair in test.pairs:
        print(str(pair["pair"][0].id), "<->", str(pair["pair"][1].id), pair["caption"])
        