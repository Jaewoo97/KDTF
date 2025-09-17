import math
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
from transformers import CLIPProcessor, CLIPModel
import copy
from utils.pllava_utils import *
from utils.pllava_eval_utils import conv_templates
from decord import VideoReader, cpu
import skimage.segmentation as segmentation

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
BOX_SIZE = [120, 120]
ORIGINAL_IMG_SIZE = [576, 720]  # h, w
# R = RX * RZ
RESOLUTION = 376
OBSTACLE_THRSHLD = [14, 28, 28, 28, 27]       # pixels
WALK_THRSHLD = [10, 20, 17, 18, 11]           # pixels
WALK_SLOW_THRSHLD = [1, 1.5, 1.5, 1.5, 1.5]      # pixels
CENTROIDS = {
    0: []
}
# IMAGE_DIRS = ['eth', 'hotel', 'zara1', 'zara2', 'students']

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
        pad1 = np.zeros((max_height - h1, w1, 3), dtype=np.uint)  # Assuming color images
        img1 = np.vstack((img1, pad1))
    if h2 < max_height:
        pad2 = np.zeros((max_height - h2, w2, 3), dtype=np.uint)  # Assuming color images
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
        self.global_pos = np.array([0, 0, 0], dtype=np.float32)
        self.bbox = [0, 0, 0, 0]
        self.caption = None
        self.rot_z = 0
        self.pose = None        
        self.visible = None
        self.robot_ori = None
        self.robot_pos = None
        self.frame = None
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

        # Load the smaller CLIP model and processor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.model = self.model.cuda()
        print('Image encoder initialized.')

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
    
    def _crop_img_dot(self, img, pos, sceneIdx):
        pos = [float(i) for i in pos]
        h, w, c = img.shape
        fig, ax = plt.subplots(figsize=(10, 10))
        
        world_coords_homogeneous = np.array([pos[0], pos[1], 1]).reshape(3, 1)

        # Apply the homography matrix to get the image coordinates in homogeneous form
        H_inv = np.linalg.inv(self.H)
        image_coords_homogeneous = np.dot(H_inv, world_coords_homogeneous)

        # Convert back from homogeneous to image coordinates
        u, v = image_coords_homogeneous[0] / image_coords_homogeneous[2], image_coords_homogeneous[1] / image_coords_homogeneous[2]
        
        if sceneIdx >1:
        # if sceneIdx != 1: 
            pos = np.array((u[0], v[0]))
        else: 
            pos = np.array((v[0], u[0]))
        assert pos[0] < 720 and pos[1] < 576
        pos = (int(round(pos[0])), int(round(pos[1])))
        
        x_start = pos[0] - BOX_SIZE[0]
        if x_start < 0:
            x_start = 0
        x_end = x_start + 2 * BOX_SIZE[0]
        if x_end > w:
            x_end = w - 1
            x_start = x_end - 2 * BOX_SIZE[0]
        y_start = pos[1] - BOX_SIZE[1]
        if y_start < 0:
            y_start = 0
        y_end = y_start + 2 * BOX_SIZE[1]
        if y_end > w:
            y_end = w - 1
            y_start = y_end - 2 * BOX_SIZE[1]
        
        # x_start, y_start = max(0, x_start), max(0, y_start)
        # x_end, y_end = min(x_end, w), min(y_end, h)
        cropped_img = img[int(y_start):int(y_end), int(x_start):int(x_end)]
        assert y_end-y_start == BOX_SIZE[0]*2 and x_end-x_start == BOX_SIZE[1]*2
        return cropped_img, img, [y_start, y_end, x_start, x_end]
    
    
    def _global2img(self, pos, sceneIdx):
        pos = [float(i) for i in pos]
        
        world_coords_homogeneous = np.array([pos[0], pos[1], 1]).reshape(3, 1)

        # Apply the homography matrix to get the image coordinates in homogeneous form
        H_inv = np.linalg.inv(self.H)
        image_coords_homogeneous = np.dot(H_inv, world_coords_homogeneous)

        # Convert back from homogeneous to image coordinates
        u, v = image_coords_homogeneous[0] / image_coords_homogeneous[2], image_coords_homogeneous[1] / image_coords_homogeneous[2]
        
        if sceneIdx >1:
        # if sceneIdx != 1: 
            pos = np.array((u[0], v[0]))
        else: 
            pos = np.array((v[0], u[0]))
        if pos[0] <= 720 and pos[1] <= 576: return pos
        else: return None
    
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

    def load_files(self, traj_dir, img_dir, homography, map):
        self.traj_data = {}  # {frame: {agent_id: data} }
        print('Loading trajectory data. . .')
        with open(traj_dir, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in tqdm.tqdm(reader):
                data_line = row[0].split('\t')
                cur_frame, cur_agent = int(float(data_line[0]))//10, int(float(data_line[1]))
                if cur_frame not in self.traj_data.keys():
                    self.traj_data[cur_frame] = {}
                if cur_agent not in self.traj_data[cur_frame].keys():
                    self.traj_data[cur_frame][cur_agent] = np.array((data_line[2], data_line[3]))  # x,y data
        self.img_dir = img_dir
        self.H = homography
        self.map = plt.imread(map) * 255.0
        self.map_boundary = segmentation.find_boundaries(self.map.astype(np.uint8), connectivity=1, mode='inner', background=1)
        
    def preprocess_frame(self, target_frame, sceneIdx, use_frame):
        ''' Process frame, target_frame is the 10 divided frame'''
        self.target_frame = f'{target_frame:06d}'
        # if visualize:
        img = cv2.imread(self.img_dir + '/' + str(int(self.target_frame)*10).zfill(4) + ".jpg")
        # self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        if self.agents is None: self.agents = {}
        
        if target_frame in self.traj_data.keys():
            for agent_id in self.traj_data[target_frame].keys():
                if agent_id not in self.agents.keys():
                    temp = Agent(int(agent_id))
                    temp.type = 'pedestrian'
                    temp.local_pos[target_frame] = self.traj_data[target_frame][agent_id]
                    # print(temp.local_pos[target_frame])
                    pixel_pos = self._global2img(temp.local_pos[target_frame], sceneIdx)
                    if pixel_pos is not None:
                        temp.pixel_pos[target_frame] = pixel_pos
                        img_cropped, _, _ = self._crop_img_dot(img, temp.local_pos[target_frame], sceneIdx)
                        temp.frame = img_cropped
                    self.agents[int(agent_id)] = temp
                else:
                    self.agents[int(agent_id)].local_pos[target_frame] = self.traj_data[target_frame][agent_id]
                    # print(self.agents[int(agent_id)].local_pos[target_frame])
                    pixel_pos = self._global2img(self.agents[int(agent_id)].local_pos[target_frame], sceneIdx)
                    if pixel_pos is not None:
                        self.agents[int(agent_id)].pixel_pos[target_frame] = pixel_pos
                        img_cropped, _, _ = self._crop_img_dot(img, self.agents[int(agent_id)].local_pos[target_frame], sceneIdx)
                        self.agents[int(agent_id)].frame = img_cropped
                
    def caption_multiframe(self, frame_idx, sceneIdx, numFrames2use):
        H, W = self.map.shape
        img = copy.deepcopy(self.map)
        img *= 255.0
        img = img.astype(np.uint8)
        img = np.expand_dims(img, axis=-1)
        img = np.concatenate((img,img,img), axis=-1)
        # img_bound = copy.deepcopy(self.map_boundary)
        # img_bound *= 255.0
        # img_bound = img_bound.astype(np.uint8)
        # img_bound = np.expand_dims(img_bound, axis=-1)
        # img_bound = np.concatenate((img_bound,img_bound,img_bound), axis=-1)
        for agent_id in self.agents.keys():
            agent_pos = np.zeros((0, 2), dtype=np.uint)
            # agent_vid = np.zeros((0, BOX_SIZE[0]*2, BOX_SIZE[1]*2, 3))
            for frame_net in range(numFrames2use,-1,-1):
                if frame_idx - frame_net in self.agents[agent_id].pixel_pos.keys():
                    agent_pos = np.concatenate((agent_pos, np.expand_dims(np.array(self.agents[agent_id].pixel_pos[frame_idx - frame_net]), axis=0)), axis=0)
            if frame_idx not in self.agents[agent_id].pixel_pos.keys(): continue
            
            min_row, max_row, min_col, max_col = agent_pos[-1, 1]-OBSTACLE_THRSHLD[sceneIdx], agent_pos[-1, 1]+OBSTACLE_THRSHLD[sceneIdx], agent_pos[-1, 0]-OBSTACLE_THRSHLD[sceneIdx], agent_pos[-1, 0]+OBSTACLE_THRSHLD[sceneIdx]
            min_row, max_row, min_col, max_col = int(max(0, min_row)), int(min(H-1, max_row)), int(max(0, min_col)), int(min(W-1, max_col))
            
            agent_pos[-1, 0], agent_pos[-1, 1] = max(0, agent_pos[-1, 0]), max(0, agent_pos[-1, 1])
            agent_pos[-1, 0], agent_pos[-1, 1] = min(W-1, agent_pos[-1, 0]), min(H-1, agent_pos[-1, 1])
            if agent_pos.shape[0] > 1: displacement = agent_pos[-1]-agent_pos[0]
            else: displacement = None
            if (self.map[min_row:max_row, min_col:max_col] == 0).sum() > 0:
                distance, angle, angle_displacement, closest_obs = find_closest_obstacle(self.map, self.map_boundary, agent_pos[-1, 0], agent_pos[-1, 1], displacement)
            else:
                distance, angle = None, None
            
            caption = ''
            if agent_pos.shape[0]==2:
                if np.linalg.norm(agent_pos[0]-agent_pos[1]) > WALK_THRSHLD[sceneIdx]:  # Is walking
                    caption += 'The person is walking.'
                elif np.linalg.norm(agent_pos[0]-agent_pos[1]) > WALK_SLOW_THRSHLD[sceneIdx]:  # Is walking slowly
                    caption += 'The person is walking slowly.'
                else:
                    caption += 'The person is standing still.'
                    
            if len(caption) > 0: caption += ' '
            
            if distance is None or distance > OBSTACLE_THRSHLD[sceneIdx]:
                caption += 'There is no obstacle around.'
            elif angle_displacement is not None:
                angle = angle - angle_displacement
                if angle > 30 and angle <100:
                    caption += 'There is an obstacle on the right.'
                elif angle <= 30 and angle >= -30:
                    caption += 'There is an obstacle in front.'
                elif angle < -30 and angle >-100:
                    caption += 'There is an obstacle on the left.'
                else:
                    caption += 'There is no obstacle in the heading direction of the person.'
            if self.caption_count == 32:
                sdfl=1
            self.agents[agent_id].caption = caption
            
            # # # # # # # # # # # For debugging # # # # # # # # # #
            # img_ = copy.deepcopy(img)
            # # agent_pos = agent_pos[-1]
            # cv2.circle(img_, (int(agent_pos[-1, 0]), int(agent_pos[-1, 1])), 5, (0, 0, 255), -1, cv2.LINE_AA)       # (x, y)
            # print(f'cap_count: {self.caption_count} / {caption}')
            # img_ = Image.fromarray(img_)
            # img_.save(f'/mnt/jaewoo4tb/textraj/temp_zara2/{self.caption_count}_frame{frame_idx}_agent{agent_id}_scene{sceneIdx}.png')
            # self.caption_count += 1
            # # # # # # # # # # # For debugging # # # # # # # # # #
            
            crop_img_small = Image.fromarray(self.agents[agent_id].frame)
            inputs = self.processor(images=crop_img_small, return_tensors="pt", padding=True)
            inputs['pixel_values'] = inputs['pixel_values'].cuda()
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            self.agents[agent_id].pose = outputs.squeeze().cpu()
            
def find_closest_obstacle(binary_map, binary_map_boundary, start_col, start_row, displacement):
    closest_dist = float('inf')  # Initialize with a large number
    closest_obstacle = None
    angle_to_obstacle = None
    
    # Dimensions of the binary map
    rows, cols = binary_map_boundary.shape
    
    # Iterate through the map to find obstacles
    for i in range(rows):
        for j in range(cols):
            if binary_map_boundary[i, j] == True:  # Obstacle found
                # Calculate the Euclidean distance
                dist = np.sqrt((i - start_row)**2 + (j - start_col)**2)
                if dist < closest_dist:  # Check if it's the closest obstacle
                    closest_dist = dist
                    closest_obstacle = (i, j)
                    # Calculate the angle using atan2
                    angle_to_obstacle = math.atan2(i - start_row, j - start_col)
    
    if angle_to_obstacle<0: angle_to_obstacle += 2*math.pi
    angle_to_obstacle *= (180 / math.pi)
    
    if binary_map[int(start_row), int(start_col)] == 0: # inside obstacle
        angle_to_obstacle += 180
        closest_dist = 1
    if displacement is not None and (displacement[0] != 0 or displacement[1] != 0):
        angle_displacement = math.atan2(displacement[1], displacement[0])
        if angle_displacement<0: angle_displacement += 2*math.pi
        angle_displacement *= (180 / math.pi)
    else:
        angle_displacement = None
    
    return closest_dist, angle_to_obstacle, angle_displacement, closest_obstacle

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
        