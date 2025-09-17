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
# R = RX * RZ

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
        self.local_pos = np.array([0, 0], dtype=np.float32)
        self.global_pos = np.array([0, 0, 0], dtype=np.float32)
        self.bbox = [0, 0, 0, 0]
        self.caption = None
        self.rot_z = 0
        self.pose = None        
        self.visible = None
        self.robot_ori = None
        self.robot_pos = None

class Socials:
    def __init__(self):
        self.clusters = {}  # key: cluster ID, value: (caption |str|, list of agent IDs |int|)
        self.interactions = {} # key: (agent_id, agent_id), value: |str| of interaction description. Order of agent_id doesn't matter
        
class Caption3DPoseEngine:
    def __init__(self, type_cap):
        self.HORIZONTAL_MARGIN = 100
        self.CROP_IMG_DIR = "/mnt/jaewoo4tb/textraj/temp/crop_imgs"
        self.caption_count = 0
        
        if type_cap is not None:
            print(f"Loading VLM, type: {type_cap}")
            if type_cap == 'vllm':
                from parser_model.caption_getter_vllm import CaptionModel
                self.caption_model = CaptionModel()
            else:
                from parser_model.caption_getter import CaptionModel
                self.caption_model = CaptionModel()
        
        # print("Loading 3D pose regressor")
        # default_cfg = bev_settings()
        # self.pose_model = BEV(default_cfg)
        # self.smpl_parser = SMPLA_parser(default_cfg.smpl_path, default_cfg.smil_path)
        # if visualize: self.renderer = Py3DR()
        # else: self.renderer = None
        # print('Caption3DPoseEngine initialized.')        
        
        # Load the smaller CLIP model and processor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        print('Image encoder initialized.')

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
    
    def _crop_img_single_dot(self, img, pos, pos_prev, sceneIdx):
        pos = [float(i) for i in pos]
        if pos_prev is not None: pos_prev = [float(i) for i in pos_prev]
        h, w, c = img.shape
        box_size = [130, 130]
        fig, ax = plt.subplots(figsize=(10, 10))
        
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
        ax.imshow(img)
        # ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if pos_prev is None:
            ax.scatter(pos[0], pos[1], s=10, color='red', marker='o')  # Use scatter for red circle
        else:
            x_0, y_0 = pos_prev[0], pos_prev[1]
            dx, dy = pos[0]-pos_prev[0], pos[1]-pos_prev[1]
            ax.arrow(x_0, y_0, dx, dy, color='red', width=0.5, length_includes_head=True, head_width=0.5, head_length=0.5)
        # cv2.circle(img, pos, radius=5, color=(255, 0, 0), thickness=-1)  # Red color circles
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # Invert y-axis to match OpenCV's coordinate system

        plt.axis('off')  # Turn off axis

        # Save the plot to a buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)

        # Convert the buffer to a NumPy array
        arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img_np = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        img_np = cv2.resize(img_np, (w,h), interpolation=cv2.INTER_LINEAR)

        # Clean up
        buf.close()
        plt.close(fig)
        
        x_start = pos[0] - box_size[0]
        x_end = x_start + 2 * box_size[0]
        y_start = pos[1] - box_size[1]
        y_end = y_start + 2 * box_size[1]
        
        x_start, y_start = max(0, x_start), max(0, y_start)
        x_end, y_end = min(x_end, w), min(y_end, h)
        foo = img_np[int(y_start):int(y_end), int(x_start):int(x_end)]
        if foo.shape[0] == 0:
            sdjklf=1
        return foo , img_np
    
    def _crop_img_single_dot_v2(self, img, pos, pos_prev, sceneIdx):
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

    def load_files(self, traj_dir, img_dir, homography):
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
        
    def load_files_v2(self, traj_dir, img_dir, homography, scene_name=None):     # load from ynet original preprocessed
        self.traj_data = {}  # {frame: {agent_id: data} }
        print('Loading trajectory data. . .')
        df = pd.read_pickle(traj_dir)
        # scene_id = 'hotel'  # Replace with the desired sceneId
        filtered_df = df[df['sceneId'] == scene_name]
        selected_data = filtered_df[['frame', 'trackId', 'x', 'y']]
        result_array = selected_data.to_numpy()
        for row_idx in tqdm.tqdm(range(result_array.shape[0])):
            row = result_array[row_idx]
            data_line = row[0].split('\t')
            cur_frame, cur_agent = int(data_line[0]), int(float(data_line[1]))
            if cur_frame not in self.traj_data.keys():
                self.traj_data[cur_frame] = {}
            if cur_agent not in self.traj_data[cur_frame].keys():
                self.traj_data[cur_frame][cur_agent] = np.array((data_line[2], data_line[3]))  # x,y data
        self.img_dir = img_dir
        self.H = homography

    def preprocess_frame(self, target_frame):
        ''' Process frame, target_frame is the 10 divided frame'''
        self.target_frame = f'{target_frame:06d}'
        # if visualize:
        if int(self.target_frame)>0: 
            self.prev_img = cv2.imread(self.img_dir + '/' + str((int(self.target_frame)-1)*10).zfill(4) + ".jpg")
        else: self.prev_img = None
        self.img = cv2.imread(self.img_dir + '/' + str(int(self.target_frame)*10).zfill(4) + ".jpg")
        # self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        self.agents, self.agents_prev = {}, {}
        
        if target_frame in self.traj_data.keys():
            for agent_id in self.traj_data[target_frame].keys():
                temp = Agent(int(agent_id))
                temp.type = 'pedestrian'
                temp.local_pos = self.traj_data[target_frame][agent_id]
                self.agents[int(agent_id)] = temp

        if target_frame-1 in self.traj_data.keys():
            for agent_id in self.traj_data[target_frame-1].keys():
                temp = Agent(int(agent_id))
                temp.type = 'pedestrian'
                temp.local_pos = self.traj_data[target_frame-1][agent_id]
                self.agents_prev[int(agent_id)] = temp
    
    def preprocess_frame_(self, target_frame):
        ''' Process frame, target_frame is the 10 divided frame'''
        # target_frame = f'{target_frame:06d}'
        # if visualize:
        if int(target_frame)>0: 
            prev_img = cv2.imread(self.img_dir + '/' + str(int(target_frame-1)*10).zfill(4) + ".jpg")
        else: prev_img = None
        img = cv2.imread(self.img_dir + '/' + str(int(target_frame)*10).zfill(4) + ".jpg")

        agents, agents_prev = {}, {}
        
        if target_frame in self.traj_data.keys():
            for agent_id in self.traj_data[target_frame].keys():
                temp = Agent(int(agent_id))
                temp.type = 'pedestrian'
                temp.local_pos = self.traj_data[target_frame][agent_id]
                agents[int(agent_id)] = temp

        if target_frame-1 in self.traj_data.keys():
            for agent_id in self.traj_data[target_frame-1].keys():
                temp = Agent(int(agent_id))
                temp.type = 'pedestrian'
                temp.local_pos = self.traj_data[target_frame-1][agent_id]
                agents_prev[int(agent_id)] = temp
        if prev_img is None: agents_prev = None
        return img, prev_img, agents, agents_prev
        
    def caption_single(self, frame_idx, sceneIdx):
        for agent_id in self.agents.keys():
            agent = self.agents[agent_id]
            agent_pos = agent.local_pos
            if agent_id in self.agents_prev.keys():
                agent_prev = self.agents_prev[agent_id]
                agent_pos_prev = agent_prev.local_pos
            else: agent_pos_prev = None
            
            temp_img = self.img.copy()
            # temp_img_prev = self.prev_img.copy()
            crop_img, full_img, crop_img_small = self._crop_img_single_dot_v2(temp_img, agent_pos, agent_pos_prev, sceneIdx)
            cv2.imwrite(f'temp/{self.caption_count}_crop_frame{frame_idx}_agent{agent_id}.jpg', crop_img)
            cv2.imwrite(f'temp/full_{self.caption_count}_dot_test_frame{frame_idx}_agent{agent_id}.jpg', full_img)
            if agent_pos_prev is None:
                if sceneIdx == 0:
                    caption = self.caption_model.get_caption(crop_img, "[INST] <image>\nThis is a outdoor street scene where pedestrains are walking around, with brown colored floor in the middle and snow covered on its side. Your task is to caption each pedestrian to help predict their future trajectory. Concisely describe if there are any obstacles around the person indicated by red dot in less than 20 words. Check if there are any obstacle in the heading direction of pedestrian. Choose from these options: -The person is walking alone, no obstacle around. -The person is walking alone, obstacle on the left. -The person is walking alone, obstacle on the right. -The person is walking alone, obstacle in front. -The person is walking in group, no obstacle around. -The person is walking in group, obstacle on the left. -The person is walking in group, obstacle on the right. -The person is walking in group, obstacle in front. -The person is standing still alone. -The person is standing still in group. Answer only by repeating the option sentence.[/INST]")
                else:
                    caption = self.caption_model.get_caption(crop_img, "[INST] <image>\nThis is a outdoor street scene where pedestrains are walking around. Your task is to caption each pedestrian to help predict their future trajectory. Concisely describe if there are any obstacles around the person indicated by red dot in less than 20 words. Check if there are any obstacle in the heading direction of pedestrian. Choose from these options: -The person is walking alone, no obstacle around. -The person is walking alone, obstacle on the left. -The person is walking alone, obstacle on the right. -The person is walking alone, obstacle in front. -The person is walking in group, no obstacle around. -The person is walking in group, obstacle on the left. -The person is walking in group, obstacle on the right. -The person is walking in group, obstacle in front. -The person is standing still alone. -The person is standing still in group. Answer only by repeating the option sentence.[/INST]")
            else:
                if sceneIdx == 0:
                    caption = self.caption_model.get_caption(crop_img, "[INST] <image>\nThis is a outdoor street scene where pedestrains are walking around, with brown colored floor in the middle and snow covered on its side. Your task is to caption each pedestrian to help predict their future trajectory. Check if there are any obstacle in the heading direction of pedestrian. Choose from these options: -The person is walking alone, no obstacle around. -The person is walking alone, obstacle on the left. -The person is walking alone, obstacle on the right. -The person is walking alone, obstacle in front. -The person is walking in group, no obstacle around. -The person is walking in group, obstacle on the left. -The person is walking in group, obstacle on the right. -The person is walking in group, obstacle in front. -The person is standing still alone. -The person is standing still in group. The red arrow shows the pedestrians displacement from previous frame. Answer only by repeating the option sentence.[/INST]")
                else:
                    caption = self.caption_model.get_caption(crop_img, "[INST] <image>\nThis is a outdoor street scene where pedestrains are walking around. Your task is to caption each pedestrian to help predict their future trajectory. Check if there are any obstacle in the heading direction of pedestrian. Choose from these options: -The person is walking alone, no obstacle around. -The person is walking alone, obstacle on the left. -The person is walking alone, obstacle on the right. -The person is walking alone, obstacle in front. -The person is walking in group, no obstacle around. -The person is walking in group, obstacle on the left. -The person is walking in group, obstacle on the right. -The person is walking in group, obstacle in front. -The person is standing still alone. -The person is standing still in group. The red arrow shows the pedestrians displacement from previous frame. Answer only by repeating the option sentence.[/INST]")
                    # caption = self.caption_model.get_caption(jpeg_image, "[INST] <image>\nThis is a outdoor street scene where pedestrains are walking around. Your task is to caption each pedestrian to help predict their future trajectory. The red arrow depicts the pedestrians displacement from previous frame, heading direction. Is there any obstacle in the heading direction of pedestrian with arrow? Answer concisely in one full sentence, less than 20 words. [/INST]")
            print(f'#: {self.caption_count}, {caption}')
            caption = caption[caption.find(next(filter(str.isalpha, caption))):]
            self.caption_count += 1
            agent.caption = caption
            
            # img feature
            crop_img_small = Image.fromarray(crop_img_small)
            inputs = self.processor(images=crop_img_small, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            agent.pose = outputs.squeeze()
            
    def caption_single_v2(self, frame_idx, sceneIdx):
        # Concat 2 frames
        
        # frame t
        img, _, agents, agents_prev = self.preprocess_frame_(frame_idx)
        # frame t-1
        img_, _, agents_, agents_prev_ = self.preprocess_frame_(frame_idx-1)
        for agent_id in agents.keys():
            agent = agents[agent_id]
            agent_pos = agent.local_pos
            if agent_id in agents_prev.keys():
                agent_prev = agents_prev[agent_id]
                agent_pos_prev = agent_prev.local_pos
            else: agent_pos_prev = None
            
            # frame t
            temp_img = img.copy()
            jpeg_image1, full_img, crop_img_small = self._crop_img_single_dot_v2(temp_img, agent_pos, agent_pos_prev, sceneIdx)
            
            # frame t-1
            if agent_id in agents_.keys():
                agent = agents_[agent_id]
                agent_pos = agent.local_pos
                if agent_id in agents_prev_.keys():
                    agent_prev = agents_prev_[agent_id]
                    agent_pos_prev = agent_prev.local_pos
                else: agent_pos_prev = None
                
            temp_img = img_.copy()
            if agent_pos_prev is None:
                jpeg_image = jpeg_image1
            else:
                jpeg_image2, full_img, _ = self._crop_img_single_dot_v2(temp_img, agent_pos, agent_pos_prev, sceneIdx)
                jpeg_image = concat_images_horizontally(jpeg_image2, jpeg_image1)
            
            cv2.imwrite(f'temp/{self.caption_count}_concat_frame{frame_idx}_agent{agent_id}.jpg', jpeg_image)
            if agent_pos_prev is None:
                if sceneIdx == 0:
                    caption = self.caption_model.get_caption(jpeg_image, "[INST] <image>\nThis is a outdoor street scene where pedestrains are walking around, with brown colored floor in the middle and snow covered on its side. Your task is to caption each pedestrian to help predict their future trajectory. Is the person static or moving? If moving, check if there are any obstacle in the heading direction of pedestrian. The red arrow depicts the pedestrians displacement from previous frame, heading direction. Red dot denotes that the person is stationary. Answer less than 20 words, one sentence.[/INST]")
                    # caption = self.caption_model.get_caption(jpeg_image, "[INST] <image>\nThis is a outdoor street scene where pedestrains are walking around, with brown colored floor in the middle and snow covered on its side. Your task is to caption each pedestrian to help predict their future trajectory. Check if there are any obstacle in the heading direction of pedestrian. Choose from these options: -The person is walking alone, no obstacle around. -The person is walking alone, obstacle on the left. -The person is walking alone, obstacle on the right. -The person is walking alone, obstacle in front. -The person is walking in group, no obstacle around. -The person is walking in group, obstacle on the left. -The person is walking in group, obstacle on the right. -The person is walking in group, obstacle in front. -The person is standing still alone. -The person is standing still in group. The red arrow depicts the pedestrians displacement from previous frame, heading direction. Answer less than 20 words, one sentence.[/INST]")
                else:
                    caption = self.caption_model.get_caption(jpeg_image, "[INST] <image>\nThis is a outdoor street scene where pedestrains are walking around. Your task is to caption each pedestrian to help predict their future trajectory. Is the person static or moving? If moving, check if there are any obstacle in the heading direction of pedestrian. The red arrow depicts the pedestrians displacement from previous frame, heading direction. Red dot denotes that the person is stationary. Answer less than 20 words, one sentence.[/INST]")
            else:
                if sceneIdx == 0:
                    # caption = self.caption_model.get_caption(jpeg_image, "[INST] <image>\nThis is a outdoor street scene where pedestrains are walking around, with brown colored floor in the middle and snow covered on its side. . Your task is to caption each pedestrian to help predict their future trajectory. Check if there are any obstacle in the heading direction of pedestrian. Left image is previous frame, right image is current frame. Choose from these options: -The person is walking alone, no obstacle around. -The person is walking alone, obstacle on the left. -The person is walking alone, obstacle on the right. -The person is walking alone, obstacle in front. -The person is walking in group, no obstacle around. -The person is walking in group, obstacle on the left. -The person is walking in group, obstacle on the right. -The person is walking in group, obstacle in front. -The person is standing still alone. -The person is standing still in group. The red arrow depicts the pedestrians displacement from previous frame, heading direction. Answer less than 20 words, one sentence. Only caption the current, right image.[/INST]")
                    caption = self.caption_model.get_caption(jpeg_image, "[INST] <image>\nThis is a outdoor street scene where pedestrains are walking around, with brown colored floor in the middle and snow covered on its side. Your task is to caption each pedestrian to help predict their future trajectory. Check if there are any obstacle in the heading direction of pedestrian. Left image is previous frame, right image is current frame. Is the person static or moving? If moving, check if there are any obstacle in the heading direction of pedestrian. The red arrow depicts the pedestrians displacement from previous frame, heading direction. Red dot denotes that the person is stationary. Answer less than 20 words, one sentence. Only caption the current, right image.[/INST]")
                else:
                    # caption = self.caption_model.get_caption(jpeg_image, "[INST] <image>\nThis is a outdoor street scene where pedestrains are walking around. Your task is to caption each pedestrian to help predict their future trajectory. Check if there are any obstacle in the heading direction of pedestrian. Left image is previous frame, right image is current frame. Choose from these options: -The person is walking alone, no obstacle around. -The person is walking alone, obstacle on the left. -The person is walking alone, obstacle on the right. -The person is walking alone, obstacle in front. -The person is walking in group, no obstacle around. -The person is walking in group, obstacle on the left. -The person is walking in group, obstacle on the right. -The person is walking in group, obstacle in front. -The person is standing still alone. -The person is standing still in group. The red arrow depicts the pedestrians displacement from previous frame, heading direction. Answer less than 20 words, one sentence. Only caption the current, right image.[/INST]")
                    caption = self.caption_model.get_caption(jpeg_image, "[INST] <image>\nThis is a outdoor street scene where pedestrains are walking around. Your task is to caption each pedestrian to help predict their future trajectory. Is the person static or moving? If moving, check if there are any obstacle in the heading direction of pedestrian. The red arrow depicts the pedestrians displacement from previous frame, heading direction. Red dot denotes that the person is stationary. Answer less than 20 words, one sentence. Only caption the current, right image.[/INST]")
                # caption = self.caption_model.get_caption(jpeg_image, "[INST] <image>\nThis is a outdoor street scene where pedestrains are walking around. Your task is to caption each pedestrian to help predict their future trajectory. The red arrow depicts the pedestrians displacement from previous frame, heading direction. Is there any obstacle in the heading direction of pedestrian with arrow? Answer concisely in one full sentence, less than 20 words. [/INST]")
            print(f'#: {self.caption_count}, {caption}')
            caption = caption[caption.find(next(filter(str.isalpha, caption))):]
            self.caption_count += 1
            agent.caption = caption
            
            # img feature
            crop_img_small = Image.fromarray(crop_img_small)
            inputs = self.processor(images=crop_img_small, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            agent.pose = outputs.squeeze()

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
    
    def regress_3dpose(self):
        for agent_id in self.agents.keys():
            agent = self.agents[agent_id]
            if int(agent.id) in self.visible_agents and np.linalg.norm(agent.local_pos)<5 and np.linalg.norm(agent.local_pos)>0.75:
                bbox = agent.bbox
                temp_img = self.img.copy()
                crop_img = self._crop_img_single(temp_img, bbox, draw_bbox=False, hori_margin=35)
                pose = self.pose_model.forward_parse(crop_img)
                if pose is not None:
                    if pose['verts'].shape[0] > 1: 
                        choose_agent = np.argmin(np.abs(pose['cam'][:,0]))
                        pose['smpl_thetas'] = np.expand_dims(pose['smpl_thetas'][choose_agent], 0)
                        pose['smpl_betas'] = np.expand_dims(pose['smpl_betas'][choose_agent], 0)
                    pose['smpl_thetas'] = rotate_root(pose['smpl_thetas'], -90, 'x')
                    verts, joints, face = self.smpl_parser(pose['smpl_betas'], pose['smpl_thetas'])
                    b_ori = get_b_ori(joints)
                    b_ori_theta = torch.atan2(b_ori[0, 1], b_ori[0, 0]) * (180/np.pi)
                    rot_z = (agent.rot_z * (180/np.pi))
                    # rot_z = (agent.rot_z * (180/np.pi)) - 180
                    robot_ori = agent.robot_ori
                    observation_angle = agent.observation_angle * (180/np.pi)
                    # print(f'Agent {agent_id}, position: {agent.global_pos}, rot_z: {rot_z}, obs_angle: {observation_angle}')
                    pose['smpl_thetas'] = rotate_root(pose['smpl_thetas'], -b_ori_theta - rot_z + robot_ori + 180, 'z')
                    # pose['smpl_thetas'] = rotate_root(pose['smpl_thetas'], -b_ori_theta + rot_z + robot_ori + 56.9233569, 'z')
                    # pose['smpl_thetas'] = rotate_root(pose['smpl_thetas'], -b_ori_theta + rot_z + robot_ori, 'z') # original
                    # pose['smpl_thetas'] = rotate_root(pose['smpl_thetas'], - observation_angle + robot_ori, 'z')
                    verts, joints, face = self.smpl_parser(pose['smpl_betas'], pose['smpl_thetas'])
                    b_ori = get_b_ori(joints)
                    pose.update({'verts': verts, 'joints': joints, 'smpl_face':face, 'b_ori':b_ori})
                    agent.pose = pose
                    agent.global_pos = agent.global_pos - [0, 0, joints[:,:,2].min().cpu().numpy()]

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

def to_verb(verb):
    if verb == 'conversation': return 'having conversation'
    else: return verb

def image2world(image_coords, scene, homo_mat, resize):
	"""
	Transform trajectories of one scene from image_coordinates to world_coordinates
	:param image_coords: torch.Tensor, shape=[num_person, (optional: num_samples), timesteps, xy]
	:param scene: string indicating current scene, options=['eth', 'hotel', 'student01', 'student03', 'zara1', 'zara2']
	:param homo_mat: dict, key is scene, value is torch.Tensor containing homography matrix (data/eth_ucy/scene_name.H)
	:param resize: float, resize factor
	:return: trajectories in world_coordinates
	"""
	traj_image2world = image_coords.clone()
	if traj_image2world.dim() == 4:
		traj_image2world = traj_image2world.reshape(-1, image_coords.shape[2], 2)
	if scene in ['eth', 'hotel']:
		# eth and hotel have different coordinate system than ucy data
		traj_image2world[:, :, [0, 1]] = traj_image2world[:, :, [1, 0]]
	traj_image2world = traj_image2world / resize
	traj_image2world = F.pad(input=traj_image2world, pad=(0, 1, 0, 0), mode='constant', value=1)
	traj_image2world = traj_image2world.reshape(-1, 3)
	traj_image2world = torch.matmul(homo_mat[scene], traj_image2world.T).T
	traj_image2world = traj_image2world / traj_image2world[:, 2:]
	traj_image2world = traj_image2world[:, :2]
	traj_image2world = traj_image2world.view_as(image_coords)
	return traj_image2world

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
        