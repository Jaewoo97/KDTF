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
sys.path.append('/mnt/jaewoo4tb/textraj')
from bev.model import BEV
from bev.cfg import bev_settings
from bev.post_parser import *
from romp.vis_human.pyrenderer import Py3DR
from utils.utils import *

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
    def __init__(self, type_cap, visualize):
        self.HORIZONTAL_MARGIN = 100
        self.CROP_IMG_DIR = "/mnt/jaewoo4tb/textraj/temp/crop_imgs"
        
        # print(f"Loading VLM, type: {type_cap}")
        # if type_cap == 'vllm':
        #     from parser_model.caption_getter_vllm import CaptionModel
        #     self.caption_model = CaptionModel()
        # else:
        #     from parser_model.caption_getter import CaptionModel
        #     self.caption_model = CaptionModel()
        
        print("Loading 3D pose regressor")
        default_cfg = bev_settings()
        self.pose_model = BEV(default_cfg)
        self.smpl_parser = SMPLA_parser(default_cfg.smpl_path, default_cfg.smil_path)
        # if visualize: self.renderer = Py3DR()
        # else: self.renderer = None
        
        print('Caption3DPoseEngine initialized.')        

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

    def _cluster_agents(self, thr=1.0):
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

    def load_files(self, label_2d, label_3d, img_dir, social):
        self.label_2d = json.load(open(label_2d))["labels"]
        self.label_3d = json.load(open(label_3d))["labels"]
        self.label_social = json.load(open(social))["labels"]
        self.img_dir = img_dir
        
    
    def preprocess_frame(self, target_frame, global_position, global_ori):
        ''' Process location '''
        self.target_frame = f'{target_frame:06d}'
        self.img = cv2.imread(self.img_dir + self.target_frame + ".jpg")
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        self.agents_temp, self.agents, self.socials = {}, {}, Socials()
        frame_2d = self.label_2d[self.target_frame + ".jpg"]
        frame_3d = self.label_3d[self.target_frame + ".pcd"]
        frame_3d_dict = {}
        for frame_3d_i in frame_3d:
            frame_3d_dict[int(frame_3d_i['label_id'].split(":")[1])] = frame_3d_i

        # Process visible agents
        self.visible_agents = []
        for agent_2d in frame_2d:
            if agent_2d["attributes"]["occlusion"] == "Severely_occluded" or agent_2d["attributes"]["occlusion"] == "Fully_occluded": #neglect occluded agents
                continue
            
            id = int(agent_2d["label_id"].split(":")[1])
            self.visible_agents.append(id)
            bbox = agent_2d["box"]
            temp = Agent(id)
            temp.type = agent_2d["label_id"].split(":")[0]
            temp.visible = True
            temp.bbox = bbox
            self.agents_temp[id] = temp
                
        for temp_id in self.agents_temp.keys():
            if temp_id not in frame_3d_dict.keys(): continue
            temp = self.agents_temp[temp_id]
            agent_3d = frame_3d_dict[temp.id]
            local_pos_temp = np.array((agent_3d["box"]["cx"], agent_3d["box"]["cy"]), dtype=np.float32)
            local_pos_temp = rotate_coordinates(local_pos_temp, global_ori)
            temp.local_pos = local_pos_temp
            temp.global_pos[:2] = local_pos_temp + global_position[:2]
            temp.rot_z = agent_3d["box"]["rot_z"]
            temp.observation_angle = agent_3d["observation_angle"]
            temp.robot_ori = global_ori
            temp.robot_pos = global_position[:2]
            self.agents[temp_id] = temp
        
        # Process social activities
        fully_occluded_agents, false_cluster = [], []
        for social_annot in self.label_social[self.target_frame + ".jpg"]:
            annot_verbs = [i for i in social_annot['action_label'].keys()]
            agent_caption = "The person is "
            current_agent_id = social_annot['label_id'].split(":")[1]
            for annot_verb_idx in range(len(annot_verbs)):
                add_str = annot_verbs[annot_verb_idx]
                add_str = add_str.replace('sth', 'something')
                add_str = add_str.replace('&', ' and ')
                agent_caption += add_str
                if annot_verb_idx != len(annot_verbs) - 1: agent_caption += ', '
                else: agent_caption += '.'
            agent_idx = int(current_agent_id)
            if agent_idx not in self.agents.keys():
                fully_occluded_agents.append(agent_idx)
                continue
            if 'unknown' not in agent_caption:
                agent_caption = agent_caption.replace('_', ' ')
                self.agents[agent_idx].caption = agent_caption
            
            # group activity
            if social_annot['social_group']['cluster_ID'] not in false_cluster:
                if list(social_annot['group_info'][0]['SSC'].keys())[0] == 'unknown' and list(social_annot['group_info'][0]['BPC'].keys())[0] == 'unknown' and list(social_annot['group_info'][0]['inter'].keys())[0] == 'unknown' and list(social_annot['group_info'][0]['location_pre'].keys())[0] == 'unknown':
                    false_cluster.append(social_annot['social_group']['cluster_ID'])
                    continue
                if social_annot['social_group']['cluster_ID'] not in self.socials.clusters.keys():
                    cluster_caption = "They are "
                    assert len(social_annot['group_info']) == 1
                    for inter_cap_idx in range(len(social_annot['group_info'][0]['inter'].keys())):
                        add_str = list(social_annot['group_info'][0]['inter'].keys())[inter_cap_idx]
                        add_str = add_str.replace('&', ' and ')
                        if inter_cap_idx+1 != len(social_annot['group_info'][0]['inter'].keys()): cluster_caption = cluster_caption + add_str + ' and '
                        else: cluster_caption = cluster_caption + add_str
                    cluster_caption += ' on '
                    assert len(social_annot['group_info'][0]['BPC'].keys()) == 1
                    assert len(social_annot['group_info'][0]['location_pre'].keys()) == len(social_annot['group_info'][0]['location_pre'].keys())
                    if list(social_annot['group_info'][0]['location_pre'].keys())[0] == 'unknown': 
                        cluster_caption = cluster_caption + list(social_annot['group_info'][0]['BPC'].keys())[0] + '.' # up to BPC
                        assert list(social_annot['group_info'][0]['SSC'].keys())[0] == 'unknown'
                        if 'unknown' in cluster_caption: continue
                        cluster_caption = cluster_caption.replace('_', ' ')
                        self.socials.clusters[social_annot['social_group']['cluster_ID']] = (cluster_caption, [agent_idx])
                    else:
                        if 'carrying' not in cluster_caption: cluster_caption = cluster_caption + list(social_annot['group_info'][0]['BPC'].keys())[0] # up to BPC
                        for pre_ssc_iter in range(len(social_annot['group_info'][0]['location_pre'].keys())):
                            cur_ssc = list(social_annot['group_info'][0]['SSC'].keys())[pre_ssc_iter]
                            cur_pre = list(social_annot['group_info'][0]['location_pre'].keys())[pre_ssc_iter]
                            cur_pre, cur_ssc = cur_pre.replace('&', ' and '), cur_ssc.replace('&', ' and ')
                            if 'carrying' in cluster_caption:                                
                                cluster_caption = cluster_caption[:-3] + cur_pre + ' ' + cur_ssc + ', '
                            else:
                                if pre_ssc_iter != 0: cluster_caption = cluster_caption + ', ' + cur_pre + ' ' + cur_ssc # up to BPC
                                else: cluster_caption = cluster_caption + ' ' + cur_pre + ' ' + cur_ssc # up to BPC
                        if 'carrying' in cluster_caption: cluster_caption = cluster_caption + 'on ' + list(social_annot['group_info'][0]['BPC'].keys())[0]
                        cluster_caption += '.'
                        cluster_caption = cluster_caption.replace('_', ' ')
                        if 'unknown' in cluster_caption:
                            false_cluster.append(social_annot['social_group']['cluster_ID'])
                        else:
                            cluster_caption = cluster_caption.replace('_', ' ')
                            self.socials.clusters[social_annot['social_group']['cluster_ID']] = (cluster_caption, [agent_idx])
                else: self.socials.clusters[social_annot['social_group']['cluster_ID']][1].append(agent_idx)   # up to SSC
            
            # dual interaction
            interaction_pair_ids = []
            for h_inter in social_annot['H-interaction']:
                h_interaction_add = int(h_inter['pair'].split(":")[1])
                if h_interaction_add in fully_occluded_agents: continue     # annotation only in social, no annotation for 3d bbox 
                interaction_pair_ids.append(h_interaction_add)
                assert len(h_inter['inter_labels'].keys()) == 1
            for unique_pair_id in np.unique(interaction_pair_ids):
                if (agent_idx, int(unique_pair_id)) not in self.socials.interactions.keys() and (int(unique_pair_id), agent_idx) not in self.socials.interactions.keys():
                    inter_caption = "They are "
                    h_inter_idxs = np.where(np.array(interaction_pair_ids) == unique_pair_id)[0]
                    for h_inter_idx in h_inter_idxs:
                        inter_caption = inter_caption + to_verb(list(social_annot['H-interaction'][int(h_inter_idx)]['inter_labels'].keys())[0]) + ' and '
                    inter_caption = inter_caption[:-5] + '.'
                    inter_caption = inter_caption.replace('sth', 'something')
                    inter_caption = inter_caption.replace('&', ' and ')
                    self.socials.interactions[tuple(sorted((agent_idx, int(unique_pair_id))))] = inter_caption
                    
    def caption_single(self):
        for agent_id in self.agents.keys():
            agent = self.agents[agent_id]
            bbox = agent.bbox
            temp_img = self.img.copy()
            crop_img = self._crop_img_single(temp_img, bbox)
            img = Image.fromarray(crop_img)
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            buffer.seek(0)
            jpeg_image = Image.open(buffer)
            caption = self.caption_model.get_caption(jpeg_image, "[INST] <image>\nConcisely describe the behavior of the person indicated by red bounding box in less than 20 words. Focus on if the person is moving or static. [/INST]")
            agent.caption = caption

    def caption_dual(self):
        pairs = self._cluster_agents()
        self.pairs = []
        for pair in pairs:
            temp_img = self.img.copy()
            pair_img = self._crop_img_dual(temp_img, pair)
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
                    robot_ori = agent.robot_ori
                    observation_angle = agent.observation_angle * (180/np.pi)
                    pose['smpl_thetas'] = rotate_root(pose['smpl_thetas'], -b_ori_theta - rot_z + robot_ori + 180, 'z')
                    verts, joints, face = self.smpl_parser(pose['smpl_betas'], pose['smpl_thetas'])
                    b_ori = get_b_ori(joints)
                    pose.update({'verts': verts, 'joints': joints, 'smpl_face':face, 'b_ori':b_ori})
                    agent.pose = pose
                    agent.global_pos = agent.global_pos - [0, 0, joints[:,:,2].min().cpu().numpy()]

def get_b_ori(joints):
    x_axis = joints[:, 2, :] - joints[:, 1, :]
    z_axis = joints[:, 0, :] - joints[:, 12, :]
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