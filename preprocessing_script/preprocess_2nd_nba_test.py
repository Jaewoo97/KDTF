
"""
v2: only by number of agents in this scene
- scene: {
    agent_position: 25 X 20 X 2
    agent_pose: 25 X 20 X 72
    agent_pose_mask: 25 X 20
    agent_description: 25 X 20 X embed_dim
    agent_description_mask: 25 X 20
    agent_mask: 25 X 20 => True 면 mask 된 정보

    object_position: N X 20 X 3 => xy, object_index
    object_mask: N X 20

    robot_position: 20 X 2
    robot_orientation: 20 X 1

    cluster_ID: 25 X 20 ⇒ cluster ID int
    cluster_embed: max(cluster ID) X 20 X embed_dim
    cluster_embed_mask: max(cluster ID) X 20

    dual_interaction: 25 X 25 X 20 X embed_dim
    dual_interaction_mask: 25 X 25 X 20
    }
"""
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
from tqdm import tqdm
import sys
sys.path.append('/mnt/jaewoo4tb/textraj/')
from utils.utils import *
from parser_model.text_encoder import TextEncoder
from parser_model.text_encoder_bert import TextEncoder_BERT, TextEncoder_TinyBERT
from itertools import permutations
from itertools import product
from scipy.spatial.transform import Rotation as R_
from HiVT.utils_hivt import TemporalData



class PreProcess:
    def __init__(self, agent_data: np.array):
        self.FRAME_LENGTH = 30
        self.FRAME_STEP = 1
        self.PAST_FRAME = 10
        self.FUT_FRAME = 20
        self.MAX_NUM_AGENTS = 11

        self.agent_data = np.transpose(agent_data, (0, 2, 1, 3)) / (94/28) # Turn to meters
        
    def __call__(self, data_idx):
        # print(f"start preprocessing from frame {initial_frame} to {initial_frame + (self.FRAME_LENGTH - 1) * self.FRAME_STEP}")
        scene = {}

        agent_mask = torch.ones(size=(self.MAX_NUM_AGENTS, self.FRAME_LENGTH), dtype=torch.bool)
        agent_position = self.agent_data[data_idx]        

        x, positions = torch.Tensor(agent_position).clone(), torch.Tensor(agent_position).clone()
        
        # padding_mask_all = torch.ones((self.MAX_NUM_AGENTS, T)).bool()
        # padding_mask = raw_file['padding']
        padding_mask_all = torch.logical_and(x[:,:,0] == 0, x[:,:,1] == 0)
        padding_mask_all = torch.Tensor(padding_mask_all).bool()
        
        bos_mask = torch.zeros(self.MAX_NUM_AGENTS, self.PAST_FRAME, dtype=torch.bool)
        bos_mask[:, 0] = ~padding_mask_all[:, 0]
        bos_mask[:, 1: self.PAST_FRAME] = padding_mask_all[:, : self.PAST_FRAME-1] & ~padding_mask_all[:, 1: self.PAST_FRAME]
        
        rotate_angles = torch.zeros(self.MAX_NUM_AGENTS, dtype=torch.float)
        valid_indices = [torch.nonzero(row[:self.PAST_FRAME], as_tuple=True)[0] for row in ~padding_mask_all]
        
        for actor_id in range(self.MAX_NUM_AGENTS):
            if len(valid_indices[actor_id]) < 2: continue   # If zero or one frame of observation, skip
            if (valid_indices[actor_id][1:]-valid_indices[actor_id][:-1]).min()>1: continue # If no consecutive frames, skip
            heading_vector = x[actor_id, valid_indices[actor_id][-1]] - x[actor_id, valid_indices[actor_id][-2]]
            rotate_angles[actor_id] = torch.atan2(heading_vector[1], heading_vector[0]) 
                
        x[:,self.PAST_FRAME:] = x[:,self.PAST_FRAME:] - x[:,self.PAST_FRAME-1].unsqueeze(-2)
        x[:,1:self.PAST_FRAME] = x[:,1:self.PAST_FRAME] - x[:,:self.PAST_FRAME-1]
        x[:,0] = torch.zeros(x.shape[0], 2)
        padding_mask_all_temp = padding_mask_all.clone()
        padding_mask_all_temp[:,1:self.PAST_FRAME] = padding_mask_all[:,1:self.PAST_FRAME] | padding_mask_all[:,:self.PAST_FRAME-1]
        x[padding_mask_all_temp] = 0
        y = x[:,self.PAST_FRAME:]
        
        # rotate_angles = rotate_angles * -1      # OPPOSITE
        
        rotate_mat = torch.empty(self.MAX_NUM_AGENTS, 3, 3)
        sin_vals = torch.sin(rotate_angles)
        cos_vals = torch.cos(rotate_angles)
        rotate_mat[:, 0, 0] = cos_vals
        rotate_mat[:, 0, 1] = -sin_vals      # original: -
        rotate_mat[:, 0, 2] = 0
        rotate_mat[:, 1, 0] = sin_vals     # original: +
        rotate_mat[:, 1, 1] = cos_vals
        rotate_mat[:, 1, 2] = 0
        rotate_mat[:, 2, 0] = 0
        rotate_mat[:, 2, 1] = 0
        rotate_mat[:, 2, 2] = 1
        if y is not None:
            y = torch.bmm(y, rotate_mat[:, :2, :2]) 
        
        agent_mask_true = (~padding_mask_all[:,:self.PAST_FRAME]).sum(-1) > (self.PAST_FRAME//2)      # Enough frames for past (True is valid data)
        num_valid_agents = agent_mask_true.sum()
        # print(num_valid_agents)
        if num_valid_agents < 2: 
            raise Exception("Something wrong")
            return None, None
        
        rotate_mat_ = torch.empty(num_valid_agents, 3, 3)
        positions_ = torch.Tensor(size=(num_valid_agents, self.FRAME_LENGTH, 2))
        x_ = torch.Tensor(size=(num_valid_agents, self.FRAME_LENGTH, 2))
        y_ = torch.Tensor(size=(num_valid_agents, self.FUT_FRAME, 2))
        padding_mask_all_ = torch.ones(size=(num_valid_agents, self.FRAME_LENGTH), dtype=torch.bool)
        bos_mask_ = torch.zeros(num_valid_agents, self.PAST_FRAME, dtype=torch.bool)
        rotate_angles_ = torch.zeros(num_valid_agents, dtype=torch.float)
        
        rotate_mat_[:agent_mask_true.sum()] = rotate_mat[agent_mask_true]
        rotate_angles_[:agent_mask_true.sum()] = rotate_angles[agent_mask_true]
        positions_[:agent_mask_true.sum()] = positions[agent_mask_true]
        # agent_position_[:agent_mask_true.sum()] = agent_position[agent_mask_true]
        padding_mask_all_[:agent_mask_true.sum()] = padding_mask_all[agent_mask_true]
        bos_mask_[:agent_mask_true.sum()] = bos_mask[agent_mask_true]
        x_[:agent_mask_true.sum()] = x[agent_mask_true]
        y_[:agent_mask_true.sum()] = y[agent_mask_true]
        
        edge_index = torch.LongTensor(list(permutations(range(num_valid_agents), 2))).t().contiguous()
        
        scene = {
                # "frames": frames, 
                'num_nodes': agent_mask_true.sum(),
                'rotate_mat': rotate_mat_,
                'x': x_[:,:self.PAST_FRAME],
                'positions': positions_,
                'rotate_angles': rotate_angles_,
                'padding_mask': padding_mask_all_, # position masking
                'edge_index': edge_index,
                'bos_mask': bos_mask_,
                'y': y_,
                }

        data = TemporalData(**scene)
        # return scene, num_valid_agents
        return data, num_valid_agents
        
    def rotate_root(pose, rotation_angle_degrees, axis):
        # Convert the angle to radians
        if torch.is_tensor(rotation_angle_degrees): 
            if rotation_angle_degrees.device.type=='cuda': rotation_angle_degrees = rotation_angle_degrees.cpu()
        pose = pose.squeeze()
        # rotation_angle_radians = np.radians(rotation_angle_degrees)
        rotation_angle_radians = rotation_angle_degrees
        rotation_matrix = R_.from_euler(axis, rotation_angle_radians).as_matrix()
        root_rotation_matrix = R_.from_rotvec(pose[:, :3]).as_matrix()
        new_root_rotation_matrix = rotation_matrix @ root_rotation_matrix
        new_root_rotation_vector = R_.from_matrix(new_root_rotation_matrix).as_rotvec()
        new_pose = np.copy(pose)
        new_pose[:, :3] = new_root_rotation_vector
        return np.expand_dims(new_pose, 0)

if __name__ == "__main__":
    from glob import glob
    parser = argparse.ArgumentParser(description="Preprocessing script for dataset.")
    
    parser.add_argument("--scene_idx_start", type=int, default=0,
                        help="Scene idx out of 5 to start.")
    parser.add_argument("--scene_idx_end", type=int, default=5,
                        help="Scene idx out of 5 to end.")
    args = parser.parse_args()

    print(f"starting 2nd preprocessing, start: {args.scene_idx_start} / end: {args.scene_idx_end}.")
    save_root = "/mnt/jaewoo4tb/textraj/preprocessed_2nd/nba_test"
    datas = "/mnt/jaewoo4tb/srtp/srtp/raw_data/nba/nba_test.npy"
    nba_data = np.load(datas)
    preprocess = PreProcess(nba_data)
    
    try:
        os.makedirs(os.path.join(save_root, 'val'), exist_ok=True)
    except:
        pass
    
    for data_idx in range(nba_data.shape[0]):
        if data_idx < 12500:
            split = 'val'
        else:
            break
        result = preprocess(data_idx)
        torch.save(result[0], f"{save_root}/{split}/{str(data_idx).zfill(5)}.pt")
