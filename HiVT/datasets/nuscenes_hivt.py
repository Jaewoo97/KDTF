import os
import os.path as osp
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pyquaternion import Quaternion
import math
import torch
torch.pi = math.pi

from multiprocessing import Process
from multiprocessing import Pool
from itertools import repeat

from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from scipy.spatial.distance import cdist
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data.dataset import files_exist
from tqdm import tqdm
import pickle as pkl
from shapely.geometry import LineString, Point
import glob
import random

import sys
sys.path.append('/mnt/ssd2/frm_lightning_backup/')
# from models.utils.util import TemporalData
# from debug_util import *

PED_CLASS = {}
VEH_CLASS = {}
SPLIT_NAME = {'nuScenes': {'train': 'train', 'val': 'train_val', 'test': 'val', 'mini_train': 'mini_train', 'mini_val': 'mini_val'},
                'Argoverse': {'train': 'train', 'val': 'val', 'test': 'test_obs', 'sample': 'forecasting_sample'}}
STOPLINE_TYPES = ['PED_CROSSING', 'TURN_STOP', 'STOP_SIGN', 'YIELD', 'TRAFFIC_LIGHT']
ACTOR_CATEGORY = {'vehicle.bus.bendy': 0, 'vehicle.bus.rigid': 1, 'vehicle.car':2, 'vehicle.construction':3, 
                    'vehicle.emergency.ambulance':4, 'vehicle.emergency.police':5, 'vehicle.motorcycle': 6, 'vehicle.trailer':7, 
                    'vehicle.truck':8, 'vehicle.bicycle': 9, 'others': 10}

class TemporalData(Data):

    def __init__(self,
                 x: Optional[torch.Tensor] = None,
                 positions: Optional[torch.Tensor] = None,
                 states: Optional[torch.Tensor] = None,
                 category: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.Tensor] = None,
                 edge_attrs: Optional[List[torch.Tensor]] = None,
                 y: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 padding_mask: Optional[torch.Tensor] = None,
                 bos_mask: Optional[torch.Tensor] = None,
                 rotate_angles: Optional[torch.Tensor] = None,
                 lane_positions: Optional[torch.Tensor] = None,
                 lane_vectors: Optional[torch.Tensor] = None,
                 lane_rotate_angles: Optional[torch.Tensor] = None,
                 lane_edge_index: Optional[torch.Tensor] = None,
                 lane_edge_type: Optional[torch.Tensor] = None,
                 lane_paddings: Optional[torch.Tensor] = None,
                 lane_lengths: Optional[torch.Tensor] = None,
                 lane_actor_index: Optional[torch.Tensor] = None,
                 lane_actor_vectors: Optional[torch.Tensor] = None,
                 lane_edge_index2_succ: Optional[torch.Tensor] = None,
                 lane_edge_index2_pred: Optional[torch.Tensor] = None,
                 lane_edge_index2_neigh: Optional[torch.Tensor] = None,
                 goal_idcs: Optional[torch.Tensor] = None,
                 has_goal: Optional[torch.Tensor] = None,
                 seq_id: Optional[int] = None,
                 **kwargs) -> None:
        if x is None:
            super(TemporalData, self).__init__()
            return
        super(TemporalData, self).__init__(x=x, positions=positions, states=states, category=category, edge_index=edge_index, y=y, num_nodes=num_nodes,
                                           padding_mask=padding_mask, bos_mask=bos_mask, rotate_angles=rotate_angles,
                                           lane_positions=lane_positions, lane_vectors=lane_vectors, lane_rotate_angles=lane_rotate_angles, 
                                           lane_edge_index=lane_edge_index, lane_edge_type=lane_edge_type,
                                           lane_paddings=lane_paddings, lane_lengths=lane_lengths,
                                           lane_actor_index=lane_actor_index, lane_actor_vectors=lane_actor_vectors,
                                           lane_edge_index2_succ=lane_edge_index2_succ, lane_edge_index2_pred=lane_edge_index2_pred, lane_edge_index2_neigh=lane_edge_index2_neigh,
                                           goal_idcs=goal_idcs, has_goal=has_goal, 
                                           seq_id=seq_id, **kwargs)
        if edge_attrs is not None:
            for t in range(self.x.size(1)):
                self[f'edge_attr_{t}'] = edge_attrs[t]

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'lane_actor_index':
            return torch.tensor([[self['lane_vectors'].size(0)], [self.num_nodes]])
        elif key == 'lane_edge_index':
            return torch.tensor([[self['lane_vectors'].size(0)], [self['lane_vectors'].size(0)]])
        elif 'lane_edge_index2' in key:
            return torch.tensor([[self['lane_actor_index'].size(1)], [self['lane_actor_index'].size(1)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class nuScenesDataset(Dataset):

    def __init__(self,
                 split: str,
                 root: str,
                 process_dir: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50,
                 process: bool = False,
                 version: str ='v1.0-trainval',
                 spec_args: Dict = None) -> None:
        self._split = split
        self._local_radius = local_radius
        for k,v in spec_args.items():
            self.__setattr__(k, v)

        self._directory = SPLIT_NAME[self.dataset][split]
        self.root = root
        self.version = version
        self.process_dir = process_dir

        temp_raw_file_names = sorted(get_prediction_challenge_split(self._directory, root))
        dir_processed_files = sorted(glob.glob(self.processed_dir+'/*.pt'))
        dir_processed_files = [os.path.basename(i) for i in dir_processed_files]
        self._raw_file_names = [i for i in temp_raw_file_names if i+'.pt' not in dir_processed_files]
        
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self.raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        super(nuScenesDataset, self).__init__(root, transform=transform)

    def _download(self):
        return
    
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.version)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.process_dir, self._directory)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths
    
    def _process(self):

        if files_exist(self.processed_paths):  # pragma: no cover
            print('Found processed files')
            return

        print('Processing...', file=sys.stderr)

        os.makedirs(self.processed_dir, exist_ok=True)
        self.process()

        print('Done!', file=sys.stderr)

    def process(self) -> None:

        from nuscenes import NuScenes
        from nuscenes.prediction import PredictHelper
        from nuscenes.map_expansion.map_api import NuScenesMap
        from nuscenes.map_expansion import arcline_path_utils


        self.nusc = NuScenes(self.version, self.root)
        self.helper = PredictHelper(self.nusc)
        self.map_locs = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']
        self.maps = {i: NuScenesMap(map_name=i, dataroot=self.helper.data.dataroot) for i in self.map_locs}
        self.arcline_path_utils = arcline_path_utils

        self.process_nuscenes(self._raw_file_names, self._local_radius)

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        data = torch.load(self.processed_paths[idx])
        if self._split == 'train':
            data = self.augment(data) 
        return data

    def augment(self, data):
        if self.random_flip:
            if random.choice([0, 1]):
                data.x = data.x * torch.tensor([-1,1])
                data.y = data.y * torch.tensor([-1,1])
                data.positions = data.positions * torch.tensor([-1,1])
                theta_x = torch.cos(data.theta)
                theta_y = torch.sin(data.theta)
                data.theta = torch.atan2(theta_y, -1*theta_x)
                angle_x = torch.cos(data.rotate_angles)
                angle_y = torch.sin(data.rotate_angles)
                data.rotate_angles = torch.atan2(angle_y, -1*angle_x)
                lane_angle_x = torch.cos(data.lane_rotate_angles)
                lane_angle_y = torch.sin(data.lane_rotate_angles)
                data.lane_rotate_angles = torch.atan2(lane_angle_y, -1*lane_angle_x)
                data.lane_positions = data.lane_positions * torch.tensor([-1,1])
                data.lane_vectors = data.lane_vectors * torch.tensor([-1,1])
                data.lane_actor_vectors = data.lane_actor_vectors * torch.tensor([-1,1])
            if random.choice([0, 1]):
                data.x = data.x * torch.tensor([1,-1])
                data.y = data.y * torch.tensor([1,-1])
                data.positions = data.positions * torch.tensor([1,-1])
                theta_x = torch.cos(data.theta)
                theta_y = torch.sin(data.theta)
                data.theta = torch.atan2(-1*theta_y, theta_x)
                angle_x = torch.cos(data.rotate_angles)
                angle_y = torch.sin(data.rotate_angles)
                data.rotate_angles = torch.atan2(-1*angle_y, angle_x)
                lane_angle_x = torch.cos(data.lane_rotate_angles)
                lane_angle_y = torch.sin(data.lane_rotate_angles)
                data.lane_rotate_angles = torch.atan2(-1*lane_angle_y, lane_angle_x)
                data.lane_positions = data.lane_positions * torch.tensor([1,-1])
                data.lane_vectors = data.lane_vectors * torch.tensor([1,-1])
                data.lane_actor_vectors = data.lane_actor_vectors * torch.tensor([1,-1])
        if self.random_rotate:
            rotate_scale = self.random_rotate_deg / 180 * torch.pi
            rotate_angle_pert = torch.clamp(torch.normal(torch.zeros_like(data.rotate_angles), torch.ones_like(data.rotate_angles)), -1, 1) * rotate_scale
            data.rotate_angles = data.rotate_angles + rotate_angle_pert
            data.rotate_angles = data.rotate_angles + 2*torch.pi*(data.rotate_angles < -torch.pi) - 2*torch.pi*(data.rotate_angles > torch.pi)

        return data

    
    def process_nuscenes(self, tokens: str,
                         radius: float) -> Dict:

        for token in tqdm(tokens):
            self.debug_token_ = token

            instance_token, sample_token = token.split("_")
            starting_annotation = self.helper.get_sample_annotation(instance_token, sample_token)
            origin = torch.tensor(starting_annotation['translation'][:2], dtype=torch.float)
            theta = torch.tensor(quaternion_yaw(Quaternion(starting_annotation['rotation'])), dtype=torch.float)

            motions, motion_states, valid_mask, agent_mask, instance_tokens, actor_types = self.get_motions(token)
            num_nodes, total_timestamp, _ = motions.shape

            agent_motion = motions[agent_mask]
            # origin = torch.tensor(agent_motion[0,self.ref_time], dtype=torch.float)
            # agent_heading_vector = origin - torch.tensor(agent_motion[0,self.ref_time-1], dtype=torch.float)
            # theta = torch.atan2(agent_heading_vector[1], agent_heading_vector[0])
            rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                    [torch.sin(theta), torch.cos(theta)]])

            # initialization
            x = torch.zeros(num_nodes, total_timestamp, 2, dtype=torch.float)
            edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous().reshape(2,-1)
            padding_mask = torch.ones(num_nodes, total_timestamp, dtype=torch.bool)
            bos_mask = torch.zeros(num_nodes, self.ref_time+1, dtype=torch.bool)
            rotate_angles = torch.zeros(num_nodes, dtype=torch.float)
            
            
            for node_idx, actor_motion in enumerate(motions):
                actor_valid = valid_mask[node_idx]
                padding_mask[node_idx] = ~torch.from_numpy(actor_valid)
                if padding_mask[node_idx, self.ref_time]:  # make no predictions for actors that are unseen at the current time step
                    padding_mask[node_idx, self.ref_time+1:] = True
                xy = torch.from_numpy(actor_motion).float()
                x[node_idx, actor_valid] = torch.matmul(xy - origin, rotate_mat)[actor_valid]

                valid_historical = actor_valid[:self.ref_time+1]
                node_historical_steps = valid_historical.nonzero()[0]
                if sum(valid_historical) > 1:  # calculate the heading of the actor (approximately)
                    heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
                    # rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])

                    starting_annotation_ = self.helper.get_sample_annotation(instance_tokens[node_idx], sample_token)
                    rotate_angle = torch.tensor(quaternion_yaw(Quaternion(starting_annotation_['rotation'])), dtype=torch.float)
                    rotate_angles[node_idx] = normalize_angle(rotate_angle - theta)
                else:  # make no predictions for the actor if the number of valid time steps is less than 2
                    padding_mask[node_idx, self.ref_time+1:] = True

            ## original ##
            # bos_mask is True if time step t is valid and time step t-1 is invalid
            bos_mask[:, 0] = ~padding_mask[:, 0]
            bos_mask[:, 1: self.ref_time+1] = padding_mask[:, : self.ref_time] & ~padding_mask[:, 1: self.ref_time+1]

            positions = x.clone()
            x[:, self.ref_time+1:] = torch.where((padding_mask[:, self.ref_time].unsqueeze(-1) | padding_mask[:, self.ref_time+1:]).unsqueeze(-1),
                                    torch.zeros(num_nodes, 12, 2),
                                    x[:, self.ref_time+1:] - x[:, self.ref_time].unsqueeze(-2))
            x[:, 1: self.ref_time+1] = torch.where((padding_mask[:, : self.ref_time] | padding_mask[:, 1: self.ref_time+1]).unsqueeze(-1),
                                    torch.zeros(num_nodes, self.ref_time, 2),
                                    x[:, 1: self.ref_time+1] - x[:, : self.ref_time])
            x[:, 0] = torch.zeros(num_nodes, 2)

            # get lane features at the current time step
            node_positions = torch.tensor(motions, dtype=torch.float32) # global coordi
            node_positions_ref = node_positions[:,self.ref_time]
            node_positions_valid = node_positions[~padding_mask]
            (lane_positions, lane_vectors, lane_lengths, lane_edges, lane_edges_type, lane_edges_dict) = \
                                self.get_lane_features(token, node_positions_ref, node_positions_valid, origin, rotate_mat, radius)

            node_positions_goal = positions[:,-1]
            node_diff_goal = positions[:,-1] - positions[:,-2]
            node_goal_mask = ~padding_mask[:,-1]

            goal_idcs, has_goal = self.get_goal_lane(node_positions_goal, node_diff_goal, node_goal_mask, lane_positions, lane_vectors)

            node_inds_ref = torch.arange(num_nodes, dtype=torch.float)[~padding_mask[:,self.ref_time]]
            lane_positions, lane_vectors, lane_rotate_angles, lane_paddings, lane_actor_index, lane_actor_vectors, lane_lengths, goal_idcs, has_goal, lane2_edge_index = \
                                self.get_lane_tensors(node_inds_ref, node_positions_ref, lane_positions, lane_vectors, lane_lengths, lane_edges_dict, goal_idcs, has_goal, origin, rotate_mat, rotate_angles, radius)
            

            y = None if (self._split == 'test' and self.dataset != 'nuScenes') else x[:, self.ref_time+1:]
            seq_id = token

            processed = {
                'x': x[:, : self.ref_time+1],  # [N, 20, 2]
                'positions': positions,  # [N, 50, 2]
                'edge_index': edge_index,  # [2, N x N - 1]
                'y': y,  # [N, 30, 2]
                'num_nodes': num_nodes,
                'padding_mask': padding_mask,  # [N, 50]
                'bos_mask': bos_mask,  # [N, 20]
                'rotate_angles': rotate_angles,  # [N]
                'lane_positions': lane_positions,  # [L, 6, 2]
                'lane_vectors': lane_vectors,  # [L, 6, 2]
                'lane_rotate_angles': lane_rotate_angles,
                'lane_edge_index': lane_edges, 
                'lane_edge_type': lane_edges_type,
                'lane_paddings': lane_paddings, # [L,6]
                'lane_lengths': lane_lengths,
                'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
                'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2]
                'lane_edge_index2_succ': lane2_edge_index['succ'],
                'lane_edge_index2_pred': lane2_edge_index['pred'],
                'lane_edge_index2_neigh': lane2_edge_index['neigh'],
                'goal_idcs': goal_idcs,
                'has_goal': has_goal,
                'seq_id': seq_id,
                'av_index': torch.tensor(np.where(agent_mask)[0][0]).item(),
                'agent_index': torch.tensor(np.where(agent_mask)[0][0]).item(),
                'origin': origin.unsqueeze(0),
                'theta': theta,
            }
            data = TemporalData(**processed)
            
            ## Visualize processed data
            # viz_data_goal(data, 'tmp')
            torch.save(data, os.path.join(self.processed_dir, token+'.pt'))
        return 

    def get_lane_tensors(self, node_inds: List[int],
                                node_positions: torch.Tensor, # global coordi
                                lane_positions: List[torch.Tensor], # local coordi
                                lane_vectors: List[torch.Tensor],
                                lane_lengths: List[int],
                                lane_edges_dict, 
                                goal_idcs: torch.Tensor,
                                has_goal, 
                                origin: torch.Tensor,
                                rotate_mat: torch.Tensor,
                                rotate_angles,
                                radius: float):

        lane_positions_ = torch.zeros(len(lane_positions), self.lseg_len, 2)
        lane_vectors_ = torch.zeros(len(lane_positions), self.lseg_len, 2)
        lane_padding_ = torch.ones(len(lane_positions), self.lseg_len)
        for lidx in range(len(lane_positions)):
            lane_length = lane_lengths[lidx]
            lane_positions_[lidx, :lane_length] = lane_positions[lidx]
            lane_vectors_[lidx, :lane_length] = lane_vectors[lidx]
            lane_padding_[lidx, :lane_length] = 0
        lane_rotate_angles_ = torch.atan2(lane_vectors_[:,0,1], lane_vectors_[:,0,0])

        node_positions = torch.matmul(node_positions - origin, rotate_mat)
        # lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors_.size(0)), node_inds.int()))).t().contiguous()
        lane_actor_index = torch.flip(torch.LongTensor(list(product(node_inds.int(), torch.arange(lane_vectors_.size(0))))).t().contiguous(), [0])
        # lane_actor_vectors = \
        #         lane_positions_[:,0,:].repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors_.size(0), 1)
        assert (torch.tensor(lane_lengths) == 0).float().sum() == 0, '0 length lanes are included'
        lane_actor_vectors = \
                lane_positions_[torch.arange(len(lane_positions)),(torch.tensor(lane_lengths)-1).long(),:].repeat(len(node_inds),1) - node_positions.repeat_interleave(lane_vectors_.size(0), dim=0)
        # mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
        actors_rotate_mat = torch.empty(node_inds.size(0),2,2)
        sin_vals, cos_vals = torch.sin(rotate_angles), torch.cos(rotate_angles)
        actors_rotate_mat[:,0,0] = cos_vals
        actors_rotate_mat[:,0,1] = -sin_vals
        actors_rotate_mat[:,1,0] = sin_vals
        actors_rotate_mat[:,1,1] = cos_vals
        # TODO: lane_actor_index의 size가 0 이라서 1 index 가 out-of-range error 뜨는것 해결
        if len(lane_actor_index) == 0:
            lane_actor_index = lane_actor_index.reshape(2,-1)
            has_goal_ = torch.zeros_like(node_inds, dtype=torch.bool)
            lane2_edges_index = {'succ': torch.LongTensor([[],[]]).reshape(2,-1), 'pred': torch.LongTensor([[],[]]).reshape(2,-1), 'neigh': torch.LongTensor([[],[]]).reshape(2,-1)}

        else:
            lane_actor_vectors_norm = torch.bmm(lane_actor_vectors.unsqueeze(1), actors_rotate_mat[lane_actor_index[1]]).squeeze(1)
            mask = (-20<lane_actor_vectors_norm[:,0]) & (lane_actor_vectors_norm[:,0]<80) & (-50<lane_actor_vectors_norm[:,1]) & (lane_actor_vectors_norm[:,1]<50)
            lane_actor_index = lane_actor_index.reshape(2,-1)[:, mask]
            lane_actor_vectors = lane_actor_vectors[mask]
            goal_idcs = goal_idcs[mask]

            has_goal_msk = has_goal[mask]
            has_goal_idx = has_goal_msk[torch.nonzero(has_goal_msk)] - 1 # 

            ## CROP
            has_goal_ = torch.zeros_like(node_inds, dtype=torch.bool)
            has_goal_[has_goal_idx.squeeze(-1).long()] = True

            unique_lanes = torch.unique(lane_actor_index[0])
            unique_actors = torch.unique(lane_actor_index[1])
            lane_actor_edge_ids = torch.arange(lane_actor_index.size(1))

            lane2_edges_index = {'succ': [], 'pred': [], 'neigh': []}
            ## CROP

            for actor_i in unique_actors:
                lane4actor = lane_actor_index[0][lane_actor_index[1]==actor_i]
                lane_actor_edge_id = lane_actor_edge_ids[lane_actor_index[1]==actor_i]

                for lane_actor_edge_i, src_lane_i in zip(lane_actor_edge_id, lane4actor):
                    succ_dst_lane_idcs = lane_edges_dict['succ'][src_lane_i]
                    pred_dst_lane_idcs = lane_edges_dict['pred'][src_lane_i]
                    neigh_dst_lane_idcs = lane_edges_dict['neigh'][src_lane_i]

                    for succ_dst_lane_i in succ_dst_lane_idcs:
                        if succ_dst_lane_i in lane4actor:
                            lane_actor_edge_j = lane_actor_edge_id[lane4actor==succ_dst_lane_i]
                            lane2_edge_index = torch.LongTensor(list(product([lane_actor_edge_i], lane_actor_edge_j))).t().contiguous()
                            lane2_edges_index['succ'].append(lane2_edge_index)

                    for pred_dst_lane_i in pred_dst_lane_idcs:
                        if pred_dst_lane_i in lane4actor:
                            lane_actor_edge_j = lane_actor_edge_id[lane4actor==pred_dst_lane_i]
                            lane2_edge_index = torch.LongTensor(list(product([lane_actor_edge_i], lane_actor_edge_j))).t().contiguous()
                            lane2_edges_index['pred'].append(lane2_edge_index)

                    for neigh_dst_lane_i in neigh_dst_lane_idcs:
                        if neigh_dst_lane_i in lane4actor:
                            lane_actor_edge_j = lane_actor_edge_id[lane4actor==neigh_dst_lane_i]
                            lane2_edge_index = torch.LongTensor(list(product([lane_actor_edge_i], lane_actor_edge_j))).t().contiguous()
                            lane2_edges_index['neigh'].append(lane2_edge_index)

            for key, lane2_edge_index in lane2_edges_index.items():
                if len(lane2_edge_index) > 0:
                    lane2_edges_index[key] = torch.LongTensor(torch.hstack(lane2_edge_index))
                else:
                    lane2_edges_index[key] = torch.LongTensor([[],[]]).reshape(2,-1)

            assert goal_idcs.sum() == has_goal_.sum(), f'goal number: {goal_idcs.sum().item()}, has_goal number: {has_goal_.sum().item()}'

        return lane_positions_, lane_vectors_, lane_rotate_angles_, lane_padding_, lane_actor_index, lane_actor_vectors, torch.LongTensor(lane_lengths), goal_idcs, has_goal_, lane2_edges_index#, lane_lane_index, lane_lane_vectors

    def get_goal_lane(self, node_positions_goal, node_diff_goal, node_goal_mask, lane_positions, lane_vectors):
        lane_num = len(lane_positions)

        goal_idcs = []
        has_goal = []
        for nidx in range(node_positions_goal.size(0)):
            if not node_goal_mask[nidx]:
                goal_idcs.append(torch.zeros(lane_num))
                has_goal.append(torch.zeros(lane_num))
            else:
                query_pose = node_positions_goal[nidx]
                query_diff = node_diff_goal[nidx]
                query_angle = torch.atan2(query_diff[1], query_diff[0])

                dist_vals = []
                angle_diffs = []
                for lidx in range(len(lane_positions)):
                    lane_poses = lane_positions[lidx]
                    lane_diffs = lane_vectors[lidx]

                    distances = torch.norm(lane_poses - query_pose, p=2, dim=-1)
                    dist_vals.append(torch.min(distances))
                    idx = torch.argmin(distances)

                    lane_angle = torch.atan2(lane_diffs[idx,1], lane_diffs[idx,0])
                    # angle_diff = torch.atan2(torch.sin(query_angle-lane_angle), torch.cos(query_angle-lane_angle))
                    # angle_diff = torch.abs(normalize_angle(angle_diff))
                    angle_diff = torch.abs(normalize_angle(query_angle-lane_angle))
                    angle_diffs.append(angle_diff)

                if torch.norm(query_diff, p=2, dim=-1) < 0.1: # if difference is too small, angle is within noise.
                    idcs_yaw = torch.arange(lane_num)
                else:
                    idcs_yaw = torch.where(torch.tensor(angle_diffs) <= self.lseg_angle_thres*torch.pi/180)[0]
                idcs_dist = torch.where(torch.tensor(dist_vals) <= self.lseg_dist_thres)[0]
                idcs = np.intersect1d(idcs_dist, idcs_yaw)

                if len(idcs) > 0:
                    assigned_node_id = idcs[int(torch.argmin(torch.tensor(dist_vals)[idcs]))]
                    goal_ = torch.zeros(lane_num)
                    goal_[assigned_node_id] = 1.
                    goal_idcs.append(goal_)

                    has_goal_ = torch.zeros(lane_num)
                    has_goal_[assigned_node_id] = nidx+1 ## agent 0은 goal이 없는 0 하고 구분이 안되니까 
                    has_goal.append(has_goal_)
                else:
                    goal_idcs.append(torch.zeros(lane_num))
                    has_goal.append(torch.zeros(lane_num))

        return torch.cat(goal_idcs,-1), torch.cat(has_goal,-1)


    def get_lane_features(self, token, node_positions, node_positions_valid, origin, rotate_mat, radius, lane_resolustion: float = 1):
        
        radius=80

        instance_token, sample_token = token.split("_")
        map_name = self.helper.get_map_name_from_sample_token(sample_token)
        map_api = self.maps[map_name]

        lane_ids = set()
        intersect_polygons = set()

        for node_position in node_positions:
            layers = map_api.get_records_in_radius(node_position[0].item(), node_position[1].item(), radius, ['lane', 'lane_connector', 'road_segment'])
            lanes = layers['lane'] + layers['lane_connector']
            lane_ids.update(lanes)

            road_segments = layers['road_segment']
            for road_segment in road_segments:
                road_segment = map_api.get('road_segment', road_segment)
                if road_segment['is_intersection']:
                    intersect_polygons.update([road_segment['polygon_token']])

        intersections = []
        for polygon_token in intersect_polygons:
            intersections.append(map_api.extract_polygon(polygon_token))

        lane_positions, lane_lengths, lane_vectors, lane_types = [], [], [], []
        lane_tokens = []
        lane_positions_glob = []

        ## filter in local coordi, include all goal lane range
        node_positions_valid = torch.matmul(node_positions_valid-origin, rotate_mat)
        x_min, x_max, y_min, y_max = node_positions_valid[:,0].min()-50, node_positions_valid[:,0].max()+50, node_positions_valid[:,1].min()-50, node_positions_valid[:,1].max()+50

        for lane_id in lane_ids:
            arcline_paths = map_api.get_arcline_path(lane_id)
            for arc_path in arcline_paths:
                arc_ctls = self.arcline_path_utils.discretize(arc_path, lane_resolustion)
                arc_type = torch.tensor(self.arcline_path_utils.compute_segment_sign(arc_path), dtype=torch.float32)

                lane_centerline_global = torch.tensor(np.vstack(arc_ctls)[:,:2], dtype=torch.float32)
                lane_centerline_ = torch.matmul(lane_centerline_global-origin, rotate_mat)

                lane_isin = torch.mul(torch.mul(x_min<lane_centerline_[:,0], lane_centerline_[:,0]<x_max), torch.mul(y_min<lane_centerline_[:,1], lane_centerline_[:,1]<y_max))
                lane_centerline = lane_centerline_[lane_isin]
                lane_centerline_global = lane_centerline_global[lane_isin]
                if lane_centerline.size(0) == 0:
                    continue
                
                n_segments = int(np.ceil(len(lane_centerline) / (self.lseg_len+1)))
                n_poses = int(np.ceil(len(lane_centerline) / n_segments))
                for n in range(n_segments):
                    lane_segment = lane_centerline[n * n_poses: (n+1) * n_poses]
                    lane_segment_global = lane_centerline_global[n * n_poses: (n+1) * n_poses]
                    if len(lane_segment)>1:
                        lane_tokens.append(lane_id)
                        lane_positions.append((lane_segment[1:] + lane_segment[:-1])/2)
                        lane_vectors.append(lane_segment[1:] - lane_segment[:-1])
                        lane_lengths.append(lane_segment[:-1].size(0))
                        lane_types.append(arc_type)

                        lane_positions_glob.append(lane_segment_global)

        e_succ = self.get_successor_edges(lane_tokens, map_api)
        e_pred = self.get_predecessor_edges(e_succ)
        e_neigh = self.get_proximal_edges(lane_positions, lane_vectors, e_succ)
        # e_inter = self.get_intersect_edges(intersections, lane_positions_glob)

        src_nodes, dest_nodes, edge_types = [], [], []
        for src_node_id in range(len(lane_positions)):
            # succ_nodes, pred_nodes, neigh_nodes, inter_nodes = e_succ[src_node_id], e_pred[src_node_id], e_neigh[src_node_id], e_inter[src_node_id]
            succ_nodes, pred_nodes, neigh_nodes = e_succ[src_node_id], e_pred[src_node_id], e_neigh[src_node_id]
            for dest_node_id in succ_nodes:
                src_nodes.append(src_node_id)
                dest_nodes.append(dest_node_id)
                edge_types.append(0.)
            for dest_node_id in pred_nodes:
                src_nodes.append(src_node_id)
                dest_nodes.append(dest_node_id)
                edge_types.append(1.)
            for dest_node_id in neigh_nodes:
                src_nodes.append(src_node_id)
                dest_nodes.append(dest_node_id)
                edge_types.append(2.)
            # for dest_node_id in inter_nodes:
            #     src_nodes.append(src_node_id)
            #     dest_nodes.append(dest_node_id)
            #     edge_types.append(3.)

        lane_edges = torch.LongTensor([src_nodes, dest_nodes])
        lane_edge_types = torch.tensor(edge_types)

        edges = {'succ': e_succ, 'pred': e_pred, 'neigh': e_neigh}

        return lane_positions, lane_vectors, lane_lengths, lane_edges, lane_edge_types, edges

    def get_motions(self, token: str):
        instance_token, sample_token = token.split("_")
        origin = self.get_target_agent_global_pose(token)

        # Load all actors for sample
        actors_pasts = self.helper.get_past_for_sample(sample_token, seconds=self.t_h, in_agent_frame=False, just_xy=True)
        actors_futures = self.helper.get_future_for_sample(sample_token, seconds=self.t_f, in_agent_frame=False, just_xy=True)
        actors_curents_ = self.helper.get_annotations_for_sample(sample_token)
        actors_curents = {}
        actors_states = {}
        actors_category = []
        for annotation in actors_curents_:
            if 'vehicle' in annotation['category_name']:
                ann_i_t = annotation['instance_token']
                if len(annotation['attribute_tokens'])>0 and ann_i_t != instance_token and 'parked' in self.nusc.get('attribute', annotation['attribute_tokens'][0])['name']:
                    continue
                present_pose = np.asarray(annotation['translation'][0:2]).reshape(1, 2)
                actors_curents[ann_i_t] = present_pose

                motion_state = self.get_past_motion_states(ann_i_t, sample_token)
                actors_states[ann_i_t] = motion_state
                try:
                    actors_category.append(ACTOR_CATEGORY[annotation['category_name']])
                except:
                    actors_category.append(ACTOR_CATEGORY['others'])
        
        num_nodes = len(actors_curents)
        current_i_ts = list(actors_curents.keys())
        assert num_nodes == len(current_i_ts)

        total_timestamp = (self.t_h + self.t_f) * self.res + 1
        motions = np.zeros((num_nodes, total_timestamp, 2))
        past_states = np.zeros((num_nodes, self.t_h*2+1, 3))
        agent_mask = np.zeros(num_nodes, dtype=bool)
        valid_mask = np.zeros((num_nodes, total_timestamp), dtype=bool)

        for idx, i_t in enumerate(current_i_ts):
            past = np.flip(np.array(actors_pasts[i_t]).reshape(-1,2), axis=0)
            current = np.array(actors_curents[i_t]).reshape(-1,2)
            future = np.array(actors_futures[i_t]).reshape(-1,2)

            motion = np.concatenate((past, current, future), axis=0)

            past_valid = np.zeros(self.t_h*self.res, dtype=bool)
            current_valid = np.ones(1, dtype=bool)
            future_valid = np.zeros(self.t_f*self.res, dtype=bool)

            past_valid[:len(past)] = True
            past_valid = np.flip(past_valid)
            future_valid[:len(future)] = True

            all_valid = np.concatenate((past_valid,current_valid,future_valid))

            motions[idx, all_valid] = motion
            past_states[idx] = actors_states[i_t]
            valid_mask[idx] = all_valid

            if i_t == instance_token:
                agent_mask[idx] = True

        return motions, past_states, valid_mask, agent_mask, current_i_ts, np.array(actors_category)
    
    def get_target_agent_global_pose(self, token: int) -> Tuple[float, float, float]:
        """
        Returns global pose of target actor
        :param idx: data index
        :return global_pose: (x, y, yaw) or target actor in global co-ordinates
        """
        i_t, s_t = token.split("_")
        sample_annotation = self.helper.get_sample_annotation(i_t, s_t)
        x, y = sample_annotation['translation'][:2]
        yaw = quaternion_yaw(Quaternion(sample_annotation['rotation']))
        yaw = correct_yaw(yaw)
        global_pose = (x, y, yaw)

        return global_pose

    def get_past_motion_states(self, i_t, s_t):
        """
        Returns past motion states: v, a, yaw_rate for a given instance and sample token over self.t_h seconds
        """
        motion_states = np.zeros((2 * self.t_h + 1, 3))
        motion_states[-1, 0] = self.helper.get_velocity_for_agent(i_t, s_t)
        motion_states[-1, 1] = self.helper.get_acceleration_for_agent(i_t, s_t)
        motion_states[-1, 2] = self.helper.get_heading_change_rate_for_agent(i_t, s_t)
        hist = self.helper.get_past_for_agent(i_t, s_t, seconds=self.t_h, in_agent_frame=True, just_xy=False)

        for k in range(len(hist)):
            motion_states[-(k + 2), 0] = self.helper.get_velocity_for_agent(i_t, hist[k]['sample_token'])
            motion_states[-(k + 2), 1] = self.helper.get_acceleration_for_agent(i_t, hist[k]['sample_token'])
            motion_states[-(k + 2), 2] = self.helper.get_heading_change_rate_for_agent(i_t, hist[k]['sample_token'])

        motion_states = np.nan_to_num(motion_states)
        return motion_states

    @staticmethod
    def get_lane_flags(lanes: List[List[Tuple]], polygons: Dict[str, List[Polygon]]) -> List[np.ndarray]:
        """
        Returns flags indicating whether each pose on lane polylines lies on polygon map layers
        like stop-lines or cross-walks
        :param lanes: list of lane poses
        :param polygons: dictionary of polygon layers
        :return lane_flags: list of ndarrays with flags
        """

        lane_flags = [np.zeros((len(lane), len(polygons.keys()))) for lane in lanes]
        for lane_num, lane in enumerate(lanes):
            for pose_num, pose in enumerate(lane):
                point = Point(pose[0], pose[1])
                for n, k in enumerate(polygons.keys()):
                    polygon_list = polygons[k]
                    for polygon in polygon_list:
                        if polygon.contains(point):
                            lane_flags[lane_num][pose_num][n] = 1
                            break

        return lane_flags

    @staticmethod
    def get_successor_edges(lane_ids: torch.Tensor, map_api) -> List[List[int]]:
        """
        Returns successor edge list for each node
        """
        e_succ = []
        for node_id, lane_id in enumerate(lane_ids):
            e_succ_node = []
            if node_id + 1 < len(lane_ids) and lane_id == lane_ids[node_id + 1]:
                e_succ_node.append(node_id + 1)
            else:
                outgoing_lane_ids = map_api.get_outgoing_lane_ids(lane_id)
                for outgoing_id in outgoing_lane_ids:
                    if outgoing_id in lane_ids:
                        e_succ_node.append(lane_ids.index(outgoing_id))

            e_succ.append(e_succ_node)

        return e_succ

    @staticmethod
    def get_predecessor_edges(e_succ) -> List[List[int]]:
        """
        Returns successor edge list for each node
        """
        num_nodes = len(e_succ)
        e_pred = [[] for _ in range(num_nodes)]

        for node_id, nodes_succ in enumerate(e_succ):
            for node_succ in nodes_succ:
                e_pred[node_succ].append(node_id)

        return e_pred
    
    @staticmethod
    def get_proximal_edges(lane_node_feats: List[np.ndarray], lane_vectors: List[np.ndarray], e_succ: List[List[int]],
                           dist_thresh=4, yaw_thresh=np.pi/4) -> List[List[int]]:
        """
        Returns proximal edge list for each node
        """
        e_prox = [[] for _ in lane_node_feats]

        for src_node_id, src_node_feats in enumerate(lane_node_feats):
            for dest_node_id in range(src_node_id + 1, len(lane_node_feats)):
                if dest_node_id not in e_succ[src_node_id] and src_node_id not in e_succ[dest_node_id]:
                    dest_node_feats = lane_node_feats[dest_node_id]

                    pairwise_dist = cdist(src_node_feats[:, :2], dest_node_feats[:, :2])
                    min_dist = np.min(pairwise_dist)
                    if min_dist <= dist_thresh:
                        src_node_diff, dest_node_diff = lane_vectors[src_node_id], lane_vectors[dest_node_id]

                        yaw_src = np.arctan2(torch.mean(src_node_diff[:,1]),
                                             torch.mean(src_node_diff[:,0]))
                        yaw_dest = np.arctan2(torch.mean(dest_node_diff[:, 1]),
                                              torch.mean(dest_node_diff[:, 0]))
                        yaw_diff = np.arctan2(np.sin(yaw_src-yaw_dest), np.cos(yaw_src-yaw_dest))
                        yaw_diff = normalize_angle(yaw_diff)
                            
                        if np.absolute(yaw_diff) <= yaw_thresh:
                            e_prox[src_node_id].append(dest_node_id)
                            e_prox[dest_node_id].append(src_node_id)

        return e_prox

    @staticmethod
    def get_intersect_edges(intersect_polygons, lane_positions_global) -> List[List[int]]:
        """
        Returns intersection edge list for each node
        """
        intersect_neighbors = [[] for _ in range(len(intersect_polygons))]
        for lane_id, lane_position in enumerate(lane_positions_global):
            mean_pos = lane_position.mean(dim=0)
            mean_pos = Point(mean_pos[0], mean_pos[1])

            for pol_id, polygon in enumerate(intersect_polygons):
                if polygon.contains(mean_pos):
                    intersect_neighbors[pol_id].append(lane_id)

        e_inter = [[] for _ in range(len(lane_positions_global))]
        for union in intersect_neighbors:
            for i, src_node_id in enumerate(union):
                for j in range(i+1, len(union)):
                    dest_node_id = union[j]
                    e_inter[src_node_id].append(dest_node_id)
                    e_inter[dest_node_id].append(src_node_id)

        return e_inter
    
def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def correct_yaw(yaw: float) -> float:
    """
    nuScenes maps were flipped over the y-axis, so we need to
    add pi to the angle needed to rotate the heading.
    :param yaw: Yaw angle to rotate the image.
    :return: Yaw after correction.
    """
    if yaw <= 0:
        yaw = -np.pi - yaw
    else:
        yaw = np.pi - yaw

    return yaw

def normalize_angle(angle):
    if angle < -torch.pi:
        angle += 2*torch.pi
    elif angle > torch.pi:
        angle -= 2*torch.pi
    return angle

if __name__ == '__main__':

    split = 'train'
    spec_args = {'dataset': 'nuScenes', 'n_jobs': 8, 't_h': 2, 't_f': 6, 'res': 2, 'ref_time':4, 'lseg_len': 10, 'lseg_angle_thres': 30, 'lseg_dist_thres': 2.5, 'random_flip': True, }
    A1D = nuScenesDataset(split, root='data/nuScenes', process_dir='data/nus_preprocessed',
                             version='v1.0-trainval', process=True, spec_args=spec_args)