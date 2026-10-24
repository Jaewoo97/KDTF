"""Stage 2: turn stage-1 per-scene dictionaries into HiVT `TemporalData` samples (one .pt per sliding window).

Each sample covers FRAME_LENGTH=20 steps (8 observed + 12 future) at 2.5 FPS and contains, per agent,
the trajectory, the SMPL pose (JRDB/SIT) or CLIP image feature (ETH/UCY), and the TinyBERT [CLS] embedding of
the agent's caption. JRDB additionally stores pairwise-interaction caption embeddings on the edges.

Output layout (one folder per scene, consumed by split_train_val.py):
    <output_dir>/<scene_name>/<initial_frame>.pt

Example:
    python preprocessing/stage2_build_samples.py --dataset jrdb \
        --input_dir data/preprocessed_1st/jrdb_v1 --output_dir data/preprocessed_2nd/jrdb_v2_fps_2_5_frame_20_withRobot
"""
import argparse
import glob
import os
import sys
from itertools import permutations

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from HiVT.utils_hivt import TemporalData  # samples are pickled with this class path; keep the import as is

# Per-dataset settings used for the paper data. FRAME_STEP converts raw annotation frames to 2.5 FPS.
DATASET_CFG = {
    'jrdb': dict(frame_step=6, position_key='global_position', pose_dim=72, pose_heading=True,
                 strict_agent_filter=True, min_agents=0, interactions=True, start_from_first_frame=False),
    'sit': dict(frame_step=4, position_key='global_position', pose_dim=72, pose_heading=True,
                strict_agent_filter=False, min_agents=1, interactions=False, start_from_first_frame=False),
    'ethucy': dict(frame_step=1, position_key='local_position', pose_dim=512, pose_heading=False,
                   strict_agent_filter=False, min_agents=2, interactions=False, start_from_first_frame=True),
}


class CachedTextEncoder:
    """TinyBERT [CLS] embedding (312-d) with a cache, since captions repeat a lot across frames."""

    def __init__(self):
        from parser_model.text_encoder_bert import TextEncoder_TinyBERT
        self.encoder = TextEncoder_TinyBERT()
        self.cache = {}

    def __call__(self, text):
        if text not in self.cache:
            self.cache[text] = self.encoder(text).cpu()
        return self.cache[text]


class PreProcess:
    TEXT_EMBED_DIM = 312  # TinyBERT
    FRAME_LENGTH = 20
    PAST_FRAME = 8
    FUT_FRAME = 12
    TEMP_NUM_AGENTS = 100  # upper bound on distinct agents inside one window

    def __init__(self, dataset, agent_data, scene_name, scene_idx, interaction_data=None, fix_pairwise_interaction=False):
        self.dataset = dataset
        self.cfg = DATASET_CFG[dataset]
        self.agent_data = agent_data
        self.interaction_data = interaction_data
        self.scene_name = scene_name
        self.scene_idx = scene_idx
        self.fix_pairwise_interaction = fix_pairwise_interaction

    def __call__(self, initial_frame, text_encoder):
        cfg = self.cfg
        A, T, P = self.TEMP_NUM_AGENTS, self.FRAME_LENGTH, self.PAST_FRAME

        agent_position = torch.zeros(A, T, 2)
        agent_ids = torch.zeros(A, dtype=torch.int) - 1
        robot_position = torch.zeros(1, T, 2)
        agent_pose = torch.zeros(A, T, cfg['pose_dim'])
        agent_pose_mask = torch.ones(A, T, dtype=torch.bool)
        agent_description = torch.zeros(A, T, self.TEXT_EMBED_DIM)
        agent_description_mask = torch.ones(A, T, dtype=torch.bool)
        agent_mask = torch.ones(A, T, dtype=torch.bool)  # True = no observation
        b_ori = torch.zeros(A, T, 2)
        pair_captions = []  # (save_idx1, save_idx2, frame, caption)
        agent_id_to_idx = {}
        frames = torch.zeros(T, dtype=torch.int)

        for frame in range(T):
            actual_frame = initial_frame + frame * cfg['frame_step']
            frames[frame] = actual_frame
            if actual_frame not in self.agent_data:
                continue
            for agent_id, value in self.agent_data[actual_frame].items():
                if 'robot_pos' in value:
                    robot_position[0, frame] = torch.tensor(value['robot_pos'][:2])
                if agent_id not in agent_id_to_idx:
                    agent_id_to_idx[agent_id] = len(agent_id_to_idx)
                idx = agent_id_to_idx[agent_id]
                if self.dataset == 'sit':
                    agent_ids[idx] = agent_id

                agent_position[idx, frame] = torch.from_numpy(np.asarray(value[cfg['position_key']][:2], dtype=np.float64))
                agent_mask[idx, frame] = False

                if value['pose'] is not None:
                    if cfg['pose_dim'] == 72:  # SMPL thetas from BEV
                        agent_pose[idx, frame] = torch.from_numpy(value['pose']['smpl_thetas'])
                        b_ori[idx, frame] = value['pose']['b_ori'].cpu()[0, :2]
                    else:  # ETH/UCY: CLIP image feature of the agent crop
                        agent_pose[idx, frame] = value['pose']
                    agent_pose_mask[idx, frame] = False

                if value['description'] is not None:
                    agent_description[idx, frame] = text_encoder(value['description'])
                    agent_description_mask[idx, frame] = False

            if cfg['interactions'] and 'dual_interaction' in self.interaction_data[actual_frame]:
                for (agent_id1, agent_id2), description in self.interaction_data[actual_frame]['dual_interaction']:
                    if agent_id1 in agent_id_to_idx and agent_id2 in agent_id_to_idx:
                        pair_captions.append((agent_id_to_idx[agent_id1], agent_id_to_idx[agent_id2], frame, description))

        # Agents with more than half of the past observed
        valid = (~agent_mask[:, :P]).sum(-1) > (P // 2)
        if not cfg['strict_agent_filter'] and valid.sum() < cfg['min_agents']:
            return None, 0

        x, positions = agent_position.clone(), agent_position.clone()
        padding_mask = agent_mask
        bos_mask = torch.zeros(A, P, dtype=torch.bool)
        bos_mask[:, 0] = ~padding_mask[:, 0]
        bos_mask[:, 1:P] = padding_mask[:, :P - 1] & ~padding_mask[:, 1:P]

        # Heading of each agent: body orientation from BEV if the pose is recent enough, else last velocity
        rotate_angles = torch.zeros(A)
        has_consecutive_obs = torch.zeros(A, dtype=torch.bool)
        valid_indices = [torch.nonzero(row[:P], as_tuple=True)[0] for row in ~padding_mask]
        valid_pose_indices = [torch.nonzero(row[:P], as_tuple=True)[0] for row in ~agent_pose_mask]
        for i in range(A):
            if len(valid_indices[i]) < 2:
                continue
            if (valid_indices[i][1:] - valid_indices[i][:-1]).min() > 1:
                continue
            has_consecutive_obs[i] = True
            if cfg['pose_heading'] and len(valid_pose_indices[i]) > 0 and valid_pose_indices[i].max() > P - 3:
                heading_vector = b_ori[i, valid_pose_indices[i].max()]
            else:
                heading_vector = x[i, valid_indices[i][-1]] - x[i, valid_indices[i][-2]]
            rotate_angles[i] = torch.atan2(heading_vector[1], heading_vector[0])

        # Past: per-step displacement; future: offset from the last observed position
        x[:, P:] = x[:, P:] - x[:, P - 1].unsqueeze(-2)
        x[:, 1:P] = x[:, 1:P] - x[:, :P - 1]
        x[:, 0] = 0
        padding_mask_tmp = padding_mask.clone()
        padding_mask_tmp[:, 1:P] = padding_mask[:, 1:P] | padding_mask[:, :P - 1]
        x[padding_mask_tmp] = 0

        sin_vals, cos_vals = torch.sin(rotate_angles), torch.cos(rotate_angles)
        rotate_mat = torch.zeros(A, 3, 3)
        rotate_mat[:, 0, 0] = cos_vals
        rotate_mat[:, 0, 1] = -sin_vals
        rotate_mat[:, 1, 0] = sin_vals
        rotate_mat[:, 1, 1] = cos_vals
        rotate_mat[:, 2, 2] = 1
        y = torch.bmm(x[:, P:], rotate_mat[:, :2, :2])

        if cfg['strict_agent_filter']:
            current_observed = ~((positions[:, P - 1, :] == 0).sum(-1) == 2)
            valid = valid & current_observed & has_consecutive_obs
        num_valid_agents = int(valid.sum())
        edge_index = torch.LongTensor(list(permutations(range(num_valid_agents), 2))).t().contiguous()

        scene = {
            'num_nodes': valid.sum(),
            'rotate_mat': rotate_mat[valid].clone(),
            'scene': self.scene_name,
            'x': x[valid][:, :P].clone(),
            'x_pose': agent_pose[valid][:, :P].clone(),
            'x_pose_mask': agent_pose_mask[valid][:, :P].clone(),
            'x_text': agent_description[valid][:, :P].clone(),
            'x_text_mask': agent_description_mask[valid][:, :P].clone(),
            'positions': positions[valid].clone(),
            'rotate_angles': rotate_angles[valid].clone(),
            'padding_mask': padding_mask[valid].clone(),
            'edge_index': edge_index,
            'bos_mask': bos_mask[valid].clone(),
            'y': y[valid].clone(),
            'y_pose': agent_pose[valid][:, P:].clone(),
            'y_pose_mask': agent_pose_mask[valid][:, P:].clone(),
            'y_text': agent_description[valid][:, P:].clone(),
            'y_text_mask': agent_description_mask[valid][:, P:].clone(),
        }
        if self.dataset != 'ethucy':
            scene.update({'robot_pos': robot_position, 'scene_idx': self.scene_idx, 'initial_frame': initial_frame})
        if self.dataset == 'sit':
            scene['agent_ids'] = agent_ids[valid].clone()
        if cfg['interactions']:
            scene['actual_frames'] = frames
            edge_attr, edge_attr_mask = self._pairwise_interaction(pair_captions, valid, edge_index, text_encoder)
            scene.update({'x_interaction': edge_attr[:, :P].clone(), 'x_interaction_mask': edge_attr_mask[:, :P].clone(),
                          'y_interaction': edge_attr[:, P:].clone(), 'y_interaction_mask': edge_attr_mask[:, P:].clone()})
        return TemporalData(**scene), num_valid_agents

    def _pairwise_interaction(self, pair_captions, valid, edge_index, text_encoder):
        """Edge features (num_edges, T, D) from JRDB-Social pairwise interaction captions."""
        A, T, D = self.TEMP_NUM_AGENTS, self.FRAME_LENGTH, self.TEXT_EMBED_DIM
        n = int(valid.sum())
        if not self.fix_pairwise_interaction:
            # NOTE: the original script selected agents with `dual[valid, valid]`, which picks only the (empty)
            # self-pairs, so every edge ends up zero and fully masked. This reproduces the data used in the paper.
            return torch.zeros(edge_index.shape[1], T, D), torch.ones(edge_index.shape[1], T, dtype=torch.bool)
        dual = torch.zeros(A, A, T, D)
        dual_mask = torch.ones(A, A, T, dtype=torch.bool)
        for i1, i2, frame, description in pair_captions:
            emb = text_encoder(description)
            dual[i1, i2, frame] = dual[i2, i1, frame] = emb
            dual_mask[i1, i2, frame] = dual_mask[i2, i1, frame] = False
        dual, dual_mask = dual[valid][:, valid], dual_mask[valid][:, valid]
        assert dual.shape[0] == n
        return dual[edge_index[0], edge_index[1]], dual_mask[edge_index[0], edge_index[1]]


def list_scenes(input_dir):
    """Stage-1 outputs are `<scene>_agents_<s>_to_<e>.pt` (+ `<scene>_interactions_<s>_to_<e>.pt` for JRDB)."""
    agent_files = sorted(glob.glob(os.path.join(input_dir, '*_agents_*.pt')))
    if not agent_files:
        raise FileNotFoundError(f'No *_agents_*.pt files in {input_dir}')
    return agent_files


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, choices=list(DATASET_CFG.keys()))
    parser.add_argument('--input_dir', required=True, help='Stage-1 output directory')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--scene_idx_start', type=int, default=0, help='Process scenes [start, end] (inclusive), for parallel runs')
    parser.add_argument('--scene_idx_end', type=int, default=10 ** 6)
    parser.add_argument('--max_samples_per_scene', type=int, default=None, help='Debug: stop after N windows per scene')
    parser.add_argument('--fix_pairwise_interaction', action='store_true',
                        help='JRDB only: actually fill pairwise-interaction edge features (the paper data has them all masked)')
    args = parser.parse_args()

    cfg = DATASET_CFG[args.dataset]
    text_encoder = CachedTextEncoder()
    agent_files = list_scenes(args.input_dir)
    for scene_idx, agent_file in enumerate(agent_files):
        if not args.scene_idx_start <= scene_idx <= args.scene_idx_end:
            continue
        scene_name = os.path.basename(agent_file)[:-3]
        print(f'[{scene_idx + 1}/{len(agent_files)}] {scene_name}')
        agent_data = torch.load(agent_file)
        interaction_data = None
        if cfg['interactions']:
            interaction_data = torch.load(agent_file.replace('_agents_', '_interactions_'))

        preprocess = PreProcess(args.dataset, agent_data, scene_name, scene_idx, interaction_data, args.fix_pairwise_interaction)
        frame_range = sorted(agent_data.keys())
        first_frame = frame_range[0] if cfg['start_from_first_frame'] else 0
        last_initial_frame = frame_range[-1] - cfg['frame_step'] * (PreProcess.FRAME_LENGTH - 1)
        scene_dir = os.path.join(args.output_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True)

        num_saved, max_agents = 0, 0
        for initial_frame in tqdm(range(first_frame, last_initial_frame)):
            data, num_agents = preprocess(initial_frame, text_encoder)
            if data is None:
                continue
            torch.save(data, os.path.join(scene_dir, f'{initial_frame}.pt'))
            num_saved += 1
            max_agents = max(max_agents, num_agents)
            if args.max_samples_per_scene is not None and num_saved >= args.max_samples_per_scene:
                break
        print(f'  saved {num_saved} samples, max agents per sample: {max_agents}')


if __name__ == '__main__':
    main()
