"""Stage 1 (JRDB): per-frame agent positions, BEV 3D pose and JRDB-Act/Social captions.

For every scene this writes
    <output_dir>/<scene>_agents_<start>_to_<end>.pt        {frame: {agent_id: {...}}}
    <output_dir>/<scene>_interactions_<start>_to_<end>.pt  {frame: {'dual_interaction': [...], 'cluster': {...}}}

Captions are built from the JRDB-Act / JRDB-Social labels (no VLM is needed for JRDB).
3D pose comes from BEV (see README: put BEV.pth / SMPLA_NEUTRAL.pth / smil_packed_info.pth under $KDTF_ROMP_DIR).

Example:
    python preprocessing/stage1_jrdb.py --jrdb_root /data/jrdb/train_dataset --output_dir data/preprocessed_1st/jrdb_v1
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from parser_model.captions import Caption3DPoseEngine

DEFAULT_ODOMETRY = os.path.join(REPO_ROOT, 'preprocessing', 'jrdb', 'odometry_processed')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--jrdb_root', required=True, help='JRDB train_dataset/ (images/, labels/)')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--odometry_dir', default=DEFAULT_ODOMETRY, help='<scene>_pos.npy / <scene>_orientation.npy per scene')
    parser.add_argument('--scenes', nargs='*', default=None, help='Only process these scenes (default: all)')
    parser.add_argument('--max_frames', type=int, default=None, help='Debug: only process the first N frames of each scene')
    parser.add_argument('--visualize', action='store_true', help='Save a 3D pose plot per frame under <output_dir>/<scene>/frames')
    args = parser.parse_args()

    label_2d_base = os.path.join(args.jrdb_root, 'labels', 'labels_2d_stitched')
    label_3d_base = os.path.join(args.jrdb_root, 'labels', 'labels_3d')
    label_social_base = os.path.join(args.jrdb_root, 'labels', 'labels_2d_activity_social_stitched')
    img_base = os.path.join(args.jrdb_root, 'images', 'image_stitched')
    scenes = sorted(os.path.basename(os.path.normpath(s)) for s in glob.glob(os.path.join(args.jrdb_root, 'images', 'image_0', '*')))
    if args.scenes:
        scenes = [s for s in scenes if s in args.scenes]
    os.makedirs(args.output_dir, exist_ok=True)
    if args.visualize:
        from utils.plot_2d_3d import plot_3d_human

    caption_engine = Caption3DPoseEngine('local', args.visualize)
    for scene_idx, scene_name in enumerate(scenes):
        print(f'Processing scene: {scene_name}, {scene_idx + 1} out of {len(scenes)} scenes.')
        # Robot odometry, relative to the first frame
        odometry_pos = np.load(os.path.join(args.odometry_dir, f'{scene_name}_pos.npy'))
        odometry_pos -= odometry_pos[0]
        odometry_ori = np.load(os.path.join(args.odometry_dir, f'{scene_name}_orientation.npy'))
        odometry_ori -= odometry_ori[0]

        caption_engine.load_files(os.path.join(label_2d_base, scene_name + '.json'), os.path.join(label_3d_base, scene_name + '.json'),
                                  os.path.join(img_base, scene_name) + '/', os.path.join(label_social_base, scene_name + '.json'))
        frame_num = len(glob.glob(os.path.join(img_base, scene_name, '*.jpg')))
        start_frame_idx, last_frame_idx = 0, frame_num
        if args.max_frames is not None:
            frame_num = min(frame_num, args.max_frames)

        save_data, save_interaction = {}, {}
        for frame_idx in tqdm(range(start_frame_idx, frame_num)):
            caption_engine.preprocess_frame(frame_idx, odometry_pos[frame_idx], odometry_ori[frame_idx])
            caption_engine.regress_3dpose()

            save_data[frame_idx] = {}
            for agent_id, agent in caption_engine.agents.items():
                save_data[frame_idx][agent_id] = {
                    'description': agent.caption,
                    'global_position': agent.global_pos,
                    'local_position': agent.local_pos,
                    'pose': agent.pose,
                    'robot_pos': agent.robot_pos,
                    'robot_ori': agent.robot_ori,
                    'rot_z': agent.rot_z,
                    'cluster_id': -1,  # default for agents with no cluster
                    'bbox': agent.bbox,
                }

            save_interaction[frame_idx] = {'cluster': {}}
            socials = caption_engine.socials
            if len(socials.interactions) > 0:
                save_interaction[frame_idx]['dual_interaction'] = [[k, v] for k, v in socials.interactions.items()]
                for cluster_id, (cluster_caption, cluster_agents) in socials.clusters.items():
                    for agent_in_cluster in cluster_agents:
                        save_data[frame_idx][int(agent_in_cluster)]['cluster_id'] = cluster_id
                    save_interaction[frame_idx]['cluster'][cluster_id] = cluster_caption

            if args.visualize:
                frame_save_dir = os.path.join(args.output_dir, scene_name, 'frames')
                os.makedirs(frame_save_dir, exist_ok=True)
                plot_3d_human(frame_idx, save_data, None, save_dir=frame_save_dir)

        suffix = f'{start_frame_idx}_to_{last_frame_idx}.pt'
        torch.save(save_data, os.path.join(args.output_dir, f'{scene_name}_agents_{suffix}'))
        torch.save(save_interaction, os.path.join(args.output_dir, f'{scene_name}_interactions_{suffix}'))


if __name__ == '__main__':
    main()
