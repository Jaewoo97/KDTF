"""Stage 1 (SIT): per-frame agent positions, BEV 3D pose and PLLaVA-13B video captions.

Each scene folder of the SIT dataset must contain cam_img/, ego_trajectory/, label_2d/ and label_3d/; scenes
missing any of these are skipped. Writes <output_dir>/<scene>_agents_<start>_to_<end>.pt per scene.

Captioning runs PLLaVA-13B (set PLLAVA_MODEL_DIR) on the last USE_FRAME=8 crops of each agent, so one scene
per process is the practical unit:
    for i in $(seq 0 62); do python preprocessing/stage1_sit.py --sit_root /data/sit --output_dir data/preprocessed_1st/sit_v1_prompt3 --scene_idx $i; done
"""
import argparse
import glob
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from parser_model.captions_sit import Caption3DPoseEngine

USE_FRAME = 8  # number of past frames given to the video captioner


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sit_root', required=True, help='Folder with one sub-folder per SIT scene')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--scene_idx', type=int, nargs='*', default=None, help='Scene indices (sorted order) to process; default: all')
    parser.add_argument('--prompt_idx', type=int, default=3, choices=[1, 2, 3], help='Captioning prompt (paper data: 3)')
    parser.add_argument('--visualize', action='store_true', help='Save 3D pose plots for the first 30 frames')
    args = parser.parse_args()

    scene_dirs = sorted(glob.glob(os.path.join(args.sit_root, '*')))
    os.makedirs(args.output_dir, exist_ok=True)
    if args.visualize:
        from utils.plot_2d_3d import plot_3d_human
    caption_engine = Caption3DPoseEngine('eval_sit', USE_FRAME)

    for scene_idx, scene_dir in enumerate(scene_dirs):
        if args.scene_idx is not None and scene_idx not in args.scene_idx:
            continue
        scene = os.path.basename(scene_dir)
        dir_items = {os.path.basename(p) for p in glob.glob(scene_dir + '/*')}
        if not {'cam_img', 'ego_trajectory', 'label_2d', 'label_3d'} <= dir_items:
            print(f'Skipping {scene}: missing annotations')
            continue
        print(f'Processing scene: {scene}, {scene_idx + 1} out of {len(scene_dirs)} scenes.')

        cap_frame_save_dir = os.path.join(args.output_dir, scene, 'cap_frames')
        os.makedirs(cap_frame_save_dir, exist_ok=True)
        caption_engine.load_files(os.path.join(scene_dir, 'label_3d'), os.path.join(scene_dir, 'label_2d'),
                                  os.path.join(scene_dir, 'cam_img'), os.path.join(scene_dir, 'ego_trajectory'))
        num_frames = len(glob.glob(os.path.join(scene_dir, 'cam_img', '1', 'data_rgb', '*.png')))

        save_data = {}
        start_frame_idx = 0
        for frame_idx in tqdm(range(start_frame_idx, num_frames)):
            caption_engine.preprocess_frame(frame_idx, scene_idx, USE_FRAME)  # trajectories + cropped images
            if frame_idx < start_frame_idx + USE_FRAME - 1:  # wait until USE_FRAME frames are gathered
                continue
            caption_engine.caption_multiframe(frame_idx, scene_idx, USE_FRAME - 1, cap_frame_save_dir, use_prompt=args.prompt_idx)
            caption_engine.regress_3dpose(frame_idx)

            save_data[frame_idx] = {}
            for agent_id, agent in caption_engine.agents.items():
                if frame_idx not in agent.global_pos:
                    continue
                save_data[frame_idx][agent_id] = {
                    'description': agent.caption,
                    'local_position': agent.global_pos[frame_idx][:2] - agent.robot_pos,
                    'global_position': agent.global_pos[frame_idx],
                    'pose': agent.pose,
                    'robot_pos': agent.robot_pos,
                    'robot_ori': agent.robot_ori,
                    'rot_z': agent.rot_z,
                }
            if args.visualize and frame_idx < 30:
                frame_save_dir = os.path.join(args.output_dir, scene, 'frames')
                os.makedirs(frame_save_dir, exist_ok=True)
                plot_3d_human(frame_idx, save_data, None, save_dir=frame_save_dir)

        torch.save(save_data, os.path.join(args.output_dir, f'{scene}_agents_{start_frame_idx}_to_{num_frames}.pt'))


if __name__ == '__main__':
    main()
