"""Stage 1 (ETH/UCY): per-frame positions, rule-based captions and CLIP image features.

Captions describe motion (walking / walking slowly / standing still, from the last USE_FRAME frames) and the
direction of the closest obstacle on the Y-Net oracle segmentation map. In place of a body pose, ETH/UCY samples
use the CLIP ViT-B/16 image feature (512-d) of the agent crop.

Expected <ethucy_root> layout:
    biwi_eth.txt biwi_hotel.txt crowds_zara01.txt crowds_zara02.txt students003.txt   (world-coordinate tracks)
    images/<scene>/<frame:04d>.jpg                                                    (ethucy_extract_frames.py)
    ynet/<scene>_oracle.png                                                           (Y-Net oracle semantic maps)

Example:
    python preprocessing/stage1_ethucy.py --ethucy_root /data/ethucy --output_dir data/preprocessed_1st/ethucy_v1
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from parser_model.captions_ethucy_v3 import Caption3DPoseEngine

SCENES = ['eth', 'hotel', 'zara1', 'zara2', 'students']
TRAJ_FILES = ['biwi_eth.txt', 'biwi_hotel.txt', 'crowds_zara01.txt', 'crowds_zara02.txt', 'students003.txt']
MAP_FILES = ['eth_oracle.png', 'hotel_oracle.png', 'zara1_oracle.png', 'zara2_oracle.png', 'students_oracle.png']
# world -> image homographies
HOMOGRAPHIES = [np.array([[2.8128700e-02, 2.0091900e-03, -4.6693600e+00], [8.0625700e-04, 2.5195500e-02, -5.0608800e+00], [3.4555400e-04, 9.2512200e-05, 4.6255300e-01]]),
                np.array([[1.1048200e-02, 6.6958900e-04, -3.3295300e+00], [-1.5966000e-03, 1.1632400e-02, -5.3951400e+00], [1.1190700e-04, 1.3617400e-05, 5.4276600e-01]]),
                np.array([[0.02174104, 0, -0.15], [0, -0.02461883, 13.77429807], [0, 0, 1]]),
                np.array([[0.02174104, 0, -0.4], [0, -0.02386598, 14.98401686], [0, 0, 1]]),
                np.array([[0.02220407, 0, -0.48], [0, -0.02477289, 13.92551292], [0, 0, 1]])]
USE_FRAME = 2  # frames used to decide whether the agent is walking


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ethucy_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--image_dir', default=None, help='Default: <ethucy_root>/images')
    parser.add_argument('--map_dir', default=None, help='Default: <ethucy_root>/ynet')
    parser.add_argument('--scenes', nargs='*', default=SCENES, choices=SCENES)
    parser.add_argument('--max_frames', type=int, default=None, help='Debug: stop after N frames per scene')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    image_dir = args.image_dir or os.path.join(args.ethucy_root, 'images')
    map_dir = args.map_dir or os.path.join(args.ethucy_root, 'ynet')
    os.makedirs(args.output_dir, exist_ok=True)
    if args.visualize:
        from utils.plot_2d_3d import plot_2d_human

    for scene_idx, scene in enumerate(SCENES):
        if scene not in args.scenes:
            continue
        print(f'Processing scene: {scene}, {scene_idx + 1} out of {len(SCENES)} scenes.')
        frame_num = len(glob.glob(os.path.join(image_dir, scene, '*.jpg')))
        conv_mode = 'eval_ethucy0' if scene_idx == 0 else 'eval_ethucy'
        caption_engine = Caption3DPoseEngine(conv_mode, USE_FRAME)
        caption_engine.load_files(os.path.join(args.ethucy_root, TRAJ_FILES[scene_idx]), os.path.join(image_dir, scene),
                                  HOMOGRAPHIES[scene_idx], os.path.join(map_dir, MAP_FILES[scene_idx]))

        save_data = {}
        start_frame_idx = list(caption_engine.traj_data.keys())[0]
        end_frame_idx = frame_num if args.max_frames is None else min(frame_num, start_frame_idx + args.max_frames)
        for frame_idx in tqdm(range(start_frame_idx, end_frame_idx)):
            caption_engine.preprocess_frame(frame_idx, scene_idx, USE_FRAME)
            if frame_idx < start_frame_idx + USE_FRAME - 1:
                continue
            caption_engine.caption_multiframe(frame_idx, scene_idx, USE_FRAME - 1)

            save_data[frame_idx] = {}
            for agent_id, agent in caption_engine.agents.items():
                if frame_idx != max(agent.local_pos.keys()):  # agent not observed in this frame
                    continue
                save_data[frame_idx][agent_id] = {
                    'description': agent.caption,
                    'local_position': agent.local_pos[frame_idx],
                    'pose': agent.pose,  # CLIP image feature
                }
            if len(save_data[frame_idx]) == 0:
                del save_data[frame_idx]
            elif args.visualize:
                frame_save_dir = os.path.join(args.output_dir, scene, 'frames')
                os.makedirs(frame_save_dir, exist_ok=True)
                plot_2d_human(frame_idx, save_data, caption_engine.img, save_dir=frame_save_dir)

        torch.save(save_data, os.path.join(args.output_dir, f'{scene}_agents_{start_frame_idx}_to_{frame_num}.pt'))


if __name__ == '__main__':
    main()
