# ### JRDB parser with image captioning via llava-next ###
# v2: use video captioning instead of image captioning
# v3: map-based captioning
# v3_withAgent: map-based captioning, also consider agent as obstacle
from PIL import Image
import cv2
import numpy as np
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.append('/mnt/jaewoo4tb/textraj')
import os.path as osp
import torch
from torch import nn
import glob
from tqdm import tqdm
import argparse
import copy
import json
from utils.utils import *
from utils.plot_2d_3d import *
from parser_model.captions_ethucy_v3 import Caption3DPoseEngine
sys.path.append('/home/user/anaconda3/envs/textraj/lib/python3.8/site-packages/bev/')
# TRAJ_FILES = ['ynet/eth_train.pkl', 'ynet/eth_test.pkl', 'ynet/hotel_train.pkl', 'ynet/hotel_test.pkl', 'ynet/zara1_train.pkl', \
#     'ynet/zara1_test.pkl', 'ynet/zara2_train.pkl', 'ynet/zara2_test.pkl', 'ynet/univ_train.pkl', 'ynet/univ_test.pkl']
# IMAGE_DIRS = ['eth', 'eth', 'hotel', 'hotel', 'zara1', 'zara1', 'zara2', 'zara2', 'students', 'students']
TRAJ_FILES = ['biwi_eth.txt', 'biwi_hotel.txt', 'crowds_zara01.txt', 'crowds_zara02.txt', 'students003.txt']       # https://github.com/MaKaRuiNah/SHENet/tree/main?tab=readme-ov-file
IMAGE_DIRS = ['eth', 'hotel', 'zara1', 'zara2', 'students']
SCENE_NAMES = ['']
# HOMOGRAPHIES = ['ynet/eth_H.txt', 'ynet/hotel_H.txt', 'ynet/zara1_H.txt', 'ynet/zara2_H.txt', 'ynet/uni_examples_H.txt']
HOMOGRAPHIES = [np.array([[2.8128700e-02, 2.0091900e-03, -4.6693600e+00],[8.0625700e-04, 2.5195500e-02, -5.0608800e+00],[3.4555400e-04, 9.2512200e-05, 4.6255300e-01]]), 
                np.array([[1.1048200e-02, 6.6958900e-04, -3.3295300e+00],[-1.5966000e-03, 1.1632400e-02, -5.3951400e+00],[1.1190700e-04, 1.3617400e-05, 5.4276600e-01]]),
                np.array([[0.02174104, 0, -0.15],[0, -0.02461883, 13.77429807],[0, 0, 1]]),
                np.array([[0.02174104, 0, -0.4],[0, -0.02386598, 14.98401686],[0, 0, 1]]),
                np.array([[0.02220407, 0, -0.48],[0, -0.02477289, 13.92551292],[0, 0, 1]]),]
MAPS = ['/ssd4tb/ethucy/ynet/eth_oracle.png', '/ssd4tb/ethucy/ynet/hotel_oracle.png', '/ssd4tb/ethucy/ynet/zara1_oracle.png', '/ssd4tb/ethucy/ynet/zara2_oracle.png', '/ssd4tb/ethucy/ynet/students_oracle.png']
USE_FRAME = 2

def main_parse():
    print("Start parsing.")
    '''Parameters'''
    VERSION = 'ethucy_v1'
    VLM_TYPE = 'vllm' # ['local', 'vllm']   for local, use bev_jw. for vllm, use textraj (conda env)
    VISUALIZE = False
    
    default_save_dir = '/mnt/jaewoo4tb/textraj/preprocessed_1st/' + VERSION
    
    img_base = '/ssd4tb/ethucy/images'
    traj_base = '/ssd4tb/ethucy'
    homo_base = '/ssd4tb/ethucy'
    # scenes = sorted(glob.glob('images/image_0/*'))


    for sceneIdx, scene in enumerate(IMAGE_DIRS):
        # if sceneIdx <4: continue
        print(f'Processing scene: {scene}, {sceneIdx+1} out of {len(IMAGE_DIRS)} scenes.')

        os.makedirs(default_save_dir, exist_ok=True)
        output_savedir = os.path.join(default_save_dir, scene)
        
        frame_save_dir, gif_save_dir, data_save_dir = os.path.join(output_savedir, 'frames'), os.path.join(output_savedir, 'gif'), os.path.join(output_savedir, 'data')
        save_data, save_interaction = {}, {}

        frame_num = len(glob.glob(f'{img_base}/{scene}/*.jpg'))
        if sceneIdx == 0: conv_mode = 'eval_ethucy0'
        else: conv_mode = 'eval_ethucy'
        caption_engine = Caption3DPoseEngine(conv_mode, USE_FRAME)
        caption_engine.load_files(os.path.join(traj_base, TRAJ_FILES[sceneIdx]), os.path.join(img_base, IMAGE_DIRS[sceneIdx]), HOMOGRAPHIES[sceneIdx], MAPS[sceneIdx])
        interaction = {}

        # range(len(imgs_stitch))
        start_frame_idx = list(caption_engine.traj_data.keys())[0]
        last_frame_idx = frame_num
        for frame_idx in tqdm(range(start_frame_idx, frame_num)):
            # if frame_idx < 40: continue
            caption_engine.preprocess_frame(frame_idx, sceneIdx, USE_FRAME)
            if frame_idx < start_frame_idx+USE_FRAME-1:   # first five frames
                continue
            else: 
                caption_engine.caption_multiframe(frame_idx, sceneIdx, USE_FRAME-1)
            # caption_engine.caption_single(frame_idx)
            # caption_engine.caption_dual()
            # caption_engine.regress_3dpose()
            
            # stopPrint(caption_engine.caption_single)
            # stopPrint(caption_engine.caption_dual)
            
            save_data[frame_idx] = {}
            for agent_id in caption_engine.agents.keys():
                if frame_idx != max(caption_engine.agents[agent_id].local_pos.keys()):
                    continue
                if agent_id not in save_data[frame_idx].keys():
                    save_data[frame_idx][agent_id] = {}
                        
                save_data[frame_idx][agent_id]["description"] = caption_engine.agents[agent_id].caption
                # save_data[frame_idx][agent_id]["global_position"] = caption_engine.agents[agent_id].global_pos
                save_data[frame_idx][agent_id]["local_position"] = caption_engine.agents[agent_id].local_pos[frame_idx]
                save_data[frame_idx][agent_id]["pose"] = caption_engine.agents[agent_id].pose
                # save_data[frame_idx][agent_id]["robot_pos"] = caption_engine.agents[agent_id].robot_pos
                # save_data[frame_idx][agent_id]["robot_ori"] = caption_engine.agents[agent_id].robot_ori
                # save_data[frame_idx][agent_id]["rot_z"] = caption_engine.agents[agent_id].rot_z
                # save_data[frame_idx][agent_id]["cluster_id"] = -1   # default for agents with no cluster
            if len(save_data[frame_idx].keys()) == 0:
                del save_data[frame_idx]    # If no agent in frame, delete frame
            
            # visualize
            if VISUALIZE:
                os.makedirs(frame_save_dir, exist_ok=True)
                plot_2d_human(frame_idx, save_data, caption_engine.img, save_dir=frame_save_dir)
                # visualize_scene(frame_idx, caption_engine.img, caption_engine.label_2d, caption_engine.label_3d, person, interaction, os.path.join(frame_save_dir, f'{str(frame_idx).zfill(4)}.png'))
            # if frame_idx == 100: break
        # break
        
        torch.save(save_data, os.path.join(default_save_dir, scene+f'_agents_{start_frame_idx}_to_{last_frame_idx}.pt'))
        # torch.save(save_interaction, os.path.join(default_save_dir, scene+f'_interactions_{start_frame_idx}_to_{last_frame_idx}.pt'))
        # break

if __name__ == "__main__":
    main_parse()