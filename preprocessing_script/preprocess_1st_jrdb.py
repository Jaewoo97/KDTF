# ### JRDB parser with image captioning via llava-next ###
from PIL import Image
# import cv2
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
from parser_model.captions import Caption3DPoseEngine
sys.path.append('/home/user/anaconda3/envs/textraj/lib/python3.8/site-packages/bev/')

def main_parse():
    print("Start parsing.")
    '''Parameters'''
    VERSION = 'egovlm_debug'
    VLM_TYPE = 'vllm' # ['local', 'vllm']   for local, use bev_jw. for vllm, use textraj (conda env)
    VISUALIZE = True
    
    default_save_dir = '/mnt/jaewoo4tb/textraj/preprocessed_1st/' + VERSION
    base_jrdb = '/ssd4tb/t2p/jrdb/train_dataset/'
    
    label_2d_base = f"{base_jrdb}labels/labels_2d_stitched/"
    label_3d_base = f"{base_jrdb}labels/labels_3d/"
    label_social_base = f"{base_jrdb}labels/labels_2d_activity_social_stitched/"
    img_base = f"{base_jrdb}images/image_stitched/"
    odometry_base = f'{base_jrdb}odometry_processed'
    
    scenes = sorted(glob.glob(base_jrdb+ 'images/image_0/*'))


    for sceneIdx, scene in enumerate(scenes):
        scene_name = os.path.basename(os.path.normpath(scene))
        print(f'Processing scene: {scene_name}, {sceneIdx} out of {len(scenes)} scenes.')

        os.makedirs(default_save_dir, exist_ok=True)
        output_savedir = os.path.join(default_save_dir, scene_name)
        annot_odometry_pos = np.load(f'{odometry_base}/{scene_name}_pos.npy')
        annot_odometry_pos -= annot_odometry_pos[0]                 # Normalize by reference to the first frame
        annot_odometry_ori = np.load(f'{odometry_base}/{scene_name}_orientation.npy')
        annot_odometry_ori -= annot_odometry_ori[0]                 # Normalize by reference to the first frame
        
        frame_save_dir, gif_save_dir, data_save_dir, pose2d_save_dir = os.path.join(output_savedir, 'frames'), os.path.join(output_savedir, 'gif'), os.path.join(output_savedir, 'data'), os.path.join(output_savedir, 'pose2d')        
        save_data, save_interaction = {}, {}

        frame_num = len(glob.glob(f'{base_jrdb}images/image_stitched/{scene_name}/*.jpg'))
        caption_engine = Caption3DPoseEngine(VLM_TYPE, VISUALIZE)
        caption_engine.load_files(label_2d_base + scene_name + ".json", label_3d_base + scene_name + ".json", img_base + scene_name + "/", label_social_base + scene_name + ".json")
        interaction = {}

        start_frame_idx = 0
        last_frame_idx = frame_num
        for frame_idx in tqdm(range(start_frame_idx, frame_num)):
            caption_engine.preprocess_frame(frame_idx, annot_odometry_pos[frame_idx], annot_odometry_ori[frame_idx])
            caption_engine.regress_3dpose()
            
            save_data[frame_idx] = {}
            for agent_id in caption_engine.agents.keys():
                if agent_id not in save_data[frame_idx].keys():
                    save_data[frame_idx][agent_id] = {}
                        
                save_data[frame_idx][agent_id]["description"] = caption_engine.agents[agent_id].caption
                save_data[frame_idx][agent_id]["global_position"] = caption_engine.agents[agent_id].global_pos
                save_data[frame_idx][agent_id]["local_position"] = caption_engine.agents[agent_id].local_pos
                save_data[frame_idx][agent_id]["pose"] = caption_engine.agents[agent_id].pose
                save_data[frame_idx][agent_id]["robot_pos"] = caption_engine.agents[agent_id].robot_pos
                save_data[frame_idx][agent_id]["robot_ori"] = caption_engine.agents[agent_id].robot_ori
                save_data[frame_idx][agent_id]["rot_z"] = caption_engine.agents[agent_id].rot_z
                save_data[frame_idx][agent_id]["cluster_id"] = -1   # default for agents with no cluster
                save_data[frame_idx][agent_id]["bbox"] = caption_engine.agents[agent_id].bbox
                        
            # save socials
            save_interaction[frame_idx] = {}
            save_interaction[frame_idx]['cluster'] = {}
            if len(caption_engine.socials.interactions.keys()) > 0:
                for int_key, int_value in caption_engine.socials.interactions.items():
                    if 'dual_interaction' not in save_interaction[frame_idx].keys():
                        save_interaction[frame_idx]['dual_interaction'] = [[int_key, int_value]]
                    else: save_interaction[frame_idx]['dual_interaction'].append([int_key, int_value])
                for cluster_id in caption_engine.socials.clusters.keys():
                    for agent_in_cluster in caption_engine.socials.clusters[cluster_id][1]:
                        save_data[frame_idx][int(agent_in_cluster)]['cluster_id'] = cluster_id
                    save_interaction[frame_idx]['cluster'][cluster_id] = caption_engine.socials.clusters[cluster_id][0]
            
            # visualize
            if VISUALIZE:
                os.makedirs(frame_save_dir, exist_ok=True)
                plot_3d_human(frame_idx, save_data, None, save_dir=frame_save_dir)
                # visualize_scene(frame_idx, caption_engine.img, caption_engine.label_2d, caption_engine.label_3d, person, interaction, os.path.join(frame_save_dir, f'{str(frame_idx).zfill(4)}.png'))
        
        torch.save(save_data, os.path.join(default_save_dir, scene_name+f'_agents_{start_frame_idx}_to_{last_frame_idx}.pt'))
        torch.save(save_interaction, os.path.join(default_save_dir, scene_name+f'_interactions_{start_frame_idx}_to_{last_frame_idx}.pt'))

if __name__ == "__main__":
    main_parse()