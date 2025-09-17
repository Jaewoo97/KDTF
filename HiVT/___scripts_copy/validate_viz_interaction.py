# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.data import DataLoader

from datasets import jrdbDataset

from datamodules import jrdbDatamodule
from models.hivt_v2_withViz_interaction import HiVT

if __name__ == '__main__':
    pl.seed_everything(2024)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='/mnt/jaewoo4tb/textraj/preprocessed_2nd/jrdb_v2_fps_2_5_frame_20_withRobot_withPoseInfo')
    # parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minADE', choices=['val_minADE', 'val_minFDE'])
    parser.add_argument('--save_top_k', type=int, default=5)
    
    # # # # # # Editables
    parser.add_argument('--log_dir', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/logs_withViz/', help='Directory where logs and checkpoints are saved')
    # parser.add_argument('--encoder_type', type=str, default='traj_pose', choices=['traj', 'traj_pose', 'traj_pose_8steps', 'traj_pose_text'])
    parser.add_argument('--encoder_type', type=str, default='traj_v3', choices=['traj', 'traj_pose', 'traj_pose_8steps', 'traj_pose_text'])
    # parser.add_argument('--encoder_type', type=str, default='traj_pose_text_intra', choices=['traj', 'traj_pose', 'traj_pose_8steps', 'traj_pose_text'])
    parser.add_argument('--log_name', type=str, default='_InstCorrected_val', help='Directory where logs and checkpoints are saved')
    # parser.add_argument('--log_name', type=str, default='_InstCorrected_traj_pose_text', help='Directory where logs and checkpoints are saved')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--emb_numHead', type=int, default=8)
    parser.add_argument('--emb_dropout', type=int, default=0.5)
    parser.add_argument('--emb_numLayers', type=int, default=1)
    # Traj Pose Text intra
    # parser.add_argument('--ckpt_path', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/logs_jrdb_curFixed/_V2InstCorrected_hivtV3_trajPoseTextIntra_onlyFullTimesteps_intra1.5/lightning_logs/version_0/checkpoints/epoch=4-step=1610.ckpt')
    
    # Traj + interaction text
    parser.add_argument('--ckpt_path', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/logs_jrdb_curFixed/_V2InstCorrected_hivtV3_traj_allTimesteps_intra1.5/lightning_logs/version_1/checkpoints/epoch=3-step=1288.ckpt')
    
    # Traj Pose
    # parser.add_argument('--ckpt_path', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/logs_jrdb_curFixed/_V2InstCorrected_hivtV2_trajPose_onlyIntra_1.5m/lightning_logs/version_0/checkpoints/epoch=4-step=1610.ckpt')
    
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    trainer = pl.Trainer.from_argparse_args(args, 
                                            default_root_dir=os.path.join(args.log_dir, args.log_name),
                                            # num_sanity_val_steps=0,
                                            )
    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=False)
    val_dataset = jrdbDataset(root=args.root, split='val')
    dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer.validate(model, dataloader)
