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
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules import jrdbDatamodule
# from HiVT.models.hivt_kd_onlyAgent import HiVT        # only full timestep x loss
from HiVT.models.hivt_kd_onlyAgent_v4 import HiVT       # with inst loss

if __name__ == '__main__':
    pl.seed_everything(4545)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='/mnt/jaewoo4tb/textraj/preprocessed_2nd/jrdb_v2_fps_2_5_frame_20')
    # parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minADE', choices=['val_minADE', 'val_minFDE'])
    parser.add_argument('--save_top_k', type=int, default=5)
    parser.add_argument('--encoder_type', type=str, default='traj_pose_text', choices=['traj', 'traj_v2', 'traj_v3', 'traj_pose', 'traj_pose_text_embedMask', 'traj_pose_text_embedMask_teacher', 'traj_pose_8steps', 'traj_pose_text', 'traj_pose_text_v2', 'traj_pose_text_v3', 'traj_text'])
    parser.add_argument('--ckpt_path', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/logs_jrdb/__KD_TrajPoseText2trajPose_fullKD_modalMask_const0.25FIXED/lightning_logs/version_0/checkpoints/epoch=29-step=9660.ckpt')
    
    # # # # # # Editables
    parser.add_argument('--log_dir', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/logs_jrdb/', help='Directory where logs and checkpoints are saved')
        
    ################# ALSO EDIT TRAINER #######################
    parser.add_argument('--encoder_type_student', type=str, default='traj_pose', choices=['traj', 'traj_v2', 'traj_v3', 'traj_pose', 'traj_pose_8steps', 'traj_pose_text', 'traj_pose_text_v2', 'traj_pose_text_v3', 'traj_text'])
    ################# ALSO EDIT TRAINER #######################
    
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--emb_numHead', type=int, default=4)
    parser.add_argument('--emb_dropout', type=int, default=0.5)
    parser.add_argument('--emb_numLayers', type=int, default=2)
    # parser.add_argument('--log_name', type=str, default='__KD_TrajPoseText2trajPose_onlyLocalKL_fixed', help='Directory where logs and checkpoints are saved')
    parser.add_argument('--log_name', type=str, default='__KD_TrajPoseText2trajPose_fullKD_modalMask_finetune5e-4', help='Directory where logs and checkpoints are saved')
    
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()
    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=False, strict=False)
    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    trainer = pl.Trainer.from_argparse_args(args, 
                                            callbacks=[model_checkpoint],
                                            default_root_dir=os.path.join(args.log_dir, args.log_name),
                                            # num_sanity_val_steps=0,
                                            )
    model = HiVT(**vars(args))
    datamodule = jrdbDatamodule.from_argparse_args(args)
    trainer.fit(model, datamodule)
