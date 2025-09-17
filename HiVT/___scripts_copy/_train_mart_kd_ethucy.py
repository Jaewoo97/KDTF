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
from models._mart_v3_kd_noReg_ethucy import mart
# from models._mart_v3_kd_noReg import mart

if __name__ == '__main__':
    pl.seed_everything(9581)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='/mnt/jaewoo4tb/textraj/preprocessed_2nd/ethucy_v2_fps_2_5_frame_20_eth')
    # parser.add_argument('--root', type=str, default='/mnt/jaewoo4tb/textraj/preprocessed_2nd/jrdb_v2_fps_2_5_frame_20_withRobot')
    # parser.add_argument('--root', type=str, default='/mnt/jaewoo4tb/textraj/preprocessed_2nd/sit_v2_fps_2_5_frame_20_withRobot_withPoseInfo')
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
    
    # # # # # # Editables
    # Teacher
    parser.add_argument('--modality', type=str, default='traj_pose_text')     # traj, traj_pose, traj_text, traj_pose_text
    parser.add_argument('--ckpt_path', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/_logs_mart/ethucy/eth/martV3_allTimesteps_trajPoseText_onlyDiff/lightning_logs/version_0/checkpoints/epoch=41-step=2226.ckpt')
    # parser.add_argument('--ckpt_path', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/_logs_mart/ethucy/zara2/martV3_allTimesteps_trajPoseText_onlyDiff/lightning_logs/version_0/checkpoints/epoch=36-step=1924.ckpt')
    
    ################# ALSO EDIT TRAINER #######################
    parser.add_argument('--modality_student', type=str, default='traj')     # traj, traj_pose, traj_text, traj_pose_text
    ################# ALSO EDIT TRAINER #######################
    
    parser.add_argument('--log_dir', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/_logs_mart/ethucy/eth', help='Directory where logs and checkpoints are saved')
    # parser.add_argument('--log_name', type=str, default='_InstCorrected_traj_pose_text', help='Directory where logs and checkpoints are saved')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--decoder_hidden_dim', type=int, default=128)
    parser.add_argument('--model_dim', type=int, default=128)
    parser.add_argument('--dropout_mart', type=float, default=0.5)
    parser.add_argument('--num_head_mart', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--aggregation', type=str, default='avg')
    parser.add_argument('--function_type', type=int, default=2)
    parser.add_argument('--hyper_scales', type=list, default=[1, 'adaptive'])
    parser.add_argument('--onlyFullTimestep', type=bool, default=True)
    parser.add_argument('--log_name', type=str, default='_KD_FIXED_martV3_onlyFullTimesteps_trajPoseText2traj_withBatchAve_withReg_freezed', help='Directory where logs and checkpoints are saved')
    
    parser = mart.add_model_specific_args(parser)
    args = parser.parse_args()

    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    trainer = pl.Trainer.from_argparse_args(args, 
                                            callbacks=[model_checkpoint],
                                            default_root_dir=os.path.join(args.log_dir, args.log_name),
                                            # num_sanity_val_steps=0,
                                            )
    model = mart(**vars(args))
    model = mart.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=False, strict=False)
    datamodule = jrdbDatamodule.from_argparse_args(args)
    if model.onlyFullTimestep != args.onlyFullTimestep:
        model.onlyFullTimestep = args.onlyFullTimestep
        print(f'Fixed to {model.onlyFullTimestep}')
    model.freeze()
    trainer.fit(model, datamodule)
