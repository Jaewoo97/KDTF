from argparse import ArgumentParser
import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules import jrdbDatamodule
from models.hivt_v2 import HiVT

if __name__ == '__main__':
    pl.seed_everything(1110)

    parser = ArgumentParser()
    # parser.add_argument('--root', type=str, default='/mnt/jaewoo4tb/textraj/preprocessed_2nd/sit_v2_fps_2_5_frame_20_withRobot_withPoseInfo')
    parser.add_argument('--root', type=str, default='/mnt/jaewoo4tb/textraj/preprocessed_2nd/jrdb_v2_fps_2_5_frame_20_withRobot')
    # parser.add_argument('--root', type=str, default='/mnt/jaewoo4tb/textraj/preprocessed_2nd/jrdb_v2_fps_2_5_frame_20_withRobot')
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
    parser.add_argument('--log_dir', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/logs_jrdb_curFixed/', help='Directory where logs and checkpoints are saved')
    # parser.add_argument('--encoder_type', type=str, default='traj_pose', choices=['traj', 'traj_pose', 'traj_pose_8steps', 'traj_pose_text'])
    parser.add_argument('--encoder_type', type=str, default='traj_pose', choices=['traj_v3', 'traj_pose', 'traj_pose_text_v3', 'traj_text', ])
    # parser.add_argument('--log_name', type=str, default='_InstCorrected_traj_pose_text', help='Directory where logs and checkpoints are saved')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--emb_numHead', type=int, default=4)
    parser.add_argument('--emb_dropout', type=int, default=0.5)
    parser.add_argument('--emb_numLayers', type=int, default=2)
    parser.add_argument('--onlyFullTimestep', type=bool, default=False)
    parser.add_argument('--log_name', type=str, default='debug', help='Directory where logs and checkpoints are saved')
    
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    trainer = pl.Trainer.from_argparse_args(args, 
                                            callbacks=[model_checkpoint],
                                            default_root_dir=os.path.join(args.log_dir, args.log_name),
                                            # num_sanity_val_steps=0,
                                            )
    model = HiVT(**vars(args))
    datamodule = jrdbDatamodule.from_argparse_args(args)
    trainer.fit(model, datamodule)
