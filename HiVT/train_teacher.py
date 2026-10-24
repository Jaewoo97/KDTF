"""Stage 1 of KDTF: train the multi-modal teacher (trajectory + pose + text).

Example:
    python train_teacher.py --dataset jrdb --root /path/to/preprocessed_2nd/jrdb_v2_fps_2_5_frame_20_withRobot
"""
from argparse import ArgumentParser
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `HiVT.*` imports
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules import jrdbDatamodule

# dataset -> (teacher model module, default teacher encoder, default seed) used for the paper experiments
TEACHER_CFG = {
    'jrdb':   ('models.hivt_v3',        'traj_pose_text_intra', 1110),
    'sit':    ('models.hivt_v3_sit',    'traj_pose_text',       1555),
    'ethucy': ('models.hivt_v2_ethucy', 'traj_pose_text',       9999),
}


def get_model_cls(dataset):
    module = __import__(TEACHER_CFG[dataset][0], fromlist=['HiVT'])
    return module.HiVT


if __name__ == '__main__':
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument('--dataset', type=str, required=True, choices=list(TEACHER_CFG.keys()))
    dataset = pre_parser.parse_known_args()[0].dataset
    HiVT = get_model_cls(dataset)
    _, default_encoder, default_seed = TEACHER_CFG[dataset]

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=list(TEACHER_CFG.keys()))
    parser.add_argument('--root', type=str, required=True, help='Preprocessed (2nd stage) dataset root containing train/ and val/')
    parser.add_argument('--seed', type=int, default=default_seed)
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
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory where logs and checkpoints are saved')
    parser.add_argument('--log_name', type=str, default=f'teacher_{dataset}', help='Sub-directory name under log_dir')
    parser.add_argument('--encoder_type', type=str, default=default_encoder,
                        help='Teacher local encoder. Paper setting: jrdb=traj_pose_text_intra, sit/ethucy=traj_pose_text')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--emb_numHead', type=int, default=4)
    parser.add_argument('--emb_dropout', type=float, default=0.5)
    parser.add_argument('--emb_numLayers', type=int, default=2)
    parser.add_argument('--onlyFullTimestep', type=bool, default=False)
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    trainer = pl.Trainer.from_argparse_args(args,
                                            callbacks=[model_checkpoint],
                                            default_root_dir=os.path.join(args.log_dir, args.log_name))
    model = HiVT(**vars(args))
    datamodule = jrdbDatamodule.from_argparse_args(args)
    trainer.fit(model, datamodule)
