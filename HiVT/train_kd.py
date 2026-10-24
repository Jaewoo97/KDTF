"""Stage 2 of KDTF: distill a frozen multi-modal teacher into a student that only sees trajectory (+pose).

The teacher checkpoint from train_teacher.py is loaded into the teacher branch of the KD model
(local_encoder / global_interactor / decoder); the student and dummy branches are initialized fresh.

Example:
    python train_kd.py --dataset jrdb --root /path/to/jrdb_v2_fps_2_5_frame_20_withRobot \
        --teacher_ckpt logs/teacher_jrdb/lightning_logs/version_0/checkpoints/epoch=4-step=1610.ckpt
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

# dataset -> (KD model module, default student encoder, default seed) used for the paper experiments
KD_CFG = {
    'jrdb':   ('HiVT.models.hivt_kd_onlyAgent_v4_dummy_teachV3', 'traj_pose', 5115),
    'sit':    ('HiVT.models.hivt_kd_onlyAgent_sit_v4_dummy',     'traj_pose', 9657),
    'ethucy': ('HiVT.models.hivt_kd_onlyAgent_ethucy_v4_nodummy', 'traj_v3',  4631),
}


def get_model_cls(dataset):
    module = __import__(KD_CFG[dataset][0], fromlist=['HiVT'])
    return module.HiVT


if __name__ == '__main__':
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument('--dataset', type=str, required=True, choices=list(KD_CFG.keys()))
    dataset = pre_parser.parse_known_args()[0].dataset
    HiVT = get_model_cls(dataset)
    _, default_student, default_seed = KD_CFG[dataset]

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=list(KD_CFG.keys()))
    parser.add_argument('--root', type=str, required=True, help='Preprocessed (2nd stage) dataset root containing train/ and val/')
    parser.add_argument('--teacher_ckpt', type=str, required=True, help='Teacher checkpoint produced by train_teacher.py')
    parser.add_argument('--encoder_type_student', type=str, default=default_student,
                        help='Student local encoder: traj_pose (trajectory+pose) or traj_v3 (trajectory only)')
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
    parser.add_argument('--log_name', type=str, default=f'kd_{dataset}', help='Sub-directory name under log_dir')
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    trainer = pl.Trainer.from_argparse_args(args,
                                            callbacks=[model_checkpoint],
                                            default_root_dir=os.path.join(args.log_dir, args.log_name))
    # Architecture hyper-parameters (encoder_type, num_modes, embed_dim, ...) come from the teacher checkpoint.
    model = HiVT.load_from_checkpoint(checkpoint_path=args.teacher_ckpt, parallel=False, strict=False,
                                      encoder_type_student=args.encoder_type_student)
    print(f'teacher encoder: {model.hparams.encoder_type}, student encoder: {args.encoder_type_student}, '
          f'num_modes: {model.num_modes}')
    # KD models override freeze(): only the teacher/dummy local encoders and global interactors are frozen.
    model.freeze()
    datamodule = jrdbDatamodule.from_argparse_args(args)
    trainer.fit(model, datamodule)
