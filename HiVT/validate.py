"""Evaluate a teacher (train_teacher.py) or KD student (train_kd.py) checkpoint on the val split.

Example:
    python validate.py --dataset jrdb --model kd --root /path/to/jrdb_v2_fps_2_5_frame_20_withRobot --ckpt path/to/kd.ckpt
"""
from argparse import ArgumentParser
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `HiVT.*` imports
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import jrdbDataset
import train_teacher
import train_kd

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['jrdb', 'sit', 'ethucy'])
    parser.add_argument('--model', type=str, required=True, choices=['teacher', 'kd'])
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    if args.model == 'teacher':
        HiVT = train_teacher.get_model_cls(args.dataset)
    else:
        HiVT = train_kd.get_model_cls(args.dataset)
    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt, parallel=False)
    val_dataset = jrdbDataset(root=args.root, split='val')
    dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, persistent_workers=args.num_workers > 0)
    trainer = pl.Trainer(gpus=args.gpus, logger=False)
    trainer.validate(model, dataloader)
