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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules import jrdbDatamodule
from models._social_transmotion_v1_ethucy import socialTransmotion

from pytorch_lightning.loggers import TensorBoardLogger
from ray import train, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
    
if __name__ == '__main__':
    pl.seed_everything(8989)
    TuneReportCallback({"ade": "val_minADE", "fde": "val_minFDE"}, on="validation_end")
    
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='/mnt/jaewoo4tb/textraj/preprocessed_2nd/ethucy_v2_fps_2_5_frame_20_zara2')
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
    parser.add_argument('--log_dir', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/_logs_socialTransmotion/ethucy/zara2_rayTune', help='Directory where logs and checkpoints are saved')
    # parser.add_argument('--log_name', type=str, default='_InstCorrected_traj_pose_text', help='Directory where logs and checkpoints are saved')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=212)
    parser.add_argument('--token_num', type=int, default=25)
    parser.add_argument('--num_layers_local', type=int, default=6)
    parser.add_argument('--num_layers_global', type=int, default=3)
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--output_scale', type=int, default=1)
    parser.add_argument('--num_head_st', type=int, default=4)
    parser.add_argument('--modality', type=str, default='traj')     # traj, traj_pose
    parser.add_argument('--log_name', type=str, default='socialTransmotionV1_3e-5_FullLoss_traj', help='Directory where logs and checkpoints are saved')
    
    parser = socialTransmotion.add_model_specific_args(parser)
    args = parser.parse_args()

    # trainer = pl.Trainer(
    #     devices="auto",
    #     accelerator="auto",
    #     strategy=RayDDPStrategy(),
    #     callbacks=[RayTrainReportCallback()],
    #     plugins=[RayLightningEnvironment()],
    #     enable_progress_bar=False,
    # )


    search_space = {
        "lr": tune.loguniform(5e-5, 1e-3), 
        "historical_steps": 8,
        "future_steps": 12,
        "num_modes": 6,
        "rotate": False,
        "embed_dim": args.embed_dim,
        "seq_len": args.seq_len,
        "token_num": args.token_num,
        "num_layers_local": args.num_layers_local,
        "num_layers_global": args.num_layers_global,
        "dim_feedforward": args.dim_feedforward,
        "output_scale": args.output_scale,
        "num_head_st": args.num_head_st,
        "dropout": tune.choice([0.0, 0.2]),
        "parallel": False,
        "weight_decay": args.weight_decay,
        "T_max": args.T_max,
        "modality": args.modality,
        "train_batch_size": args.train_batch_size,
        "val_batch_size": args.val_batch_size,
        "shuffle": args.shuffle,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "max_epochs": args.max_epochs,
        "root": args.root
        }

    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={"CPU": 4, "GPU": 1}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_minADE",
            checkpoint_score_order="min",
        ),
    )
    
    def train_func(args_):
        trainer = pl.Trainer(
                            max_epochs=args_['max_epochs'],
                            devices="auto",
                            accelerator="auto",
                            strategy=RayDDPStrategy(find_unused_parameters=True),
                            callbacks=[RayTrainReportCallback()],
                            plugins=[RayLightningEnvironment()],
                            enable_progress_bar=False,
                            default_root_dir=os.path.join(args.log_dir, args.log_name),
                            # num_sanity_val_steps=0,
                            )

        model = socialTransmotion(**args_)
        datamodule = jrdbDatamodule(
            root=args_['root'],
            train_batch_size=args_['train_batch_size'],
            val_batch_size=args_['val_batch_size'],
            shuffle=args_['shuffle'],
            num_workers=args_['num_workers'],
            pin_memory=args_['pin_memory'],
            persistent_workers=args_['persistent_workers']
        )
        trainer = prepare_trainer(trainer)
        trainer.fit(model, datamodule)
        
    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    # The maximum training epochs
    NUM_EPOCHS = 40
    NUM_SAMPLES = 20
    def tune_mart_asha(num_samples=NUM_SAMPLES):
        scheduler = ASHAScheduler(max_t=NUM_EPOCHS, grace_period=4, reduction_factor=2)
        tuner = tune.Tuner(
            ray_trainer,
            param_space={"train_loop_config": search_space},
            tune_config=tune.TuneConfig(
                metric="val_minADE",
                mode="min",
                num_samples=num_samples,
                scheduler=scheduler,
            ),
        )
        return tuner.fit()

    results = tune_mart_asha()
    print(results.get_best_result(metric="val_minADE", mode="min"))
    sdfsdf = 1