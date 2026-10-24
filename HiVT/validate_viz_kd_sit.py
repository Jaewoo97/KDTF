from argparse import ArgumentParser
import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.data import DataLoader

from datasets import jrdbDataset

from datamodules import jrdbDatamodule
# from HiVT.models.hivt_kd_onlyAgent import HiVT        # only full timestep x loss (only reg KD)
# from HiVT.models.hivt_kd_onlyAgent_v4_dummy_featKD import HiVT        # only feat KD with reg KD
# from HiVT.models.hivt_kd_onlyAgent_v4_dummy import HiVT       # with inst loss

from HiVT.models.hivt_kd_onlyAgent_v4_dummy_teachV3_VIZ_sit import HiVT       # teacher v3
# from HiVT.models.hivt_kd_onlyAgent_v4_dummy_teachV3_noReg import HiVT       # teacher v3 with no Reg
# from HiVT.models.hivt_kd_onlyAgent_v4_dummy_teachV3_onlyFullTimestep import HiVT       # teacher v3 only Full timestep loss
# from HiVT.models.hivt_kd_onlyAgent_v4_dummy_teachV3_onlyFullTimestep_noReg import HiVT       # teacher v3 only Full timestep loss with no Reg

if __name__ == '__main__':
    pl.seed_everything(8556)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='/mnt/jaewoo4tb/textraj/preprocessed_2nd/sit_v2_fps_2_5_frame_20_withRobot_withPoseInfo_v2')
    # parser.add_argument('--root', type=str, default='/mnt/jaewoo4tb/textraj/preprocessed_2nd/jrdb_v2_fps_2_5_frame_20_withRobot_withPoseInfo')
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
    parser.add_argument('--encoder_type', type=str, default='traj_pose_text', choices=['traj', 'traj_v2', 'traj_v3', 'traj_pose', 'traj_pose_8steps', 'traj_pose_text', 'traj_pose_text_embedMask', 'traj_pose_text_v1', 'traj_pose_text_v2', 'traj_pose_text_v3', 'traj_text', 'traj_pose_text_onlyEgo', 'traj_pose_text_egoOther', 'traj_pose_text_egoOtherInteraction'])
    
    # Traj pose KD
    parser.add_argument('--ckpt_path', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/logs_sit_curFixed/__KD_FIXED_TrajPoseText2trajPose_dummy_regScaled_fixed_freezed_intra/lightning_logs/version_0/checkpoints/epoch=14-step=1005.ckpt')
    
    # # # # # # Editables
    parser.add_argument('--log_dir', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/logs_sit_curFixed/', help='Directory where logs and checkpoints are saved')
        
    ################# ALSO EDIT TRAINER #######################
    parser.add_argument('--encoder_type_student', type=str, default='traj_pose', choices=['traj', 'traj_v2', 'traj_v3', 'traj_pose', 'traj_pose_8steps', 'traj_pose_text', 'traj_pose_text_v2', 'traj_pose_text_v3', 'traj_text'])
    ################# ALSO EDIT TRAINER #######################
    
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--emb_numHead', type=int, default=4)
    parser.add_argument('--emb_dropout', type=int, default=0.5)
    parser.add_argument('--emb_numLayers', type=int, default=2)
    # parser.add_argument('--log_name', type=str, default='__KD_TrajPoseText2trajPose_onlyLocalKL_fixed', help='Directory where logs and checkpoints are saved')
    parser.add_argument('--log_name', type=str, default='__VIZ', help='Directory where logs and checkpoints are saved')
    # parser.add_argument('--log_name', type=str, default='__KD_FIXED_TrajPoseText2trajPose_FullKD_dummy_regPriority_freezed', help='Directory where logs and checkpoints are saved')

    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()
    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    trainer = pl.Trainer.from_argparse_args(args, 
                                            callbacks=[model_checkpoint],
                                            default_root_dir=os.path.join(args.log_dir, args.log_name),
                                            # num_sanity_val_steps=0,
                                            )
    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=False)
    val_dataset = jrdbDataset(root=args.root, split='val')
    dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer.validate(model, dataloader)