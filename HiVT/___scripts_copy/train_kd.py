from argparse import ArgumentParser
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules import jrdbDatamodule
# from HiVT.models.hivt_kd_onlyAgent import HiVT        # only full timestep x loss (only reg KD)
# from HiVT.models.hivt_kd_onlyAgent_v4_dummy_featKD import HiVT        # only feat KD with reg KD
# from HiVT.models.hivt_kd_onlyAgent_v4_dummy import HiVT       # with inst loss

from HiVT.models.hivt_kd_onlyAgent_v4_dummy_teachV3 import HiVT       # teacher v3
# from HiVT.models.hivt_kd_onlyAgent_v4_dummy_teachV3_noReg import HiVT       # teacher v3 with no Reg
# from HiVT.models.hivt_kd_onlyAgent_v4_dummy_teachV3_onlyFullTimestep import HiVT       # teacher v3 only Full timestep loss
# from HiVT.models.hivt_kd_onlyAgent_v4_dummy_teachV3_onlyFullTimestep_noReg import HiVT       # teacher v3 only Full timestep loss with no Reg

# FINAL
# from HiVT.models.hivt_kd_onlyAgent_v4_dummy_teachV3_FINAL import HiVT
# from HiVT.models.hivt_kd_onlyAgent_v4_dummy_teachV3_noReg import HiVT
# from HiVT.models.hivt_kd_onlyAgent_v4_dummy_teachV3_FINAL_onlyLocal import HiVT
# from HiVT.models.hivt_kd_onlyAgent_v4_dummy_teachV3_FINAL_onlyGlobal import HiVT

if __name__ == '__main__':
    pl.seed_everything(5115)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='/mnt/jaewoo4tb/textraj/preprocessed_2nd/jrdb_v2_fps_2_5_frame_20_withRobot')
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
    parser.add_argument('--encoder_type', type=str, default='traj_pose_text_intra', choices=['traj', 'traj_v2', 'traj_v3', 'traj_pose', 'traj_pose_8steps', 'traj_pose_text', 'traj_pose_text_embedMask', 'traj_pose_text_v1', 'traj_pose_text_v2', 'traj_pose_text_v3', 'traj_text', 'traj_pose_text_onlyEgo', 'traj_pose_text_egoOther', 'traj_pose_text_egoOtherInteraction'])
    
    # V3
    # parser.add_argument('--ckpt_path', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/logs_jrdb_curFixed/_V2InstCorrected_hivtV3_trajPoseTextIntra_onlyIntra_allTimesteps_1.5/lightning_logs/version_0/checkpoints/epoch=4-step=1610.ckpt')
    
    # V3 num mode 1
    parser.add_argument('--ckpt_path', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/logs_jrdb_curFixed/_V2InstCorrected_hivtV3_trajPoseTextIntra_allFullTimesteps_intra1.5_numMode1/lightning_logs/version_1/checkpoints/epoch=4-step=1610.ckpt')
    
    # V3 num mode 10
    # parser.add_argument('--ckpt_path', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/logs_jrdb_curFixed/_V2InstCorrected_hivtV3_trajPoseTextIntra_allFullTimesteps_intra1.5_numMode10/lightning_logs/version_0/checkpoints/epoch=4-step=1610.ckpt')
    
    # V3 num mode 20
    # parser.add_argument('--ckpt_path', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/logs_jrdb_curFixed/_V2InstCorrected_hivtV3_trajPoseTextIntra_allFullTimesteps_intra1.5_numMode20/lightning_logs/version_0/checkpoints/epoch=4-step=1610.ckpt')
    
    # V3 only Full
    # parser.add_argument('--ckpt_path', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/logs_jrdb_curFixed/_V2InstCorrected_hivtV3_trajPoseTextIntra_onlyFullTimesteps_intra1.5/lightning_logs/version_0/checkpoints/epoch=4-step=1610.ckpt')

    # V2
    # parser.add_argument('--ckpt_path', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/logs_jrdb_curFixed/_V2InstCorrected_hivtV2_trajPoseTextIntra_onlyIntra_allTimesteps_1.5/lightning_logs/version_0/checkpoints/epoch=4-step=1610.ckpt')
    
    # # # # # # Editables
    parser.add_argument('--log_dir', type=str, default='/mnt/jaewoo4tb/textraj/HiVT/logs_jrdb_curFixed/', help='Directory where logs and checkpoints are saved')
        
    ################# ALSO EDIT TRAINER #######################
    parser.add_argument('--encoder_type_student', type=str, default='traj_pose', choices=['traj', 'traj_v2', 'traj_v3', 'traj_pose', 'traj_pose_8steps', 'traj_pose_text', 'traj_pose_text_v2', 'traj_pose_text_v3', 'traj_text'])
    ################# ALSO EDIT TRAINER #######################
    
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--emb_numHead', type=int, default=4)
    parser.add_argument('--emb_dropout', type=int, default=0.5)
    parser.add_argument('--emb_numLayers', type=int, default=2)
    # parser.add_argument('--log_name', type=str, default='__KD_TrajPoseText2trajPose_onlyLocalKL_fixed', help='Directory where logs and checkpoints are saved')
    parser.add_argument('--log_name', type=str, default='__KD_FIXED_TrajPoseText2trajPose_teachIntraV3_1.5_teacherAllTimesteps_studentAllTimesteps_numMode1', help='Directory where logs and checkpoints are saved')
    # parser.add_argument('--log_name', type=str, default='__KD_FIXED_TrajPoseText2trajPose_FullKD_dummy_regPriority_freezed', help='Directory where logs and checkpoints are saved')

    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()
    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    trainer = pl.Trainer.from_argparse_args(args, 
                                            callbacks=[model_checkpoint],
                                            default_root_dir=os.path.join(args.log_dir, args.log_name),
                                            # num_sanity_val_steps=0,
                                            )
    model = HiVT(**vars(args))
    model = HiVT.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=False, strict=False)
    print(f'num_modes: {model.num_modes}')
    model.freeze()
    datamodule = jrdbDatamodule.from_argparse_args(args)
    trainer.fit(model, datamodule)
