# version v1_1: consider global coords as well
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from losses import LaplaceNLLLoss
from losses import SoftTargetCrossEntropyLoss
from metrics import ADE
from metrics import FDE
from metrics import MR
from models import *
from utils_hivt import TemporalData
from visualization import viz_trajectory
import copy
from utils_hivt_socialTransmotion import *


class socialTransmotion(pl.LightningModule):

    def __init__(self,
                 historical_steps: int,
                 future_steps: int,
                 num_modes: int,
                 rotate: bool,
                 embed_dim: int,
                 seq_len: int,
                 token_num: int,
                 num_layers_local: int,
                 num_layers_global: int,
                 dim_feedforward: int,
                 output_scale: int,
                 num_head_st: int,
                 dropout: float,
                 parallel: bool,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 onlyFullTimestep = 'False',
                 activation = 'relu',
                 modality = 'traj', # 'traj', 'traj_pose'
                 **kwargs) -> None:
        super(socialTransmotion, self).__init__()
        self.save_hyperparameters()
        if 'jrdb' in kwargs['root']: self.datasetType = 'jrdb'
        elif 'sit' in kwargs['root']: self.datasetType = 'sit'
        elif 'ethucy' in kwargs['root']: self.datasetType = 'ethucy'
        else: raise Exception('Wrong dataset?')
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.max_epochs = kwargs['max_epochs']
        self.obs_and_pred = historical_steps + future_steps
        self.output_scale = output_scale
        self.token_num = token_num
        self.joints_pose = 24
        self.seq_len = seq_len
        self.nhid = embed_dim
        self.modality = modality
        self.onlyFullTimestep = onlyFullTimestep
        # self.device = 'cuda'
        
        # self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.reg_loss = torch.nn.MSELoss()
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')
        self.ave_time = 0
        self.inf_count = 0
        self.minADE = ADE()
        self.minFDE = FDE()
        self.minADE_pose = ADE()
        self.minFDE_pose = FDE()
        self.minADE_pose_inst = ADE()
        self.minFDE_pose_inst = FDE()
        self.minADE_inst_all = ADE()
        self.minFDE_inst_all = FDE()
        self.minADE_1frame_all = ADE()
        self.minFDE_1frame_all = FDE()
        
        # Social transmotion part
        
        self.fc_in_traj = nn.Linear(4,embed_dim)
        self.fc_out_traj = nn.Linear(embed_dim, 2)
        self.pi = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1))
        self.angle_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout))
        self.pi_time = nn.Linear(future_steps*embed_dim, embed_dim)
        self.double_id_encoder = LearnedTrajandIDEncoding(embed_dim, dropout, seq_len=self.obs_and_pred) 
        self.id_encoder = LearnedIDEncoding(embed_dim, dropout, seq_len=self.obs_and_pred)

        self.scale = torch.sqrt(torch.FloatTensor([embed_dim]))

        self.fc_in_3dpose = nn.Linear(512, embed_dim)
        self.pose3d_encoder = Learnedpose3dEncoding(embed_dim, dropout)

        encoder_layer_local = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_head_st,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation=activation)
        self.local_former = AuxilliaryEncoderCMT(encoder_layer_local, num_layers=num_layers_local, norm=nn.LayerNorm(embed_dim))

        encoder_layer_global = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_head_st,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation=activation)
        self.global_former = AuxilliaryEncoderST(encoder_layer_global, num_layers=num_layers_global, norm=nn.LayerNorm(embed_dim))
        self.num_mode_mlp = nn.Linear(embed_dim, embed_dim * num_modes)
        self.pose_token = nn.Parameter(torch.Tensor(historical_steps, 72))
        self.traj_token = nn.Parameter(torch.Tensor(historical_steps, 2))
        nn.init.normal_(self.pose_token, mean=0., std=.02)
        nn.init.normal_(self.traj_token, mean=0., std=.02)

    def forward(self, data: TemporalData, given_frames, metamask=None):
        if self.modality == 'traj_pose':
            b_x, b_y, padding_mask, b_x_p, b_y_p, pad_mask_p, batch_mask, b_angle, robot_pos = agent2batch_ethucy(data)
            B_, N_, F_, _ = b_x.shape
            b_x, b_y = torch.cat((b_x, torch.zeros(b_x.shape[:-1]).unsqueeze(-1).cuda()), dim=-1), torch.cat((b_y, torch.zeros(b_y.shape[:-1]).unsqueeze(-1).cuda()), dim=-1)
            b_x_p = b_x_p.reshape(B_, N_, F_, -1)
            # tgt = torch.cat((b_x.unsqueeze(-2), b_x_p), dim=-2)
            tgt = b_x.unsqueeze(-2)
            tgt = tgt[:,:,:8]
            B, N, in_F, J, K = tgt.shape

            F = 20
            out_F = F - in_F
            assert out_F==12
            ## keep padding
            pad_idx = np.repeat([in_F - 1], out_F)
            i_idx = np.append(np.arange(0, in_F), pad_idx)  
            tgt = tgt[:, :, i_idx]
            tgt = tgt.permute(0, 2, 1, 3, 4)
            padding_mask_indexed = padding_mask[:, :, i_idx]
            padding_mask_indexed_pose = pad_mask_p[:, :, i_idx]
            
            ## add mask
            if metamask: # training
                mask_ratio_traj = 0.1 
                mask_ratio = 0.1 
                mask_ratio_modality = 0.3 
            else:
                mask_ratio_traj = 0.0 
                mask_ratio = 0.0 
                mask_ratio_modality = 0.0 

            tgt_traj = tgt[:,:,:,0,:2]
            traj_mask = torch.rand((B,F,N)).float().to(self.device) > mask_ratio_traj
            traj_mask = traj_mask.unsqueeze(3).repeat_interleave(2,dim=-1)
            tgt_traj = tgt_traj*traj_mask
            
            ## mask for specific modality for whole observation horizon
            modality_selection_3dpose = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).repeat(1,in_F,1,512) 
            modality_selection = modality_selection_3dpose

            tgt_3dpose = b_x_p.permute(0,2,1,3) * modality_selection
    
            #### joints-mask
            joints_3d_mask = torch.rand((B,in_F,N,512)).float().to(self.device) > mask_ratio
            tgt_3dpose = tgt_3dpose * joints_3d_mask
            
            ############
            # Transformer
            ###########
            tgt_traj = self.fc_in_traj(tgt_traj) 
            tgt_traj = self.double_id_encoder(tgt_traj, num_people=N) 

            tgt_3dpose = self.fc_in_3dpose(tgt_3dpose) 

            padding_mask_trj = padding_mask_indexed.reshape(-1, F)
            padding_mask_pose = padding_mask_indexed_pose.reshape(-1, F)[:, :in_F]
            tgt_padding_mask_global = padding_mask_indexed.reshape(B, -1)           
            
            tgt_traj = torch.transpose(tgt_traj,0,1).reshape(F,-1,self.nhid)  # F, BXN, D
            tgt_3dpose = torch.transpose(tgt_3dpose, 0,1).reshape(in_F, -1, self.nhid) # in_FXJ, BXN, D
            
            tgt_angle = self.angle_mlp(b_angle.unsqueeze(-1)).unsqueeze(0).reshape(1,-1,self.nhid)
            angle_mask = torch.zeros((tgt_angle.shape[1], 1), dtype=torch.bool).cuda()
            tgt_padding_mask_local = torch.cat((padding_mask_trj, padding_mask_pose, angle_mask), dim=-1)        # BXN , 20+192+1


            tgt = torch.cat((tgt_traj, tgt_3dpose, tgt_angle),0) 
            out_local = self.local_former(tgt, mask=None, src_key_padding_mask=tgt_padding_mask_local)
            # out_local = torch.nan_to_num(out_local, nan=0.0)

            ##### local residual ######
            out_local = out_local * self.output_scale + tgt
            
            out_local = out_local[:self.obs_and_pred+1].reshape(self.obs_and_pred+1,B,N,self.nhid).permute(2,0,1,3).reshape(-1,B,self.nhid)     # transformed into NT, B, D
            tgt_padding_mask_global = torch.cat((tgt_padding_mask_global, torch.zeros((B, N), dtype=torch.bool).cuda()), dim=-1)
            out_global = self.global_former(out_local, mask=None, src_key_padding_mask=tgt_padding_mask_global)     # mask: B X NT

            ##### global residual ######
            out_global = out_global * self.output_scale + out_local
            out_global = out_global.reshape(N,F+1,-1,self.nhid).permute(2,0,1,3) # change into into B, N, F, D
            out_final = self.num_mode_mlp(out_global).reshape(B,N,F+1,self.num_modes,self.nhid).permute(3,0,1,2,4)
            
            out_pi = self.pi_time(out_final.permute(0,1,2,4,3)[:, :, :, :, -(self.future_steps+1):-1].reshape(self.num_modes,B,N,-1))
            out_pi = self.pi(out_pi)[:, batch_mask].squeeze().permute(1,0)
            out_final = self.fc_out_traj(out_final)[:,:,:,-(self.future_steps+1):-1]
        elif self.modality == 'traj':
            b_x, b_y, padding_mask, b_x_p, b_y_p, pad_mask_p, batch_mask, b_angle, global_pos = agent2batch_ethucy(data)
            B_, N_, F_, _ = b_x.shape
            b_x, b_y = torch.cat((b_x, global_pos), dim=-1), torch.cat((b_y, torch.zeros(b_y.shape[:-1]).unsqueeze(-1).cuda()), dim=-1)
            tgt = b_x.unsqueeze(-2)
            tgt = tgt[:,:,:8]
            B, N, in_F, J, K = tgt.shape 

            F = 20
            out_F = F - in_F
            assert out_F==12
            ## keep padding
            pad_idx = np.repeat([in_F - 1], out_F)
            i_idx = np.append(np.arange(0, in_F), pad_idx)  
            tgt = tgt[:, :, i_idx]
            tgt = tgt.permute(0, 2, 1, 3, 4)
            padding_mask_indexed = padding_mask[:, :, i_idx]
            padding_mask_indexed_pose = pad_mask_p[:, :, i_idx]
            
            ## add mask
            if metamask: # training
                mask_ratio_traj = 0.1  
                mask_ratio = 0.1 
                mask_ratio_modality = 0.3 
            else:
                mask_ratio_traj = 0.0 
                mask_ratio = 0.0 
                mask_ratio_modality = 0.0 

            tgt_traj = tgt[:,:,:,0,:4]
            traj_mask = torch.rand((B,F,N)).float().to(self.device) > mask_ratio_traj
            traj_mask = traj_mask.unsqueeze(3).repeat_interleave(4,dim=-1)
            tgt_traj = tgt_traj*traj_mask
            
            ## mask for specific modality for whole observation horizon
            modality_selection_3dpose = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).unsqueeze(4).repeat(1,F,1,self.joints_pose,K) 
            modality_selection = modality_selection_3dpose

            ############
            # Transformer
            ###########
            tgt_traj = self.fc_in_traj(tgt_traj) 
            tgt_traj = self.double_id_encoder(tgt_traj, num_people=N) 
            
            padding_mask_trj = padding_mask_indexed.reshape(-1, F)
            padding_mask_pose = padding_mask_indexed_pose.reshape(-1, F)[:, :in_F].repeat(1, self.joints_pose)
            tgt_padding_mask_global = padding_mask_indexed.reshape(B, -1)           
            
            tgt_traj = torch.transpose(tgt_traj,0,1).reshape(F,-1,self.nhid)  # F, BXN, D
            
            tgt_angle = self.angle_mlp(b_angle.unsqueeze(-1)).unsqueeze(0).reshape(1,-1,self.nhid)
            angle_mask = torch.zeros((tgt_angle.shape[1], 1), dtype=torch.bool).cuda()
            tgt_padding_mask_local = torch.cat((padding_mask_trj, angle_mask), dim=-1)        # BXN , 20+1

            tgt = torch.cat((tgt_traj, tgt_angle),0) 
            out_local = self.local_former(tgt, mask=None, src_key_padding_mask=tgt_padding_mask_local)  # 21 X BN X D
            # out_local = torch.nan_to_num(out_local, nan=0.0)

            ##### local residual ######
            out_local = out_local * self.output_scale + tgt
            
            out_local = out_local[:self.obs_and_pred+1].reshape(self.obs_and_pred+1,B,N,self.nhid).permute(2,0,1,3).reshape(-1,B,self.nhid)     # transformed into NT, B, D
            tgt_padding_mask_global = torch.cat((tgt_padding_mask_global, torch.zeros((B, N), dtype=torch.bool).cuda()), dim=-1)
            out_global = self.global_former(out_local, mask=None, src_key_padding_mask=tgt_padding_mask_global)     # mask: B X NT

            ##### global residual ######
            out_global = out_global * self.output_scale + out_local
            out_global = out_global.reshape(N,F+1,-1,self.nhid).permute(2,0,1,3) # change into into B, N, F, D
            out_final = self.num_mode_mlp(out_global).reshape(B,N,F+1,self.num_modes,self.nhid).permute(3,0,1,2,4)
            
            out_pi = self.pi_time(out_final.permute(0,1,2,4,3)[:, :, :, :, -(self.future_steps+1):-1].reshape(self.num_modes,B,N,-1))
            out_pi = self.pi(out_pi)[:, batch_mask].squeeze().permute(1,0)
            out_final = self.fc_out_traj(out_final)[:,:,:,-(self.future_steps+1):-1]
            
            
        else: raise Exception("Modality other than traj, traj_pose is not supported")
        
            
        return out_final[:, batch_mask], out_pi

    def training_step(self, data, batch_idx):
        inv_rot = data.rotate_mat[:, :2, :2].transpose(1, 2) 
        # org_y = data.y @ inv_rot
        org_y = data.y
        original_data = copy.deepcopy(data)
        # Full timesteps
        y_hat, pi = self(copy.deepcopy(original_data), 8, metamask=True)
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        valid_steps = reg_mask.sum(dim=-1)
        cls_mask = valid_steps > 0
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - org_y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        # l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], org_y[reg_mask])
        # reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()
        cls_loss = self.cls_loss(pi[cls_mask], soft_target)
        
        if self.onlyFullTimestep:
            loss = reg_loss + cls_loss
            reg_loss_1frames, cls_loss_1frames, reg_loss_2frames, cls_loss_2frames = 0, 0, 0, 0
        else:
            # 2 timesteps
            data = copy.deepcopy(original_data)
            valid_indices_ = [torch.nonzero(row[:self.historical_steps], as_tuple=True)[0] for row in ~data['padding_mask']]
            valid_indices = []
            for agent_idx in range(len(valid_indices_)):
                if len(valid_indices_[agent_idx]) == 0: 
                    raise Exception("This should not happen")
                valid_indices.append(max(valid_indices_[agent_idx]).item())
            valid_indices = torch.tensor(valid_indices, device=data['x'].device)
            pad_mask_inst = data['padding_mask'][:, :self.historical_steps].clone()
            bos_mask_inst = torch.zeros(data['bos_mask'].shape, dtype=torch.bool, device=data['bos_mask'].device)
            for i in range(pad_mask_inst.shape[0]):
                pad_mask_inst[i, :valid_indices[i] - 1] = True
                data['x'][i, :valid_indices[i]] = 0
                data['x_text'][i, :valid_indices[i]] = 0
                data['x_pose'][i, :valid_indices[i]] = 0
                data['x_text_mask'][i, :valid_indices[i]] = True
                data['x_pose_mask'][i, :valid_indices[i]] = True
                if self.datasetType == 'jrdb':
                    data['x_interaction'][i, :valid_indices[i]] = 0
                    data['x_interaction_mask'][i, :valid_indices[i]] = True
            bos_mask_inst[:, 0] = ~pad_mask_inst[:, 0]
            bos_mask_inst[:, 1: self.historical_steps] = pad_mask_inst[:, : self.historical_steps-1] & ~pad_mask_inst[:, 1: self.historical_steps]
            data['padding_mask'][:, :self.historical_steps] = pad_mask_inst
            data['bos_mask'] = bos_mask_inst
            y_hat, pi = self(data, 2, metamask=True)
            reg_mask = ~data['padding_mask'][:, self.historical_steps:]
            valid_steps = reg_mask.sum(dim=-1)
            cls_mask = valid_steps > 0
            l2_norm = (torch.norm(y_hat[:, :, :, : 2] - org_y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
            # l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
            best_mode = l2_norm.argmin(dim=0)
            y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
            reg_loss_2frames = self.reg_loss(y_hat_best[reg_mask], org_y[reg_mask])
            # reg_loss_2frames = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
            soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()
            cls_loss_2frames = self.cls_loss(pi[cls_mask], soft_target)
            
            # 1 timesteps
            data = copy.deepcopy(original_data)
            valid_indices_ = [torch.nonzero(row[:self.historical_steps], as_tuple=True)[0] for row in ~data['padding_mask']]
            valid_indices = []
            for agent_idx in range(len(valid_indices_)):
                if len(valid_indices_[agent_idx]) == 0: 
                    raise Exception("This should not happen")
                valid_indices.append(max(valid_indices_[agent_idx]).item())
            valid_indices = torch.tensor(valid_indices, device=data['x'].device)
            pad_mask_inst = data['padding_mask'][:, :self.historical_steps].clone()
            bos_mask_inst = torch.zeros(data['bos_mask'].shape, dtype=torch.bool, device=data['bos_mask'].device)
            for i in range(pad_mask_inst.shape[0]):
                pad_mask_inst[i, :valid_indices[i]] = True
                data['x'][i, :valid_indices[i]+1] = 0
                data['x_text'][i, :valid_indices[i]] = 0
                data['x_pose'][i, :valid_indices[i]] = 0
                data['x_text_mask'][i, :valid_indices[i]] = True
                data['x_pose_mask'][i, :valid_indices[i]] = True
                if self.datasetType == 'jrdb':
                    data['x_interaction'][i, :valid_indices[i]] = 0
                    data['x_interaction_mask'][i, :valid_indices[i]] = True
            # inst_invalid_mask = (~pad_mask_inst).sum(-1) == 0   # These have no past observation, should not make predictions
            bos_mask_inst[:, 0] = ~pad_mask_inst[:, 0]
            bos_mask_inst[:, 1: self.historical_steps] = pad_mask_inst[:, : self.historical_steps-1] & ~pad_mask_inst[:, 1: self.historical_steps]
            data['padding_mask'][:, :self.historical_steps] = pad_mask_inst
            # data['padding_mask'][inst_invalid_mask, self.historical_steps:] = True
            data['bos_mask'] = bos_mask_inst
            y_hat, pi = self(data, 1, metamask=True)
            reg_mask = ~data['padding_mask'][:, self.historical_steps:]
            valid_steps = reg_mask.sum(dim=-1)
            cls_mask = valid_steps > 0
            l2_norm = (torch.norm(y_hat[:, :, :, : 2] - org_y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
            # l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
            best_mode = l2_norm.argmin(dim=0)
            y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
            reg_loss_1frames = self.reg_loss(y_hat_best[reg_mask], org_y[reg_mask])
            # reg_loss_1frames = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
            soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()
            cls_loss_1frames = self.cls_loss(pi[cls_mask], soft_target)
            
            loss = reg_loss *3 + cls_loss + reg_loss_1frames + cls_loss_1frames + reg_loss_2frames + cls_loss_2frames
        self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss', cls_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss_1frames', reg_loss_1frames, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss_1frames', cls_loss_1frames, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss_2frames', reg_loss_2frames, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss_2frames', cls_loss_2frames, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, data, batch_idx):
        inv_rot = data.rotate_mat[:, :2, :2].transpose(1, 2) 
        # org_y = data.y @ inv_rot
        org_y = data.y
        data_original = copy.deepcopy(data)
        y_hat, pi = self(data, 8)
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - org_y.unsqueeze(0), p=2, dim=-1) * reg_mask.unsqueeze(0)).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], org_y[reg_mask])
        self.log('val_reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        # viz_trajectory(y_hat_best, data.y, data, "viz", batch_idx)
        self.minADE.update(y_hat_best[..., :2], org_y, reg_mask)
        self.minFDE.update(y_hat_best[..., :2], org_y, reg_mask)
        batch_size_ = (reg_mask.sum(-1)>0).sum()
        self.log('val_minADE', self.minADE, True, on_step=False, on_epoch=True, batch_size=batch_size_)
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        
        # Instantaneous TP 1 frame
        data = copy.deepcopy(data_original)
        valid_indices_ = [torch.nonzero(row[:self.historical_steps], as_tuple=True)[0] for row in ~data['padding_mask']]
        valid_indices = []
        for agent_idx in range(len(valid_indices_)):
            if len(valid_indices_[agent_idx]) == 0: 
                raise Exception("This should not happen")
            valid_indices.append(max(valid_indices_[agent_idx]).item())
        valid_indices = torch.tensor(valid_indices, device=data['x'].device)
        pad_mask_inst = data['padding_mask'][:, :self.historical_steps].clone()
        bos_mask_inst = torch.zeros(data['bos_mask'].shape, dtype=torch.bool, device=data['bos_mask'].device)
        for i in range(pad_mask_inst.shape[0]):
            pad_mask_inst[i, :valid_indices[i]] = True
            data['x'][i, :valid_indices[i]+1] = 0
            data['x_text'][i, :valid_indices[i]] = 0
            data['x_pose'][i, :valid_indices[i]] = 0
            data['x_text_mask'][i, :valid_indices[i]] = True
            data['x_pose_mask'][i, :valid_indices[i]] = True
            if self.datasetType == 'jrdb':
                data['x_interaction'][i, :valid_indices[i]] = 0
                data['x_interaction_mask'][i, :valid_indices[i]] = True
        bos_mask_inst[:, 0] = ~pad_mask_inst[:, 0]
        bos_mask_inst[:, 1: self.historical_steps] = pad_mask_inst[:, : self.historical_steps-1] & ~pad_mask_inst[:, 1: self.historical_steps]
        data['padding_mask'][:, :self.historical_steps] = pad_mask_inst
        data['bos_mask'] = bos_mask_inst
        y_hat, pi = self(data, 1)
        
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - org_y.unsqueeze(0), p=2, dim=-1) * reg_mask.unsqueeze(0)).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        self.minADE_1frame_all.update(y_hat_best[..., :2], org_y, reg_mask)
        self.minFDE_1frame_all.update(y_hat_best[..., :2], org_y, reg_mask)
        batch_size_ = (reg_mask.sum(-1)>0).sum()
        self.log('val_minADE_1frame_all', self.minADE_1frame_all, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        self.log('val_minFDE_1frame_all', self.minFDE_1frame_all, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        
        # Instantaneous TP / most recent timestep
        data = copy.deepcopy(data_original)
        valid_indices_ = [torch.nonzero(row[:self.historical_steps], as_tuple=True)[0] for row in ~data['padding_mask']]
        valid_indices = []
        for agent_idx in range(len(valid_indices_)):
            if len(valid_indices_[agent_idx]) == 0: 
                raise Exception("This should not happen")
            valid_indices.append(max(valid_indices_[agent_idx]).item())
        valid_indices = torch.tensor(valid_indices, device=data['x'].device)
        pad_mask_inst = data['padding_mask'][:, :self.historical_steps].clone()
        bos_mask_inst = torch.zeros(data['bos_mask'].shape, dtype=torch.bool, device=data['bos_mask'].device)
        for i in range(pad_mask_inst.shape[0]):
            pad_mask_inst[i, :valid_indices[i] - 1] = True
            data['x'][i, :valid_indices[i]] = 0
            data['x_text'][i, :valid_indices[i]] = 0
            data['x_pose'][i, :valid_indices[i]] = 0
            data['x_text_mask'][i, :valid_indices[i]] = True
            data['x_pose_mask'][i, :valid_indices[i]] = True
            if self.datasetType == 'jrdb':
                data['x_interaction'][i, :valid_indices[i]] = 0
                data['x_interaction_mask'][i, :valid_indices[i]] = True
        # inst_invalid_mask = (~pad_mask_inst).sum(-1) == 0   # These have no past observation, should not make predictions
        bos_mask_inst[:, 0] = ~pad_mask_inst[:, 0]
        bos_mask_inst[:, 1: self.historical_steps] = pad_mask_inst[:, : self.historical_steps-1] & ~pad_mask_inst[:, 1: self.historical_steps]
        data['padding_mask'][:, :self.historical_steps] = pad_mask_inst
        # data['padding_mask'][inst_invalid_mask, self.historical_steps:] = True
        data['bos_mask'] = bos_mask_inst
        y_hat, pi = self(data, 2)
        
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - org_y.unsqueeze(0), p=2, dim=-1) * reg_mask.unsqueeze(0)).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        self.minADE_inst_all.update(y_hat_best[..., :2], org_y, reg_mask)
        self.minFDE_inst_all.update(y_hat_best[..., :2], org_y, reg_mask)
        batch_size_ = (reg_mask.sum(-1)>0).sum()
        self.log('val_minADE_inst_all', self.minADE_inst_all, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        self.log('val_minFDE_inst_all', self.minFDE_inst_all, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        
        # Eval on agents with pose / inst
        data = copy.deepcopy(data_original)
        data['padding_mask'][:, self.historical_steps:] = data['padding_mask'][:, self.historical_steps:] | data['y_pose_mask']
        valid_indices_ = [torch.nonzero(row[:self.historical_steps], as_tuple=True)[0] for row in ~data['padding_mask']]
        valid_indices = []
        for agent_idx in range(len(valid_indices_)):
            if len(valid_indices_[agent_idx]) == 0: 
                raise Exception("This should not happen")
            valid_indices.append(max(valid_indices_[agent_idx]).item())
        valid_indices = torch.tensor(valid_indices, device=data['x'].device)
        pad_mask_inst = data['padding_mask'][:, :self.historical_steps].clone()
        bos_mask_inst = torch.zeros(data['bos_mask'].shape, dtype=torch.bool, device=data['bos_mask'].device)
        for i in range(pad_mask_inst.shape[0]):
            pad_mask_inst[i, :valid_indices[i] - 1] = True
            data['x'][i, :valid_indices[i]] = 0
            data['x_text'][i, :valid_indices[i]] = 0
            data['x_text_mask'][i, :valid_indices[i]] = True
            if self.datasetType == 'jrdb':
                data['x_interaction'][i, :valid_indices[i]] = 0
                data['x_interaction_mask'][i, :valid_indices[i]] = True
        # inst_invalid_mask = (~pad_mask_inst).sum(-1) == 0   # These have no past observation, should not make predictions
        bos_mask_inst[:, 0] = ~pad_mask_inst[:, 0]
        bos_mask_inst[:, 1: self.historical_steps] = pad_mask_inst[:, : self.historical_steps-1] & ~pad_mask_inst[:, 1: self.historical_steps]
        data['padding_mask'][:, :self.historical_steps] = pad_mask_inst
        # data['padding_mask'][inst_invalid_mask, self.historical_steps:] = True
        data['bos_mask'] = bos_mask_inst
        y_hat, pi = self(data, 2)
        
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - org_y.unsqueeze(0), p=2, dim=-1) * reg_mask.unsqueeze(0)).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        self.minADE_pose_inst.update(y_hat_best[..., :2], org_y, reg_mask)
        self.minFDE_pose_inst.update(y_hat_best[..., :2], org_y, reg_mask)
        batch_size_ = (reg_mask.sum(-1)>0).sum()
        self.log('val_minADE_pose_inst', self.minADE_pose_inst, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        self.log('val_minFDE_pose_inst', self.minFDE_pose_inst, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        
        
        # Eval on agents with pose / regular
        data = copy.deepcopy(data_original)
        data['padding_mask'][:, self.historical_steps:] = data['padding_mask'][:, self.historical_steps:] | data['y_pose_mask']
        y_hat, pi = self(data, 8)
        
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - org_y.unsqueeze(0), p=2, dim=-1) * reg_mask.unsqueeze(0)).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        self.minADE_pose.update(y_hat_best[..., :2], org_y, reg_mask)
        self.minFDE_pose.update(y_hat_best[..., :2], org_y, reg_mask)
        batch_size_ = (reg_mask.sum(-1)>0).sum()
        self.log('val_minADE_pose', self.minADE_pose, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        self.log('val_minFDE_pose', self.minFDE_pose, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)        
        

    def validation_step_withoutInvRot(self, data, batch_idx):
        data_original = copy.deepcopy(data)
        y_hat, pi = self(data)
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y.unsqueeze(0), p=2, dim=-1) * reg_mask.unsqueeze(0)).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        self.log('val_reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        # viz_trajectory(y_hat_best, data.y, data, "viz", batch_idx)
        self.minADE.update(y_hat_best[..., :2], data.y, reg_mask)
        self.minFDE.update(y_hat_best[..., :2], data.y, reg_mask)
        batch_size_ = (reg_mask.sum(-1)>0).sum()
        self.log('val_minADE', self.minADE, True, on_step=False, on_epoch=True, batch_size=batch_size_)
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        
        # Instantaneous TP 1 frame
        data = copy.deepcopy(data_original)
        valid_indices_ = [torch.nonzero(row[:self.historical_steps], as_tuple=True)[0] for row in ~data['padding_mask']]
        valid_indices = []
        for agent_idx in range(len(valid_indices_)):
            if len(valid_indices_[agent_idx]) == 0: 
                raise Exception("This should not happen")
            valid_indices.append(max(valid_indices_[agent_idx]).item())
        valid_indices = torch.tensor(valid_indices, device=data['x'].device)
        pad_mask_inst = data['padding_mask'][:, :self.historical_steps].clone()
        bos_mask_inst = torch.zeros(data['bos_mask'].shape, dtype=torch.bool, device=data['bos_mask'].device)
        for i in range(pad_mask_inst.shape[0]):
            pad_mask_inst[i, :valid_indices[i]] = True
            data['x'][i, :valid_indices[i]+1] = 0
            data['x_text'][i, :valid_indices[i]] = 0
            data['x_pose'][i, :valid_indices[i]] = 0
            data['x_text_mask'][i, :valid_indices[i]] = True
            data['x_pose_mask'][i, :valid_indices[i]] = True
            if self.datasetType == 'jrdb':
                data['x_interaction'][i, :valid_indices[i]] = 0
                data['x_interaction_mask'][i, :valid_indices[i]] = True
        bos_mask_inst[:, 0] = ~pad_mask_inst[:, 0]
        bos_mask_inst[:, 1: self.historical_steps] = pad_mask_inst[:, : self.historical_steps-1] & ~pad_mask_inst[:, 1: self.historical_steps]
        data['padding_mask'][:, :self.historical_steps] = pad_mask_inst
        data['bos_mask'] = bos_mask_inst
        y_hat, pi = self(data)
        
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y.unsqueeze(0), p=2, dim=-1) * reg_mask.unsqueeze(0)).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        self.minADE_1frame_all.update(y_hat_best[..., :2], data.y, reg_mask)
        self.minFDE_1frame_all.update(y_hat_best[..., :2], data.y, reg_mask)
        batch_size_ = (reg_mask.sum(-1)>0).sum()
        self.log('val_minADE_1frame_all', self.minADE_1frame_all, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        self.log('val_minFDE_1frame_all', self.minFDE_1frame_all, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        
        # Instantaneous TP / most recent timestep
        data = copy.deepcopy(data_original)
        valid_indices_ = [torch.nonzero(row[:self.historical_steps], as_tuple=True)[0] for row in ~data['padding_mask']]
        valid_indices = []
        for agent_idx in range(len(valid_indices_)):
            if len(valid_indices_[agent_idx]) == 0: 
                raise Exception("This should not happen")
            valid_indices.append(max(valid_indices_[agent_idx]).item())
        valid_indices = torch.tensor(valid_indices, device=data['x'].device)
        pad_mask_inst = data['padding_mask'][:, :self.historical_steps].clone()
        bos_mask_inst = torch.zeros(data['bos_mask'].shape, dtype=torch.bool, device=data['bos_mask'].device)
        for i in range(pad_mask_inst.shape[0]):
            pad_mask_inst[i, :valid_indices[i] - 1] = True
            data['x'][i, :valid_indices[i]] = 0
            data['x_text'][i, :valid_indices[i]] = 0
            data['x_pose'][i, :valid_indices[i]] = 0
            data['x_text_mask'][i, :valid_indices[i]] = True
            data['x_pose_mask'][i, :valid_indices[i]] = True
            if self.datasetType == 'jrdb':
                data['x_interaction'][i, :valid_indices[i]] = 0
                data['x_interaction_mask'][i, :valid_indices[i]] = True
        # inst_invalid_mask = (~pad_mask_inst).sum(-1) == 0   # These have no past observation, should not make predictions
        bos_mask_inst[:, 0] = ~pad_mask_inst[:, 0]
        bos_mask_inst[:, 1: self.historical_steps] = pad_mask_inst[:, : self.historical_steps-1] & ~pad_mask_inst[:, 1: self.historical_steps]
        data['padding_mask'][:, :self.historical_steps] = pad_mask_inst
        # data['padding_mask'][inst_invalid_mask, self.historical_steps:] = True
        data['bos_mask'] = bos_mask_inst
        y_hat, pi = self(data)
        
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y.unsqueeze(0), p=2, dim=-1) * reg_mask.unsqueeze(0)).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        self.minADE_inst_all.update(y_hat_best[..., :2], data.y, reg_mask)
        self.minFDE_inst_all.update(y_hat_best[..., :2], data.y, reg_mask)
        batch_size_ = (reg_mask.sum(-1)>0).sum()
        self.log('val_minADE_inst_all', self.minADE_inst_all, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        self.log('val_minFDE_inst_all', self.minFDE_inst_all, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        
        # Eval on agents with pose / inst
        data = copy.deepcopy(data_original)
        data['padding_mask'][:, self.historical_steps:] = data['padding_mask'][:, self.historical_steps:] | data['y_pose_mask']
        valid_indices_ = [torch.nonzero(row[:self.historical_steps], as_tuple=True)[0] for row in ~data['padding_mask']]
        valid_indices = []
        for agent_idx in range(len(valid_indices_)):
            if len(valid_indices_[agent_idx]) == 0: 
                raise Exception("This should not happen")
            valid_indices.append(max(valid_indices_[agent_idx]).item())
        valid_indices = torch.tensor(valid_indices, device=data['x'].device)
        pad_mask_inst = data['padding_mask'][:, :self.historical_steps].clone()
        bos_mask_inst = torch.zeros(data['bos_mask'].shape, dtype=torch.bool, device=data['bos_mask'].device)
        for i in range(pad_mask_inst.shape[0]):
            pad_mask_inst[i, :valid_indices[i] - 1] = True
            data['x'][i, :valid_indices[i]] = 0
            data['x_text'][i, :valid_indices[i]] = 0
            data['x_text_mask'][i, :valid_indices[i]] = True
            if self.datasetType == 'jrdb':
                data['x_interaction'][i, :valid_indices[i]] = 0
                data['x_interaction_mask'][i, :valid_indices[i]] = True
        # inst_invalid_mask = (~pad_mask_inst).sum(-1) == 0   # These have no past observation, should not make predictions
        bos_mask_inst[:, 0] = ~pad_mask_inst[:, 0]
        bos_mask_inst[:, 1: self.historical_steps] = pad_mask_inst[:, : self.historical_steps-1] & ~pad_mask_inst[:, 1: self.historical_steps]
        data['padding_mask'][:, :self.historical_steps] = pad_mask_inst
        # data['padding_mask'][inst_invalid_mask, self.historical_steps:] = True
        data['bos_mask'] = bos_mask_inst
        y_hat, pi = self(data)
        
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y.unsqueeze(0), p=2, dim=-1) * reg_mask.unsqueeze(0)).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        self.minADE_pose_inst.update(y_hat_best[..., :2], data.y, reg_mask)
        self.minFDE_pose_inst.update(y_hat_best[..., :2], data.y, reg_mask)
        batch_size_ = (reg_mask.sum(-1)>0).sum()
        self.log('val_minADE_pose_inst', self.minADE_pose_inst, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        self.log('val_minFDE_pose_inst', self.minFDE_pose_inst, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        
        
        # Eval on agents with pose / regular
        data = copy.deepcopy(data_original)
        data['padding_mask'][:, self.historical_steps:] = data['padding_mask'][:, self.historical_steps:] | data['y_pose_mask']
        y_hat, pi = self(data)
        
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y.unsqueeze(0), p=2, dim=-1) * reg_mask.unsqueeze(0)).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        self.minADE_pose.update(y_hat_best[..., :2], data.y, reg_mask)
        self.minFDE_pose.update(y_hat_best[..., :2], data.y, reg_mask)
        batch_size_ = (reg_mask.sum(-1)>0).sum()
        self.log('val_minADE_pose', self.minADE_pose, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        self.log('val_minFDE_pose', self.minFDE_pose, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)        

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('socialTransmotion')
        parser.add_argument('--historical_steps', type=int, default=8)
        parser.add_argument('--future_steps', type=int, default=12)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--rotate', type=bool, default=True)
        parser.add_argument('--node_dim', type=int, default=2)
        parser.add_argument('--edge_dim', type=int, default=2)
        # parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_temporal_layers', type=int, default=4)
        parser.add_argument('--num_global_layers', type=int, default=3)
        parser.add_argument('--local_radius', type=float, default=50)
        parser.add_argument('--parallel', type=bool, default=False)
        parser.add_argument('--lr', type=float, default=0.0005025656874575494)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser
