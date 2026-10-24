'''
v2: inst 에 대해서도 kd
v3: traj prediction 에 대해서도 kd
v4: inst kd
teachV3: Teacher model이 global interactor에서 text 쓴 v3 모델
'''
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import glob

from losses import LaplaceNLLLoss
from losses import SoftTargetCrossEntropyLoss
from metrics import ADE
from metrics import FDE
from metrics import MR
from models import *
from utils_hivt import *
from visualization import viz_trajectory
import copy
import os
from utils_viz_jw import VizScene


class HiVT(pl.LightningModule):

    def __init__(self,
                 historical_steps: int,
                 future_steps: int,
                 num_modes: int,
                 rotate: bool,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 num_temporal_layers: int,
                 num_global_layers: int,
                 local_radius: float,
                 parallel: bool,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 encoder_type: str,
                 encoder_type_student='traj_pose',
                 emb_numHead=4,
                 emb_dropout=0.5,
                 emb_numLayers=2,
                 **kwargs) -> None:
        super(HiVT, self).__init__()
        self.save_hyperparameters()
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.baseline_results_base_ = None
        # self.mask_ratio = torch.linspace(0, 0.1, kwargs['max_epochs'])
        # self.mask_ratio = 0.1 * torch.exp(-5 * self.mask_ratio)
        self.mask_ratio = [0.0] * kwargs['max_epochs']
        self.encoder_type_teacher = encoder_type
        self.local_encoder = build_encoder(encoder_type, historical_steps, node_dim, edge_dim, embed_dim, num_heads, dropout, num_temporal_layers, local_radius, parallel, emb_numHead, emb_dropout, emb_numLayers)
        self.local_encoder_student = build_encoder(encoder_type_student, historical_steps, node_dim, edge_dim, embed_dim, num_heads, dropout, num_temporal_layers, local_radius, parallel, emb_numHead, emb_dropout, emb_numLayers)
        self.local_encoder_dummy = build_encoder(encoder_type, historical_steps, node_dim, edge_dim, embed_dim, num_heads, dropout, num_temporal_layers, local_radius, parallel, emb_numHead, emb_dropout, emb_numLayers)
        self.global_interactor = GlobalInteractor(historical_steps=historical_steps,
                                                  embed_dim=embed_dim,
                                                  edge_dim=edge_dim,
                                                  num_modes=num_modes,
                                                  num_heads=num_heads,
                                                  num_layers=num_global_layers,
                                                  dropout=dropout,
                                                  rotate=rotate)
        self.global_interactor_dummy = GlobalInteractor(historical_steps=historical_steps,
                                                  embed_dim=embed_dim,
                                                  edge_dim=edge_dim,
                                                  num_modes=num_modes,
                                                  num_heads=num_heads,
                                                  num_layers=num_global_layers,
                                                  dropout=dropout,
                                                  rotate=rotate)
        self.global_interactor_student = GlobalInteractor(historical_steps=historical_steps,
                                                  embed_dim=embed_dim,
                                                  edge_dim=edge_dim,
                                                  num_modes=num_modes,
                                                  num_heads=num_heads,
                                                  num_layers=num_global_layers,
                                                  dropout=dropout,
                                                  rotate=rotate)
        self.decoder = MLPDecoder(local_channels=embed_dim,
                                  global_channels=embed_dim,
                                  future_steps=future_steps,
                                  num_modes=num_modes,
                                  uncertain=True)
        self.decoder_student = MLPDecoder(local_channels=embed_dim,
                                  global_channels=embed_dim,
                                  future_steps=future_steps,
                                  num_modes=num_modes,
                                  uncertain=True)
        self.reg_loss = LaplaceNLLLoss(reduction='mean')
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
        self.check_valid = torch.tensor((0,0)).cuda()
        # self.minMR = MR()
    
    def freeze(self):
        for param in self.local_encoder.parameters():
            param.requires_grad = False
        for param in self.local_encoder_dummy.parameters():
            param.requires_grad = False
        for param in self.global_interactor.parameters():
            param.requires_grad = False
        for param in self.global_interactor_dummy.parameters():
            param.requires_grad = False
        print('freezed!')

    def forward(self, data: TemporalData):
        # if self.rotate:
        #     rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
        #     sin_vals = torch.sin(data['rotate_angles'])
        #     cos_vals = torch.cos(data['rotate_angles'])
        #     rotate_mat[:, 0, 0] = cos_vals
        #     rotate_mat[:, 0, 1] = -sin_vals
        #     rotate_mat[:, 1, 0] = sin_vals
        #     rotate_mat[:, 1, 1] = cos_vals
        #     if data.y is not None:
        #         data.y = torch.bmm(data.y, rotate_mat)
        #     data['rotate_mat'] = rotate_mat
        # else:
        #     data['rotate_mat'] = None
        if data['rotate_mat'].shape[-1] == 3: data['rotate_mat'] = data['rotate_mat'][:,:2,:2]
        self.local_encoder.eval()
        self.global_interactor.eval()
        self.local_encoder_dummy.eval()
        self.global_interactor_dummy.eval()
        self.decoder.eval()
        local_embed_student, local_embed_student_stacked = self.local_encoder_student(data=data)
        global_embed_student = self.global_interactor_student(data=data, local_embed=local_embed_student)
        with torch.no_grad():
            if 'embedMask' in self.encoder_type_teacher: 
                local_embed, local_embed_stacked = self.local_encoder(data=data, mask_ratio=self.mask_ratio[self.current_epoch])
                local_embed_dummy, local_embed_stacked_dummy = self.local_encoder_dummy(data=data, mask_ratio=self.mask_ratio[self.current_epoch])
            else: 
                local_embed, local_embed_stacked = self.local_encoder(data=data)
                local_embed_dummy, local_embed_stacked_dummy = self.local_encoder_dummy(data=data)
            global_embed = self.global_interactor(data=data, local_embed=local_embed)
            global_embed_dummy = self.global_interactor_dummy(data=data, local_embed=local_embed_dummy)
        y_hat, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)
        y_hat_student, pi_student = self.decoder_student(local_embed=local_embed_student, global_embed=global_embed_student)
        return y_hat.detach(), pi.detach(), local_embed.detach(), local_embed_student, global_embed.detach(), global_embed_student, \
            local_embed_stacked.detach(), local_embed_student_stacked, y_hat_student, pi_student, local_embed_dummy, global_embed_dummy

    def forward_1frame(self, data: TemporalData):
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
            data['x_interaction'][i, :valid_indices[i]] = 0
            data['x_text_mask'][i, :valid_indices[i]] = True
            data['x_interaction_mask'][i, :valid_indices[i]] = True
        # inst_invalid_mask = (~pad_mask_inst).sum(-1) == 0   # These have no past observation, should not make predictions
        bos_mask_inst[:, 0] = ~pad_mask_inst[:, 0]
        bos_mask_inst[:, 1: self.historical_steps] = pad_mask_inst[:, : self.historical_steps-1] & ~pad_mask_inst[:, 1: self.historical_steps]
        data['padding_mask'][:, :self.historical_steps] = pad_mask_inst
        # data['padding_mask'][inst_invalid_mask, self.historical_steps:] = True
        data['bos_mask'] = bos_mask_inst
        if data['rotate_mat'].shape[-1] == 3: data['rotate_mat'] = data['rotate_mat'][:,:2,:2]
        self.local_encoder.eval()
        self.global_interactor.eval()
        self.local_encoder_dummy.eval()
        self.global_interactor_dummy.eval()
        self.decoder.eval()
        local_embed_student, local_embed_student_stacked = self.local_encoder_student(data=data)
        global_embed_student = self.global_interactor_student(data=data, local_embed=local_embed_student)
        with torch.no_grad():
            if 'embedMask' in self.encoder_type_teacher: 
                local_embed, local_embed_stacked = self.local_encoder(data=data, mask_ratio=self.mask_ratio[self.current_epoch])
                local_embed_dummy, local_embed_stacked_dummy = self.local_encoder_dummy(data=data, mask_ratio=self.mask_ratio[self.current_epoch])
            else: 
                local_embed, local_embed_stacked = self.local_encoder(data=data)
                local_embed_dummy, local_embed_stacked_dummy = self.local_encoder_dummy(data=data)
            global_embed = self.global_interactor(data=data, local_embed=local_embed)
            global_embed_dummy = self.global_interactor_dummy(data=data, local_embed=local_embed_dummy)
        y_hat, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)
        y_hat_student, pi_student = self.decoder_student(local_embed=local_embed_student, global_embed=global_embed_student)
        return y_hat.detach(), pi.detach(), local_embed.detach(), local_embed_student, global_embed.detach(), global_embed_student, \
            local_embed_stacked.detach(), local_embed_student_stacked, y_hat_student, pi_student, local_embed_dummy, global_embed_dummy
    
    def forward_2frame(self, data: TemporalData):
        
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
            data['x_interaction'][i, :valid_indices[i]] = 0
            data['x_text_mask'][i, :valid_indices[i]] = True
            data['x_interaction_mask'][i, :valid_indices[i]] = True
        # inst_invalid_mask = (~pad_mask_inst).sum(-1) == 0   # These have no past observation, should not make predictions
        bos_mask_inst[:, 0] = ~pad_mask_inst[:, 0]
        bos_mask_inst[:, 1: self.historical_steps] = pad_mask_inst[:, : self.historical_steps-1] & ~pad_mask_inst[:, 1: self.historical_steps]
        data['padding_mask'][:, :self.historical_steps] = pad_mask_inst
        # data['padding_mask'][inst_invalid_mask, self.historical_steps:] = True
        data['bos_mask'] = bos_mask_inst
        if data['rotate_mat'].shape[-1] == 3: data['rotate_mat'] = data['rotate_mat'][:,:2,:2]
        self.local_encoder.eval()
        self.global_interactor.eval()
        self.local_encoder_dummy.eval()
        self.global_interactor_dummy.eval()
        self.decoder.eval()
        local_embed_student, local_embed_student_stacked = self.local_encoder_student(data=data)
        global_embed_student = self.global_interactor_student(data=data, local_embed=local_embed_student)
        with torch.no_grad():
            if 'embedMask' in self.encoder_type_teacher: 
                local_embed, local_embed_stacked = self.local_encoder(data=data, mask_ratio=self.mask_ratio[self.current_epoch])
                local_embed_dummy, local_embed_stacked_dummy = self.local_encoder_dummy(data=data, mask_ratio=self.mask_ratio[self.current_epoch])
            else: 
                local_embed, local_embed_stacked = self.local_encoder(data=data)
                local_embed_dummy, local_embed_stacked_dummy = self.local_encoder_dummy(data=data)
            global_embed = self.global_interactor(data=data, local_embed=local_embed)
            global_embed_dummy = self.global_interactor_dummy(data=data, local_embed=local_embed_dummy)
        y_hat, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)
        y_hat_student, pi_student = self.decoder_student(local_embed=local_embed_student, global_embed=global_embed_student)
        return y_hat.detach(), pi.detach(), local_embed.detach(), local_embed_student, global_embed.detach(), global_embed_student, \
            local_embed_stacked.detach(), local_embed_student_stacked, y_hat_student, pi_student, local_embed_dummy, global_embed_dummy

    def training_step(self, data, batch_idx):
        y_hat, pi, local_embed, local_embed_student, global_embed, global_embed_student, local_embed_stacked, local_embed_student_stacked, y_hat_student, pi_student, local_embed_dummy, global_embed_dummy = self(copy.deepcopy(data))   # full timestep forward
        
        # Regression & classification loss
        current_true = data.positions[:, self.historical_steps-1,:] == self.check_valid
        current_true = (current_true.sum(-1) == 2).unsqueeze(-1).repeat(1, self.future_steps) # If true, invalid agent
        reg_mask = ~torch.logical_or(data['padding_mask'][:, self.historical_steps:], current_true)
        valid_steps = reg_mask.sum(dim=-1)
        cls_mask = valid_steps > 0
        l2_norm = (torch.norm(y_hat_student[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat_student[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()
        cls_loss = self.cls_loss(pi_student[cls_mask], soft_target)
        
        # distillation loss
        local_agent_kd_loss = cosine_similarity_loss(local_embed_student, local_embed)
        global_agent_kd_loss = cosine_similarity_loss(global_embed_student, global_embed)
        y_hat_1f, pi_1f, local_embed_1f, local_embed_student_1f, global_embed_1f, global_embed_student_1f, local_embed_stacked_1f, local_embed_student_stacked_1f, y_hat_student_1f, pi_student_1f, local_embed_1f_dummy, global_embed_1f_dummy = self.forward_1frame(copy.deepcopy(data))   # 1frame
        local_agent_kd_loss_1f = cosine_similarity_loss(local_embed_student_1f, local_embed_1f)
        global_agent_kd_loss_1f = cosine_similarity_loss(global_embed_student_1f, global_embed_1f)
        l2_norm = (torch.norm(y_hat_student_1f[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat_student_1f[best_mode, torch.arange(data.num_nodes)]
        reg_loss_1f = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        y_hat_2f, pi_2f, local_embed_2f, local_embed_student_2f, global_embed_2f, global_embed_student_2f, local_embed_stacked_2f, local_embed_student_stacked_2f, y_hat_student_2f, pi_student_2f, local_embed_2f_dummy, global_embed_2f_dummy = self.forward_2frame(copy.deepcopy(data))   # 2frame
        local_agent_kd_loss_2f = cosine_similarity_loss(local_embed_student_2f, local_embed_2f)
        global_agent_kd_loss_2f = cosine_similarity_loss(global_embed_student_2f, global_embed_2f)
        l2_norm = (torch.norm(y_hat_student_2f[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat_student_2f[best_mode, torch.arange(data.num_nodes)]
        reg_loss_2f = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        
        # dummy distillation loss
        local_agent_kd_loss_dummy = kl_divergence_loss(local_embed_student, local_embed_dummy)
        global_agent_kd_loss_dummy = kl_divergence_loss(global_embed_student, global_embed_dummy)
        local_agent_kd_loss_1f_dummy = kl_divergence_loss(local_embed_student_1f, local_embed_1f_dummy)
        global_agent_kd_loss_1f_dummy = kl_divergence_loss(global_embed_student_1f, global_embed_1f_dummy)
        local_agent_kd_loss_2f_dummy = kl_divergence_loss(local_embed_student_2f, local_embed_2f_dummy)
        global_agent_kd_loss_2f_dummy = kl_divergence_loss(global_embed_student_2f, global_embed_2f_dummy)
        
        # loss = reg_loss * 3 + cls_loss + local_agent_kd_loss + global_agent_kd_loss + local_agent_kd_loss_1f + global_agent_kd_loss_1f + local_agent_kd_loss_2f + global_agent_kd_loss_2f + reg_loss_1f + reg_loss_2f 
        loss = reg_loss * 3 + cls_loss + local_agent_kd_loss *0.1 + global_agent_kd_loss *0.1 + local_agent_kd_loss_1f *0.1 + global_agent_kd_loss_1f *0.1 + local_agent_kd_loss_2f *0.1 + global_agent_kd_loss_2f *0.1 + reg_loss_1f + reg_loss_2f \
            + local_agent_kd_loss_dummy + global_agent_kd_loss_dummy + local_agent_kd_loss_1f_dummy + global_agent_kd_loss_1f_dummy + local_agent_kd_loss_2f_dummy + global_agent_kd_loss_2f_dummy
        # loss = reg_loss * 3 + cls_loss + local_agent_kd_loss *0.1 + global_agent_kd_loss *0.1 + local_agent_kd_loss_dummy + global_agent_kd_loss_dummy
        self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss', cls_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_localAgent_kd_loss', local_agent_kd_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        # self.log('train_localAgent_kd_loss_1frame', local_agent_kd_loss_1frame, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        # self.log('train_localAgent_kd_loss_2frame', local_agent_kd_loss_2frame, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_globalAgent_kd_loss', global_agent_kd_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_localAgent_kd_loss_dummy', local_agent_kd_loss_dummy, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_globalAgent_kd_loss_dummy', global_agent_kd_loss_dummy, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)

        return loss

    def validation_step(self, data, batch_idx):
        # if batch_idx%4 !=0 : return None
        if batch_idx < 50: return None
        # if batch_idx != 1112: return None
        
        # if batch_idx != 710: return None
        PRINT_AGENT = None
        # PRINT_AGENT = None
        
        # if batch_idx != 710: return None
        # PRINT_AGENT = [10, 11, 12]
        
        data_original = copy.deepcopy(data)
        
        verts = data['x_pose_verts']
        faces = data['x_pose_face']
        pose_mask = data['x_pose_mask']
        # save_root = '/mnt/jaewoo4tb/textraj/HiVT/__viz_qual_v2/kd'
        save_root = '/mnt/jaewoo4tb/textraj/HiVT/__viz_qual_sit/KD'
        if self.baseline_results_base_ is None:
            self.baseline_results_base_ = sorted(glob.glob('/mnt/jaewoo4tb/textraj/HiVT/__viz_modality_save_sit/traj_pose/*.pt'))
        baseline_results_base = torch.load(self.baseline_results_base_[batch_idx])
        os.makedirs(save_root, exist_ok=True)
        
        _, _, _, _, _, _, _, _, y_hat, pi, _, _ = self(data)
        current_true = data.positions[:, self.historical_steps-1,:] == self.check_valid
        current_true = (current_true.sum(-1) == 2).unsqueeze(-1).repeat(1, self.future_steps) # If true, invalid agent
        reg_mask = ~torch.logical_or(data['padding_mask'][:, self.historical_steps:], current_true)
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y.unsqueeze(0), p=2, dim=-1) * reg_mask.unsqueeze(0)).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        self.log('val_reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
        self.minADE.update(y_hat_best[..., :2], data.y, reg_mask)
        self.minFDE.update(y_hat_best[..., :2], data.y, reg_mask)
        batch_size_ = (reg_mask.sum(-1)>0).sum()
        self.log('val_minADE', self.minADE, True, on_step=False, on_epoch=True, batch_size=batch_size_)
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        
        
        # visualization of original data
        plot_x = data['positions'][:, :self.historical_steps]
        plot_y = data['positions'][:, self.historical_steps:]
        plot_padding_mask = data['padding_mask']
        plot_verts = verts
        plot_faces = faces
        plot_pose_mask = pose_mask
        scene_name = data['scene'][0]
        robot_position = data['robot_pos']
        initial_frame = data['initial_frame'].item()
        plot_save_name = str(batch_idx).zfill(5) + "-whole_frame"
        preprocessed_1st_root = "/mnt/jaewoo4tb/textraj/preprocessed_1st/sit_v1/"
        preprocessed_1st = torch.load(preprocessed_1st_root + scene_name + ".pt")
        # interaction_scene_name = scene_name.replace("agents", "interactions")
        # preprocessed_1st_interaction = torch.load(preprocessed_1st_root + interaction_scene_name + ".pt")

        caption_target_frame = initial_frame + (self.historical_steps - 1) * 6 # 6 for frame step
        ids = data['agent_ids'].tolist()
        captions = [preprocessed_1st[caption_target_frame][id]['description'] 
                    if id in preprocessed_1st[caption_target_frame] else None for id in ids]

        # interactions = preprocessed_1st_interaction[caption_target_frame]['dual_interaction']
        original_viz = VizScene(plot_x, plot_y, data['rotate_mat'], plot_padding_mask, y_hat_best[:, :, :2], plot_verts, plot_faces, plot_pose_mask, robot_position, initial_frame, save_root, plot_save_name, scene_name, y_hat_best_baseline=baseline_results_base['full'][:,:,:2])
        frames = [0]
        original_viz.draw_frames(frames, PRINT_AGENT)
        original_viz.save_meta(captions, None, ids)
         ##################################### 여기까지 수정함 #####################################################
        
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
            # data['x_interaction'][i, :valid_indices[i]] = 0
            data['x_text_mask'][i, :valid_indices[i]] = True
            data['x_pose_mask'][i, :valid_indices[i]] = True
            # data['x_interaction_mask'][i, :valid_indices[i]] = True
        # inst_invalid_mask = (~pad_mask_inst).sum(-1) == 0   # These have no past observation, should not make predictions
        bos_mask_inst[:, 0] = ~pad_mask_inst[:, 0]
        bos_mask_inst[:, 1: self.historical_steps] = pad_mask_inst[:, : self.historical_steps-1] & ~pad_mask_inst[:, 1: self.historical_steps]
        data['padding_mask'][:, :self.historical_steps] = pad_mask_inst
        # data['padding_mask'][inst_invalid_mask, self.historical_steps:] = True
        data['bos_mask'] = bos_mask_inst
        _, _, _, _, _, _, _, _, y_hat, pi, _, _ = self(data)
        
        current_true = data.positions[:, self.historical_steps-1,:] == self.check_valid
        current_true = (current_true.sum(-1) == 2).unsqueeze(-1).repeat(1, self.future_steps) # If true, invalid agent
        reg_mask = ~torch.logical_or(data['padding_mask'][:, self.historical_steps:], current_true)
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y.unsqueeze(0), p=2, dim=-1) * reg_mask.unsqueeze(0)).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        self.minADE_1frame_all.update(y_hat_best[..., :2], data.y, reg_mask)
        self.minFDE_1frame_all.update(y_hat_best[..., :2], data.y, reg_mask)
        batch_size_ = (reg_mask.sum(-1)>0).sum()
        self.log('val_minADE_1frame_all', self.minADE_1frame_all, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        self.log('val_minFDE_1frame_all', self.minFDE_1frame_all, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)

        # visualization of one frame predection
        plot_x = data['positions'][:, :self.historical_steps]
        # plot_x = data['positions'][:, self.historical_steps-1:self.historical_steps]
        plot_y = data['positions'][:, self.historical_steps:]
        # plot_padding_mask = data['padding_mask'][: , self.historical_steps-1:]
        plot_padding_mask = data['padding_mask']
        plot_verts = verts
        # plot_verts = verts[:, self.historical_steps-1:]
        plot_faces = faces
        # plot_faces = faces[:, self.historical_steps-1:]
        # plot_pose_mask = pose_mask[:, self.historical_steps-1:]
        plot_pose_mask = pose_mask
        plot_save_name = str(batch_idx).zfill(5) + "-one_frame"
        one_frame_viz = VizScene(plot_x, plot_y, data['rotate_mat'], plot_padding_mask, y_hat_best[:, :, :2], plot_verts, plot_faces, plot_pose_mask, robot_position, initial_frame, save_root, plot_save_name, scene_name, y_hat_best_baseline=baseline_results_base['1f'][:,:,:2])
        # one_frame_viz = VizScene(plot_x, plot_y, data['rotate_mat'], plot_padding_mask, y_hat_best[:, :, :2], plot_verts, plot_faces, plot_pose_mask, robot_position, initial_frame, save_root, plot_save_name, scene_name, y_hat_best_baseline=baseline_result['1f'][:,:,:2])
        frames = [7]
        one_frame_viz.draw_frames(frames, PRINT_AGENT)
        
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
            # data['x_interaction'][i, :valid_indices[i]] = 0
            data['x_text_mask'][i, :valid_indices[i]] = True
            data['x_pose_mask'][i, :valid_indices[i]] = True
            # data['x_interaction_mask'][i, :valid_indices[i]] = True
        # inst_invalid_mask = (~pad_mask_inst).sum(-1) == 0   # These have no past observation, should not make predictions
        bos_mask_inst[:, 0] = ~pad_mask_inst[:, 0]
        bos_mask_inst[:, 1: self.historical_steps] = pad_mask_inst[:, : self.historical_steps-1] & ~pad_mask_inst[:, 1: self.historical_steps]
        data['padding_mask'][:, :self.historical_steps] = pad_mask_inst
        # data['padding_mask'][inst_invalid_mask, self.historical_steps:] = True
        data['bos_mask'] = bos_mask_inst
        _, _, _, _, _, _, _, _, y_hat, pi, _, _ = self(data)
        
        current_true = data.positions[:, self.historical_steps-1,:] == self.check_valid
        current_true = (current_true.sum(-1) == 2).unsqueeze(-1).repeat(1, self.future_steps) # If true, invalid agent
        reg_mask = ~torch.logical_or(data['padding_mask'][:, self.historical_steps:], current_true)
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y.unsqueeze(0), p=2, dim=-1) * reg_mask.unsqueeze(0)).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        self.minADE_inst_all.update(y_hat_best[..., :2], data.y, reg_mask)
        self.minFDE_inst_all.update(y_hat_best[..., :2], data.y, reg_mask)
        batch_size_ = (reg_mask.sum(-1)>0).sum()
        self.log('val_minADE_inst_all', self.minADE_inst_all, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        self.log('val_minFDE_inst_all', self.minFDE_inst_all, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size_)
        
        plot_x = data['positions'][:, :self.historical_steps]
        # plot_x = data['positions'][:, self.historical_steps-2:self.historical_steps]
        plot_y = data['positions'][:, self.historical_steps:]
        # plot_padding_mask = data['padding_mask'][:, self.historical_steps-2:]
        plot_padding_mask = data['padding_mask']
        plot_verts = verts
        plot_faces = faces
        # plot_verts = verts[:, self.historical_steps-2:]
        # plot_faces = faces[:, self.historical_steps-2:]
        plot_pose_mask = pose_mask
        # plot_pose_mask = pose_mask[:, self.historical_steps-2:]
        plot_save_name = str(batch_idx).zfill(5) + "-two_frame"
        two_frame_viz = VizScene(plot_x, plot_y, data['rotate_mat'], plot_padding_mask, y_hat_best[:, :, :2], plot_verts, plot_faces, plot_pose_mask, robot_position, initial_frame, save_root, plot_save_name, scene_name, y_hat_best_baseline=baseline_results_base['2f'][:,:,:2])
        # two_frame_viz = VizScene(plot_x, plot_y, data['rotate_mat'], plot_padding_mask, y_hat_best[:, :, :2], plot_verts, plot_faces, plot_pose_mask, robot_position, initial_frame, save_root, plot_save_name, scene_name, y_hat_best_baseline=baseline_result['2f'][:,:,:2])
        frames = [6]
        two_frame_viz.draw_frames(frames, PRINT_AGENT)

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
        param_dict = {param_name: param for param_name, param in self.named_parameters() if 'student' in param_name}    # Only update student

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay)) if param_name in param_dict.keys()],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay)) if param_name in param_dict.keys()],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('HiVT')
        parser.add_argument('--historical_steps', type=int, default=8)
        parser.add_argument('--future_steps', type=int, default=12)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--rotate', type=bool, default=True)
        parser.add_argument('--node_dim', type=int, default=2)
        parser.add_argument('--edge_dim', type=int, default=2)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_temporal_layers', type=int, default=4)
        parser.add_argument('--num_global_layers', type=int, default=3)
        parser.add_argument('--local_radius', type=float, default=50)
        parser.add_argument('--parallel', type=bool, default=False)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser

def build_encoder(encoder_type, historical_steps, node_dim, edge_dim, embed_dim, num_heads, dropout, num_temporal_layers, local_radius, parallel, emb_numHead, emb_dropout, emb_numLayers):
    if encoder_type == 'traj':
        local_encoder = LocalEncoder_onlyTraj(historical_steps=historical_steps,
                                        node_dim=node_dim,
                                        edge_dim=edge_dim,
                                        embed_dim=embed_dim,
                                        num_heads=num_heads,
                                        dropout=dropout,
                                        num_temporal_layers=num_temporal_layers,
                                        local_radius=local_radius,
                                        parallel=parallel)
    elif encoder_type == 'traj_v2':
        local_encoder = LocalEncoder_onlyTraj_v2(historical_steps=historical_steps,
                                        node_dim=node_dim,
                                        edge_dim=edge_dim,
                                        embed_dim=embed_dim,
                                        num_heads=num_heads,
                                        dropout=dropout,
                                        num_temporal_layers=num_temporal_layers,
                                        local_radius=local_radius,
                                        parallel=parallel,
                                        emb_numHead=emb_numHead,
                                        emb_dropout=emb_dropout,
                                        emb_numLayers=emb_numLayers)
    elif encoder_type == 'traj_v3':
        local_encoder = LocalEncoder_onlyTraj_v3(historical_steps=historical_steps,
                                        node_dim=node_dim,
                                        edge_dim=edge_dim,
                                        embed_dim=embed_dim,
                                        num_heads=num_heads,
                                        dropout=dropout,
                                        num_temporal_layers=num_temporal_layers,
                                        local_radius=local_radius,
                                        parallel=parallel,
                                        emb_numHead=emb_numHead,
                                        emb_dropout=emb_dropout,
                                        emb_numLayers=emb_numLayers)
    elif encoder_type == 'traj_pose':
        local_encoder = LocalEncoder_wPose(historical_steps=historical_steps,
                            node_dim=node_dim,
                            edge_dim=edge_dim,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            num_temporal_layers=num_temporal_layers,
                            local_radius=local_radius,
                            parallel=parallel,
                            emb_numHead=emb_numHead,
                            emb_dropout=emb_dropout,
                            emb_numLayers=emb_numLayers)
    elif encoder_type == 'traj_pose_joint':
        local_encoder = LocalEncoder_wPose_joint(historical_steps=historical_steps,
                            node_dim=node_dim,
                            edge_dim=edge_dim,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            num_temporal_layers=num_temporal_layers,
                            local_radius=local_radius,
                            parallel=parallel,
                            emb_numHead=emb_numHead,
                            emb_dropout=emb_dropout,
                            emb_numLayers=emb_numLayers)
    elif encoder_type == 'traj_pose_withAngle':
        local_encoder = LocalEncoder_wPose_withAngle(historical_steps=historical_steps,
                            node_dim=node_dim,
                            edge_dim=edge_dim,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            num_temporal_layers=num_temporal_layers,
                            local_radius=local_radius,
                            parallel=parallel,
                            emb_numHead=emb_numHead,
                            emb_dropout=emb_dropout,
                            emb_numLayers=emb_numLayers)
    elif encoder_type == 'traj_pose_woRotation':
        local_encoder = LocalEncoder_wPose_withoutRotation(historical_steps=historical_steps,
                            node_dim=node_dim,
                            edge_dim=edge_dim,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            num_temporal_layers=num_temporal_layers,
                            local_radius=local_radius,
                            parallel=parallel,
                            emb_numHead=emb_numHead,
                            emb_dropout=emb_dropout,
                            emb_numLayers=emb_numLayers)
    elif encoder_type == 'traj_pose_8steps':
        local_encoder = LocalEncoder_wPose_8steps(historical_steps=historical_steps,
                            node_dim=node_dim,
                            edge_dim=edge_dim,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            num_temporal_layers=num_temporal_layers,
                            local_radius=local_radius,
                            parallel=parallel)
    elif encoder_type == 'traj_pose_text':   # Main
        local_encoder = LocalEncoder_wPoseText_sit(historical_steps=historical_steps,
                            node_dim=node_dim,
                            edge_dim=edge_dim,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            num_temporal_layers=num_temporal_layers,
                            local_radius=local_radius,
                            parallel=parallel,
                            emb_numHead=emb_numHead,
                            emb_dropout=emb_dropout,
                            emb_numLayers=emb_numLayers)
    elif encoder_type == 'traj_pose_text_intra':   # Main
        local_encoder = LocalEncoder_wPoseText_intra(historical_steps=historical_steps,
                            node_dim=node_dim,
                            edge_dim=edge_dim,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            num_temporal_layers=num_temporal_layers,
                            local_radius=local_radius,
                            parallel=parallel,
                            emb_numHead=emb_numHead,
                            emb_dropout=emb_dropout,
                            emb_numLayers=emb_numLayers)
    elif encoder_type == 'traj_pose_text_intra_joint':   # Main
        local_encoder = LocalEncoder_wPoseText_intra_joint(historical_steps=historical_steps,
                            node_dim=node_dim,
                            edge_dim=edge_dim,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            num_temporal_layers=num_temporal_layers,
                            local_radius=local_radius,
                            parallel=parallel,
                            emb_numHead=emb_numHead,
                            emb_dropout=emb_dropout,
                            emb_numLayers=emb_numLayers)
    elif encoder_type == 'traj_pose_text_embedMask_teacher':
        local_encoder = LocalEncoder_wPoseText_embedMask_teacher(historical_steps=historical_steps,
                            node_dim=node_dim,
                            edge_dim=edge_dim,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            num_temporal_layers=num_temporal_layers,
                            local_radius=local_radius,
                            parallel=parallel,
                            emb_numHead=emb_numHead,
                            emb_dropout=emb_dropout,
                            emb_numLayers=emb_numLayers)
    elif encoder_type == 'traj_pose_text_v1':
        local_encoder = LocalEncoder_wPoseText_v1(historical_steps=historical_steps,
                            node_dim=node_dim,
                            edge_dim=edge_dim,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            num_temporal_layers=num_temporal_layers,
                            local_radius=local_radius,
                            parallel=parallel,
                            emb_numHead=emb_numHead,
                            emb_dropout=emb_dropout,
                            emb_numLayers=emb_numLayers)
    elif encoder_type == 'traj_pose_text_v2':
        local_encoder = LocalEncoder_wPoseText_v2(historical_steps=historical_steps,
                            node_dim=node_dim,
                            edge_dim=edge_dim,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            num_temporal_layers=num_temporal_layers,
                            local_radius=local_radius,
                            parallel=parallel,
                            emb_numHead=emb_numHead,
                            emb_dropout=emb_dropout,
                            emb_numLayers=emb_numLayers)
    elif encoder_type == 'traj_pose_text_v3':
        local_encoder = LocalEncoder_wPoseText_v3(historical_steps=historical_steps,
                            node_dim=node_dim,
                            edge_dim=edge_dim,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            num_temporal_layers=num_temporal_layers,
                            local_radius=local_radius,
                            parallel=parallel,
                            emb_numHead=emb_numHead,
                            emb_dropout=emb_dropout,
                            emb_numLayers=emb_numLayers)
    elif encoder_type == 'traj_text':
        local_encoder = LocalEncoder_wText(historical_steps=historical_steps,
                            node_dim=node_dim,
                            edge_dim=edge_dim,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            num_temporal_layers=num_temporal_layers,
                            local_radius=local_radius,
                            parallel=parallel,
                            emb_numHead=emb_numHead,
                            emb_dropout=emb_dropout,
                            emb_numLayers=emb_numLayers)
    elif encoder_type == 'traj_pose_text_onlyEgo':
        local_encoder = LocalEncoder_wPoseText_onlyEgo(historical_steps=historical_steps,
                            node_dim=node_dim,
                            edge_dim=edge_dim,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            num_temporal_layers=num_temporal_layers,
                            local_radius=local_radius,
                            parallel=parallel,
                            emb_numHead=emb_numHead,
                            emb_dropout=emb_dropout,
                            emb_numLayers=emb_numLayers)
    elif encoder_type == 'traj_pose_text_egoOther':
        local_encoder = LocalEncoder_wPoseText_egoOther(historical_steps=historical_steps,
                            node_dim=node_dim,
                            edge_dim=edge_dim,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            num_temporal_layers=num_temporal_layers,
                            local_radius=local_radius,
                            parallel=parallel,
                            emb_numHead=emb_numHead,
                            emb_dropout=emb_dropout,
                            emb_numLayers=emb_numLayers)
    elif encoder_type == 'traj_pose_text_egoOtherInteraction':
        local_encoder = LocalEncoder_wPoseText_egoOtherInteraction(historical_steps=historical_steps,
                            node_dim=node_dim,
                            edge_dim=edge_dim,
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            num_temporal_layers=num_temporal_layers,
                            local_radius=local_radius,
                            parallel=parallel,
                            emb_numHead=emb_numHead,
                            emb_dropout=emb_dropout,
                            emb_numLayers=emb_numLayers)
    return local_encoder

