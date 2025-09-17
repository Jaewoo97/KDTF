# V2: change encoding method: MLP => TRM

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
from utils_hivt_mart import *


class mart(pl.LightningModule):

    def __init__(self,
                 historical_steps: int,
                 future_steps: int,
                 num_modes: int,
                 rotate: bool,
                 hidden_dim: int,
                 decoder_hidden_dim: int,
                 model_dim: int,
                 dropout_mart: float,
                 num_head_mart: int,
                 num_layers: int,
                 aggregation: str,
                 function_type: int,
                 hyper_scales: list,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 modality = 'traj_pose_text', # modality of teacher
                 modality_student = 'traj',
                 onlyFullTimestep = True,
                 **kwargs) -> None:
        super(mart, self).__init__()
        self.save_hyperparameters()
        if 'jrdb' in kwargs['root']: self.datasetType = 'jrdb'
        elif 'sit' in kwargs['root']: self.datasetType = 'sit'
        elif 'ethucy' in kwargs['root']: self.datasetType = 'ethucy'
        else: raise Exception('Wrong dataset?')
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max
        self.max_epochs = kwargs['max_epochs']
        self.obs_and_pred = historical_steps + future_steps
        self.modality = modality
        self.modality_student = modality_student
        self.onlyFullTimestep = onlyFullTimestep
        print(f'onlyFullTimestep: {self.onlyFullTimestep}')
        
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
        self.check_valid = torch.tensor((0,0)).cuda()
        
        # mart part
        self.model_dim = model_dim
        self.num_layers = num_layers
        module_args = {
            'num_layers': num_layers,
            'num_heads': num_head_mart,
            'node_dim': model_dim,
            'node_hidden_dim': hidden_dim,
            'edge_dim': model_dim,
            'edge_hidden_dim_1': hidden_dim,
            'edge_hidden_dim_2': hidden_dim,
            'dropout': dropout_mart,
        }
        self.input_dim = 4
        
        # Teacher 
        if 'pose' in self.modality:
            self.pose_fc = nn.Sequential(
			nn.Linear(72, model_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(model_dim, model_dim),
			nn.Dropout(0.1))
        if 'text' in self.modality:
            self.text_fc = nn.Sequential(
			nn.Linear(312, model_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(model_dim, model_dim),
			nn.Dropout(0.1))
        self.input_fc = nn.Sequential(
			nn.Linear(self.input_dim, model_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(model_dim, model_dim),
			nn.Dropout(0.1))
        self.pos_encoder = PositionalAgentEncoding(model_dim, 0.1, concat=True)
        self.pair_encoders = nn.ModuleList()
        self.hyper_encoders = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.pair_encoders.append(RT(**module_args))
            else:
                self.pair_encoders.append(RTNoEdgeInit(**module_args))
        module_args['function_type'] = function_type
        for i in range(num_layers):
            if i == 0:
                self.hyper_encoders.append(HRT(**module_args))
            else:
                self.hyper_encoders.append(HRTNoEdgeInit(**module_args))
        if modality == 'traj':
            self.temporal_encoder = TemporalEncoder(historical_steps=self.historical_steps+1,
                                                embed_dim=model_dim,
                                                num_heads=4,
                                                dropout=0.1,
                                                num_layers=2)
        elif modality == 'traj_pose' or modality == 'traj_text':
            self.temporal_encoder = TemporalEncoder(historical_steps=(self.historical_steps*2)+1,
                                                embed_dim=model_dim,
                                                num_heads=4,
                                                dropout=0.1,
                                                num_layers=2)
        elif modality == 'traj_pose_text':
            self.temporal_encoder = TemporalEncoder(historical_steps=(self.historical_steps*3)+1,
                                                embed_dim=model_dim,
                                                num_heads=4,
                                                dropout=0.1,
                                                num_layers=2)
        else:
            raise Exception('Unsupported modality')
        self.angle_mlp = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_mart),
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout_mart))
        
        # Regularization 
        if 'pose' in self.modality:
            self.pose_fc_reg = nn.Sequential(
			nn.Linear(72, model_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(model_dim, model_dim),
			nn.Dropout(0.1))
        if 'text' in self.modality:
            self.text_fc_reg = nn.Sequential(
			nn.Linear(312, model_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(model_dim, model_dim),
			nn.Dropout(0.1))
        self.input_fc_reg = nn.Sequential(
			nn.Linear(self.input_dim, model_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(model_dim, model_dim),
			nn.Dropout(0.1))
        self.pos_encoder_reg = PositionalAgentEncoding(model_dim, 0.1, concat=True)
        self.pair_encoders_reg = nn.ModuleList()
        self.hyper_encoders_reg = nn.ModuleList()
        module_args = {
            'num_layers': num_layers,
            'num_heads': num_head_mart,
            'node_dim': model_dim,
            'node_hidden_dim': hidden_dim,
            'edge_dim': model_dim,
            'edge_hidden_dim_1': hidden_dim,
            'edge_hidden_dim_2': hidden_dim,
            'dropout': dropout_mart,
        }
        for i in range(num_layers):
            if i == 0:
                self.pair_encoders_reg.append(RT(**module_args))
            else:
                self.pair_encoders_reg.append(RTNoEdgeInit(**module_args))
        module_args['function_type'] = function_type
        for i in range(num_layers):
            if i == 0:
                self.hyper_encoders_reg.append(HRT(**module_args))
            else:
                self.hyper_encoders_reg.append(HRTNoEdgeInit(**module_args))
        if modality == 'traj':
            self.temporal_encoder_reg = TemporalEncoder(historical_steps=self.historical_steps+1,
                                                embed_dim=model_dim,
                                                num_heads=4,
                                                dropout=0.1,
                                                num_layers=2)
        elif modality == 'traj_pose' or modality == 'traj_text':
            self.temporal_encoder_reg = TemporalEncoder(historical_steps=(self.historical_steps*2)+1,
                                                embed_dim=model_dim,
                                                num_heads=4,
                                                dropout=0.1,
                                                num_layers=2)
        elif modality == 'traj_pose_text':
            self.temporal_encoder_reg = TemporalEncoder(historical_steps=(self.historical_steps*3)+1,
                                                embed_dim=model_dim,
                                                num_heads=4,
                                                dropout=0.1,
                                                num_layers=2)
        else:
            raise Exception('Unsupported modality')
        self.angle_mlp_reg = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_mart),
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout_mart))
        
        # Student
        if 'pose' in self.modality_student:
            self.pose_fc_student = nn.Sequential(
			nn.Linear(72, model_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(model_dim, model_dim),
			nn.Dropout(0.1))
        if 'text' in self.modality_student:
            self.text_fc_student = nn.Sequential(
			nn.Linear(312, model_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(model_dim, model_dim),
			nn.Dropout(0.1))
        self.input_fc_student = nn.Sequential(
			nn.Linear(self.input_dim, model_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(model_dim, model_dim),
			nn.Dropout(0.1))
        self.pos_encoder_student = PositionalAgentEncoding(model_dim, 0.1, concat=True)
        self.pair_encoders_student = nn.ModuleList()
        self.hyper_encoders_student = nn.ModuleList()
        module_args = {
            'num_layers': num_layers,
            'num_heads': num_head_mart,
            'node_dim': model_dim,
            'node_hidden_dim': hidden_dim,
            'edge_dim': model_dim,
            'edge_hidden_dim_1': hidden_dim,
            'edge_hidden_dim_2': hidden_dim,
            'dropout': dropout_mart,
        }
        for i in range(num_layers):
            if i == 0:
                self.pair_encoders_student.append(RT(**module_args))
            else:
                self.pair_encoders_student.append(RTNoEdgeInit(**module_args))
        module_args['function_type'] = function_type
        for i in range(num_layers):
            if i == 0:
                self.hyper_encoders_student.append(HRT(**module_args))
            else:
                self.hyper_encoders_student.append(HRTNoEdgeInit(**module_args))
        if modality_student == 'traj':
            self.temporal_encoder_student = TemporalEncoder(historical_steps=self.historical_steps+1,
                                                embed_dim=model_dim,
                                                num_heads=4,
                                                dropout=0.1,
                                                num_layers=2)
        elif modality_student == 'traj_pose' or modality_student == 'traj_text':
            self.temporal_encoder_student = TemporalEncoder(historical_steps=(self.historical_steps*2)+1,
                                                embed_dim=model_dim,
                                                num_heads=4,
                                                dropout=0.1,
                                                num_layers=2)
        elif modality_student == 'traj_pose_text':
            self.temporal_encoder_student = TemporalEncoder(historical_steps=(self.historical_steps*3)+1,
                                                embed_dim=model_dim,
                                                num_heads=4,
                                                dropout=0.1,
                                                num_layers=2)
        else:
            raise Exception('Unsupported modality')
        self.angle_mlp_student = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_mart),
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout_mart))

        # Only for student
        self.num_mode_mlp_student = nn.Linear(model_dim*3, model_dim * num_modes)
        self.decoder_student = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim, self.future_steps * 2))
        self.pi_student = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim, 1))
        # self.modality_encoder = MultipleInputEmbedding_mask_v3(in_channels=[node_dim, 72, 312], out_channel=embed_dim, emb_numHead=emb_numHead, emb_dropout=emb_dropout, emb_numLayers=emb_numLayers)
        # self.pi_time = nn.Linear(historical_steps*embed_dim, embed_dim)
        
    def freeze(self):
        if 'pose' in self.modality:
            for param in self.pose_fc.parameters():
                param.requires_grad = False
            for param in self.pose_fc_reg.parameters():
                param.requires_grad = False
        if 'text' in self.modality:
            for param in self.text_fc.parameters():
                param.requires_grad = False
            for param in self.text_fc_reg.parameters():
                param.requires_grad = False
        for param in self.input_fc.parameters():
            param.requires_grad = False
        for param in self.pos_encoder.parameters():
            param.requires_grad = False
        for param in self.pair_encoders.parameters():
            param.requires_grad = False
        for param in self.hyper_encoders.parameters():
            param.requires_grad = False
        for param in self.temporal_encoder.parameters():
            param.requires_grad = False
        for param in self.angle_mlp.parameters():
            param.requires_grad = False
        print("freezed!")
        
    def forward(self, data: TemporalData, given_frames, metamask=None):
        # DEBUG
        true_pos = data.positions[~data.padding_mask]
        wrong_data = true_pos == torch.tensor((0,0)).cuda()
        wrong_data = wrong_data.sum(-1) ==2
        if wrong_data.sum()>0:
            sdflkjsd=1
        
        b_x, b_y, padding_mask, b_x_p, b_y_p, pad_mask_p, batch_mask, b_angle, b_x_pos, b_x_t, pad_mask_t = agent2batch_mart(data)
        B, N, T_PAST, _ = b_x.shape
        inputs = []
        batch_size, num_agents, length, _ = b_x_pos.shape
        inputs.append(b_x)
        if self.input_dim > 2:
            inputs.append(b_x_pos)
        inputs = torch.cat(inputs, dim=-1)
        inputs = inputs.view(batch_size*num_agents, length, -1).contiguous()
        
        if self.modality == 'traj_pose_text':
            # Teacher part
            inputs_fc = self.input_fc(inputs).view(batch_size*num_agents, length, self.model_dim)
            inputs_pose = self.pose_fc(b_x_p).view(batch_size*num_agents, length, self.model_dim)
            inputs_text = self.text_fc(b_x_t).view(batch_size*num_agents, length, self.model_dim)
            angle_token = self.angle_mlp(b_angle.unsqueeze(-1))
            angle_token = angle_token.reshape(-1, angle_token.shape[-1])
            
            angle_token = angle_token.unsqueeze(0)
            fused_feat_ = torch.cat((inputs_fc.permute(1,0,2), inputs_pose.permute(1,0,2), inputs_text.permute(1,0,2), angle_token), dim=0)
            fused_mask_ = torch.cat((padding_mask[:,:,:self.historical_steps].reshape(-1, self.historical_steps), 
                                        pad_mask_p[:,:,:self.historical_steps].reshape(-1, self.historical_steps), 
                                        pad_mask_t[:,:,:self.historical_steps].reshape(-1, self.historical_steps), 
                                        torch.zeros((B*N, 1), dtype=torch.bool).cuda()), dim=-1)
            fused_feat_ = self.temporal_encoder(fused_feat_, fused_mask_.permute(1,0))
            fused_feat_ = torch.where(torch.isnan(fused_feat_), torch.zeros_like(fused_feat_).cuda(), fused_feat_)
            n_initial = fused_feat_.reshape(B, N, self.model_dim)
            n_pair, e_pair = n_initial, None
            n_group, e_group, G = n_initial, None, None        
            for i in range(self.num_layers):
                n_pair, e_pair = self.pair_encoders[i](n_pair, e_pair, batch_mask, return_edge=True)
                n_group, e_group, G = self.hyper_encoders[i](n_group, e_group, G, batch_mask, return_edge=True)        
            n_final = torch.cat([n_initial, n_pair, n_group], dim=-1)
            
            # Regularization part
            inputs_fc_reg = self.input_fc_reg(inputs).view(batch_size*num_agents, length, self.model_dim)
            inputs_pose_reg = self.pose_fc_reg(b_x_p).view(batch_size*num_agents, length, self.model_dim)
            inputs_text_reg = self.text_fc_reg(b_x_t).view(batch_size*num_agents, length, self.model_dim)
            angle_token_reg = self.angle_mlp_reg(b_angle.unsqueeze(-1))
            angle_token_reg = angle_token_reg.reshape(-1, angle_token_reg.shape[-1])
            
            angle_token_reg = angle_token_reg.unsqueeze(0)
            fused_feat__reg = torch.cat((inputs_fc_reg.permute(1,0,2), inputs_pose_reg.permute(1,0,2), inputs_text_reg.permute(1,0,2), angle_token_reg), dim=0)
            fused_mask__reg = torch.cat((padding_mask[:,:,:self.historical_steps].reshape(-1, self.historical_steps), 
                                        pad_mask_p[:,:,:self.historical_steps].reshape(-1, self.historical_steps), 
                                        pad_mask_t[:,:,:self.historical_steps].reshape(-1, self.historical_steps), 
                                        torch.zeros((B*N, 1), dtype=torch.bool).cuda()), dim=-1)
            fused_feat__reg = self.temporal_encoder_reg(fused_feat__reg, fused_mask__reg.permute(1,0))
            fused_feat__reg = torch.where(torch.isnan(fused_feat__reg), torch.zeros_like(fused_feat__reg).cuda(), fused_feat__reg)
            n_initial_reg = fused_feat__reg.reshape(B, N, self.model_dim)
            n_pair_reg, e_pair_reg = n_initial_reg, None
            n_group_reg, e_group_reg, G_reg = n_initial_reg, None, None        
            for i in range(self.num_layers):
                n_pair_reg, e_pair_reg = self.pair_encoders_reg[i](n_pair_reg, e_pair_reg, batch_mask, return_edge=True)
                n_group_reg, e_group_reg, G_reg = self.hyper_encoders_reg[i](n_group_reg, e_group_reg, G_reg, batch_mask, return_edge=True)        
            n_final_reg = torch.cat([n_initial_reg, n_pair_reg, n_group_reg], dim=-1)
        elif self.modality == 'traj_text':
            # Teacher part
            inputs_fc = self.input_fc(inputs).view(batch_size*num_agents, length, self.model_dim)
            inputs_text = self.text_fc(b_x_t).view(batch_size*num_agents, length, self.model_dim)
            angle_token = self.angle_mlp(b_angle.unsqueeze(-1))
            angle_token = angle_token.reshape(-1, angle_token.shape[-1])
            
            angle_token = angle_token.unsqueeze(0)
            fused_feat_ = torch.cat((inputs_fc.permute(1,0,2), inputs_text.permute(1,0,2), angle_token), dim=0)
            fused_mask_ = torch.cat((padding_mask[:,:,:self.historical_steps].reshape(-1, self.historical_steps), 
                                        pad_mask_t[:,:,:self.historical_steps].reshape(-1, self.historical_steps), 
                                        torch.zeros((B*N, 1), dtype=torch.bool).cuda()), dim=-1)
            fused_feat_ = self.temporal_encoder(fused_feat_, fused_mask_.permute(1,0))
            fused_feat_ = torch.where(torch.isnan(fused_feat_), torch.zeros_like(fused_feat_).cuda(), fused_feat_)
            n_initial = fused_feat_.reshape(B, N, self.model_dim)
            n_pair, e_pair = n_initial, None
            n_group, e_group, G = n_initial, None, None        
            for i in range(self.num_layers):
                n_pair, e_pair = self.pair_encoders[i](n_pair, e_pair, batch_mask, return_edge=True)
                n_group, e_group, G = self.hyper_encoders[i](n_group, e_group, G, batch_mask, return_edge=True)        
            n_final = torch.cat([n_initial, n_pair, n_group], dim=-1)
            
            # Regularization part
            inputs_fc_reg = self.input_fc_reg(inputs).view(batch_size*num_agents, length, self.model_dim)
            inputs_text_reg = self.text_fc_reg(b_x_t).view(batch_size*num_agents, length, self.model_dim)
            angle_token_reg = self.angle_mlp_reg(b_angle.unsqueeze(-1))
            angle_token_reg = angle_token_reg.reshape(-1, angle_token_reg.shape[-1])
            
            angle_token_reg = angle_token_reg.unsqueeze(0)
            fused_feat__reg = torch.cat((inputs_fc_reg.permute(1,0,2), inputs_text_reg.permute(1,0,2), angle_token_reg), dim=0)
            fused_mask__reg = torch.cat((padding_mask[:,:,:self.historical_steps].reshape(-1, self.historical_steps), 
                                        pad_mask_t[:,:,:self.historical_steps].reshape(-1, self.historical_steps), 
                                        torch.zeros((B*N, 1), dtype=torch.bool).cuda()), dim=-1)
            fused_feat__reg = self.temporal_encoder_reg(fused_feat__reg, fused_mask__reg.permute(1,0))
            fused_feat__reg = torch.where(torch.isnan(fused_feat__reg), torch.zeros_like(fused_feat__reg).cuda(), fused_feat__reg)
            n_initial_reg = fused_feat__reg.reshape(B, N, self.model_dim)
            n_pair_reg, e_pair_reg = n_initial_reg, None
            n_group_reg, e_group_reg, G_reg = n_initial_reg, None, None        
            for i in range(self.num_layers):
                n_pair_reg, e_pair_reg = self.pair_encoders_reg[i](n_pair_reg, e_pair_reg, batch_mask, return_edge=True)
                n_group_reg, e_group_reg, G_reg = self.hyper_encoders_reg[i](n_group_reg, e_group_reg, G_reg, batch_mask, return_edge=True)        
            n_final_reg = torch.cat([n_initial_reg, n_pair_reg, n_group_reg], dim=-1)
            
        if self.modality_student == 'traj_pose':
            
            inputs_fc_student = self.input_fc_student(inputs).view(batch_size*num_agents, length, self.model_dim)
            inputs_pose_student = self.pose_fc_student(b_x_p).view(batch_size*num_agents, length, self.model_dim)
            angle_token_student = self.angle_mlp_student(b_angle.unsqueeze(-1))
            angle_token_student = angle_token_student.reshape(-1, angle_token.shape[-1])
            
            fused_mask__student = torch.cat((padding_mask[:,:,:self.historical_steps].reshape(-1, self.historical_steps), pad_mask_p[:,:,:self.historical_steps].reshape(-1, self.historical_steps)), dim=-1)
            angle_token_student = angle_token_student.unsqueeze(0)
            fused_feat__student = torch.cat((inputs_fc_student.permute(1,0,2), inputs_pose_student.permute(1,0,2), angle_token_student), dim=0)
            fused_mask__student = torch.cat((fused_mask__student, torch.zeros((B*N, 1), dtype=torch.bool).cuda()), dim=-1)
            fused_feat__student = self.temporal_encoder_student(fused_feat__student, fused_mask__student.permute(1,0))
            fused_feat__student = torch.where(torch.isnan(fused_feat__student), torch.zeros_like(fused_feat__student).cuda(), fused_feat__student)
            
            n_initial_student = fused_feat__student.reshape(B, N, self.model_dim)
            
            n_pair_student, e_pair_student = n_initial_student, None
            n_group_student, e_group_student, G_student = n_initial_student, None, None
            
            # Add angle as additional input
            
            for i in range(self.num_layers):
                n_pair_student, e_pair_student = self.pair_encoders_student[i](n_pair_student, e_pair_student, batch_mask, return_edge=True)
                n_group_student, e_group_student, G_student = self.hyper_encoders_student[i](n_group_student, e_group_student, G_student, batch_mask, return_edge=True)
            
            n_final_student = torch.cat([n_initial_student, n_pair_student, n_group_student], dim=-1)
            
            # Unified decoding method
            out_final_feat_student = self.num_mode_mlp_student(n_final_student)
            B, N, _ = out_final_feat_student.shape
            out_final_feat_student = out_final_feat_student.reshape(B, N, self.num_modes, -1)
            out_pi_student = self.pi_student(out_final_feat_student).squeeze()
            out_final_student = self.decoder_student(out_final_feat_student.permute(2,0,1,3)).reshape(self.num_modes, B, N, self.future_steps, 2)
            
        elif self.modality_student == 'traj':
            
            inputs_fc_student = self.input_fc_student(inputs).view(batch_size*num_agents, length, self.model_dim)
            angle_token_student = self.angle_mlp_student(b_angle.unsqueeze(-1))
            angle_token_student = angle_token_student.reshape(-1, angle_token_student.shape[-1])
            
            fused_feat__student = inputs_fc_student.permute(1,0,2)
            fused_mask__student = padding_mask[:,:,:self.historical_steps].reshape(-1, self.historical_steps)
            angle_token_student = angle_token_student.unsqueeze(0)
            fused_feat__student = torch.cat((fused_feat__student, angle_token_student), dim=0)
            fused_mask__student = torch.cat((fused_mask__student, torch.zeros((B*N, 1), dtype=torch.bool).cuda()), dim=-1)
            fused_feat__student = self.temporal_encoder_student(fused_feat__student, fused_mask__student.permute(1,0))
            fused_feat__student = torch.where(torch.isnan(fused_feat__student), torch.zeros_like(fused_feat__student).cuda(), fused_feat__student)
            
            n_initial_student = fused_feat__student.reshape(B, N, self.model_dim)
            
            n_pair_student, e_pair_student = n_initial_student, None
            n_group_student, e_group_student, G_student = n_initial_student, None, None
            
            # Add angle as additional input
            
            for i in range(self.num_layers):
                n_pair_student, e_pair_student = self.pair_encoders_student[i](n_pair_student, e_pair_student, batch_mask, return_edge=True)
                n_group_student, e_group_student, G_student = self.hyper_encoders_student[i](n_group_student, e_group_student, G_student, batch_mask, return_edge=True)
            
            n_final_student = torch.cat([n_initial_student, n_pair_student, n_group_student], dim=-1)

            # Unified decoding method
            out_final_feat_student = self.num_mode_mlp_student(n_final_student)
            B, N, _ = out_final_feat_student.shape
            out_final_feat_student = out_final_feat_student.reshape(B, N, self.num_modes, -1)
            out_pi_student = self.pi_student(out_final_feat_student).squeeze()
            out_final_student = self.decoder_student(out_final_feat_student.permute(2,0,1,3)).reshape(self.num_modes, B, N, self.future_steps, 2)
            
        else: raise Exception("Student modality unsupported")
        
            
        return out_final_student[:, batch_mask], out_pi_student[batch_mask], n_final[batch_mask], n_final_student[batch_mask], n_final_reg[batch_mask]

    def training_step(self, data, batch_idx):
        inv_rot = data.rotate_mat[:, :2, :2].transpose(1, 2) 
        org_y = data.y
        # org_y = data.y @ inv_rot
        original_data = copy.deepcopy(data)
        # Full timesteps
        y_hat, pi, feat_teacher, feat_student, feat_reg = self(copy.deepcopy(original_data), 8, metamask=True)
        feat_teacher = feat_teacher.detach()
        feat_reg = feat_reg.detach()
        current_true = data.positions[:, self.historical_steps-1,:] == self.check_valid
        current_true = (current_true.sum(-1) == 2).unsqueeze(-1).repeat(1, self.future_steps) # If true, invalid agent
        reg_mask = ~torch.logical_or(data['padding_mask'][:, self.historical_steps:], current_true)
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
        
        local_embed_student, global_embed_student = feat_student[..., :self.model_dim], feat_student[..., self.model_dim:]
        local_embed_teacher, global_embed_teacher = feat_teacher[..., :self.model_dim], feat_teacher[..., self.model_dim:]
        local_embed_reg, global_embed_reg = feat_reg[..., :self.model_dim], feat_reg[..., self.model_dim:]
        
        local_agent_kd_loss = kl_divergence_loss(local_embed_student, local_embed_teacher)
        global_agent_kd_loss = kl_divergence_loss(global_embed_student, global_embed_teacher)
        # local_agent_kd_loss_reg = kl_divergence_loss(local_embed_student, local_embed_reg)
        # global_agent_kd_loss_reg = kl_divergence_loss(global_embed_student, global_embed_reg)
        
        if self.onlyFullTimestep:
            loss = reg_loss *3 + cls_loss + local_agent_kd_loss + global_agent_kd_loss
            reg_loss_1frames, cls_loss_1frames, reg_loss_2frames, cls_loss_2frames = 0, 0, 0, 0
            local_agent_kd_loss_1f, global_agent_kd_loss_1f, local_agent_kd_loss_reg_1f, global_agent_kd_loss_reg_1f, local_agent_kd_loss_2f, global_agent_kd_loss_2f, local_agent_kd_loss_reg_2f, global_agent_kd_loss_reg_2f = 0,0,0,0,0,0,0,0            
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
            y_hat, pi, feat_teacher_2f, feat_student_2f, feat_reg_2f = self(data, 2, metamask=True)
            feat_teacher_2f = feat_teacher_2f.detach()
            feat_reg_2f = feat_reg_2f.detach()
            current_true = data.positions[:, self.historical_steps-1,:] == self.check_valid
            current_true = (current_true.sum(-1) == 2).unsqueeze(-1).repeat(1, self.future_steps) # If true, invalid agent
            reg_mask = ~torch.logical_or(data['padding_mask'][:, self.historical_steps:], current_true)
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
            y_hat, pi, feat_teacher_1f, feat_student_1f, feat_reg_1f = self(data, 1, metamask=True)
            feat_teacher_1f = feat_teacher_1f.detach()
            feat_reg_1f = feat_reg_1f.detach()
            current_true = data.positions[:, self.historical_steps-1,:] == self.check_valid
            current_true = (current_true.sum(-1) == 2).unsqueeze(-1).repeat(1, self.future_steps) # If true, invalid agent
            reg_mask = ~torch.logical_or(data['padding_mask'][:, self.historical_steps:], current_true)
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
            
            
            # loss
            local_embed_student_1f, global_embed_student_1f = feat_student_1f[..., :self.model_dim], feat_student_1f[..., self.model_dim:]
            local_embed_teacher_1f, global_embed_teacher_1f = feat_teacher_1f[..., :self.model_dim], feat_teacher_1f[..., self.model_dim:]
            local_embed_reg_1f, global_embed_reg_1f = feat_reg_1f[..., :self.model_dim], feat_reg_1f[..., self.model_dim:]
            
            local_agent_kd_loss_1f = kl_divergence_loss(local_embed_student_1f, local_embed_teacher_1f)
            global_agent_kd_loss_1f = kl_divergence_loss(global_embed_student_1f, global_embed_teacher_1f)
            # local_agent_kd_loss_reg_1f = kl_divergence_loss(local_embed_student_1f, local_embed_reg_1f)
            # global_agent_kd_loss_reg_1f = kl_divergence_loss(global_embed_student_1f, global_embed_reg_1f)
            
            local_embed_student_2f, global_embed_student_2f = feat_student_2f[..., :self.model_dim], feat_student_2f[..., self.model_dim:]
            local_embed_teacher_2f, global_embed_teacher_2f = feat_teacher_2f[..., :self.model_dim], feat_teacher_2f[..., self.model_dim:]
            local_embed_reg_2f, global_embed_reg_2f = feat_reg_2f[..., :self.model_dim], feat_reg_2f[..., self.model_dim:]
            
            local_agent_kd_loss_2f = kl_divergence_loss(local_embed_student_2f, local_embed_teacher_2f)
            global_agent_kd_loss_2f = kl_divergence_loss(global_embed_student_2f, global_embed_teacher_2f)
            # local_agent_kd_loss_reg_2f = kl_divergence_loss(local_embed_student_2f, local_embed_reg_2f)
            # global_agent_kd_loss_reg_2f = kl_divergence_loss(global_embed_student_2f, global_embed_reg_2f)
            
            loss = reg_loss *3 + cls_loss + reg_loss_1frames + reg_loss_2frames + \
                local_agent_kd_loss + global_agent_kd_loss + local_agent_kd_loss_1f + global_agent_kd_loss_1f + local_agent_kd_loss_2f + global_agent_kd_loss_2f
                
                
        self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss', cls_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss_1frames', reg_loss_1frames, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss_1frames', cls_loss_1frames, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss_2frames', reg_loss_2frames, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss_2frames', cls_loss_2frames, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        
        self.log('local_agent_kd_loss', local_agent_kd_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('global_agent_kd_loss', global_agent_kd_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        # self.log('local_agent_kd_loss_reg', local_agent_kd_loss_reg, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        # self.log('global_agent_kd_loss_reg', global_agent_kd_loss_reg, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, data, batch_idx):
        inv_rot = data.rotate_mat[:, :2, :2].transpose(1, 2) 
        org_y = data.y
        # org_y = data.y @ inv_rot
        data_original = copy.deepcopy(data)
        y_hat, pi, _, _, _ = self(data, 8)
        current_true = data.positions[:, self.historical_steps-1,:] == self.check_valid
        current_true = (current_true.sum(-1) == 2).unsqueeze(-1).repeat(1, self.future_steps) # If true, invalid agent
        reg_mask = ~torch.logical_or(data['padding_mask'][:, self.historical_steps:], current_true)
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
        y_hat, pi, _, _, _ = self(data, 1)
        
        current_true = data.positions[:, self.historical_steps-1,:] == self.check_valid
        current_true = (current_true.sum(-1) == 2).unsqueeze(-1).repeat(1, self.future_steps) # If true, invalid agent
        reg_mask = ~torch.logical_or(data['padding_mask'][:, self.historical_steps:], current_true)
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
        y_hat, pi, _, _, _ = self(data, 2)
        
        current_true = data.positions[:, self.historical_steps-1,:] == self.check_valid
        current_true = (current_true.sum(-1) == 2).unsqueeze(-1).repeat(1, self.future_steps) # If true, invalid agent
        reg_mask = ~torch.logical_or(data['padding_mask'][:, self.historical_steps:], current_true)
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
        y_hat, pi, _, _, _ = self(data, 2)
        
        current_true = data.positions[:, self.historical_steps-1,:] == self.check_valid
        current_true = (current_true.sum(-1) == 2).unsqueeze(-1).repeat(1, self.future_steps) # If true, invalid agent
        reg_mask = ~torch.logical_or(data['padding_mask'][:, self.historical_steps:], current_true)
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
        y_hat, pi, _, _, _ = self(data, 8)
        
        current_true = data.positions[:, self.historical_steps-1,:] == self.check_valid
        current_true = (current_true.sum(-1) == 2).unsqueeze(-1).repeat(1, self.future_steps) # If true, invalid agent
        reg_mask = ~torch.logical_or(data['padding_mask'][:, self.historical_steps:], current_true)
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
        param_dict = {param_name: param for param_name, param in self.named_parameters() if 'student' in param_name}
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
        parser.add_argument('--lr', type=float, default=0.0012765545541836703)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser
