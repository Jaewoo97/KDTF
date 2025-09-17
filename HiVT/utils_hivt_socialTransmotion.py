import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import argparse
from datetime import datetime

class AuxilliaryEncoderCMT(nn.TransformerEncoder):
    def __init__(self, encoder_layer_local, num_layers, norm=None):
        super(AuxilliaryEncoderCMT, self).__init__(encoder_layer=encoder_layer_local,
                                            num_layers=num_layers,
                                            norm=norm)

    def forward(self, src, mask=None, src_key_padding_mask=None, get_attn=False):
        output = src
        attn_matrices = []

        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
    

class AuxilliaryEncoderST(nn.TransformerEncoder):
    def __init__(self, encoder_layer_local, num_layers, norm=None):
        super(AuxilliaryEncoderST, self).__init__(encoder_layer=encoder_layer_local,
                                            num_layers=num_layers,
                                            norm=norm)

    def forward(self, src, mask=None, src_key_padding_mask=None, get_attn=False):
        output = src
        attn_matrices = []

        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        
        if self.norm is not None:
            output = self.norm(output)

        return output
    
class LearnedIDEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=21, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.person_encoding = nn.Embedding(1000, d_model, max_norm=True).to(device)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor, num_people=1) -> torch.Tensor:

        x = x + self.person_encoding(torch.arange(num_people).repeat_interleave(self.seq_len, dim=0).to(self.device)).unsqueeze(1)
        return self.dropout(x)
    

class LearnedTrajandIDEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=21, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model//2, max_norm=True).to(device)
        self.person_encoding = nn.Embedding(1000, d_model//2, max_norm=True).to(device)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor, num_people=1) -> torch.Tensor:
   
        half = x.size(3)//2 ## 124

        x[:,:,:,0:half*2:2] = x[:,:,:,0:half*2:2] + self.learned_encoding(torch.arange(self.seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)
        x[:,:,:,1:half*2:2] = x[:,:,:,1:half*2:2] + self.person_encoding(torch.arange(num_people).unsqueeze(0).repeat_interleave(self.seq_len, dim=0).to(self.device)).unsqueeze(0)


        return self.dropout(x)

class LearnedTrajandIDEncoding_v2(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=20, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model//2, max_norm=True).to(device)
        self.person_encoding = nn.Embedding(1000, d_model//2, max_norm=True).to(device)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor, seq_len, batch=None) -> torch.Tensor:
        _, counts = torch.unique_consecutive(batch, return_counts=True)
        batch_counts = torch.cat([torch.arange(c) for c in counts])

        half = x.size(2)//2 ## 124
        seq_len_ = torch.arange(self.seq_len).to(self.device)[-seq_len:]
        x[:,:,0:half*2:2] = x[:,:,0:half*2:2] + self.learned_encoding(seq_len_).unsqueeze(0)
        x[:,:,1:half*2:2] = x[:,:,1:half*2:2] + self.person_encoding(batch_counts.unsqueeze(1).repeat_interleave(seq_len, dim=1).to(self.device))
        # x[:,:,1:half*2:2] = x[:,:,1:half*2:2] + self.person_encoding(torch.arange(num_people).unsqueeze(0).repeat_interleave(self.seq_len, dim=0).to(self.device)).unsqueeze(0)


        return self.dropout(x)

class Learnedbb3dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=9, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model, max_norm=True).to(device)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
   
        x = x + self.learned_encoding(torch.arange(self.seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)
    
class Learnedbb2dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=9, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model, max_norm=True).to(device)
        self._seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      
        x = x + self.learned_encoding(torch.arange(self.seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)
    
class Learnedpose3dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=198, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(1)
        
        x = x + self.learned_encoding(torch.arange(seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)
    
class Learnedpose2dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=198, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(1)
        x = x + self.learned_encoding(torch.arange(seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)
    
def agent2batch(data):  # data is dict
    batch_size = data.batch.max()+1
    max_num_person = torch.bincount(data.batch).max()
    past_F, fut_F = data.x.shape[1], data.y.shape[1]
    joint = torch.zeros((batch_size, max_num_person, past_F+fut_F, 24, 2)) # B x N x F x 1 x 3
    traj = torch.index_select(data.x, 0, data.batch)
    
    batch_x = torch.zeros(batch_size, max_num_person, past_F, 2).cuda()
    batch_x_pos = torch.zeros(batch_size, max_num_person, past_F, 2).cuda()
    batch_y = torch.zeros(batch_size, max_num_person, fut_F, 2).cuda()
    batch_angle = torch.zeros(batch_size, max_num_person).cuda()
    batch_counts = torch.bincount(data.batch)
    padded_mask = torch.ones((batch_size, max_num_person, past_F+fut_F), dtype=torch.bool).cuda()
    batch_x_pose = torch.zeros(batch_size, max_num_person, past_F, 72).cuda()
    batch_y_pose = torch.zeros(batch_size, max_num_person, fut_F, 72).cuda()
    padded_mask_pose = torch.ones((batch_size, max_num_person, past_F+fut_F), dtype=torch.bool).cuda()
    
    mask = torch.zeros(batch_size, max_num_person, dtype=torch.bool).cuda()
    batch_ranges = torch.arange(max_num_person).unsqueeze(0).expand(batch_size, max_num_person).cuda()
    mask = (batch_ranges < batch_counts.unsqueeze(1))   # True is real data
    
    # 1.
    
    # 1. Ave by batch
    # local_pos = data.positions - (data.positions[~data.padding_mask]).mean(0)
    # batch_x_pos[mask] = local_pos[:, :past_F]
    
    # 2. Ave by scene
    padded_mask[mask] = data.padding_mask
    padded_mask_cur = padded_mask[:,:,:8].sum(-1)
    batch_x_pos[mask] = data.positions[:,:8]
    batch_x_pos_ = batch_x_pos.sum(-2).sum(1) / padded_mask_cur.sum(-1).unsqueeze(-1)
    batch_x_pos = batch_x_pos - batch_x_pos_.unsqueeze(1).unsqueeze(1)
    
    # 2.
    data.padding_mask[:, 1:] = data.padding_mask[:, 1:] | data.padding_mask[:, :-1]        # If using difference
    data.padding_mask[:, 0] = True
    batch_x[mask] = data.x[:, :past_F]
    batch_y[mask] = data.y
    
    batch_x_pose[mask] = data.x_pose
    batch_y_pose[mask] = data.y_pose
    padded_mask_pose[:, :, :past_F][mask] = data.x_pose_mask
    padded_mask_pose[:, :, past_F:][mask] = data.y_pose_mask
    batch_angle[mask] = data.rotate_angles
    
    return batch_x, batch_y, padded_mask, batch_x_pose, batch_y_pose, padded_mask_pose, mask, batch_angle, batch_x_pos

def agent2batch_ethucy(data):  # data is dict
    batch_size = data.batch.max()+1
    max_num_person = torch.bincount(data.batch).max()
    past_F, fut_F = data.x.shape[1], data.y.shape[1]
    joint = torch.zeros((batch_size, max_num_person, past_F+fut_F, 24, 2)) # B x N x F x 1 x 3
    traj = torch.index_select(data.x, 0, data.batch)
    
    batch_x = torch.zeros(batch_size, max_num_person, past_F, 2).cuda()
    batch_x_pos = torch.zeros(batch_size, max_num_person, past_F, 2).cuda()
    batch_y = torch.zeros(batch_size, max_num_person, fut_F, 2).cuda()
    batch_angle = torch.zeros(batch_size, max_num_person).cuda()
    batch_counts = torch.bincount(data.batch)
    padded_mask = torch.ones((batch_size, max_num_person, past_F+fut_F), dtype=torch.bool).cuda()
    batch_x_pose = torch.zeros(batch_size, max_num_person, past_F, 512).cuda()
    batch_y_pose = torch.zeros(batch_size, max_num_person, fut_F, 512).cuda()
    padded_mask_pose = torch.ones((batch_size, max_num_person, past_F+fut_F), dtype=torch.bool).cuda()
    
    mask = torch.zeros(batch_size, max_num_person, dtype=torch.bool).cuda()
    batch_ranges = torch.arange(max_num_person).unsqueeze(0).expand(batch_size, max_num_person).cuda()
    mask = (batch_ranges < batch_counts.unsqueeze(1))   # True is real data
    
    # 1.
    # robot_pos_batch = data.robot_pos[data.batch]
    # local_pos = data.positions - robot_pos_batch
    # batch_x[mask] = local_pos[:, :past_F]
    # batch_y[mask] = local_pos[:, past_F:]
    local_pos = data.positions
    batch_x_pos[mask] = local_pos[:, :past_F]
    
    # 2.
    data.padding_mask[:, 1:] = data.padding_mask[:, 1:] | data.padding_mask[:, :-1]        # If using difference
    data.padding_mask[:, 0] = True
    batch_x[mask] = data.x[:, :past_F]
    batch_y[mask] = data.y
    
    padded_mask[mask] = data.padding_mask
    batch_x_pose[mask] = data.x_pose
    batch_y_pose[mask] = data.y_pose
    padded_mask_pose[:, :, :past_F][mask] = data.x_pose_mask
    padded_mask_pose[:, :, past_F:][mask] = data.y_pose_mask
    batch_angle[mask] = data.rotate_angles
    
    return batch_x, batch_y, padded_mask, batch_x_pose, batch_y_pose, padded_mask_pose, mask, batch_angle, batch_x_pos


def agent2batch_embed(data, embed, padding_mask_indexed):  # data is dict
    B = data.batch.max()+1
    max_num_person = torch.bincount(data.batch).max()
    past_F, fut_F = data.x.shape[1], data.y.shape[1]
    
    F, BN, D = embed.shape
    batch_embed = torch.zeros((F*max_num_person.item(), B.item(), D)).cuda()
    batch_embed_mask = torch.zeros((F*max_num_person.item(), B.item()), dtype=torch.bool).cuda()
    batch_ranges = torch.arange(max_num_person).unsqueeze(0).expand(B, max_num_person).cuda()
    batch_counts = torch.bincount(data.batch)
    mask = (batch_ranges < batch_counts.unsqueeze(1)).permute(1,0).repeat(F,1)   # True is real data
    
    embed = embed.permute(1,0,2)        # into BN, F, D
    batch_embed[mask,:] = embed.reshape(-1, 128)
    batch_embed_mask[mask] = padding_mask_indexed.reshape(-1)
    
    return batch_embed, batch_embed_mask