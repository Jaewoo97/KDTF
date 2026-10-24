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
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation as R_
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph

from models import MultipleInputEmbedding, SingleInputEmbedding, TrajPoseEmbedding, MultipleInputEmbedding_mask, MultipleInputEmbedding_mask_v3, MultipleInputEmbedding_mask_v2
from utils_hivt import DistanceDropEdge
from utils_hivt import TemporalData
from utils_hivt import init_weights


class LocalEncoder_wPoseText_woRotation(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_temporal_layers: int = 4,
                 local_radius: float = 50,
                 parallel: bool = False,
                 emb_numHead = None,
                 emb_dropout = None,
                 emb_numLayers = None) -> None:
        super(LocalEncoder_wPoseText_woRotation, self).__init__()
        self.historical_steps = historical_steps
        self.parallel = parallel
        print('LocalEncoder_wPoseText_woRotation')

        self.drop_edge = DistanceDropEdge()
        # self.drop_edge = DistanceDropEdge(local_radius)
        self.aa_encoder = AAEncoder(historical_steps=historical_steps,
                                    node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout,
                                    parallel=parallel,
                                    emb_numHead=emb_numHead,
                                    emb_dropout=emb_dropout,
                                    emb_numLayers=emb_numLayers)
        self.temporal_encoder = TemporalEncoder(historical_steps=historical_steps,
                                                embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                num_layers=num_temporal_layers)
        

    def forward(self, data: TemporalData) -> torch.Tensor:
        for t in range(1, self.historical_steps):
            data[f'edge_index_{t}'], _, edge_mask = subgraph(subset=~data['padding_mask'][:, t], edge_index=data.edge_index, return_edge_mask=True)
            data[f'edge_attr_{t}'] = \
                data['positions'][data[f'edge_index_{t}'][0], t] - data['positions'][data[f'edge_index_{t}'][1], t]
            data[f'edge_attr_text_{t}'] = data['x_interaction'][edge_mask, t]
            data[f'edge_attr_text_mask_{t}'] = data['x_interaction_mask'][edge_mask, t]
        out = [None] * (self.historical_steps -1)
        for t in range(1, self.historical_steps):
            edge_index, edge_attr = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_{t}'])
            _, edge_attr_text = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_text_{t}'])
            _, edge_attr_text_mask = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_text_mask_{t}'])
            out[t-1] = self.aa_encoder(x=data.x[:, t], x_pose=data.x_pose[:, t], x_pose_mask=data.x_pose_mask[:, t], x_text=data.x_text[:, t], x_text_mask=data.x_text_mask[:, t], 
                                       rotate_angles=data.rotate_angles, t=t, edge_index=edge_index, edge_attr=edge_attr, edge_attr_text=edge_attr_text, edge_attr_text_mask=edge_attr_text_mask, bos_mask=data['bos_mask'][:, t], rotate_mat=data['rotate_mat'])
        out_stacked = torch.stack(out)  # [T, N, D]
        out = self.temporal_encoder(x=out_stacked, padding_mask=data['padding_mask'][:,1 : self.historical_steps])
        # edge_index, edge_attr = self.drop_edge(data['lane_actor_index'], data['lane_actor_vectors'])
        # out = self.al_encoder(x=(data['lane_vectors'], out), edge_index=edge_index, edge_attr=edge_attr,
        #                       is_intersections=data['is_intersections'], turn_directions=data['turn_directions'],
        #                       traffic_controls=data['traffic_controls'], rotate_mat=data['rotate_mat'])
        return out, out_stacked


class AAEncoder(MessagePassing):

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 parallel: bool = False,
                 emb_numHead = None,
                 emb_dropout = None,
                 emb_numLayers = None,
                 **kwargs) -> None:
        super(AAEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.parallel = parallel

        self.center_embed = MultipleInputEmbedding_mask_v3(in_channels=[node_dim, 72, 312], out_channel=embed_dim, emb_numHead=emb_numHead, emb_dropout=emb_dropout, emb_numLayers=emb_numLayers)
        self.nbr_embed = MultipleInputEmbedding_mask_v3(in_channels=[node_dim, edge_dim, 72, 312], out_channel=embed_dim, emb_numHead=emb_numHead, emb_dropout=emb_dropout, emb_numLayers=emb_numLayers)
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        self.bos_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))
        self.pose_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))
        self.neigh_x_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))
        self.text_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))
        self.interaction_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))
        self.edge_rotated_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))
        nn.init.normal_(self.bos_token, mean=0., std=.02)
        nn.init.normal_(self.pose_token, mean=0., std=.02)
        nn.init.normal_(self.neigh_x_token, mean=0., std=.02)
        nn.init.normal_(self.text_token, mean=0., std=.02)
        nn.init.normal_(self.interaction_token, mean=0., std=.02)
        nn.init.normal_(self.edge_rotated_token, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                x_pose: torch.Tensor,
                x_pose_mask: torch.Tensor,
                x_text: torch.Tensor,
                x_text_mask: torch.Tensor,
                t: Optional[int],
                edge_index: Adj,
                edge_attr: torch.Tensor,
                edge_attr_text: torch.Tensor,
                edge_attr_text_mask: torch.Tensor,
                bos_mask: torch.Tensor,
                rotate_angles: torch.Tensor,
                rotate_mat: Optional[torch.Tensor] = None,
                size: Size = None) -> torch.Tensor:
        if rotate_mat is None:
            raise Exception('Not implemented')
            center_embed = self.center_embed(x)
        else:
            rotated_x = torch.bmm(x.unsqueeze(-2), rotate_mat)
            if len(rotate_angles) != 0:
                rotated_pose = rotate_root(x_pose, 0, 'z')
            else:
                rotated_pose = x_pose
                sdd=1
            center_embed = self.center_embed([rotated_x, rotated_pose, x_text], [bos_mask, x_pose_mask, x_text_mask], 
                                             [self.bos_token[t], self.pose_token[t], self.text_token[t]]).squeeze(-2)
        # center_embed = torch.where(bos_mask.unsqueeze(-1), self.bos_token[t], center_embed)
        x_mask = torch.logical_and(x[..., 0] == 0, x[..., 1] == 0 )
        self.t = t
        center_embed = center_embed + self._mha_block(self.norm1(center_embed), x, x_mask, x_pose, x_pose_mask, edge_index,
                                                      edge_attr, edge_attr_text, edge_attr_text_mask, rotate_mat, rotate_angles, size)
        center_embed = center_embed + self._ff_block(self.norm2(center_embed))
        return center_embed

    def message(self,
                edge_index: Adj,
                center_embed_i: torch.Tensor,
                x_j: torch.Tensor,
                x_pose_j: torch.Tensor,
                x_mask: torch.Tensor,
                x_pose_mask: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_attr_text: torch.Tensor,
                edge_attr_text_mask: torch.Tensor,
                rotate_mat: Optional[torch.Tensor],
                rotate_angles: torch.Tensor,
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int], 
                ) -> torch.Tensor:
        if rotate_mat is None:
            raise Exception('Not implemented yet')
            nbr_embed = self.nbr_embed([x_j, edge_attr])
        else:
            # print(rotate_angles.shape)
            if self.parallel:
                center_rotate_mat = rotate_mat.repeat(self.historical_steps, 1, 1)[edge_index[1]]
            else:
                center_rotate_mat = rotate_mat[edge_index[1]]
                center_rotate_angle = rotate_angles[edge_index[1]]
            x_j_rotated = torch.bmm(x_j.unsqueeze(-2), center_rotate_mat).squeeze(-2)
            edge_attr_rotated = torch.bmm(edge_attr.unsqueeze(-2), center_rotate_mat).squeeze(-2)
            if len(center_rotate_angle) != 0:
                x_pose_j_rotated = rotate_root(x_pose_j, 0, 'z')
            else:
                x_pose_j_rotated = x_pose_j
                dsf=1
            # x_j_rotated = torch.where(x_mask[edge_index[1]].unsqueeze(-1), self.neigh_x_token[self.t], x_j_rotated)
            nbr_embed = self.nbr_embed([x_j_rotated, edge_attr_rotated, x_pose_j_rotated, edge_attr_text], 
                                       [x_mask[edge_index[1]], torch.zeros(edge_attr_rotated.shape[:-1], dtype=torch.bool, device='cuda'), x_pose_mask[edge_index[1]], edge_attr_text_mask], 
                                       [self.neigh_x_token[self.t], self.edge_rotated_token[self.t], self.pose_token[self.t], self.interaction_token[self.t]])
        
        query = self.lin_q(center_embed_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               center_embed: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(center_embed))
        return inputs + gate * (self.lin_self(center_embed) - inputs)

    def _mha_block(self,
                   center_embed: torch.Tensor,
                   x: torch.Tensor,
                   x_mask: torch.Tensor,
                   x_pose: torch.Tensor,
                   x_pose_mask: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   edge_attr_text: torch.Tensor,
                   edge_attr_text_mask: torch.Tensor,
                   rotate_mat: Optional[torch.Tensor],
                   rotate_angles: torch.Tensor,
                   size: Size,) -> torch.Tensor:
        center_embed = self.out_proj(self.propagate(edge_index=edge_index, x=x, x_mask=x_mask, x_pose=x_pose, x_pose_mask=x_pose_mask, 
                                                    center_embed=center_embed, edge_attr=edge_attr, edge_attr_text=edge_attr_text, edge_attr_text_mask=edge_attr_text_mask,
                                                    rotate_mat=rotate_mat, rotate_angles=rotate_angles, size=size,))
        return self.proj_drop(center_embed)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TemporalEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoder, self).__init__()
        historical_steps -= 1
        encoder_layer = TemporalEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers,
                                                         norm=nn.LayerNorm(embed_dim))
        self.padding_token = nn.Parameter(torch.Tensor(historical_steps, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))
        attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)
        self.register_buffer('attn_mask', attn_mask)
        nn.init.normal_(self.padding_token, mean=0., std=.02)
        nn.init.normal_(self.cls_token, mean=0., std=.02)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        x = torch.where(padding_mask.t().unsqueeze(-1), self.padding_token, x)
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat((x, expand_cls_token), dim=0)
        x = x + self.pos_embed
        out = self.transformer_encoder(src=x, mask=self.attn_mask, src_key_padding_mask=None)
        return out[-1]  # [N, D]

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TemporalEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                is_causal: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self,
                  x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return self.dropout2(x)


def rotate_root(pose, rotation_angle_degrees, axis):
    # Convert the angle to radians
    if torch.is_tensor(rotation_angle_degrees): 
        if rotation_angle_degrees.device.type=='cuda': rotation_angle_degrees = rotation_angle_degrees.cpu()
    if torch.is_tensor(pose): pose = pose.cpu().detach().numpy()
    pose = pose.squeeze()
    # rotation_angle_radians = np.radians(rotation_angle_degrees)
    rotation_angle_radians = rotation_angle_degrees     # original: +
    rotation_matrix = R_.from_euler(axis, rotation_angle_radians).as_matrix()
    root_rotation_matrix = R_.from_rotvec(pose[:, :3]).as_matrix()
    new_root_rotation_matrix = rotation_matrix @ root_rotation_matrix
    new_root_rotation_vector = R_.from_matrix(new_root_rotation_matrix).as_rotvec()
    new_pose = np.copy(pose)
    new_pose[:, :3] = new_root_rotation_vector
    return torch.tensor(new_pose, device='cuda')
    # return np.expand_dims(new_pose, 0)