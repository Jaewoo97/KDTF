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
from typing import List, Optional

import torch
import torch.nn as nn

from utils_hivt import init_weights
import torch.nn.functional as F


class SingleInputEmbedding(nn.Module):

    def __init__(self,
                 in_channel: int,
                 out_channel: int) -> None:
        super(SingleInputEmbedding, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)

class TrajPoseEmbedding(nn.Module):

    def __init__(self,
                 in_channel_traj: int,
                 in_channel_pose: int,
                 out_channel: int) -> None:
        super(TrajPoseEmbedding, self).__init__()
        self.embedTraj = nn.Sequential(
            nn.Linear(in_channel_traj, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.embedPose = nn.Sequential(
            nn.Linear(in_channel_pose, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.fuseLayer = nn.Sequential(
            nn.Linear(out_channel*2, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, x_pose: torch.Tensor, bos_token: torch.Tensor, pose_token: torch.Tensor, bos_mask: torch.Tensor, x_pose_mask: torch.Tensor) -> torch.Tensor:
        embedTraj = self.embedTraj(x.squeeze())
        embedPose = self.embedPose(x_pose.squeeze())
        embedTraj = torch.where(bos_mask.unsqueeze(-1), bos_token, embedTraj)
        embedPose = torch.where(x_pose_mask.unsqueeze(-1), pose_token, embedPose)
        return self.fuseLayer(torch.cat((embedTraj, embedPose), dim=-1))

class MultipleInputEmbedding(nn.Module):

    def __init__(self,
                 in_channels: List[int],
                 out_channel: int) -> None:
        super(MultipleInputEmbedding, self).__init__()
        self.module_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channel, out_channel),
                           nn.LayerNorm(out_channel),
                           nn.ReLU(inplace=True),
                           nn.Linear(out_channel, out_channel))
             for in_channel in in_channels])
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self,
                continuous_inputs: List[torch.Tensor],
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i])
        output = torch.stack(continuous_inputs).sum(dim=0)
        if categorical_inputs is not None:
            output += torch.stack(categorical_inputs).sum(dim=0)
        return self.aggr_embed(output)

class MultipleInputEmbedding_mask(nn.Module):

    def __init__(self,
                 in_channels: List[int],
                 out_channel: int) -> None:
        super(MultipleInputEmbedding_mask, self).__init__()
        self.module_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channel, out_channel),
                           nn.LayerNorm(out_channel),
                           nn.ReLU(inplace=True),
                           nn.Linear(out_channel, out_channel))
             for in_channel in in_channels])
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self,
                continuous_inputs: List[torch.Tensor],
                continuous_inputs_mask: List[torch.Tensor],
                continuous_inputs_token: List[torch.Tensor],
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i].squeeze())
            if continuous_inputs_mask[i] is not None:
                continuous_inputs[i] = torch.where(continuous_inputs_mask[i].unsqueeze(-1), continuous_inputs_token[i], continuous_inputs[i])
        output = torch.stack(continuous_inputs).sum(dim=0)
        if categorical_inputs is not None:
            output += torch.stack(categorical_inputs).sum(dim=0)
        return self.aggr_embed(output)
    import torch
import torch.nn as nn
from typing import List, Optional

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

class MultipleInputEmbedding_mask_v2(nn.Module):

    def __init__(self,
                 in_channels: List[int],
                 out_channel: int) -> None:
        super(MultipleInputEmbedding_mask_v2, self).__init__()
        self.module_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channel, out_channel),
                           nn.LayerNorm(out_channel),
                           nn.ReLU(inplace=True),
                           nn.Linear(out_channel, out_channel))
             for in_channel in in_channels])
        
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel)
        )
        
        self.continuous_attention = nn.MultiheadAttention(embed_dim=out_channel, num_heads=4, batch_first=True)
        
        self.apply(init_weights)

    def forward(self,
                continuous_inputs: List[torch.Tensor],
                continuous_inputs_mask: List[torch.Tensor],
                continuous_inputs_token: List[torch.Tensor],
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        
        # Process continuous inputs with masking
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i].squeeze())
            if continuous_inputs_mask[i] is not None:
                continuous_inputs[i] = torch.where(continuous_inputs_mask[i].unsqueeze(-1), continuous_inputs_token[i], continuous_inputs[i])
        
        continuous_inputs = torch.stack(continuous_inputs)  # Shape: (num_inputs, batch_size, out_channel)
        
        # Apply attention to continuous inputs to handle sparsity
        key_padding_mask = torch.stack(continuous_inputs_mask)
        continuous_inputs = continuous_inputs.permute(1, 0, 2)  # Shape: (S, N, E) -> (batch_size, num_inputs, out_channel)
        key_padding_mask = key_padding_mask.permute(1,0)
        attn_output, _ = self.continuous_attention(continuous_inputs, continuous_inputs, continuous_inputs, key_padding_mask=key_padding_mask.bool())   # default: no ~ on key_padding_mask
        
        # Sum the attention output and apply aggregation
        attn_output[key_padding_mask, :] = 0
        # attn_output = attn_output * ~key_padding_mask.permute(1,0).unsqueeze(-1)
        output = attn_output.sum(dim=1)
        valid_count = (~key_padding_mask).sum(1)
        output = output/valid_count.unsqueeze(-1)
        
        return self.aggr_embed(output)

class MultipleInputEmbedding_mask_v3(nn.Module):

    def __init__(self,
                 in_channels: List[int],
                 out_channel: int,
                 emb_numHead = 4,
                 emb_dropout = 0.5,
                 emb_numLayers = 2) -> None:
        super(MultipleInputEmbedding_mask_v3, self).__init__()
        self.module_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channel, out_channel),
                           nn.LayerNorm(out_channel),
                           nn.ReLU(inplace=True),
                           nn.Linear(out_channel, out_channel))
             for in_channel in in_channels])
        
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel)
        )
        
        # self.continuous_attention = nn.MultiheadAttention(embed_dim=out_channel, num_heads=1)
        self.temporal_encoder = TemporalEncoder(historical_steps=len(in_channels),
                                                embed_dim=out_channel,
                                                num_heads=emb_numHead,
                                                dropout=emb_dropout,
                                                num_layers=emb_numLayers)
        print(f'MultipleInputEmbedding_mask_v3 / numhead: {emb_numHead}, dropout: {emb_dropout}, num_layers: {emb_numLayers}')
        
        self.apply(init_weights)

    def forward(self,
                continuous_inputs: List[torch.Tensor],
                continuous_inputs_mask: List[torch.Tensor],
                continuous_inputs_token: List[torch.Tensor] = None,
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        
        # Process continuous inputs with masking
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i].squeeze())
            # if continuous_inputs_mask[i] is not None:
            #     continuous_inputs[i] = torch.where(continuous_inputs_mask[i].unsqueeze(-1), continuous_inputs_token[i], continuous_inputs[i])
        
        continuous_inputs = torch.stack(continuous_inputs)  # Shape: (num_inputs, batch_size, out_channel)
        key_padding_mask = torch.stack(continuous_inputs_mask)
        if continuous_inputs.shape[1] == 0:
            return self.aggr_embed(continuous_inputs[-1])
        continuous_inputs = self.temporal_encoder(continuous_inputs, key_padding_mask)     # Default: no ~ on key_padding_mask
        # attn_output, _ = self.continuous_attention(continuous_inputs, continuous_inputs, continuous_inputs, key_padding_mask=~key_padding_mask.bool())
        
        return self.aggr_embed(continuous_inputs)


class MultipleInputEmbedding_mask_v4(nn.Module):
    # V4: add embedding mask from v3
    def __init__(self,
                 in_channels: List[int],
                 out_channel: int,
                 emb_numHead = None,
                 emb_dropout = None,
                 emb_numLayers = None) -> None:
        super(MultipleInputEmbedding_mask_v4, self).__init__()
        self.module_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channel, out_channel),
                           nn.LayerNorm(out_channel),
                           nn.ReLU(inplace=True),
                           nn.Linear(out_channel, out_channel))
             for in_channel in in_channels])
        
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel)
        )
        
        # self.continuous_attention = nn.MultiheadAttention(embed_dim=out_channel, num_heads=1)
        self.temporal_encoder = TemporalEncoder(historical_steps=len(in_channels),
                                                embed_dim=out_channel,
                                                num_heads=emb_numHead,
                                                dropout=emb_dropout,
                                                num_layers=emb_numLayers)
        self.embed_dim = out_channel
        self.modality_token = nn.Parameter(torch.Tensor(len(in_channels), out_channel))
        nn.init.normal_(self.modality_token, mean=0., std=.02)
        print(f'MultipleInputEmbedding_mask_v4 / numhead: {emb_numHead}, dropout: {emb_dropout}, num_layers: {emb_numLayers}')
        
        self.apply(init_weights)

    def forward(self,
                continuous_inputs: List[torch.Tensor],
                continuous_inputs_mask: List[torch.Tensor],
                mask_ratio: torch.Tensor,
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        
        # Process continuous inputs with masking
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i].squeeze())
            # if continuous_inputs_mask[i] is not None:
            #     continuous_inputs[i] = torch.where(continuous_inputs_mask[i].unsqueeze(-1), continuous_inputs_token[i], continuous_inputs[i])
        continuous_inputs = torch.stack(continuous_inputs)  # Shape: (num_inputs, batch_size, out_channel)
        key_padding_mask = torch.stack(continuous_inputs_mask)
        
        embed_mask = (torch.rand(len(continuous_inputs), continuous_inputs.shape[1], self.embed_dim) > mask_ratio).to('cuda')  # Mask shape: [batch_size, embedding_dim], True is masking
        modality_token = self.modality_token.unsqueeze(1).repeat(1, continuous_inputs.shape[1], 1)
        continuous_inputs = torch.where(embed_mask, modality_token, continuous_inputs)        
        
        if continuous_inputs.shape[1] == 0:
            return self.aggr_embed(continuous_inputs[-1])
        continuous_inputs = self.temporal_encoder(continuous_inputs, key_padding_mask)     # Default: no ~ on key_padding_mask
        # attn_output, _ = self.continuous_attention(continuous_inputs, continuous_inputs, continuous_inputs, key_padding_mask=~key_padding_mask.bool())
        
        return self.aggr_embed(continuous_inputs)
    
class MultipleInputEmbedding_mask_v5(nn.Module):
    # V4: add embedding mask from v3
    # V5: modality-specific mask
    def __init__(self,
                 in_channels: List[int],
                 out_channel: int,
                 emb_numHead = None,
                 emb_dropout = None,
                 emb_numLayers = None) -> None:
        super(MultipleInputEmbedding_mask_v5, self).__init__()
        self.module_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channel, out_channel),
                           nn.LayerNorm(out_channel),
                           nn.ReLU(inplace=True),
                           nn.Linear(out_channel, out_channel))
             for in_channel in in_channels])
        
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel)
        )
        
        # self.continuous_attention = nn.MultiheadAttention(embed_dim=out_channel, num_heads=1)
        self.temporal_encoder = TemporalEncoder(historical_steps=len(in_channels),
                                                embed_dim=out_channel,
                                                num_heads=emb_numHead,
                                                dropout=emb_dropout,
                                                num_layers=emb_numLayers)
        self.embed_dim = out_channel
        print(f'MultipleInputEmbedding_mask_v5 / numhead: {emb_numHead}, dropout: {emb_dropout}, num_layers: {emb_numLayers}')
        
        self.apply(init_weights)

    def forward(self,
                continuous_inputs: List[torch.Tensor],
                continuous_inputs_mask: List[torch.Tensor],
                mask_ratio: torch.Tensor,
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        
        # Process continuous inputs with masking
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i].squeeze())
            # if continuous_inputs_mask[i] is not None:
            #     continuous_inputs[i] = torch.where(continuous_inputs_mask[i].unsqueeze(-1), continuous_inputs_token[i], continuous_inputs[i])
        continuous_inputs = torch.stack(continuous_inputs)  # Shape: (num_inputs, batch_size, out_channel)
        key_padding_mask = torch.stack(continuous_inputs_mask)
        
        embed_mask = (torch.rand(len(continuous_inputs), continuous_inputs.shape[1]) < mask_ratio).to('cuda')  # Mask shape: [batch_size, embedding_dim], True is masking
        all_masked = embed_mask.sum(dim=0) == 3  # Identifies rows where all are masked
        # If all modalities are masked, randomly unmask one
        if all_masked.any():
            # Randomly select one modality to unmask for the completely masked instances
            random_idx = torch.randint(0, continuous_inputs.shape[0], (all_masked.sum(),)).to('cuda')
            embed_mask[random_idx, all_masked] = False  # Unmask the randomly selected modality
        
        unimodal = (~key_padding_mask).sum(0) == 1
        key_padding_mask[:, ~unimodal] = torch.logical_or(key_padding_mask[:, ~unimodal], embed_mask[:, ~unimodal])
        
        # modality_token = self.modality_token.unsqueeze(1).repeat(1, continuous_inputs.shape[1], 1)
        # continuous_inputs = torch.where(embed_mask, modality_token, continuous_inputs)        
        
        if continuous_inputs.shape[1] == 0:
            return self.aggr_embed(continuous_inputs[-1])
        continuous_inputs = self.temporal_encoder(continuous_inputs, key_padding_mask)     # Default: no ~ on key_padding_mask
        # attn_output, _ = self.continuous_attention(continuous_inputs, continuous_inputs, continuous_inputs, key_padding_mask=~key_padding_mask.bool())
        
        return self.aggr_embed(continuous_inputs)


class TemporalEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoder, self).__init__()
        encoder_layer = TemporalEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers,
                                                         norm=nn.LayerNorm(embed_dim))
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))
        nn.init.normal_(self.cls_token, mean=0., std=.02)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        # x = torch.where(padding_mask.t().unsqueeze(-1), self.padding_token, x) T N D
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat((x, expand_cls_token), dim=0)
        x = x + self.pos_embed
        padding_mask = torch.cat((padding_mask, torch.zeros((1, padding_mask.shape[-1]), dtype=bool, device=padding_mask.device)), dim=0)
        out = self.transformer_encoder(src=x, src_key_padding_mask=padding_mask.permute(1,0))
        return out[-1]  # [N, D]


class TemporalEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(TemporalEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)   # Default: batch_first = False
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

