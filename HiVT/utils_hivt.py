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
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F

class TemporalData(Data):

    def __init__(self,
                num_nodes=None,
                rotate_mat=None,
                scene=None,
                x=None,
                x_pose=None,
                x_pose_mask=None,
                x_text=None,
                x_text_mask=None,
                x_interaction=None,
                x_interaction_mask=None,
                positions=None,
                rotate_angles=None,
                padding_mask=None,
                padding_mask_total=None,
                edge_index=None,
                bos_mask=None,
                y=None,
                y_pose=None,
                y_pose_mask=None,
                y_text=None,
                y_text_mask=None,
                y_interaction=None,
                y_interaction_mask=None,
                 **kwargs) -> None:
        if x is None:
            super(TemporalData, self).__init__()
            return
        super(TemporalData, self).__init__(num_nodes=num_nodes, rotate_mat=rotate_mat, scene=scene, x=x, x_pose=x_pose, x_pose_mask=x_pose_mask,
                x_text=x_text, x_text_mask=x_text_mask, x_interaction=x_interaction, x_interaction_mask=x_interaction_mask, positions=positions,
                rotate_angles=rotate_angles, padding_mask=padding_mask, padding_mask_total=padding_mask_total, edge_index=edge_index, bos_mask=bos_mask,
                y=y, y_pose=y_pose, y_pose_mask=y_pose_mask, y_text=y_text, y_text_mask=y_text_mask, y_interaction=y_interaction, y_interaction_mask=y_interaction_mask, **kwargs)
        
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'lane_actor_index':
            return torch.tensor([[self['lane_vectors'].size(0)], [self.num_nodes]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

class DistanceDropEdge(object):

    def __init__(self, max_distance: Optional[float] = None) -> None:
        self.max_distance = max_distance

    def __call__(self,
                 edge_index: torch.Tensor,
                 edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.max_distance is None:
            return edge_index, edge_attr
        if isinstance(edge_attr, List):
            edge_attr_text = edge_attr[0]
            edge_attr = edge_attr[1]
            row, col = edge_index
            mask = torch.norm(edge_attr, p=2, dim=-1) < self.max_distance
            edge_index = torch.stack([row[mask], col[mask]], dim=0)
            edge_attr = edge_attr_text[mask]
        else:
            row, col = edge_index
            mask = torch.norm(edge_attr, p=2, dim=-1) < self.max_distance
            edge_index = torch.stack([row[mask], col[mask]], dim=0)
            edge_attr = edge_attr[mask]
        return edge_index, edge_attr


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)

def kl_divergence_loss(student_output, teacher_output, temperature=1.0):
    """
    Implements KL Divergence loss for knowledge distillation.

    Args:
        student_output (torch.Tensor): Output of the student model (logits), shape (B, D)
        teacher_output (torch.Tensor): Output of the teacher model (logits), shape (B, D)
        temperature (float): Temperature scaling factor. Default is 1.0.

    Returns:
        torch.Tensor: The KL divergence loss.
    """
    # Step 1: Apply softmax to the teacher output and temperature scaling
    teacher_probs = F.softmax(teacher_output / temperature, dim=1)
    
    # Step 2: Apply log_softmax to the student output and temperature scaling
    student_log_probs = F.log_softmax(student_output / temperature, dim=1)
    
    # Step 3: Compute the KL Divergence loss
    loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

    return loss


def cosine_similarity_loss(student_features, teacher_features):
    """
    Computes the cosine similarity loss between student and teacher features using nn.CosineSimilarity.

    Args:
        student_features (torch.Tensor): The feature map from the student model, shape (batch_size, feature_dim)
        teacher_features (torch.Tensor): The feature map from the teacher model, shape (batch_size, feature_dim)

    Returns:
        torch.Tensor: The cosine similarity loss
    """
    # Initialize CosineSimilarity module
    cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-8)
    
    # Compute cosine similarity
    similarity = cosine_similarity(student_features, teacher_features)
    
    # Cosine similarity loss (we want to maximize similarity, hence minimize 1 - similarity)
    loss = 1 - similarity.mean()
    
    return loss

def NormalizationLoss(D):
    # Calculate the mean and standard deviation of the feature
    mean_D = D.mean(-1)
    std_D = D.std(-1)
    
    # Loss for mean being close to 0
    mean_loss = torch.mean((mean_D - 0) ** 2)
    
    # Loss for std being close to 1
    std_loss = torch.mean((std_D - 1) ** 2)
    
    # Combine the losses (you can adjust the weighting if needed)
    loss = mean_loss + std_loss
    
    return loss
