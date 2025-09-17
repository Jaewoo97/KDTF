from typing import List, Optional
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def agent2batch_mart(data):  # data is dict
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
    batch_x_pose = torch.zeros(batch_size, max_num_person, past_F, data.x_pose.shape[-1]).cuda()
    batch_y_pose = torch.zeros(batch_size, max_num_person, fut_F, data.x_pose.shape[-1]).cuda()
    padded_mask_pose = torch.ones((batch_size, max_num_person, past_F+fut_F), dtype=torch.bool).cuda()
    batch_x_text = torch.zeros(batch_size, max_num_person, past_F, data.x_text.shape[-1]).cuda()
    padded_mask_text = torch.ones((batch_size, max_num_person, past_F+fut_F), dtype=torch.bool).cuda()
    
    mask = torch.zeros(batch_size, max_num_person, dtype=torch.bool).cuda()
    batch_ranges = torch.arange(max_num_person).unsqueeze(0).expand(batch_size, max_num_person).cuda()
    mask = (batch_ranges < batch_counts.unsqueeze(1))   # True is real data
    
    # 1. Ave by batch
    # local_pos = data.positions - (data.positions[~data.padding_mask]).mean(0)
    # batch_x_pos[mask] = local_pos[:, :past_F]
    
    # 2. Ave by scene
    padded_mask[mask] = data.padding_mask
    padded_mask_cur = padded_mask[:,:,:8].sum(-1)
    batch_x_pos[mask] = data.positions[:,:8]
    batch_x_pos_ = batch_x_pos.sum(-2).sum(1) / padded_mask_cur.sum(-1).unsqueeze(-1)
    batch_x_pos = batch_x_pos - batch_x_pos_.unsqueeze(1).unsqueeze(1)
    
    # 3. zeroed by robot pos
    # robot_pos = data.robot_pos.unsqueeze(1).repeat(1,mask.shape[1],1,1)
    # batch_x_pos[mask] = data.positions[:,:8]
    # batch_x_pos = batch_x_pos - robot_pos[:,:,-1:]
    # batch_x_pos[~mask] = 0
    
    data.padding_mask[:, 1:] = data.padding_mask[:, 1:] | data.padding_mask[:, :-1]        
    data.padding_mask[:, 0] = True
    batch_x[mask] = data.x[:, :past_F]
    batch_y[mask] = data.y
    
    padded_mask[mask] = data.padding_mask
    batch_x_pose[mask] = data.x_pose
    batch_x_text[mask] = data.x_text
    batch_y_pose[mask] = data.y_pose
    padded_mask_pose[:, :, :past_F][mask] = data.x_pose_mask
    padded_mask_pose[:, :, past_F:][mask] = data.y_pose_mask
    padded_mask_text[:, :, :past_F][mask] = data.x_text_mask
    padded_mask_text[:, :, past_F:][mask] = data.y_text_mask
    batch_angle[mask] = data.rotate_angles
    
    return batch_x, batch_y, padded_mask, batch_x_pose, batch_y_pose, padded_mask_pose, mask, batch_angle, batch_x_pos, batch_x_text, padded_mask_text


def agent2batch_mart_ethucy(data):  # data is dict
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


class MLP_model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(1024, 512), activation='relu'):
        super(MLP_model, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_dims)
        dims.append(output_dim)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU(inplace=True))
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class PositionalAgentEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_t_len=200, concat=True):
        super(PositionalAgentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        if concat:
            self.fc = nn.Linear(2 * d_model, d_model)

        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]
        pe = pe[None].repeat(num_a, 1, 1)
        return pe

    def get_agent_enc(self, num_t, num_a, a_offset):
        ae = self.ae[a_offset: num_a + a_offset, :]
        ae = ae.repeat(num_t, 1, 1)
        return ae

    def forward(self, x, num_a, t_offset=0):
        num_t = x.shape[1]
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset) #(N, T, D)
        if self.concat:
            feat = [x, pos_enc]
            x = torch.cat(feat, dim=-1)
            x = self.fc(x)
        else:
            x += pos_enc
        return self.dropout(x) #(N, T, D)


class Decoder(nn.Module):
    def __init__(self, hyper_scales, model_dim, future_length, decoder_hidden_dim):
        super().__init__()
        self.multiplier = len(hyper_scales) + 1
        self.decoder_mlp = MLP_model(
            model_dim*self.multiplier,
            (future_length*2)+1,
            hidden_dims=(
                decoder_hidden_dim,
                decoder_hidden_dim // 2
            )
        )
        self.future_length = future_length

    def forward(self, final_feature):
        outputs = self.decoder_mlp(final_feature)
        pi = outputs[...,-1].unsqueeze(-1)
        outputs = outputs[...,:-1]
        outputs = outputs.view(-1, self.future_length, 2)
        
        return outputs, pi


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x


class AdaptiveGroupEstimator(nn.Module):
    def __init__(self, function_type=1):
        super().__init__()
        
        self.th = nn.Parameter(torch.Tensor([0.5]))
        self.ste = BinaryThreshold(function_type=function_type)

    def forward(self, node_features, padding_mask):
        node_features = node_features * padding_mask.unsqueeze(-1)
        temp = F.normalize(node_features, p=2, dim=2)
        corr_mat = torch.matmul(temp, temp.permute(0, 2, 1))
        G = self.ste(corr_mat - self.th.clamp(-0.9999, 0.9999))

        return G


class BinaryThresholdFunctionType1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        # approximation 1
        grad_input[torch.abs(input) > 0.5] = 0

        return grad_input


class BinaryThresholdFunctionType2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        # approximation 2
        grad_input[torch.abs(input) <= 0.5] *= 2 - 4. * torch.abs(input[torch.abs(input) <= 0.5])
        grad_input[torch.abs(input) > 0.5] = 0
        
        return grad_input


class BinaryThresholdFunctionType3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        # approximation 3
        grad_input[torch.abs(input) <= 0.4] *= 2 - 4. * torch.abs(input[torch.abs(input) <= 0.4])
        grad_input[torch.logical_and(torch.abs(input) > 0.4, torch.abs(input) <= 1)] *= 0.4
        grad_input[torch.abs(input) > 1] = 0

        return grad_input


class BinaryThreshold(nn.Module):
    def __init__(self, function_type=1):
        super(BinaryThreshold, self).__init__()
        self.function_type = function_type
        print('[INFO] Binary Threshold Function Type: {}'.format(function_type))

    def forward(self, input):
        if self.function_type == 1:
            return BinaryThresholdFunctionType1.apply(input)
        elif self.function_type == 2:
            return BinaryThresholdFunctionType2.apply(input)
        elif self.function_type == 3:
            return BinaryThresholdFunctionType3.apply(input)
        else:
            raise NotImplementedError


class HRT(nn.Module):
    def __init__(
            self,
            num_layers: int = 3,
            num_heads: int = 4,
            node_dim: int = 64,
            node_hidden_dim: int = 128,
            edge_dim: int = 64,
            edge_hidden_dim_1: int = 128,
            edge_hidden_dim_2: int = 128,
            dropout: float = 0.0,
            aggregation: str = 'avg',
            scale: int = 2,
            function_type: int = 1
    ):
        super(HRT, self).__init__()
        self.scale = scale
        layer = HRTTransformerLayer(
            num_heads,
            node_dim,
            node_hidden_dim,
            edge_dim,
            edge_hidden_dim_1,
            edge_hidden_dim_2,
            dropout,
            aggregation
        )
        
        print('[INFO] HRT Agg: {}'.format(aggregation))
        
        self.aggregation = aggregation # add or att
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.node2edge_mlp = MLP(input_dim=node_dim, output_dim=node_dim, hidden_size=(node_hidden_dim,))
        if self.aggregation == 'att':
            self.attention_mlp = MLP(input_dim=node_dim + edge_dim, output_dim=1, hidden_size=(32,))
        
        if self.scale > 1:
            self.group_gen = AdaptiveGroupEstimator(function_type=function_type)
                
    def init_group_adj(self, node_features, padding_mask):
        G = self.group_gen(node_features, padding_mask)
        
        return G

    def init_pair_adj(self, num_agents, batch_size):
        off_diag = np.ones([num_agents, num_agents])
        
        rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
        rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
        
        G = rel_rec + rel_send
        G = torch.from_numpy(G).type(torch.float).cuda()
        G = G.unsqueeze(0).repeat(batch_size, 1, 1)
        
        return G

    def init_edge(self, x, G):
        N = x.shape[1]
        if self.aggregation == 'avg':
            div = torch.sum(G, dim=-1)[:, :, None]
            div[div==0] = 1e5        # NOTE: prevent nan
            edges = torch.matmul(G, x)
            edges = self.node2edge_mlp(edges / div)

        elif self.aggregation == 'sum':
            edges = torch.matmul(G, x)
            edges = self.node2edge_mlp(edges)
            
        elif self.aggregation == 'att':
            x = self.node2edge_mlp(x)
            edge_init = torch.matmul(G, x)
            E = edge_init.shape[1]
            x_rep = (x[:, :, None, :].transpose(2, 1)).repeat(1, E, 1, 1)
            edge_rep = edge_init[:, :, None, :].repeat(1, 1, N, 1)
            node_edge_cat = torch.cat((x_rep, edge_rep), dim=-1)
            attention_weight = self.attention_mlp(node_edge_cat)[:, :, :, 0]
            G_weight = attention_weight * G
            G_weight = F.softmax(G_weight, dim=2)
            G_weight = G_weight * G
            edges = torch.matmul(G_weight, x)
            
        return edges
        
    def forward(self, node_features, _edge_features, _G, padding_mask, return_edge=False):
        if self.scale == 1:
            G = self.init_pair_adj(node_features.shape[1], node_features.shape[0])
        else:
            G = self.init_group_adj(node_features, padding_mask)
        edge_features = self.init_edge(node_features, G)
        
        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, G, padding_mask)
        
        if return_edge:
            return node_features, edge_features, G
        
        return node_features, G


class HRTNoEdgeInit(nn.Module):
    def __init__(
            self,
            num_layers: int = 3,
            num_heads: int = 4,
            node_dim: int = 64,
            node_hidden_dim: int = 128,
            edge_dim: int = 64,
            edge_hidden_dim_1: int = 128,
            edge_hidden_dim_2: int = 128,
            dropout: float = 0.0,
            aggregation: str = 'avg',
            scale: int = 2,
            function_type: int = 1
    ):
        super(HRTNoEdgeInit, self).__init__()
        layer = HRTTransformerLayer(
            num_heads,
            node_dim,
            node_hidden_dim,
            edge_dim,
            edge_hidden_dim_1,
            edge_hidden_dim_2,
            dropout,
            aggregation
        )
        
        self.aggregation = aggregation # add or att
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
                
    def forward(self, node_features, edge_features, G, padding_mask, return_edge=False):
        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, G, padding_mask)
        
        if return_edge:
            return node_features, edge_features, G
            
        return node_features, G


class HRTTransformerLayer(nn.Module):
    def __init__(
        self,
        num_heads: int = 4,
        node_dim: int = 64,
        node_hidden_dim: int = 128,
        edge_dim: int = 64,
        edge_hidden_dim_1: int = 128,
        edge_hidden_dim_2: int = 128,
        dropout: float = 0.0,
        aggregation: str = 'avg'
    ):
        super(HRTTransformerLayer, self).__init__()
        self.aggregation = aggregation
        self.edge_update = True
        
        self.node_attention_layer = HRTAttentionLayer(num_heads, node_dim, edge_dim, aggregation)
            
        self.dropout = nn.Dropout(dropout)
        self.linear_net_n = nn.Sequential(
            nn.Linear(node_dim, node_hidden_dim),
            # nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(node_hidden_dim, node_dim),
        )
        
        self.norm1_n = nn.LayerNorm(node_dim)
        self.norm2_n = nn.LayerNorm(node_dim)
        
        if self.edge_update:
            self.linear_net1_e = nn.Sequential(
                nn.Linear(node_dim + edge_dim, edge_hidden_dim_1),
                # nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                nn.Linear(edge_hidden_dim_1, edge_dim),
            )
            
            self.linear_net2_e = nn.Sequential(
                nn.Linear(edge_dim, edge_hidden_dim_2),
                # nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                nn.Linear(edge_hidden_dim_2, edge_dim),
            )
            
            self.norm1_e = nn.LayerNorm(edge_dim)
            self.norm2_e = nn.LayerNorm(edge_dim)
    
    def forward(self, node_features, edge_features, G, padding_mask):
        node_attn_out = self.node_attention_layer(node_features, edge_features, G, padding_mask)
        node_features = node_features + self.dropout(node_attn_out)
        node_features = self.norm1_n(node_features)
        
        linear_out = self.linear_net_n(node_features)
        node_features = node_features + self.dropout(linear_out)
        node_features = self.norm2_n(node_features)
        
        if self.edge_update:
            if self.aggregation == 'avg':
                div = torch.sum(G, dim=-1)[:, :, None]
                div[div==0] = 1e5
                concatenated_inputs = torch.cat((edge_features, torch.matmul(G, node_features) / div), dim=-1)
            elif self.aggregation == 'sum':
                concatenated_inputs = torch.cat((edge_features, torch.matmul(G, node_features)), dim=-1)
            else:
                raise NotImplementedError
            edge_features = edge_features + self.dropout(self.linear_net1_e(concatenated_inputs))
            edge_features = self.norm1_e(edge_features)
            
            edge_features = edge_features + self.dropout(self.linear_net2_e(edge_features))
            edge_features = self.norm2_e(edge_features)
        
        return node_features, edge_features


class HRTAttentionLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        node_dim: int,
        edge_dim: int,
        aggregation: str = 'avg'
    ):
        super(HRTAttentionLayer, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.aggregation = aggregation
        
        self.proj_qkv_n = nn.Linear(node_dim, 3 * node_dim)
        self.proj_qkv_e = nn.Linear(edge_dim, 3 * node_dim)
        self.proj_o = nn.Linear(node_dim, node_dim)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, node_features, edge_features, G, padding_mask):
        B, N, _ = node_features.shape
        B, E, _ = edge_features.shape
        qkv_n = self.proj_qkv_n(node_features)
        qkv_e = self.proj_qkv_e(edge_features)
        
        qkv_n = qkv_n.reshape(B, N, self.num_heads, 3 * self.head_dim)
        qkv_n = qkv_n.permute(0, 2, 1, 3)
        q_n, k_n, v_n = qkv_n.chunk(3, dim=-1)
        
        qkv_e = qkv_e.reshape(B, E, self.num_heads, 3 * self.head_dim)
        qkv_e = qkv_e.permute(0, 2, 1, 3)
        q_e, k_e, v_e = qkv_e.chunk(3, dim=-1)

        if self.aggregation == 'avg':
            div = torch.sum(G, dim=1)[:, None, :, None]
            div[div==0] = 1e5       # NOTE: prevent nan
        elif self.aggregation == 'sum':
            div = 1.0
            
        q = q_n + torch.einsum('bmn,bhnc->bhmc', G.transpose(-2, -1), q_e) / div
        k = k_n + torch.einsum('bmn,bhnc->bhmc', G.transpose(-2, -1), k_e) / div
        v = v_n + torch.einsum('bmn,bhnc->bhmc', G.transpose(-2, -1), v_e) / div
        
        qk = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Mask qk based on padding_mask
        padding_mask = (padding_mask.unsqueeze(1) & padding_mask.unsqueeze(-1)).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        qk = qk.masked_fill(padding_mask == False, -float('inf'))
        
        att_dist = F.softmax(qk, dim=-1)
        att_dist = torch.where(torch.isnan(att_dist), torch.tensor(0.0, device=att_dist.device), att_dist)

        new_nodes = torch.matmul(att_dist, v) # (B, H, N, D)
        new_nodes = new_nodes.permute(0, 2, 1, 3) # (B, N, H, D)
        new_nodes = new_nodes.reshape(B, N, self.num_heads * self.head_dim) # (B, N, H*D)
        new_nodes = self.proj_o(new_nodes)
        
        return new_nodes



def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


class RT(nn.Module):
    '''
    This is pytorch implementation of RT model.
    We implement the model based on the github repo:
    https://github.com/CameronDiao/relational-transformer
    '''
    def __init__(
            self,
            num_layers: int = 3,
            num_heads: int = 4,
            node_dim: int = 64,
            node_hidden_dim: int = 128,
            edge_dim: int = 64,
            edge_hidden_dim_1: int = 128,
            edge_hidden_dim_2: int = 128,
            dropout: float = 0.0,
    ):
        super(RT, self).__init__()
        layer = RTTransformerLayer(
            num_heads,
            node_dim,
            node_hidden_dim,
            edge_dim,
            edge_hidden_dim_1,
            edge_hidden_dim_2,
            dropout,
        )
        # self.aggregation = aggregation # 'cat', 'avg', 'sum', 'att'
        self.aggregation = 'cat'
        print('[INFO] RT Agg: {}'.format(self.aggregation))
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.node2edge_mlp = MLP(input_dim=node_dim * 2 if self.aggregation == 'cat' else node_dim, output_dim=node_dim, hidden_size=(node_hidden_dim,))
        if self.aggregation == 'att':
            self.attention_mlp = MLP(input_dim=node_dim + edge_dim, output_dim=1, hidden_size=(32,))

    def init_adj(self, num_nodes, batch):
        off_diag = np.ones([num_nodes, num_nodes])
        
        rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
        rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
        
        rel_rec = torch.FloatTensor(rel_rec).cuda()
        rel_send = torch.FloatTensor(rel_send).cuda()
        
        rel_rec = rel_rec[None, :, :].repeat(batch, 1, 1)
        rel_send = rel_send[None, :, :].repeat(batch, 1, 1)
        
        # rel_rec = rel_rec.reshape(batch, num_nodes, num_nodes, -1)
        # rel_send = rel_send.reshape(batch, num_nodes, num_nodes, -1)
        
        return rel_rec, rel_send

    def init_edge(self, x, rel_rec, rel_send):
        B, N, _ = x.shape
        if self.aggregation == 'cat':
            receivers = torch.matmul(rel_rec, x)
            senders = torch.matmul(rel_send, x)
            edges = self.node2edge_mlp(torch.cat([receivers, senders], dim=-1))
            
        elif self.aggregation == 'avg':
            H = rel_rec + rel_send
            edges = torch.matmul(H, x) / 2
            edges = self.node2edge_mlp(edges)

        elif self.aggregation == 'sum':
            H = rel_rec + rel_send
            edges = torch.matmul(H, x)
            edges = self.node2edge_mlp(edges)
            
        elif self.aggregation == 'att':
            H = rel_rec + rel_send
            x = self.node2edge_mlp(x)
            edge_init = torch.matmul(H, x)
            E = edge_init.shape[1]
            x_rep = (x[:, :, None, :].transpose(2, 1)).repeat(1, E, 1, 1)
            edge_rep = edge_init[:, :, None, :].repeat(1, 1, N, 1)
            node_edge_cat = torch.cat((x_rep, edge_rep), dim=-1)
            attention_weight = self.attention_mlp(node_edge_cat)[:, :, :, 0]
            H_weight = attention_weight * H
            H_weight = F.softmax(H_weight, dim=2)
            H_weight = H_weight * H
            edges = torch.matmul(H_weight, x)
            
        edges = edges.reshape(B, N, N, -1)
            
        return edges
        
    def forward(self, node_features, edge_features_, padding_mask, return_edge=False):
        batch = node_features.shape[0]
        num_nodes = node_features.shape[1]
        
        rel_rec, rel_send = self.init_adj(num_nodes, batch)
        edge_features = self.init_edge(node_features, rel_rec, rel_send)
        
        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, padding_mask)
        
        if return_edge:
            return node_features, edge_features
        
        return node_features


class RTNoEdgeInit(nn.Module):
    def __init__(
            self,
            num_layers: int = 3,
            num_heads: int = 4,
            node_dim: int = 64,
            node_hidden_dim: int = 128,
            edge_dim: int = 64,
            edge_hidden_dim_1: int = 128,
            edge_hidden_dim_2: int = 128,
            dropout: float = 0.0,
    ):
        super(RTNoEdgeInit, self).__init__()
        layer = RTTransformerLayer(
            num_heads,
            node_dim,
            node_hidden_dim,
            edge_dim,
            edge_hidden_dim_1,
            edge_hidden_dim_2,
            dropout,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, node_features, edge_features, padding_mask, return_edge=False):
        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, padding_mask)
        
        if return_edge:
            return node_features, edge_features
        
        return node_features


class RTTransformerLayer(nn.Module):
    def __init__(
        self,
        num_heads: int = 4,
        node_dim: int = 64,
        node_hidden_dim: int = 128,
        edge_dim: int = 64,
        edge_hidden_dim_1: int = 128,
        edge_hidden_dim_2: int = 128,
        dropout: float = 0.0,
    ):
        super(RTTransformerLayer, self).__init__()
        self.edge_update = True
        self.attention_layer = RTAttentionLayer(num_heads, node_dim, edge_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.linear_net_n = nn.Sequential(
            nn.Linear(node_dim, node_hidden_dim),
            # nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(node_hidden_dim, node_dim),
        )
        
        self.norm1_n = nn.LayerNorm(node_dim)
        self.norm2_n = nn.LayerNorm(node_dim)

        if self.edge_update:
            self.linear_net1_e = nn.Sequential(
                nn.Linear(node_dim * 2 + edge_dim * 2, edge_hidden_dim_1),
                # nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                nn.Linear(edge_hidden_dim_1, edge_dim),
            )
            
            self.linear_net2_e = nn.Sequential(
                nn.Linear(edge_dim, edge_hidden_dim_2),
                # nn.Dropout(dropout),
                nn.ReLU(inplace=True),
                nn.Linear(edge_hidden_dim_2, edge_dim),
            )
            
            self.norm1_e = nn.LayerNorm(edge_dim)
            self.norm2_e = nn.LayerNorm(edge_dim)
    
    def forward(self, node_features, edge_features, padding_mask):
        _, N, _ = node_features.shape
        attn_out = self.attention_layer(node_features, edge_features, padding_mask)
        node_features = node_features + self.dropout(attn_out)
        node_features = self.norm1_n(node_features)
        
        linear_out = self.linear_net_n(node_features)
        node_features = node_features + self.dropout(linear_out)
        node_features = self.norm2_n(node_features)
        
        if self.edge_update:
            source_nodes = node_features.unsqueeze(1)
            expanded_source_nodes = source_nodes.repeat(1, N, 1, 1)
            
            target_nodes = node_features.unsqueeze(2)
            expanded_target_nodes = target_nodes.repeat(1, 1, N, 1)
            
            reversed_edge_features = edge_features.permute(0, 2, 1, 3)
            
            concatenated_inputs = torch.cat([edge_features, reversed_edge_features, expanded_source_nodes, expanded_target_nodes], dim=-1)
            
            edge_features = edge_features + self.dropout(self.linear_net1_e(concatenated_inputs))
            edge_features = self.norm1_e(edge_features)
            
            edge_features = edge_features + self.dropout(self.linear_net2_e(edge_features))
            edge_features = self.norm2_e(edge_features)
        
        return node_features, edge_features


class RTAttentionLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        node_dim: int,
        edge_dim: int,
    ):
        super(RTAttentionLayer, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        
        self.proj_qkv_n = nn.Linear(node_dim, 3 * node_dim)
        self.proj_qkv_e = nn.Linear(edge_dim, 3 * node_dim)
        self.proj_o = nn.Linear(node_dim, node_dim)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, node_features, edge_features, padding_mask):
        B, N, _ = node_features.shape
        qkv_n = self.proj_qkv_n(node_features)
        qkv_e = self.proj_qkv_e(edge_features)
        
        qkv_n = qkv_n.reshape(B, N, self.num_heads, 3 * self.head_dim)
        qkv_n = qkv_n.permute(0, 2, 1, 3)
        q_n, k_n, v_n = qkv_n.chunk(3, dim=-1)
        
        qkv_e = qkv_e.reshape(B, N, N, self.num_heads, 3 * self.head_dim)
        qkv_e = qkv_e.permute(0, 3, 1, 2, 4)
        q_e, k_e, v_e = qkv_e.chunk(3, dim=-1)
        
        q = q_n.reshape(B, self.num_heads, N, 1, self.head_dim) + q_e
        k = k_n.reshape(B, self.num_heads, 1, N, self.head_dim) + k_e
        
        q = q.reshape(B, self.num_heads, N, N, 1, self.head_dim)
        k = q.reshape(B, self.num_heads, N, N, self.head_dim, 1)
        
        qk = torch.matmul(q, k)
        qk = qk.reshape(B, self.num_heads, N, N)
        
        qk = qk * self.scale
        
        # Mask qk based on padding_mask
        padding_mask = (padding_mask.unsqueeze(1) & padding_mask.unsqueeze(-1)).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        qk = qk.masked_fill(padding_mask == False, -float('inf'))

        att_dist = F.softmax(qk, dim=-1)
        att_dist = torch.where(torch.isnan(att_dist), torch.tensor(0.0, device=att_dist.device), att_dist)
        
        att_dist = att_dist.reshape(B, self.num_heads, N, 1, N)
        v = v_n.reshape(B, self.num_heads, 1, N, self.head_dim) + v_e
        
        new_nodes = torch.matmul(att_dist, v)
        new_nodes = new_nodes.reshape(B, self.num_heads, N, self.head_dim)
        new_nodes = new_nodes.permute(0, 2, 1, 3)
        new_nodes = new_nodes.reshape(B, N, self.num_heads * self.head_dim)
        new_nodes = self.proj_o(new_nodes)
        
        return new_nodes


class MultipleInputEmbedding_mask_v3(nn.Module):

    def __init__(self,
                 in_channels: List[int],
                 out_channel: int,
                 emb_numHead = None,
                 emb_dropout = None,
                 emb_numLayers = None) -> None:
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

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


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
