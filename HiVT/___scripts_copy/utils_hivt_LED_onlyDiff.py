from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
from torch.nn import Module, Linear
import torch.nn.functional as F

def agent2batch_led(data):  # data is dict
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
    batch_x_text = torch.zeros(batch_size, max_num_person, past_F, data.x_text.shape[-1]).cuda()
    batch_y_pose = torch.zeros(batch_size, max_num_person, fut_F, data.x_pose.shape[-1]).cuda()
    padded_mask_pose = torch.ones((batch_size, max_num_person, past_F+fut_F), dtype=torch.bool).cuda()
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
    
    data.padding_mask[:, 1:] = data.padding_mask[:, 1:] | data.padding_mask[:, :-1]        # If using difference
    data.padding_mask[:, 0] = True
    batch_x[mask] = data.x[:, :past_F]
    # inv_rot = data.rotate_mat[:, :2, :2].transpose(1, 2)  		######################################################################
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


class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super().__init__()

		self.dropout = nn.Dropout(p=dropout)
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(
			torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
		)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer("pe", pe)

	def forward(self, x):
		x = x + self.pe[: x.size(0), :]
		return self.dropout(x)

class MLP(nn.Module):
	def __init__(self, in_feat, out_feat, hid_feat=(1024, 256), activation=None, dropout=-1):
		super(MLP, self).__init__()
		dims = (in_feat, ) + hid_feat + (out_feat, )

		self.layers = nn.ModuleList()
		for i in range(len(dims) - 1):
			self.layers.append(nn.Linear(dims[i], dims[i + 1]))

		self.activation = activation if activation is not None else lambda x: x
		self.dropout = nn.Dropout(dropout) if dropout != -1 else lambda x: x

	def forward(self, x):
		for i in range(len(self.layers)):
			x = self.activation(x)
			x = self.dropout(x)
			x = self.layers[i](x)
		return x

class ConcatSquashLinear(Module):
	def __init__(self, dim_in, dim_out, dim_ctx):
		super(ConcatSquashLinear, self).__init__()
		self._layer = Linear(dim_in, dim_out)
		self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
		self._hyper_gate = Linear(dim_ctx, dim_out)

	def forward(self, ctx, x):
		# ctx: (B, 1, F+3)
		# x: (B, T, 2)
		gate = torch.sigmoid(self._hyper_gate(ctx))
		bias = self._hyper_bias(ctx)
		# if x.dim() == 3:
		#     gate = gate.unsqueeze(1)
		#     bias = bias.unsqueeze(1)
		ret = self._layer(x) * gate + bias
		return ret
	
	def batch_generate(self, ctx, x):
		# ctx: (B, n, 1, F+3)
		# x: (B, n, T, 2)
		gate = torch.sigmoid(self._hyper_gate(ctx))
		bias = self._hyper_bias(ctx)
		# if x.dim() == 3:
		#     gate = gate.unsqueeze(1)
		#     bias = bias.unsqueeze(1)
		ret = self._layer(x) * gate + bias
		return ret

class st_encoder(nn.Module):
	def __init__(self):
		super().__init__()
		channel_in = 2
		channel_out = 32
		dim_kernel = 3
		self.dim_embedding_key = 128
		self.spatial_conv_pre = nn.Conv1d(channel_in, channel_out, 1, stride=1, padding=0)
		self.spatial_conv = nn.Conv1d(channel_out, channel_out, dim_kernel, stride=1, padding=1)
		self.temporal_encoder = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)

		self.relu = nn.ReLU()

		self.reset_parameters()
		self.emb_token = nn.Parameter(torch.Tensor(8, channel_out)).cuda()
		nn.init.normal_(self.emb_token, mean=0., std=.02)

	def reset_parameters(self):
		nn.init.kaiming_normal_(self.spatial_conv.weight)
		nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
		nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
		nn.init.zeros_(self.spatial_conv.bias)
		nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
		nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

	def forward(self, X, mask):
		'''
		X: b, T, 2

		return: b, F
		'''
		B, N, T = mask.shape
		X_t = torch.transpose(X, 1, 2)
		X_t_pre = self.relu(self.spatial_conv_pre(X_t))
  
		X_t_pre_ = X_t_pre.permute(0,2,1)
		mask = mask[:,:,:8].reshape(B*N, -1)
		X_t_pre_ = torch.where( (~(mask==0)).unsqueeze(-1), self.emb_token, X_t_pre_)
		X_t_pre = X_t_pre_.permute(0,2,1)
		X_after_spatial = self.relu(self.spatial_conv(X_t_pre))
		X_embed = torch.transpose(X_after_spatial, 1, 2)

		output_x, state_x = self.temporal_encoder(X_embed)
		state_x = state_x.squeeze(0)

		return state_x


class social_transformer(nn.Module):
	def __init__(self, past_timesteps):
		super(social_transformer, self).__init__()
		self.encode_past = nn.Linear(past_timesteps*2, 128, bias=False)
		# self.encode_past = nn.Linear(48, 128, bias=False)
		self.layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=128)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)

	def forward(self, h, mask):
		'''
		h: batch_size, t, 2
		'''
		# print(h.shape)
		h_feat = self.encode_past(h.reshape(h.size(0), -1)).unsqueeze(1)
		# print(h_feat.shape)
		# n_samples, 1, 64
		h_feat_ = self.transformer_encoder(h_feat, mask)
		h_feat = h_feat + h_feat_

		return h_feat

class social_transformer_withPose(nn.Module):
	def __init__(self, past_timesteps):
		super(social_transformer_withPose, self).__init__()
		self.encode_past = nn.Linear(past_timesteps*2, 64, bias=False).cuda()
		self.encode_past_pose = nn.Linear(past_timesteps*24*3, 32, bias=False).cuda()
		# self.encode_past = nn.Linear(48, 128, bias=False)
		self.layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=128)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)
		self.traj_token = nn.Parameter(torch.Tensor(past_timesteps, 2)).cuda()
		self.pose_token = nn.Parameter(torch.Tensor(past_timesteps, 72)).cuda()
		nn.init.normal_(self.pose_token, mean=0., std=.02)
		nn.init.normal_(self.traj_token, mean=0., std=.02)
		self.angle_mlp = nn.Sequential(
			nn.Linear(1, 32),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(32, 32),
			nn.Dropout(0.1))

	def forward(self, h, h_pose, mask, mask_pose, angle):
		'''
		h: batch_size, t, 2
		'''
		# print(h.shape)
		B, N, T_PAST, _ = h.shape
		mask_, mask_pose_ = ~(mask==0), ~(mask_pose==0)
		h = torch.where(mask_[:,:,:8].unsqueeze(-1), self.traj_token, h)
		h = h.reshape(B*N, -1)
		h_pose = torch.where(mask_pose_[:,:,:8].unsqueeze(-1), self.pose_token, h_pose)
		h_pose = h_pose.reshape(B*N, -1)
  
		h_feat = self.encode_past(h.reshape(h.size(0), -1)).unsqueeze(1)
		h_feat_pose = self.encode_past_pose(h_pose.reshape(h_pose.size(0), -1)).unsqueeze(1)
		h_feat_angle = self.angle_mlp(angle.unsqueeze(-1)).reshape(h_pose.size(0), -1).unsqueeze(1)
  
  
		h_fused = torch.cat((h_feat, h_feat_pose, h_feat_angle), dim=-1)
		

		mask_trm = ( (~mask_[:,:,:8]).sum(-1)>1 ).reshape(B, N)
		h_fused = h_fused.reshape(B, N, h_fused.shape[-1]).permute(1,0,2)
		h_fused_ = self.transformer_encoder(h_fused, src_key_padding_mask=mask_trm)
		h_fused_ = h_fused + h_fused_

		return h_fused_

class social_transformer_traj(nn.Module):
	def __init__(self, past_timesteps=8):
		super(social_transformer_traj, self).__init__()
		self.encode_past = nn.Sequential(
			nn.Linear(2, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(128, 128),
			nn.Dropout(0.1))
  
		# self.temporal_encoder = nn.GRU(96, 96, 1, batch_first=True)
		# self.encode_past = nn.Linear(48, 128, bias=False)
		self.temporal_encoder = TemporalEncoder(historical_steps=past_timesteps+1,
											embed_dim=128,
											num_heads=4,
											dropout=0.1,
											num_layers=2)
  
		self.layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=128)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)
		self.traj_token = nn.Parameter(torch.Tensor(past_timesteps, 128)).cuda()
		nn.init.normal_(self.traj_token, mean=0., std=.02)
		self.angle_mlp = nn.Sequential(
			nn.Linear(1, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(128, 128),
			nn.Dropout(0.1))
	def forward(self, h, mask, h_pose, mask_pose, h_text, mask_text, angle, batch_mask):
		'''
		h: batch_size, t, 2
		'''
		# print(h.shape)
		B, N, T_PAST, _ = h.shape
		mask_ = ~(mask==0)
		# h = h.reshape(B*N, -1)
  
		h_feat = self.encode_past(h)
		h_feat = h_feat.reshape(B*N, T_PAST, -1)

		h_feat = torch.where(mask_[:,:,:8].reshape(B*N, -1).unsqueeze(-1), self.traj_token, h_feat)
		h_feat_angle = self.angle_mlp(angle.unsqueeze(-1)).reshape(h_feat.size(0), -1).unsqueeze(1)
		h_feat = torch.cat((h_feat, h_feat_angle), dim=1)
		temp_mask = mask_[:,:,:8].reshape(-1,T_PAST).permute(1,0)
		temp_mask = torch.cat((temp_mask, torch.zeros((1, temp_mask.shape[1]), dtype=torch.bool).cuda()), dim=0)
  
		h_feat = self.temporal_encoder(h_feat.permute(1,0,2), temp_mask)
		h_feat = h_feat.squeeze().unsqueeze(1)

		# mask_trm = ( (~mask_[:,:,:8]).sum(-1)>1 ).reshape(B, N)
		h_feat = h_feat.reshape(B, N, h_feat.shape[-1]).permute(1,0,2)
		h_feat_ = self.transformer_encoder(h_feat, src_key_padding_mask=~batch_mask)
		h_feat_ = h_feat + h_feat_

		return h_feat_

class social_transformer_trajPose(nn.Module):
	def __init__(self, past_timesteps=8):
		super(social_transformer_trajPose, self).__init__()
		self.encode_past = nn.Sequential(
			nn.Linear(2, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(128, 128),
			nn.Dropout(0.1))
		self.encode_pose = nn.Sequential(
			nn.Linear(72, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(128, 128),
			nn.Dropout(0.1))
		# self.temporal_encoder = nn.GRU(96, 96, 1, batch_first=True)
		# self.encode_past = nn.Linear(48, 128, bias=False)
		self.temporal_encoder = TemporalEncoder(historical_steps=(past_timesteps*2)+1,
											embed_dim=128,
											num_heads=4,
											dropout=0.1,
											num_layers=2)
  
		self.layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=128)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)
		self.traj_token = nn.Parameter(torch.Tensor(past_timesteps, 128)).cuda()
		nn.init.normal_(self.traj_token, mean=0., std=.02)
		self.angle_mlp = nn.Sequential(
			nn.Linear(1, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(128, 128),
			nn.Dropout(0.1))
	def forward(self, h, mask, h_pose, mask_pose, h_text, mask_text, angle, batch_mask):
		'''
		h: batch_size, t, 2
		'''
		# print(h.shape)
		B, N, T_PAST, _ = h.shape
		mask_ = ~(mask==0)
		# h = h.reshape(B*N, -1)
  
		h_feat = self.encode_past(h)
		h_feat = h_feat.reshape(B*N, T_PAST, -1)
		h_feat_pose = self.encode_pose(h_pose)
		h_feat_pose = h_feat_pose.reshape(B*N, T_PAST, -1)

		# h_feat = torch.where(mask_[:,:,:8].reshape(B*N, -1).unsqueeze(-1), self.traj_token, h_feat)
		h_feat_angle = self.angle_mlp(angle.unsqueeze(-1)).reshape(h_feat.size(0), -1).unsqueeze(1)
		h_feat = torch.cat((h_feat, h_feat_pose, h_feat_angle), dim=1)
		temp_mask = mask_[:,:,:8].reshape(-1,T_PAST).permute(1,0)
		temp_mask_pose = mask_pose[:,:,:8].reshape(-1,T_PAST).permute(1,0)
		temp_mask = torch.cat((temp_mask, temp_mask_pose, torch.zeros((1, temp_mask.shape[1]), dtype=torch.bool).cuda()), dim=0)
  
		h_feat = self.temporal_encoder(h_feat.permute(1,0,2), temp_mask)
		h_feat = h_feat.squeeze().unsqueeze(1)

		# mask_trm = ( (~mask_[:,:,:8]).sum(-1)>1 ).reshape(B, N)
		h_feat = h_feat.reshape(B, N, h_feat.shape[-1]).permute(1,0,2)
		h_feat_ = self.transformer_encoder(h_feat, src_key_padding_mask=~batch_mask)
		h_feat_ = h_feat + h_feat_

		return h_feat_

class social_transformer_trajText(nn.Module):
	def __init__(self, past_timesteps=8):
		super(social_transformer_trajText, self).__init__()
		self.encode_past = nn.Sequential(
			nn.Linear(2, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(128, 128),
			nn.Dropout(0.1))
		self.encode_text = nn.Sequential(
			nn.Linear(312, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(128, 128),
			nn.Dropout(0.1))
		# self.temporal_encoder = nn.GRU(96, 96, 1, batch_first=True)
		# self.encode_past = nn.Linear(48, 128, bias=False)
		self.temporal_encoder = TemporalEncoder(historical_steps=(past_timesteps*2)+1,
											embed_dim=128,
											num_heads=4,
											dropout=0.1,
											num_layers=2)
  
		self.layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=128)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)
		self.traj_token = nn.Parameter(torch.Tensor(past_timesteps, 128)).cuda()
		nn.init.normal_(self.traj_token, mean=0., std=.02)
		self.angle_mlp = nn.Sequential(
			nn.Linear(1, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(128, 128),
			nn.Dropout(0.1))
	def forward(self, h, mask, h_pose, mask_pose, h_text, mask_text, angle, batch_mask):
		'''
		h: batch_size, t, 2
		'''
		# print(h.shape)
		B, N, T_PAST, _ = h.shape
		mask_ = ~(mask==0)
		# h = h.reshape(B*N, -1)
  
		h_feat = self.encode_past(h)
		h_feat = h_feat.reshape(B*N, T_PAST, -1)
		h_feat_text = self.encode_text(h_text)
		h_feat_text = h_feat_text.reshape(B*N, T_PAST, -1)

		# h_feat = torch.where(mask_[:,:,:8].reshape(B*N, -1).unsqueeze(-1), self.traj_token, h_feat)
		h_feat_angle = self.angle_mlp(angle.unsqueeze(-1)).reshape(h_feat.size(0), -1).unsqueeze(1)
		h_feat = torch.cat((h_feat, h_feat_text, h_feat_angle), dim=1)
		temp_mask = mask_[:,:,:8].reshape(-1,T_PAST).permute(1,0)
		temp_mask_text = mask_text[:,:,:8].reshape(-1,T_PAST).permute(1,0)
		temp_mask = torch.cat((temp_mask, temp_mask_text, torch.zeros((1, temp_mask.shape[1]), dtype=torch.bool).cuda()), dim=0)
  
		h_feat = self.temporal_encoder(h_feat.permute(1,0,2), temp_mask)
		h_feat = h_feat.squeeze().unsqueeze(1)

		# mask_trm = ( (~mask_[:,:,:8]).sum(-1)>1 ).reshape(B, N)
		h_feat = h_feat.reshape(B, N, h_feat.shape[-1]).permute(1,0,2)
		h_feat_ = self.transformer_encoder(h_feat, src_key_padding_mask=~batch_mask)
		h_feat_ = h_feat + h_feat_

		return h_feat_

class social_transformer_trajPoseText(nn.Module):
	def __init__(self, past_timesteps=8):
		super(social_transformer_trajPoseText, self).__init__()
		self.encode_past = nn.Sequential(
			nn.Linear(2, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(128, 128),
			nn.Dropout(0.1))
		self.encode_text = nn.Sequential(
			nn.Linear(312, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(128, 128),
			nn.Dropout(0.1))
		self.encode_pose = nn.Sequential(
			nn.Linear(72, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(128, 128),
			nn.Dropout(0.1))
		# self.temporal_encoder = nn.GRU(96, 96, 1, batch_first=True)
		# self.encode_past = nn.Linear(48, 128, bias=False)
		self.temporal_encoder = TemporalEncoder(historical_steps=(past_timesteps*3)+1,
											embed_dim=128,
											num_heads=4,
											dropout=0.1,
											num_layers=2)
  
		self.layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=128)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)
		self.traj_token = nn.Parameter(torch.Tensor(past_timesteps, 128)).cuda()
		nn.init.normal_(self.traj_token, mean=0., std=.02)
		self.angle_mlp = nn.Sequential(
			nn.Linear(1, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(128, 128),
			nn.Dropout(0.1))
	def forward(self, h, mask, h_pose, mask_pose, h_text, mask_text, angle, batch_mask):
		'''
		h: batch_size, t, 2
		'''
		# print(h.shape)
		B, N, T_PAST, _ = h.shape
		mask_ = ~(mask==0)
		# h = h.reshape(B*N, -1)
  
		h_feat = self.encode_past(h)
		h_feat = h_feat.reshape(B*N, T_PAST, -1)
		h_feat_text = self.encode_text(h_text)
		h_feat_text = h_feat_text.reshape(B*N, T_PAST, -1)
		h_feat_pose = self.encode_pose(h_pose)
		h_feat_pose = h_feat_pose.reshape(B*N, T_PAST, -1)

		# h_feat = torch.where(mask_[:,:,:8].reshape(B*N, -1).unsqueeze(-1), self.traj_token, h_feat)
		h_feat_angle = self.angle_mlp(angle.unsqueeze(-1)).reshape(h_feat.size(0), -1).unsqueeze(1)
		h_feat = torch.cat((h_feat, h_feat_pose, h_feat_text, h_feat_angle), dim=1)
		temp_mask = mask_[:,:,:8].reshape(-1,T_PAST).permute(1,0)
		temp_mask_text = mask_text[:,:,:8].reshape(-1,T_PAST).permute(1,0)
		temp_mask_pose = mask_pose[:,:,:8].reshape(-1,T_PAST).permute(1,0)
		temp_mask = torch.cat((temp_mask, temp_mask_pose, temp_mask_text, torch.zeros((1, temp_mask.shape[1]), dtype=torch.bool).cuda()), dim=0)
		
		h_feat = self.temporal_encoder(h_feat.permute(1,0,2), temp_mask)
		h_feat = h_feat.squeeze().unsqueeze(1)

		# mask_trm = ( (~mask_[:,:,:8]).sum(-1)>1 ).reshape(B, N)
		h_feat = h_feat.reshape(B, N, h_feat.shape[-1]).permute(1,0,2)
		h_feat_ = self.transformer_encoder(h_feat, src_key_padding_mask=~batch_mask)
		h_feat_ = h_feat + h_feat_

		return h_feat_



# class social_transformer_trajPose(nn.Module):
# 	def __init__(self, past_timesteps=8):
# 		super(social_transformer_trajPose, self).__init__()
# 		self.encode_past = nn.Sequential(
# 			nn.Linear(2, 128),
# 			nn.ReLU(inplace=True),
# 			nn.Dropout(0.1),
# 			nn.Linear(128, 128),
# 			nn.Dropout(0.1))
# 		self.encode_pose = nn.Sequential(
# 			nn.Linear(72, 128),
# 			nn.ReLU(inplace=True),
# 			nn.Dropout(0.1),
# 			nn.Linear(128, 128),
# 			nn.Dropout(0.1))
  
# 		# self.temporal_encoder = nn.GRU(96, 96, 1, batch_first=True)
# 		# self.encode_past = nn.Linear(48, 128, bias=False)
# 		self.temporal_encoder = TemporalEncoder(historical_steps=past_timesteps+1,
# 											embed_dim=128,
# 											num_heads=4,
# 											dropout=0.1,
# 											num_layers=2)
# 		self.mode_encoder = TemporalEncoder(historical_steps=2, # traj, pose
#                                                 embed_dim=128,
#                                                 num_heads=4,
#                                                 dropout=0.1,
#                                                 num_layers=2)
  
# 		self.layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=128)
# 		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)
# 		self.traj_token = nn.Parameter(torch.Tensor(past_timesteps, 128)).cuda()
# 		nn.init.normal_(self.traj_token, mean=0., std=.02)
# 		self.angle_mlp = nn.Sequential(
# 			nn.Linear(1, 128),
# 			nn.ReLU(inplace=True),
# 			nn.Dropout(0.1),
# 			nn.Linear(128, 128),
# 			nn.Dropout(0.1))

# 	def forward(self, h, mask, h_pose, mask_pose, h_text, mask_text, angle, batch_mask):
# 		'''
# 		h: batch_size, t, 2
# 		'''
# 		# print(h.shape)
# 		B, N, T_PAST, _ = h.shape
# 		batch_mask_ = batch_mask.reshape(-1).unsqueeze(-1).repeat(1,T_PAST).reshape(-1)
# 		mask_ = ~(mask==0)
# 		# h = h.reshape(B*N, -1)
  
# 		h_feat = self.encode_past(h)
# 		h_feat = h_feat.reshape(B*N*T_PAST, -1).unsqueeze(0)
# 		h_feat_pose = self.encode_pose(h_pose).reshape(B*N*T_PAST, -1).unsqueeze(0)
# 		h_feat_fused = torch.cat((h_feat, h_feat_pose), dim=0)
# 		fused_mask = torch.cat((mask_[:,:,:T_PAST].reshape(1, -1), mask_pose[:,:,:T_PAST].reshape(1, -1)), dim=0)
# 		fused_feat_ = torch.empty((B, N, T_PAST, 128)).cuda()
# 		fused_feat = self.mode_encoder(h_feat_fused[:, batch_mask_], fused_mask[:, batch_mask_])
# 		fused_feat_[batch_mask] = fused_feat.reshape(-1, T_PAST, 128)
# 		fused_feat_ = fused_feat_.reshape(B*N, T_PAST, -1)
  
# 		h_feat_angle = self.angle_mlp(angle.unsqueeze(-1)).reshape(B*N, -1).unsqueeze(1)
# 		fused_feat_ = torch.cat((fused_feat_, h_feat_angle), dim=1)
# 		temp_mask = mask_[:,:,:8].reshape(-1,T_PAST).permute(1,0)
# 		temp_mask = torch.cat((temp_mask, torch.zeros((1, temp_mask.shape[1]), dtype=torch.bool).cuda()), dim=0)
  
# 		h_feat = self.temporal_encoder(fused_feat_.permute(1,0,2), temp_mask)
# 		h_feat = h_feat.squeeze().unsqueeze(1)

# 		# mask_trm = ( (~mask_[:,:,:8]).sum(-1)>1 ).reshape(B, N)
# 		h_feat = h_feat.reshape(B, N, h_feat.shape[-1]).permute(1,0,2)
# 		h_feat_ = self.transformer_encoder(h_feat, src_key_padding_mask=~batch_mask)
# 		h_feat_ = h_feat + h_feat_

# 		return h_feat_


class TransformerDenoisingModel(Module):

	def __init__(self, modality='traj', context_dim=64, tf_layer=2):
		super().__init__()
		if modality == 'traj':
			self.encoder_context = social_transformer_traj()
		elif modality == 'traj_pose':
			self.encoder_context = social_transformer_trajPose()
		elif modality == 'traj_pose_text':
			self.encoder_context = social_transformer_trajPoseText()
		elif modality == 'traj_text':
			self.encoder_context = social_transformer_trajText()
		else:
			raise Exception('modality not supported')
		self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=12)
		self.concat1 = ConcatSquashLinear(2, 2*context_dim, context_dim*2+3)
		self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=2, dim_feedforward=2*context_dim)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
		self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim*2+3)
		self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim*2+3)
		self.linear = ConcatSquashLinear(context_dim//2, 2, context_dim*2+3)


	def forward(self, x, beta, context, mask, batch_mask):
		batch_size = x.size(0)
		beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
		mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
		context = self.encoder_context(context, mask)
		# context = context.view(batch_size, 1, -1)   # (B, 1, F)

		time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
		ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
		
		x = self.concat1(ctx_emb, x)
		final_emb = x.permute(1,0,2)
		final_emb = self.pos_emb(final_emb)
		
		trans = self.transformer_encoder(final_emb).permute(1,0,2)
		trans = self.concat3(ctx_emb, trans)
		trans = self.concat4(ctx_emb, trans)
		return self.linear(ctx_emb, trans)
	

	def generate_accelerate(self, x, beta, context, mask, x_pose, mask_pose, x_text, mask_text, batch_size, rotate_angles, batch_mask):
		BN = context.shape[0]
		B, N = batch_size, BN//batch_size
		# batch_size = x.size(0)
		beta = beta.view(beta.size(0), 1, 1)          # (B, 1, 1)
		mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
		context = context.reshape(B, N, -1, context.shape[-1])
		mask = mask.reshape(B, N, -1)
		context = self.encoder_context(context, mask, x_pose, mask_pose, x_text, mask_text, rotate_angles, batch_mask)
		# context = context.view(batch_size, 1, -1)   # (B, 1, F)
		context = context.reshape(B*N, 1, -1)
		context = torch.where(torch.isnan(context), torch.zeros_like(context), context)
		time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
		# time_emb: [11, 1, 3]
		# context: [11, 1, 128]
		ctx_emb = torch.cat([time_emb, context], dim=-1).repeat(1, 6, 1).unsqueeze(2)
		# x: 11, 10, 20, 2
		# ctx_emb: 11, 10, 1, 259
		x = self.concat1.batch_generate(ctx_emb, x).contiguous().view(-1, 12, 128)
		# x: 110, 20, 128
		final_emb = x.permute(1, 0, 2)
		final_emb = self.pos_emb(final_emb)
		
		trans = self.transformer_encoder(final_emb).permute(1, 0, 2).contiguous().view(-1, 6, 12, 128)
		# trans: 11, 10, 20, 128
		trans = self.concat3.batch_generate(ctx_emb, trans)
		trans = self.concat4.batch_generate(ctx_emb, trans)
		return self.linear.batch_generate(ctx_emb, trans)

class InitializationModel(nn.Module):
	def __init__(self, t_h: int=8, d_h: int=6, t_f: int=40, d_f: int=2, k_pred: int=20):
		'''
		Parameters
		----
		t_h: history timestamps,
		d_h: dimension of each historical timestamp,
		t_f: future timestamps,
		d_f: dimension of each future timestamp,
		k_pred: number of predictions.

		'''
		super(InitializationModel, self).__init__()
		self.n = k_pred
		self.input_dim = t_h * d_h
		self.output_dim = t_f * d_f * k_pred
		self.fut_len = t_f

		self.social_encoder = social_transformer_traj(t_h)
		self.ego_var_encoder = st_encoder()
		self.ego_mean_encoder = st_encoder()
		self.ego_scale_encoder = st_encoder()

		self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())

		self.var_decoder = MLP(128*2+32, self.output_dim, hid_feat=(1024, 1024), activation=nn.ReLU())
		self.mean_decoder = MLP(128*2, t_f * d_f, hid_feat=(128, 128), activation=nn.ReLU())
		self.scale_decoder = MLP(128*2, 1, hid_feat=(128, 128), activation=nn.ReLU())

	
	# def forward(self, x, x_pose, b_angle, mask=None):
	def forward(self, x, x_pose, b_angle, mask, mask_pose, batch_mask, x_text, mask_text):
		'''
		x: batch size, t_p, 6
		'''
		B, N, T_PAST, _ = x.shape
		mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
		social_embed = self.social_encoder(x, mask, x_pose, mask_pose, x_text, mask_text, b_angle, batch_mask)
		social_embed = social_embed.squeeze(1)
		social_embed = social_embed.reshape(B*N, -1).squeeze(1)
		x = x.reshape(B*N, T_PAST, -1)
		# B, 128
		
		ego_var_embed = self.ego_var_encoder(x, mask)
		ego_mean_embed = self.ego_mean_encoder(x, mask)
		ego_scale_embed = self.ego_scale_encoder(x, mask)
		# B, 128

		mean_total = torch.cat((ego_mean_embed, social_embed), dim=-1)
		
		guess_mean = self.mean_decoder(mean_total).contiguous().view(-1, self.fut_len, 2)

		scale_total = torch.cat((ego_scale_embed, social_embed), dim=-1)
		guess_scale = self.scale_decoder(scale_total)

		guess_scale_feat = self.scale_encoder(guess_scale)
		var_total = torch.cat((ego_var_embed, social_embed, guess_scale_feat), dim=-1)
		guess_var = self.var_decoder(var_total).reshape(x.size(0), self.n, self.fut_len, 2)

		return guess_var, guess_mean, guess_scale


class InitializationModel_trajPose(nn.Module):
	def __init__(self, t_h: int=8, d_h: int=6, t_f: int=40, d_f: int=2, k_pred: int=20):
		'''
		Parameters
		----
		t_h: history timestamps,
		d_h: dimension of each historical timestamp,
		t_f: future timestamps,
		d_f: dimension of each future timestamp,
		k_pred: number of predictions.

		'''
		super(InitializationModel_trajPose, self).__init__()
		self.n = k_pred
		self.input_dim = t_h * d_h
		self.output_dim = t_f * d_f * k_pred
		self.fut_len = t_f

		self.social_encoder = social_transformer_trajPose(t_h)
		self.ego_var_encoder = st_encoder()
		self.ego_mean_encoder = st_encoder()
		self.ego_scale_encoder = st_encoder()

		self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())

		self.var_decoder = MLP(128*2+32, self.output_dim, hid_feat=(1024, 1024), activation=nn.ReLU())
		self.mean_decoder = MLP(128*2, t_f * d_f, hid_feat=(128, 128), activation=nn.ReLU())
		self.scale_decoder = MLP(128*2, 1, hid_feat=(128, 128), activation=nn.ReLU())

	
	# def forward(self, x, x_pose, b_angle, mask=None):
	def forward(self, x, x_pose, b_angle, mask, mask_pose, batch_mask, x_text, mask_text):
		'''
		x: batch size, t_p, 6
		'''
		B, N, T_PAST, _ = x.shape
		mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
		social_embed = self.social_encoder(x, mask, x_pose, mask_pose, x_text, mask_text, b_angle, batch_mask)
		social_embed = social_embed.squeeze(1)
		social_embed = social_embed.reshape(B*N, -1).squeeze(1)
		x = x.reshape(B*N, T_PAST, -1)
		# B, 128
		
		ego_var_embed = self.ego_var_encoder(x, mask)
		ego_mean_embed = self.ego_mean_encoder(x, mask)
		ego_scale_embed = self.ego_scale_encoder(x, mask)
		# B, 128

		mean_total = torch.cat((ego_mean_embed, social_embed), dim=-1)
		
		guess_mean = self.mean_decoder(mean_total).contiguous().view(-1, self.fut_len, 2)

		scale_total = torch.cat((ego_scale_embed, social_embed), dim=-1)
		guess_scale = self.scale_decoder(scale_total)

		guess_scale_feat = self.scale_encoder(guess_scale)
		var_total = torch.cat((ego_var_embed, social_embed, guess_scale_feat), dim=-1)
		guess_var = self.var_decoder(var_total).reshape(x.size(0), self.n, self.fut_len, 2)

		return guess_var, guess_mean, guess_scale


class InitializationModel_trajPoseText(nn.Module):
	def __init__(self, t_h: int=8, d_h: int=6, t_f: int=40, d_f: int=2, k_pred: int=20):
		'''
		Parameters
		----
		t_h: history timestamps,
		d_h: dimension of each historical timestamp,
		t_f: future timestamps,
		d_f: dimension of each future timestamp,
		k_pred: number of predictions.

		'''
		super(InitializationModel_trajPoseText, self).__init__()
		self.n = k_pred
		self.input_dim = t_h * d_h
		self.output_dim = t_f * d_f * k_pred
		self.fut_len = t_f

		self.social_encoder = social_transformer_trajPoseText(t_h)
		self.ego_var_encoder = st_encoder()
		self.ego_mean_encoder = st_encoder()
		self.ego_scale_encoder = st_encoder()

		self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())

		self.var_decoder = MLP(128*2+32, self.output_dim, hid_feat=(1024, 1024), activation=nn.ReLU())
		self.mean_decoder = MLP(128*2, t_f * d_f, hid_feat=(128, 128), activation=nn.ReLU())
		self.scale_decoder = MLP(128*2, 1, hid_feat=(128, 128), activation=nn.ReLU())

	
	# def forward(self, x, x_pose, b_angle, mask=None):
	def forward(self, x, x_pose, b_angle, mask, mask_pose, batch_mask, x_text, mask_text):
		'''
		x: batch size, t_p, 6
		'''
		B, N, T_PAST, _ = x.shape
		mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
		social_embed = self.social_encoder(x, mask, x_pose, mask_pose, x_text, mask_text, b_angle, batch_mask)
		social_embed = social_embed.squeeze(1)
		social_embed = social_embed.reshape(B*N, -1).squeeze(1)
		x = x.reshape(B*N, T_PAST, -1)
		# B, 128
		
		ego_var_embed = self.ego_var_encoder(x, mask)
		ego_mean_embed = self.ego_mean_encoder(x, mask)
		ego_scale_embed = self.ego_scale_encoder(x, mask)
		# B, 128

		mean_total = torch.cat((ego_mean_embed, social_embed), dim=-1)
		
		guess_mean = self.mean_decoder(mean_total).contiguous().view(-1, self.fut_len, 2)

		scale_total = torch.cat((ego_scale_embed, social_embed), dim=-1)
		guess_scale = self.scale_decoder(scale_total)

		guess_scale_feat = self.scale_encoder(guess_scale)
		var_total = torch.cat((ego_var_embed, social_embed, guess_scale_feat), dim=-1)
		guess_var = self.var_decoder(var_total).reshape(x.size(0), self.n, self.fut_len, 2)

		return guess_var, guess_mean, guess_scale


class InitializationModel_trajText(nn.Module):
	def __init__(self, t_h: int=8, d_h: int=6, t_f: int=40, d_f: int=2, k_pred: int=20):
		'''
		Parameters
		----
		t_h: history timestamps,
		d_h: dimension of each historical timestamp,
		t_f: future timestamps,
		d_f: dimension of each future timestamp,
		k_pred: number of predictions.

		'''
		super(InitializationModel_trajText, self).__init__()
		self.n = k_pred
		self.input_dim = t_h * d_h
		self.output_dim = t_f * d_f * k_pred
		self.fut_len = t_f

		self.social_encoder = social_transformer_trajText(t_h)
		self.ego_var_encoder = st_encoder()
		self.ego_mean_encoder = st_encoder()
		self.ego_scale_encoder = st_encoder()

		self.scale_encoder = MLP(1, 32, hid_feat=(4, 16), activation=nn.ReLU())

		self.var_decoder = MLP(128*2+32, self.output_dim, hid_feat=(1024, 1024), activation=nn.ReLU())
		self.mean_decoder = MLP(128*2, t_f * d_f, hid_feat=(128, 128), activation=nn.ReLU())
		self.scale_decoder = MLP(128*2, 1, hid_feat=(128, 128), activation=nn.ReLU())

	
	# def forward(self, x, x_pose, b_angle, mask=None):
	def forward(self, x, x_pose, b_angle, mask, mask_pose, batch_mask, x_text, mask_text):
		'''
		x: batch size, t_p, 6
		'''
		B, N, T_PAST, _ = x.shape
		mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
		social_embed = self.social_encoder(x, mask, x_pose, mask_pose, x_text, mask_text, b_angle, batch_mask)
		social_embed = social_embed.squeeze(1)
		social_embed = social_embed.reshape(B*N, -1).squeeze(1)
		x = x.reshape(B*N, T_PAST, -1)
		# B, 128
		
		ego_var_embed = self.ego_var_encoder(x, mask)
		ego_mean_embed = self.ego_mean_encoder(x, mask)
		ego_scale_embed = self.ego_scale_encoder(x, mask)
		# B, 128

		mean_total = torch.cat((ego_mean_embed, social_embed), dim=-1)
		
		guess_mean = self.mean_decoder(mean_total).contiguous().view(-1, self.fut_len, 2)

		scale_total = torch.cat((ego_scale_embed, social_embed), dim=-1)
		guess_scale = self.scale_decoder(scale_total)

		guess_scale_feat = self.scale_encoder(guess_scale)
		var_total = torch.cat((ego_var_embed, social_embed, guess_scale_feat), dim=-1)
		guess_var = self.var_decoder(var_total).reshape(x.size(0), self.n, self.fut_len, 2)

		return guess_var, guess_mean, guess_scale

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
                
