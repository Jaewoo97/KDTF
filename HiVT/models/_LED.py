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
from utils_hivt_LED import *
NUM_Tau = 5


class LED(pl.LightningModule):

    def __init__(self,
                 historical_steps: int,
                 future_steps: int,
                 num_modes: int,
                 diff_steps: int,
                 diff_beta_start: float,
                 diff_beta_end: float,
                 diff_beta_schedule: str,
                 modality = 'traj', # 'traj', 'traj_pose'
                 onlyFullTimestep = 'False',
                 **kwargs) -> None:
        super(LED, self).__init__()
        self.save_hyperparameters()
        if 'jrdb' in kwargs['root']: self.datasetType = 'jrdb'
        elif 'sit' in kwargs['root']: self.datasetType = 'sit'
        elif 'ethucy' in kwargs['root']: self.datasetType = 'ethucy'
        else: raise Exception('Wrong dataset?')
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.batch_size = kwargs['train_batch_size']
        self.num_modes = num_modes
        self.max_epochs = kwargs['max_epochs']
        self.obs_and_pred = historical_steps + future_steps
        self.modality = modality
        self.lr = kwargs['lr']
        self.weight_decay = kwargs['weight_decay']
        self.T_max = kwargs['T_max']
        self.onlyFullTimestep = onlyFullTimestep
        
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
        
        # LED part
        if self.modality == 'traj':
            self.model_initializer = InitializationModel(t_h=8, d_h=2, t_f=12, d_f=2, k_pred=6)
        elif self.modality == 'traj_pose': 
            self.model_initializer = InitializationModel_trajPose(t_h=8, d_h=2, t_f=12, d_f=2, k_pred=6)
        elif self.modality == 'traj_pose_text': 
            self.model_initializer = InitializationModel_trajPoseText(t_h=8, d_h=2, t_f=12, d_f=2, k_pred=6)
        elif self.modality == 'traj_text': 
            self.model_initializer = InitializationModel_trajText(t_h=8, d_h=2, t_f=12, d_f=2, k_pred=6)

        # ------------------------- define diffusion parameters -------------------------
        self.n_steps = diff_steps # define total diffusion steps

        # make beta schedule and calculate the parameters used in denoising process.
        self.betas = self.make_beta_schedule(
            schedule=diff_beta_schedule, n_timesteps=self.n_steps, 
            start=diff_beta_start, end=diff_beta_end).cuda()

        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0).cuda()
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod).cuda()
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod).cuda()
        
		# ------------------------- define models -------------------------
        self.model = TransformerDenoisingModel(self.modality)
        
        self.pose_token = nn.Parameter(torch.Tensor(historical_steps, 72))
        self.traj_token = nn.Parameter(torch.Tensor(historical_steps, 4))
        nn.init.normal_(self.pose_token, mean=0., std=.02)
        nn.init.normal_(self.traj_token, mean=0., std=.02)
        self.traj_scale = 5
        self.temporal_reweight = torch.FloatTensor([13 - i for i in range(1, 13)]).cuda().unsqueeze(0).unsqueeze(0) / 10
        # self.modality_encoder = MultipleInputEmbedding_mask_v3(in_channels=[node_dim, 72, 312], out_channel=embed_dim, emb_numHead=emb_numHead, emb_dropout=emb_dropout, emb_numLayers=emb_numLayers)
        # self.pi_time = nn.Linear(historical_steps*embed_dim, embed_dim)

    
    def make_beta_schedule(self, schedule: str = 'linear', 
            n_timesteps: int = 1000, 
            start: float = 1e-5, end: float = 1e-2) -> torch.Tensor:
        '''
        Make beta schedule.

        Parameters
        ----
        schedule: str, in ['linear', 'quad', 'sigmoid'],
        n_timesteps: int, diffusion steps,
        start: float, beta start, `start<end`,
        end: float, beta end,

        Returns
        ----
        betas: Tensor with the shape of (n_timesteps)

        '''
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        return betas

    def extract(self, input, t, x):
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    def noise_estimation_loss(self, x, y_0, mask):
        batch_size = x.shape[0]
        # Select a random step for each example
        t = torch.randint(0, self.n_steps, size=(batch_size // 2 + 1,)).to(x.device)
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[:batch_size]
        # x0 multiplier
        a = self.extract(self.alphas_bar_sqrt, t, y_0)
        beta = self.extract(self.betas, t, y_0)
        # eps multiplier
        am1 = self.extract(self.one_minus_alphas_bar_sqrt, t, y_0)
        e = torch.randn_like(y_0)
        # model input
        y = y_0 * a + e * am1
        output = self.model(y, beta, x, mask)
        # batch_size, 20, 2
        return (e - output).square().mean()

    def p_sample(self, x, mask, cur_y, t):
        if t==0:
            z = torch.zeros_like(cur_y).to(x.device)
        else:
            z = torch.randn_like(cur_y).to(x.device)
        t = torch.tensor([t]).cuda()
        # Factor to the model output
        eps_factor = ((1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
        # Model output
        beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
        eps_theta = self.model(cur_y, beta, x, mask)
        mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
        # Generate z
        z = torch.randn_like(cur_y).to(x.device)
        # Fixed sigma
        sigma_t = self.extract(self.betas, t, cur_y).sqrt()
        sample = mean + sigma_t * z
        return (sample)

    def p_sample_accelerate(self, x, mask, x_pose, mask_pose, x_text, mask_text, cur_y, t, rotate_angles, batch_mask):
        if t==0:
            z = torch.zeros_like(cur_y).to(x.device)
        else:
            z = torch.randn_like(cur_y).to(x.device)
        t = torch.tensor([t]).cuda()
        # Factor to the model output
        eps_factor = ((1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
        # Model output
        beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
        eps_theta = self.model.generate_accelerate(cur_y, beta, x, mask, x_pose, mask_pose, x_text, mask_text, self.batch_size, rotate_angles, batch_mask)
        mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
        # Generate z
        z = torch.randn_like(cur_y).to(x.device)
        # Fixed sigma
        sigma_t = self.extract(self.betas, t, cur_y).sqrt()
        sample = mean + sigma_t * z * 0.00001
        return (sample)
    
    def p_sample_loop(self, x, mask, shape):
        self.model.eval()
        prediction_total = torch.Tensor().cuda()
        for _ in range(20):
            cur_y = torch.randn(shape).to(x.device)
            for i in reversed(range(self.n_steps)):
                cur_y = self.p_sample(x, mask, cur_y, i)
            prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
        return prediction_total

    def p_sample_loop_mean(self, x, mask, loc):
        prediction_total = torch.Tensor().cuda()
        for loc_i in range(1):
            cur_y = loc
            for i in reversed(range(NUM_Tau)):
                cur_y = self.p_sample(x, mask, cur_y, i)
            prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
        return prediction_total

    def p_sample_loop_accelerate(self, x, mask, x_pose, mask_pose, x_text, mask_text, loc, rotate_angles, batch_mask):
        '''
        Batch operation to accelerate the denoising process.

        x: [11, 10, 6]
        mask: [11, 11]
        cur_y: [11, 10, 20, 2]
        '''
        prediction_total = torch.Tensor().cuda()
        cur_y = loc[:, :10]
        for i in reversed(range(NUM_Tau)):
            cur_y = self.p_sample_accelerate(x, mask, x_pose, mask_pose, x_text, mask_text, cur_y, i, rotate_angles, batch_mask)
        # cur_y_ = loc[:, 10:]
        # for i in reversed(range(NUM_Tau)):
        #     cur_y_ = self.p_sample_accelerate(x, mask, cur_y_, i)
        # # shape: B=b*n, K=10, T, 2
        prediction_total = cur_y
        # prediction_total = torch.cat((cur_y_, cur_y), dim=1)
        return prediction_total

    # def forward(self, data: TemporalData, given_frames, metamask=None):
    #     if self.modality == 'traj_pose':
            
    #     elif self.modality == 'traj':
            
    #     else: raise Exception("Modality other than traj, traj_pose is not supported")
        
            
    #     return out_final[:, batch_mask], out_pi[batch_mask]

    def training_step(self, data, batch_idx):
        original_data = copy.deepcopy(data)
        inv_rot = data.rotate_mat[:, :2, :2].transpose(1, 2) 
        current_true = data.positions[:, self.historical_steps-1,:] == self.check_valid
        current_true = (current_true.sum(-1) == 2).unsqueeze(-1).repeat(1, self.future_steps) # If true, invalid agent
        data['padding_mask'][:, self.historical_steps:] = torch.logical_or(data['padding_mask'][:, self.historical_steps:], current_true)
        # org_y = data.y @ inv_rot
        # org_y = data.y
        b_x, b_y, padding_mask, b_x_p, b_y_p, pad_mask_p, batch_mask, b_angle, batch_x_pos, b_x_t, pad_mask_t = agent2batch_led(data)
        b_x = torch.cat((b_x, batch_x_pos), dim=-1)
        B, N, T_PAST, _ = b_x.shape
        _, _, T_FUT, _ = b_y.shape
        b_y = b_y / self.traj_scale

        sample_prediction, mean_estimation, variance_estimation = self.model_initializer(b_x, b_x_p, b_angle, padding_mask, pad_mask_p, batch_mask, b_x_t, pad_mask_t)
        sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
        loc = sample_prediction + mean_estimation[:, None]
        loc = torch.where(torch.isnan(loc), torch.zeros_like(loc).cuda(), loc)
        variance_estimation = torch.where(torch.isnan(variance_estimation), torch.zeros_like(variance_estimation).cuda(), variance_estimation)
        
        b_x = b_x.reshape(B*N, T_PAST, -1)
        padding_mask_ = padding_mask[:,:,:T_PAST].reshape(B*N, T_PAST)
        generated_y = self.p_sample_loop_accelerate(b_x, padding_mask_, b_x_p, pad_mask_p, b_x_t, pad_mask_t, loc, b_angle, batch_mask)        # 여기 debug
        b_y = b_y.reshape(B*N, T_FUT, -1)
        reg_mask_fut = (~padding_mask[:,:,T_PAST:]).reshape(-1, T_FUT).unsqueeze(1).repeat(1, self.num_modes, 1)
        
        valid_mask = reg_mask_fut[:,0,:].sum(-1) > 0
        loss_dist_all = (	(generated_y - b_y.unsqueeze(dim=1)).norm(p=2, dim=-1) * reg_mask_fut * self.temporal_reweight
                    ).mean(dim=-1).min(dim=1)[0]
        loss_dist_all = loss_dist_all[valid_mask].mean()
        loss_uncertainty_all = (torch.exp(-variance_estimation)
                                *
                            ((generated_y - b_y.unsqueeze(dim=1)).norm(p=2, dim=-1) * reg_mask_fut ).mean(dim=(1, 2)) 
                                + 
                            variance_estimation
                            ).mean()
        loss_all = loss_dist_all*50 + loss_uncertainty_all
        if self.onlyFullTimestep:
            loss = loss_all
            loss_1, loss_2 = 0, 0
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
            
            current_true = data.positions[:, self.historical_steps-1,:] == self.check_valid
            current_true = (current_true.sum(-1) == 2).unsqueeze(-1).repeat(1, self.future_steps) # If true, invalid agent
            data['padding_mask'][:, self.historical_steps:] = torch.logical_or(data['padding_mask'][:, self.historical_steps:], current_true)
            b_x, b_y, padding_mask, b_x_p, b_y_p, pad_mask_p, batch_mask, b_angle, b_x_pos, b_x_t, pad_mask_t = agent2batch_led(data)
            b_x = torch.cat((b_x, b_x_pos), dim=-1)
            B, N, T_PAST, _ = b_x.shape
            _, _, T_FUT, _ = b_y.shape
            b_y = b_y / self.traj_scale

            sample_prediction, mean_estimation, variance_estimation = self.model_initializer(b_x, b_x_p, b_angle, padding_mask, pad_mask_p, batch_mask, b_x_t, pad_mask_t)
            sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
            loc = sample_prediction + mean_estimation[:, None]
            loc = torch.where(torch.isnan(loc), torch.zeros_like(loc).cuda(), loc)
            variance_estimation = torch.where(torch.isnan(variance_estimation), torch.zeros_like(variance_estimation).cuda(), variance_estimation)
            
            b_x = b_x.reshape(B*N, T_PAST, -1)
            padding_mask_ = padding_mask[:,:,:T_PAST].reshape(B*N, T_PAST)
            generated_y = self.p_sample_loop_accelerate(b_x, padding_mask_, b_x_p, pad_mask_p, b_x_t, pad_mask_t, loc, b_angle, batch_mask)
            b_y = b_y.reshape(B*N, T_FUT, -1)
            reg_mask_fut = (~padding_mask[:,:,T_PAST:]).reshape(-1, T_FUT).unsqueeze(1).repeat(1, self.num_modes, 1)
            
            valid_mask = reg_mask_fut[:,0,:].sum(-1) > 0
            loss_dist_2 = (	(generated_y - b_y.unsqueeze(dim=1)).norm(p=2, dim=-1) * reg_mask_fut * self.temporal_reweight
                        ).mean(dim=-1).min(dim=1)[0]
            loss_dist_2 = loss_dist_2[valid_mask].mean()
            loss_uncertainty_2 = (torch.exp(-variance_estimation)
                                    *
                                ((generated_y - b_y.unsqueeze(dim=1)).norm(p=2, dim=-1) * reg_mask_fut ).mean(dim=(1, 2)) 
                                    + 
                                variance_estimation
                                ).mean()
            loss_2 = loss_dist_2*50 + loss_uncertainty_2
                    
            
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
            
            current_true = data.positions[:, self.historical_steps-1,:] == self.check_valid
            current_true = (current_true.sum(-1) == 2).unsqueeze(-1).repeat(1, self.future_steps) # If true, invalid agent
            data['padding_mask'][:, self.historical_steps:] = torch.logical_or(data['padding_mask'][:, self.historical_steps:], current_true)
            b_x, b_y, padding_mask, b_x_p, b_y_p, pad_mask_p, batch_mask, b_angle, b_x_pos, b_x_t, pad_mask_t = agent2batch_led(data)
            b_x = torch.cat((b_x, b_x_pos), dim=-1)
            B, N, T_PAST, _ = b_x.shape
            _, _, T_FUT, _ = b_y.shape
            b_y = b_y / self.traj_scale

            sample_prediction, mean_estimation, variance_estimation = self.model_initializer(b_x, b_x_p, b_angle, padding_mask, pad_mask_p, batch_mask, b_x_t, pad_mask_t)
            sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
            loc = sample_prediction + mean_estimation[:, None]
            loc = torch.where(torch.isnan(loc), torch.zeros_like(loc).cuda(), loc)
            variance_estimation = torch.where(torch.isnan(variance_estimation), torch.zeros_like(variance_estimation).cuda(), variance_estimation)
            
            b_x = b_x.reshape(B*N, T_PAST, -1)
            padding_mask_ = padding_mask[:,:,:T_PAST].reshape(B*N, T_PAST)
            generated_y = self.p_sample_loop_accelerate(b_x, padding_mask_, b_x_p, pad_mask_p, b_x_t, pad_mask_t, loc, b_angle, batch_mask)
            b_y = b_y.reshape(B*N, T_FUT, -1)
            reg_mask_fut = (~padding_mask[:,:,T_PAST:]).reshape(-1, T_FUT).unsqueeze(1).repeat(1, self.num_modes, 1)
            
            valid_mask = reg_mask_fut[:,0,:].sum(-1) > 0
            loss_dist_1 = (	(generated_y - b_y.unsqueeze(dim=1)).norm(p=2, dim=-1) * reg_mask_fut * self.temporal_reweight
                        ).mean(dim=-1).min(dim=1)[0]
            loss_dist_1 = loss_dist_1[valid_mask].mean()
            loss_uncertainty_1 = (torch.exp(-variance_estimation)
                                    *
                                ((generated_y - b_y.unsqueeze(dim=1)).norm(p=2, dim=-1) * reg_mask_fut ).mean(dim=(1, 2)) 
                                    + 
                                variance_estimation
                                ).mean()
            loss_1 = loss_dist_1*50 + loss_uncertainty_1
            loss = loss_all*3 + loss_1 + loss_2
            
        self.log('train_dist_loss_all', loss_dist_all, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_loss_all', loss_all, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_loss_1', loss_1, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_loss_2', loss_2, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, data, batch_idx):
        data_original = copy.deepcopy(data)
        inv_rot = data.rotate_mat[:, :2, :2].transpose(1, 2) 
        # org_y = data.y @ inv_rot
        org_y = data.y 
        b_x, b_y, padding_mask, b_x_p, b_y_p, pad_mask_p, batch_mask, b_angle, b_x_pos, b_x_t, pad_mask_t = agent2batch_led(data)
        b_x = torch.cat((b_x, b_x_pos), dim=-1)
        B, N, T_PAST, _ = b_x.shape
        _, _, T_FUT, _ = b_y.shape
        b_y = b_y / self.traj_scale
        
        sample_prediction, mean_estimation, variance_estimation = self.model_initializer(b_x, b_x_p, b_angle, padding_mask, pad_mask_p, batch_mask, b_x_t, pad_mask_t)
        sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
        loc = sample_prediction + mean_estimation[:, None]
        loc = torch.where(torch.isnan(loc), torch.zeros_like(loc).cuda(), loc)
        variance_estimation = torch.where(torch.isnan(variance_estimation), torch.zeros_like(variance_estimation).cuda(), variance_estimation)
        
        b_x = b_x.reshape(B*N, T_PAST, -1)
        padding_mask_ = padding_mask[:,:,:T_PAST].reshape(B*N, T_PAST)
        y_hat = self.p_sample_loop_accelerate(b_x, padding_mask_, b_x_p, pad_mask_p, b_x_t, pad_mask_t, loc, b_angle, batch_mask)
        y_hat = y_hat[batch_mask.reshape(-1)].permute(1,0,2,3) * self.traj_scale
        
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
        b_x, b_y, padding_mask, b_x_p, b_y_p, pad_mask_p, batch_mask, b_angle, b_x_pos, b_x_t, pad_mask_t = agent2batch_led(data)
        b_x = torch.cat((b_x, b_x_pos), dim=-1)
        B, N, T_PAST, _ = b_x.shape
        _, _, T_FUT, _ = b_y.shape
        b_y = b_y / self.traj_scale
        
        sample_prediction, mean_estimation, variance_estimation = self.model_initializer(b_x, b_x_p, b_angle, padding_mask, pad_mask_p, batch_mask, b_x_t, pad_mask_t)
        sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
        loc = sample_prediction + mean_estimation[:, None]
        loc = torch.where(torch.isnan(loc), torch.zeros_like(loc).cuda(), loc)
        variance_estimation = torch.where(torch.isnan(variance_estimation), torch.zeros_like(variance_estimation).cuda(), variance_estimation)
        
        b_x = b_x.reshape(B*N, T_PAST, -1)
        padding_mask_ = padding_mask[:,:,:T_PAST].reshape(B*N, T_PAST)
        y_hat = self.p_sample_loop_accelerate(b_x, padding_mask_, b_x_p, pad_mask_p, b_x_t, pad_mask_t, loc, b_angle, batch_mask)
        y_hat = y_hat[batch_mask.reshape(-1)].permute(1,0,2,3) * self.traj_scale
        
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
        b_x, b_y, padding_mask, b_x_p, b_y_p, pad_mask_p, batch_mask, b_angle, b_x_pos, b_x_t, pad_mask_t = agent2batch_led(data)
        b_x = torch.cat((b_x, b_x_pos), dim=-1)
        B, N, T_PAST, _ = b_x.shape
        _, _, T_FUT, _ = b_y.shape
        b_y = b_y / self.traj_scale
        
        sample_prediction, mean_estimation, variance_estimation = self.model_initializer(b_x, b_x_p, b_angle, padding_mask, pad_mask_p, batch_mask, b_x_t, pad_mask_t)
        sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
        loc = sample_prediction + mean_estimation[:, None]
        loc = torch.where(torch.isnan(loc), torch.zeros_like(loc).cuda(), loc)
        variance_estimation = torch.where(torch.isnan(variance_estimation), torch.zeros_like(variance_estimation).cuda(), variance_estimation)
        
        b_x = b_x.reshape(B*N, T_PAST, -1)
        padding_mask_ = padding_mask[:,:,:T_PAST].reshape(B*N, T_PAST)
        y_hat = self.p_sample_loop_accelerate(b_x, padding_mask_, b_x_p, pad_mask_p, b_x_t, pad_mask_t, loc, b_angle, batch_mask)
        y_hat = y_hat[batch_mask.reshape(-1)].permute(1,0,2,3) * self.traj_scale
        
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
        b_x, b_y, padding_mask, b_x_p, b_y_p, pad_mask_p, batch_mask, b_angle, b_x_pos, b_x_t, pad_mask_t = agent2batch_led(data)
        b_x = torch.cat((b_x, b_x_pos), dim=-1)
        B, N, T_PAST, _ = b_x.shape
        _, _, T_FUT, _ = b_y.shape
        b_y = b_y / self.traj_scale
        
        sample_prediction, mean_estimation, variance_estimation = self.model_initializer(b_x, b_x_p, b_angle, padding_mask, pad_mask_p, batch_mask, b_x_t, pad_mask_t)
        sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
        loc = sample_prediction + mean_estimation[:, None]
        loc = torch.where(torch.isnan(loc), torch.zeros_like(loc).cuda(), loc)
        variance_estimation = torch.where(torch.isnan(variance_estimation), torch.zeros_like(variance_estimation).cuda(), variance_estimation)
        
        b_x = b_x.reshape(B*N, T_PAST, -1)
        padding_mask_ = padding_mask[:,:,:T_PAST].reshape(B*N, T_PAST)
        y_hat = self.p_sample_loop_accelerate(b_x, padding_mask_, b_x_p, pad_mask_p, b_x_t, pad_mask_t, loc, b_angle, batch_mask)
        y_hat = y_hat[batch_mask.reshape(-1)].permute(1,0,2,3) * self.traj_scale
        
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
        b_x, b_y, padding_mask, b_x_p, b_y_p, pad_mask_p, batch_mask, b_angle, b_x_pos, b_x_t, pad_mask_t = agent2batch_led(data)
        b_x = torch.cat((b_x, b_x_pos), dim=-1)
        B, N, T_PAST, _ = b_x.shape
        _, _, T_FUT, _ = b_y.shape
        b_y = b_y / self.traj_scale
        
        sample_prediction, mean_estimation, variance_estimation = self.model_initializer(b_x, b_x_p, b_angle, padding_mask, pad_mask_p, batch_mask, b_x_t, pad_mask_t)
        sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
        loc = sample_prediction + mean_estimation[:, None]
        loc = torch.where(torch.isnan(loc), torch.zeros_like(loc).cuda(), loc)
        variance_estimation = torch.where(torch.isnan(variance_estimation), torch.zeros_like(variance_estimation).cuda(), variance_estimation)
        
        b_x = b_x.reshape(B*N, T_PAST, -1)
        padding_mask_ = padding_mask[:,:,:T_PAST].reshape(B*N, T_PAST)
        y_hat = self.p_sample_loop_accelerate(b_x, padding_mask_, b_x_p, pad_mask_p, b_x_t, pad_mask_t, loc, b_angle, batch_mask)
        y_hat = y_hat[batch_mask.reshape(-1)].permute(1,0,2,3) * self.traj_scale
        
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
        
    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        ignore_count = 0
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                # print(full_param_name)
                if 'bias' in param_name:
                    if full_param_name in decay: 
                        ignore_count += 1
                        continue
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        if full_param_name in no_decay: 
                            ignore_count += 1
                            continue
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        if full_param_name in decay: 
                            ignore_count += 1
                            continue
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    if full_param_name in decay: 
                        ignore_count += 1
                        continue
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
        parser.add_argument('--lr', type=float, default=5e-5)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser
