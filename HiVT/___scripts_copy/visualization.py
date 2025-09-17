import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

def viz_trajectory(pred, gt, data, output_dir, batch_idx):
    # print(f'Visualizing trajectory of batch {batch_idx}')
    if not os.path.isdir(os.path.join(output_dir, 'viz_results', 'trajectory')): os.makedirs(os.path.join(output_dir, 'viz_results', 'trajectory'))
    # pred: B*N, T, 15, 3
    B = data.num_graphs
    T1 = data.x.shape[1]
    T2 = gt.shape[1] 
    x_gt = data.x.cpu().numpy()
    y_gt = gt.cpu().numpy()
    y_pred = pred[:, :, :2].cpu().detach().numpy()
    pos = data.positions.cpu().numpy()
    thetas = data.theta.cpu().numpy()

    
    for scene_idx in range(B):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        xy_mean = np.zeros((0, 2))
        scene_filter = (data.batch==scene_idx).cpu().numpy()
        x_scene_gt = x_gt[scene_filter]
        y_scene_gt = y_gt[scene_filter]
        y_scene_pred = y_pred[scene_filter]
        pos0 = pos[scene_filter, 19, :]
        theta = thetas[scene_idx]
        inv_rot = np.array([[np.cos(theta), np.sin(theta)],
                               [-np.sin(theta), np.cos(theta)]])
            
        N = x_scene_gt.shape[0]
        if N < 1: continue

        for agent_idx in range(N):
            
            agent_pos0 = pos0[agent_idx, :]
            
            x_scene_agent_gt = np.cumsum(x_scene_gt[agent_idx, :, :], axis=0)
            x_scene_agent_gt = np.matmul(x_scene_agent_gt, inv_rot) 
            x_scene_agent_gt = agent_pos0 - x_scene_agent_gt[19] + x_scene_agent_gt
            
            y_scene_agent_gt = y_scene_gt[agent_idx, :, :]
            y_scene_agent_gt = np.matmul(y_scene_agent_gt, inv_rot)
            y_scene_agent_gt = agent_pos0 - y_scene_agent_gt[0] + y_scene_agent_gt

            y_scene_agent_pred = y_scene_pred[agent_idx, :, :]
            y_scene_agent_pred = np.matmul(y_scene_agent_pred, inv_rot)
            y_scene_agent_pred = agent_pos0 - y_scene_agent_pred[0] + y_scene_agent_pred


            ax.plot(x_scene_agent_gt[:, 0], x_scene_agent_gt[:, 1], 'ko-', linewidth=0.25, markersize=0.5)
            ax.plot(y_scene_agent_gt[:, 0], y_scene_agent_gt[:, 1], 'bo-', linewidth=0.25, markersize=0.5)
            ax.plot(y_scene_agent_pred[:, 0], y_scene_agent_pred[:, 1], 'ro-', linewidth=0.25, markersize=0.5)
            
            #xy_mean = np.concatenate((xy_mean, np.expand_dims(x_scene_gt.mean(0)[:2], axis=0)), axis=0)
        #ax.set_xlim([xy_mean.mean(0)[0]-5, xy_mean.mean(0)[0]+5])
        #ax.set_ylim([xy_mean.mean(0)[1]-5, xy_mean.mean(0)[1]+5])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.savefig(f'{output_dir}/viz_results/trajectory/batch_{batch_idx}_scene_{scene_idx}.png')
        plt.close()
        plt.cla()
    # import pdb;pdb.set_trace()
