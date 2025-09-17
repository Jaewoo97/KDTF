import cv2
import numpy as np
import open3d as o3d
import torch
import yaml
from tqdm import tqdm
import natsort

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import figure
import random
from multiprocessing import Process

from PIL import Image
import os
import glob

class ObjectGetter:
    def __init__(self, panoptic_result_root, output_root, pcd_roots):
        global_config = "/mnt/jaewoo4tb/textraj/parser_model/calibration/defaults.yaml"
        camera_config = "/mnt/jaewoo4tb/textraj/parser_model/calibration/cameras.yaml"

        with open(global_config) as f:
            self.global_config_dict = yaml.safe_load(f)

        with open(camera_config) as f:
            self.camera_config_dict = yaml.safe_load(f)
        
        self.__panoptic_result_root = panoptic_result_root
        self.__output_root = output_root
        self.__pcd_roots = pcd_roots

    def __project_velo_to_ref(self, pointcloud):
        pointcloud = pointcloud[:, [1, 2, 0]]
        pointcloud[:, 0] *= -1
        pointcloud[:, 1] *= -1

        return pointcloud    

    def __move_lidar_to_camera_frame(self, pointcloud, upper=True):
        # assumed only rotation about z axis
        if upper:
            pointcloud[:, :3] = \
                pointcloud[:, :3] - torch.Tensor(self.global_config_dict['calibrated']
                                                    ['lidar_upper_to_rgb']['translation']).type(pointcloud.type())
            theta = self.global_config_dict['calibrated']['lidar_upper_to_rgb']['rotation'][-1]
        else:
            pointcloud[:, :3] = \
                pointcloud[:, :3] - torch.Tensor(self.global_config_dict['calibrated']
                                                    ['lidar_lower_to_rgb']['translation']).type(pointcloud.type())
            theta = self.global_config_dict['calibrated']['lidar_lower_to_rgb']['rotation'][-1]

        rotation_matrix = torch.Tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).type(
            pointcloud.type())
        pointcloud[:, :2] = torch.matmul(rotation_matrix, pointcloud[:, :2].unsqueeze(2)).squeeze()
        pointcloud[:, :3] = self.__project_velo_to_ref(pointcloud[:, :3])
        return pointcloud

    def __calculate_median_param_value(self, param):
        if param == 'f_y':
            idx = 4
        elif param == 'f_x':
            idx = 0
        elif param == 't_y':
            idx = 5
        elif param == 't_x':
            idx = 2
        elif param == 's':
            idx = 1
        else:
            raise 'Wrong parameter!'

        omni_camera = ['sensor_0', 'sensor_2', 'sensor_4', 'sensor_6', 'sensor_8']
        parameter_list = []
        for sensor, camera_params in self.camera_config_dict['cameras'].items():
            if sensor not in omni_camera:
                continue
            K_matrix = camera_params['K'].split(' ')
            parameter_list.append(float(K_matrix[idx]))
        return np.median(parameter_list)

    def __project_ref_to_image_torch(self, pointcloud):
        img_shape = 3, self.global_config_dict['image']['height'], self.global_config_dict['image']['width']
        median_focal_length_y = self.__calculate_median_param_value(param='f_y')
        median_optical_center_y = self.__calculate_median_param_value(param='t_y')
        theta = (torch.atan2(pointcloud[:, 0], pointcloud[:, 2]) + np.pi) % (2 * np.pi)
        horizontal_fraction = theta / (2 * np.pi)
        x = (horizontal_fraction * img_shape[2]) % img_shape[2]
        y = - median_focal_length_y * (
                pointcloud[:, 1] * torch.cos(theta) / pointcloud[:, 2]) + median_optical_center_y
        pts_2d = torch.stack([x, y], dim=1)

        return pts_2d

    def __get_object_center(self, obj_points) -> torch.Tensor:
        object_center = {}
        for id, points in obj_points.items():
            object_center[id] = torch.mean(points, 0)
        
        return object_center    

    def get_object_points(self, frame, pc_proj_save=False):
        frame_str = f"{frame:06d}"
        seg_map_dir = f"{self.__panoptic_result_root}{frame_str}_seg_map.pt"
        seg_map = torch.load(seg_map_dir).cpu().numpy()
        points_2d = torch.Tensor()
        points_3d = torch.Tensor()
        pcd_dirs = [f"{self.__pcd_roots[0]}{frame_str}.pcd", f"{self.__pcd_roots[1]}{frame_str}.pcd"]
        for i, pcd_dir in enumerate(pcd_dirs):
            pcd = o3d.io.read_point_cloud(pcd_dir)
            points = torch.tensor(pcd.points).type(torch.float32)
            points_3d = torch.concat([points_3d, points])
            
            points = self.__move_lidar_to_camera_frame(points, upper=True if i == 0 else False)
            points = self.__project_ref_to_image_torch(points)
            points = torch.round(points).to(torch.int32)
            points_2d = torch.concat([points_2d, points])
        
        points_2d = points_2d.to(torch.int32)
        xs = torch.clamp(points_2d[:, 0], 0, 3760-1)
        ys = torch.clamp(points_2d[:, 1], 0, 480-1)
        points_2d = torch.stack([xs, ys]).T

        object_points = {}
        for point_2d, point_3d in zip(points_2d, points_3d):
            id = seg_map[point_2d[1]][point_2d[0]]
            if id in object_points:
                object_points[id].append(point_3d)
            
            else:
                object_points[id] = [point_3d]
        
        for id in object_points:
            object_points[id] = torch.stack(object_points[id])


        ### pc projection
        if pc_proj_save:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot()
            panoptic_img = plt.imread(f"{self.__panoptic_result_root}{frame_str}_seg_img.jpg")
            ax.imshow(panoptic_img)
            print(points_2d.shape)
            ax.scatter(points_2d[:, 0], points_2d[:, 1], s=0.1, linewidths=0)
            fig.savefig(f"{self.__output_root}{frame_str}_pc_project.jpg", dpi=1000)

        torch.save(object_points, f"{self.__output_root}{frame_str}_object_points.pt")

        object_center = self.__get_object_center(object_points)
        torch.save(object_center, f"{self.__output_root}{frame_str}_object_center.pt")



class PCViz:
    def __init__(self, panop_dict_root, pc_cluster_root, r=10, z_off=10):
        self.__panop_dict_root = panop_dict_root
        self.__pc_cluster_root = pc_cluster_root
        self.__stuff_list = torch.load("/mnt/jaewoo4tb/textraj/parser_model/stuff.pt")
        self.__stuff_colors = torch.load("/mnt/jaewoo4tb/textraj/parser_model/stuff_colors.pt")
        self.fig = plt.figure(figsize=(10, 10))

        self.__r = r
        self.__z_off = z_off

    def __draw_frame(self, frame, unwanted, ax):
        frame_str = f"{frame:06d}"
        object_points = torch.load(f"{self.__pc_cluster_root}{frame_str}_object_points.pt")
        object_center = torch.load(f"{self.__pc_cluster_root}{frame_str}_object_center.pt")
        stuff_dict = torch.load(f"{self.__panop_dict_root}{frame_str}_seg_dict.pt")
        
        font = {'family': 'sans-serif',
                'weight': 'normal',
                'size'  : 10}

        for id in range(1, len(object_points)):
            if id in unwanted:
                continue

            try:
                obj_color = np.array([self.__stuff_colors[stuff_dict[id - 1]["category_id"]]]) / 255 # use color pre-setted by category_id
                # obj_color = [[random.random() for _ in range(3)]] # use random color
                ax.scatter(object_points[id][:, 0].detach().cpu(), object_points[id][:, 1].detach().cpu(), object_points[id][:, 2].detach().cpu(), s=1.0, c=obj_color)
        
            except:
                # print("id:", id)
                # print("frame:", frame)
                pass
        
        for id in range(1, len(object_points)):
            if id in unwanted:
                continue
            
            try:
                ax.scatter(object_center[id][0], object_center[id][1], object_center[id][2], s=2, c='red')
                ax.text(object_center[id][0], object_center[id][1], object_center[id][2], self.__stuff_list[stuff_dict[id - 1]["category_id"]] + ": " + str(stuff_dict[id - 1]["id"]), fontdict=font)
            
            except:
                pass

    def __animate(self, i):
        self.fig.clear()
        ax = self.fig.add_subplot(projection="3d")

        ax.set_xlim([-self.__r, self.__r])
        ax.set_ylim([-self.__r, self.__r])
        ax.set_zlim([self.__z_off-self.__r, self.__z_off+self.__r])

        frame = i
        self.__draw_frame(frame, [], ax)

        azim = 30
        ax.view_init(elev=20, azim=azim)
        # print(i)
        return self.fig,

    def __anim_and_save(self, frame_from, frame_to):
        print(f"[SAVE] {frame_from} - {frame_to} SAVING...")
        
        anim = animation.FuncAnimation(self.fig, self.__animate, frames=range(frame_from, frame_to), blit=False)
        anim.save(f"/mnt/jaewoo4tb/textraj/parser_model/{frame_from}.gif", fps=60)
        print(f"[SAVE] {frame_from} - {frame_to} SAVE COMPLETE")
    
    def viz(self, name, process_num=5, step=18, total_frame=200):
        start = 0
        while start < total_frame:
            processes = []
            for _ in range(process_num):
                end = min(start + step, total_frame)
                if start == end:
                    break

                process = Process(target=self.__anim_and_save, args=(start, end))
                start = end
                processes.append(process)
            
            for process in processes:
                process.start()
            
            for process in processes:
                process.join()
        
        images = []
        gif_names = natsort.natsorted(glob.glob("/mnt/jaewoo4tb/textraj/parser_model/*.gif"))
        print(gif_names)
        for gif_name in gif_names:
            gif = Image.open(gif_name)
            images.append(gif)

        print(f"total {len(images)} gifs")

        # 첫 번째 이미지를 기준으로 새로운 GIF를 만듭니다.
        images[0].save(f"/mnt/jaewoo4tb/textraj/parser_model/pc_clustered_viz/{name}.gif",
                        save_all=True,
                        append_images=images[1:], 
                        duration=100, 
                        loop=0)
        
        for gif_name in gif_names:
            os.remove(gif_name)


if __name__ == "__main__":
    test = ObjectGetter("/mnt/jaewoo4tb/textraj/parser_model/panoptic_results/", \
                        "/mnt/jaewoo4tb/textraj/parser_model/pc_clustered/", \
                        ["/mnt/jaewoo4tb/t2p/jrdb/train_dataset/pointclouds/upper_velodyne/gates-to-clark-2019-02-28_1/", \
                         "/mnt/jaewoo4tb/t2p/jrdb/train_dataset/pointclouds/lower_velodyne/gates-to-clark-2019-02-28_1/", ])
    
    for i in tqdm(range(0, 100)):
        # test.get_object_points(i)
        pass

    viz = PCViz("/mnt/jaewoo4tb/textraj/parser_model/panoptic_results/", "/mnt/jaewoo4tb/textraj/parser_model/pc_clustered/", r=5, z_off=5)
    viz.viz("close_look", process_num=5, step=18, total_frame=100)
