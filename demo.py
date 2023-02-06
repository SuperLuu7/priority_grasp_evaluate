""" Demo to show prediction results with model checkpoint-rs.tar.
    Author: lll
"""
"""
Command using grasp-baseline demo.py:
1 CUDA_VISIBLE_DEVICES=0 python demo.py --checkpoint_path logs/checkpoint-rs.tar
2 bash ./command_demo.sh

To use demo.py you need to:
Use aligned color and depth .png image, and the corresponding camera parameters file.txt.
When changing data, change all three(color.png depth.png camera_intrinsic.txt) at the 
same time. Make them correspond. Note that the factor_depth in realsense camera is 1000.

Generate the camera parameters corresponding to the image using the program: 
./realsense/rs_align_rgb_depth.py
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from scipy.fftpack import fft
from scipy import fft
from PIL import Image
import pandas as pd
import torch
from graspnetAPI import GraspGroup
import cv2
import linecache
from plyfile import PlyData, PlyElement
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(data_dir):
    # load data
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth'] #stands for the scale for depth value to be transformed into meters. 

    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud
#lll to process realsenese image and depth data, finally get point data
# def get_and_process_rs_data(data_dir, index):
def get_and_process_rs_data(data_dir):
    # load data
    index = 121 # file name
    color = np.array(Image.open('../realsense/priority_test4/color_'+str(index)+'.png'), dtype=np.float32) / 255.0
    # print(color.shape[0])
    # print(color.shape[1])
    depth = np.array(Image.open('../realsense/priority_test4/depth_'+str(index)+'.png'))
    # color = np.array(Image.open('../realsense/color_25.png'), dtype=np.float32) / 255.0
    # depth = np.array(Image.open('../realsense/depth_25.png'))
    # color = np.array(Image.open('../realsense/priority_test4/color_490.png'), dtype=np.float32) / 255.0
    # depth = np.array(Image.open('../realsense/priority_test4/depth_490.png'))
    # print(sum(sum(depth[:,:,0]==depth[:,:,1]))) # use for debug                          
    # print(sum(sum(depth[:,:,1]==depth[:,:,2]))) # use for debug
    # depth = depth[:,:, 0] # obtain one-dimensional depth
    #print(sum(depth > 0))
    #print(sum(depth < 0))
    
    # depth raw
    # depthraw = np.array(open("../realsense/14_Depth.raw", "rb").read())
    # # print(depthraw.shape)
    # # print(depthraw)
    # print(depthraw.dtype)
    
    # workspace_mask of demo
    # workspace_mask = cv2.imread(os.path.join(data_dir, 'workspace_mask.png'))
    # workspace_mask = cv2.resize(workspace_mask, (color.shape[1],color.shape[0]))
    # workspace_mask = workspace_mask[:,:, 0]
    # workspace_mask = np.array(workspace_mask, dtype = bool)
    
   
    #workspace_mask lll   
    workspace_mask = np.array(np.zeros([color.shape[0],color.shape[1]]), dtype = bool)
    # img=Image.fromarray(workspace_mask)
    # img.show()
    workspace_mask[70:-50,110:-130]=1 # up down right left 
    img=Image.fromarray(workspace_mask)
    # img.show()
    # print(workspace_mask)

    #Camera intrinsic for realsense viewer data
    # meta = pd.read_csv("../realsense/36_Depth_metadata.csv")   
    factor_depth = 1000. #1420. #stands for the scale for depth value to be transformed into meters. 
    # camera = CameraInfo(color.shape[1], color.shape[0], np.array(pd.to_numeric(meta.iloc[8][0])),  
    #                     np.array(pd.to_numeric(meta.iloc[9][0])), np.array(pd.to_numeric(meta.iloc[10][0])),
    #                     np.array(pd.to_numeric(meta.iloc[11][0])), factor_depth)
    # Camera intrinsic for video stream
    # txt: depth_scale fx fy ppx ppy
    camera = CameraInfo(color.shape[1], color.shape[0], 
                        float(linecache.getline('../realsense/camera_intrinsic.txt', 2)), 
                        float(linecache.getline('../realsense/camera_intrinsic.txt', 3)), 
                        float(linecache.getline('../realsense/camera_intrinsic.txt', 4)), 
                        float(linecache.getline('../realsense/camera_intrinsic.txt', 5)), factor_depth)
    # generate cloud
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    ## get valid points
    mask = (workspace_mask & (depth > 0))
    # mask = np.array(depth, dtype=bool)
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

#lll deiply different color between display and outlier points
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8]) # gray
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

def get_and_process_cloud_data(data_dir):
    # load data
    # Visualize ply cloud data
    cloud = o3d.io.read_point_cloud("../realsense/24.ply")
    # o3d.visualization.draw_geometries([cloud], zoom=0.3412,  #相机的缩放    
    #                               front=[0.4257, -0.2125, -0.8795], # 相机的前向量
    #                               lookat=[2.6172, 2.0475, 1.532], # 相机的观察向量
    #                               up=[-0.0694, -0.9768, 0.2024])  # 相机的向上向量
    # print(cloud)
    # print(len(cloud.points)) #use for debug

    #cloud = np.asarray(cloud.points) #using for debug
    # print(cloud.size)
    
    # remove outlier points
    # cl, ind = cloud.remove_statistical_outlier(nb_neighbors=200,  #用于指定邻域点的数量，以便计算平均距离。
    #                                                 std_ratio=1.0) #基于点云的平均距离的标准差来设置阈值。阈值越小，滤波效果越明显。
    # display_inlier_outlier(cloud, ind)
    # o3d.visualization.draw_geometries([cloud], zoom=0.3412, #相机的缩放
    #                               front=[0.4257, -0.2125, -0.8795], # 相机的前向量
    #                               lookat=[2.6172, 2.0475, 1.532], # 相机的观察向量
    #                               up=[-0.0694, -0.9768, 0.2024]) # 相机的向上向量
    # # sample points
    if len(cloud.points) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud.points), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud))
        idxs2 = np.random.choice(len(cloud.points), cfgs.num_point-len(cloud.points), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = np.asarray(cloud.points)[idxs]
    color_sampled = np.asarray(cloud.colors)[idxs]

    # convert data
    #cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud.points)
    cloud.colors = o3d.utility.Vector3dVector(cloud.colors)
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)   
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled
    
    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms() # non maximum suppression
    gg.sort_by_score()
    gg = gg[:50]    
    # gg = gg[18:19]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud])  #lll use for debug
    o3d.visualization.draw_geometries([cloud, *grippers])
    

def demos(data_dir):
    net = get_net()
    for i in [25, 50, 75, 100,125,150,175,200]:
        print(i)
        # end_points, cloud = get_and_process_data(data_dir)
        end_points, cloud = get_and_process_rs_data(data_dir, i) 
        # end_points, cloud = get_and_process_rs_data(data_dir)   
        # end_points, cloud = get_and_process_cloud_data(data_dir)

        gg = get_grasps(net, end_points)
        if cfgs.collision_thresh > 0:
            gg = collision_detection(gg, np.array(cloud.points))
        vis_grasps(gg, cloud)

def demo(data_dir):
    net = get_net()
    #end_points, cloud = get_and_process_data(data_dir)
    end_points, cloud = get_and_process_rs_data(data_dir) 
    # end_points, cloud = get_and_process_cloud_data(data_dir)

    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    vis_grasps(gg, cloud)

if __name__=='__main__':
    # data_dir = 'doc/example_data'
    data_dir = '../realsense' # in order to fuse the results of the two models
    #demos(data_dir) # used for multi-image estimation
    demo(data_dir)
