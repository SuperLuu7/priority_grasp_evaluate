""" Demo to evaluate predicting correct rate of priority_grasp.py results with model checkpoint-rs.tar by AP@50.
    Author: lll
"""
"""
Command using grasp-baseline demo.py:
1 CUDA_VISIBLE_DEVICES=0 python ./work/evaluate_priority_grasp.py --checkpoint_path logs/checkpoint-rs.tar
2 bash ./command_demo.sh

To use evaluate_priority_grasp.py you need to:


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
import math
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

# Process realsenese image and depth data, finally get point data:
# end_points, cloud, work_cloud
# end_points: means obj sampling points, used for compute grippers
# cloud: used for draw obj cloud
# work_cloud: used for dram whole image work spaces
# def get_and_process_rs_data(data_dir, index):
def get_and_process_rs_data(data_dir, class_num, i):
    # load data
    # color = np.array(Image.open('../realsense/color_'+str(index)+'.png'), dtype=np.float32) / 255.0
    # # print(color.shape[0])
    # # print(color.shape[1])
    # depth = np.array(Image.open('../realsense/depth_'+str(index)+'.png'))
    name_str = str(i) # file name number str
    color = np.array(Image.open('../realsense/priority_test5/color_'+ name_str+ '.png'), dtype=np.float32) / 255.0
    depth = np.array(Image.open('../realsense/priority_test5/depth_'+ name_str+ '.png'))
    # color = np.array(Image.open('../realsense/63_Color.png'), dtype=np.float32) / 255.0
    # depth = np.array(Image.open('../realsense/63_Depth.png'))
    # print(sum(sum(depth[:,:,0]==depth[:,:,1]))) # use for debug                          
    # print(sum(sum(depth[:,:,1]==depth[:,:,2]))) # use for debug
    #depth = (depth[:,:, 0] + depth[:,:, 1] + depth[:,:, 2])/3 # obtain one-dimensional depth
    #print(sum(depth > 0))
    #print(sum(depth < 0))
    
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
    
    # img=Image.fromarray(workspace_mask)
    # img.show()

    # get priority grasp obj  
    # lines = np.loadtxt('../yolov7-u7/seg/runs/predict-seg/exp6/63_Color.txt')#load str txt as numpu directly    
    lines = np.loadtxt('../yolov7-u7/seg/runs/predict-seg/exp23/color_'+ name_str+ '.txt')#load str txt as numpu directly
    # print(lines.shape[0]) # how many line
    # print(lines.shape[1]) # number of values each line

      
    workspace_mask = np.array(np.zeros([color.shape[0],color.shape[1]]), dtype = bool)
    workspace_mask[70:-50,110:-130]=1 # up down right left
    end_points_list = []
    cloud_list = []
    obj_xc_img = []
    obj_yc_img = []
    obj_w_img = []# w one class obj in one image
    obj_h_img = []
    for i in range(lines.shape[0]):
        #print(i)
        if class_num == np.array(lines[i][0]):

            score_obj = lines[i][5]
            if score_obj < 0.5: #filter out lower scoring targets
                continue
            x1 = round(lines[i][1]*color.shape[1])
            y1 = round(lines[i][2]*color.shape[0])
            w = round(lines[i][3]*color.shape[1])
            h = round(lines[i][4]*color.shape[0])
            # print(x1)
            # print(y1)
            # print(w)
            # print(h)
            pri_obj_mask = np.array(np.zeros([color.shape[0],color.shape[1]]), dtype = bool)
            pri_obj_mask[y1:y1+h, x1:x1+w] = 1
    
            # img=Image.fromarray(pri_obj_mask)
            # img.show()

            ## get valid points
            mask = (pri_obj_mask & (depth > 0))
            # mask = np.array(depth, dtype=bool)
            # generate cloud
            cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
            cloud_masked = cloud[mask]
            color_masked = color[mask]

            # sample obj points
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

            end_points_list.append(end_points)# points cloud zones for all obj. Uesd for predicting graspers
            cloud_list.append(cloud) # Cloud uesd for drawing
            obj_xc_img.append(lines[i][1]) 
            obj_yc_img.append(lines[i][2])
            obj_w_img.append(lines[i][3])
            obj_h_img.append(lines[i][4])

    # Generate entire point cloud work_cloud
    work_cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    work_mask = (workspace_mask & (depth > 0))
    
    work_cloud_masked = work_cloud[work_mask]
    work_color_masked = color[work_mask]
    
    # sample pointsf for cloud work_cloud
    if len(work_cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(work_cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(work_cloud_masked))
        idxs2 = np.random.choice(len(work_cloud_masked), cfgs.num_point-len(work_cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = work_cloud_masked[idxs]
    color_sampled = work_color_masked[idxs]

    # convert data for work_cloud
    work_cloud = o3d.geometry.PointCloud()
    work_cloud.points = o3d.utility.Vector3dVector(work_cloud_masked.astype(np.float32))
    work_cloud.colors = o3d.utility.Vector3dVector(work_color_masked.astype(np.float32))
    # print(len(end_points_list))
    return end_points_list, cloud_list, work_cloud, obj_xc_img, obj_yc_img, obj_w_img, obj_h_img

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

def vis_grasps(gg, cloud, work_cloud):
    gg.nms()  # non maximum suppression
    gg.sort_by_score()
    # gg = gg[:10]
    gg = gg[:50]
    # print(gg)

    grippers = gg.to_open3d_geometry_list()
    # o3d.visualization.draw_geometries([cloud])  #lll use for debug
    # o3d.visualization.draw_geometries([cloud, *grippers])
    # o3d.visualization.draw_geometries([work_cloud])  #lll use for debug
    # o3d.visualization.draw_geometries([work_cloud, *grippers])
    o3d.visualization.draw_geometries(geometry_list=[work_cloud, *grippers])

def evaluate_priority_grasp(data_dir):
    net = get_net() 
    threshold = 3 # 2,3,4
    print('threshold:obj_w_h/', threshold)
    #cor_len = [] # The number of correct predictions for one obj 
    #len_gg = [] # Total number of predicted poses for one obj
    cor_len_all = [] # The number of correct predictions for all obj in one dataset 
    len_gg_all = [] # Total number of predicted poses for one obj
    avg_correct = []
    for class_num in [2, 0, 1]: #class num repeats


        for i in range(1,151): #file names repeat
            end_points_list, cloud_list, work_cloud, obj_xc_img, obj_yc_img, obj_w_img, obj_h_img = get_and_process_rs_data(data_dir, class_num, i)
            
            for i in range(len(end_points_list)): # Loop for each target on each images
                gg = get_grasps(net, end_points_list[i])
                if cfgs.collision_thresh > 0:
                    gg = collision_detection(gg, np.array(work_cloud.points))
                gg.nms()  # non maximum suppression
                gg.sort_by_score()
   
                gg = gg[:50]

                cor_len = len(gg) # initial correct predict num
                len_gg = len(gg)
                for num_gg in range(len(gg)): # all prippers for one obj on one img
                    # if gg[num_gg].width < obj_w_img[i] and gg[num_gg].width < obj_h_img[i]:
                    #     cor_len = cor_len - 1                        
                    #     continue
                    if gg[num_gg].depth > obj_w_img[i]/threshold and gg[num_gg].depth > obj_h_img[i]/threshold:
                        cor_len = cor_len - 1
                        continue
                    # dis_graspc_objc = math.sqrt((gg[num_gg].translation[0] -obj_xc_img[i])**2 + (gg[num_gg].translation[1] -obj_yc_img[i])**2) #distance from grasp center to obj center
                    # print(dis_graspc_objc)
                    # print(gg[num_gg].translation[0])
                    # print(obj_xc_img[i])
                    # print(gg[num_gg].translation[1])
                    # print(obj_yc_img[i])
                    # if  dis_graspc_objc > obj_w_img[i]/2 and dis_graspc_objc > obj_h_img[i]/2:
                    #     cor_len = cor_len - 1
                    #     continue
                cor_len_all.append(cor_len)
                len_gg_all.append(len_gg)
        pre_correct = sum(cor_len_all)/sum(len_gg_all)
        print('Class_num: ', class_num, 'Prediction accuracy:', pre_correct)
        avg_correct.append(pre_correct)
    avg_correct = np.mean(avg_correct)
    print('Average prediction accuracy:', avg_correct)

def demo(data_dir):
    net = get_net()
    end_points_list, cloud_list, work_cloud = get_and_process_rs_data(data_dir) 
    # end_points, cloud = get_and_process_cloud_data(data_dir)
    for i in range(len(end_points_list)):
        gg = get_grasps(net, end_points_list[i])
        if cfgs.collision_thresh > 0:
            print(np.array(work_cloud.points))
            print(np.array(work_cloud.points).shape)
            gg = collision_detection(gg, np.array(work_cloud.points))
            print(len(gg))
            # print(gg[0])
        #vis_grasps(gg, cloud)
        vis_grasps(gg, cloud_list[i], work_cloud)

if __name__=='__main__':
    data_dir = '../realsense' # in order to fuse the results of the two models
    #demo(data_dir) # used for multi-image estimation
    evaluate_priority_grasp(data_dir)