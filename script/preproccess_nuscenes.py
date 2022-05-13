from nuscenes.nuscenes import NuScenes
from os import path
import cv2
from tqdm import tqdm
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import os
import argparse

parser = argparse.ArgumentParser("NuScenes")
parser.add_argument("--dataset", "-d", \
        type=str, help="Dataset folder of NuScenes", \
        default="~/repo/nuscenes/v1.0-mini")
parser.add_argument("--size", "-s", \
        type=int, help="Size of the image", \
        default=256)
parser.add_argument("--output", "-o", \
        type=str, help="Output folder",
        default="./dataset")
args = parser.parse_args()

nusc = NuScenes(version='v1.0-trainval', dataroot='/home/kbuyukburc/repo/nuscenes/v1.0-trainval01_blobs', verbose=True)

def calibDictToMatrix(calib_dict : dict) -> np.ndarray:
    quat = Quaternion(calib_dict['rotation'])
    matrix = np.eye(4)
    matrix[:3,:3] = quat.rotation_matrix
    matrix[:3, 3] = calib_dict['translation']
    return matrix

scene_dict = nusc.scene[0]
first_sample_token = scene_dict['first_sample_token']
last_sample_token = scene_dict['last_sample_token']
first_sample = nusc.get('sample', first_sample_token)
last_sample = nusc.get('sample', last_sample_token)

dataset_folder = path.join(f"{args.output}_{args.size}")
projected_folder = path.join(f"{dataset_folder}", "PROJECTED")
CAMERA_SENSORS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

try:
    os.mkdir(dataset_folder)
    os.mkdir(path.join(dataset_folder, 'LIDAR'))
except:
    pass
try:
    os.mkdir(projected_folder)    
except:
    pass
for cam_sensor in CAMERA_SENSORS:
    try:
        os.mkdir(path.join(dataset_folder, cam_sensor))      
    except:
        pass

for cam_sensor in CAMERA_SENSORS:
    try:
        os.mkdir(path.join(dataset_folder, 'LIDAR', cam_sensor))
    except:
        pass


for cam_sensor in CAMERA_SENSORS:
    try:
        os.mkdir(path.join(projected_folder, cam_sensor))
    except:
        pass

cntr = 0
scenes_num = len(nusc.scene)

for scene_num in tqdm(range(scenes_num)):
    scene_dict = nusc.scene[scene_num]
    first_sample_token = scene_dict['first_sample_token']
    last_sample_token = scene_dict['last_sample_token']
    first_sample = nusc.get('sample', first_sample_token)
    last_sample = nusc.get('sample', last_sample_token)
    current_sample = first_sample

    while current_sample['token'] != last_sample['token']:
        for cam_sensor in CAMERA_SENSORS:
            cam = nusc.get('sample_data', current_sample['data'][cam_sensor])
            cam_img_path = path.join(nusc.dataroot, cam['filename'])
            cam_img = cv2.imread(cam_img_path)
            cam = nusc.get('sample_data', current_sample['data'][cam_sensor])
            lidar_top = nusc.get('sample_data', current_sample['data']['LIDAR_TOP'])
            lidar_top_path = path.join(nusc.dataroot, lidar_top['filename'])
            lidar_top_bin = np.fromfile(lidar_top_path, dtype=np.float32).reshape(-1,5)[..., :4]    
            lidar_top_pointcloud = o3d.geometry.PointCloud()
            lidar_reflection = lidar_top_bin[:, 3]
            lidar_points = lidar_top_bin[..., :3]
            colors = np.c_[lidar_top_bin[..., 3], lidar_top_bin[..., 3], lidar_top_bin[..., 3]] / 255
            lidar_top_pointcloud.points = o3d.utility.Vector3dVector(lidar_points)
            lidar_top_pointcloud.colors = o3d.utility.Vector3dVector(colors)
            
            # Transformation Dict
            calib_lidar_to_vechile = nusc.get('calibrated_sensor', lidar_top['calibrated_sensor_token'])
            calib_vechile_to_global_lidar_time = nusc.get('ego_pose', lidar_top['ego_pose_token'])
            calib_vechile_to_global_camera_time = nusc.get('ego_pose', cam['ego_pose_token'])
            calib_camera_to_vechile =  nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
            # Transformation Matrix
            calib_lidar_to_vechile_matrix = calibDictToMatrix(calib_lidar_to_vechile)
            calib_vechile_to_global_lidar_time_matrix = calibDictToMatrix(calib_vechile_to_global_lidar_time)
            calib_vechile_to_global_camera_time_matrix = calibDictToMatrix(calib_vechile_to_global_camera_time)
            calib_camera_to_vechile_matrix = calibDictToMatrix(calib_camera_to_vechile)
            calib_lidar_to_camera_matrix = np.linalg.inv(calib_camera_to_vechile_matrix) @\
                                            np.linalg.inv(calib_vechile_to_global_camera_time_matrix) @ \
                                                calib_vechile_to_global_lidar_time_matrix @ \
                                                calib_lidar_to_vechile_matrix
            camera_top_pointcloud = lidar_top_pointcloud.transform(calib_lidar_to_camera_matrix)
            camera_points = np.asarray(lidar_top_pointcloud.points)

            rvec = R.from_matrix(calib_lidar_to_camera_matrix[:3, :3]).as_rotvec()
            tvec = calib_lidar_to_camera_matrix[:3, 3]
            depth = camera_points[:, 2]
            img_points, joc = cv2.projectPoints(lidar_points.T, rvec, tvec, np.array(calib_camera_to_vechile['camera_intrinsic']), np.array([[0,0,0,0,0]], dtype=np.float32))
            flag = (img_points[:,0, 0] < cam_img.shape[1]) & (img_points[:,0, 0] > 0) & \
                (img_points[:,0, 1] < cam_img.shape[0]) & (img_points[:,0, 1] > 0) & (depth > 0)
            img_points_flag = img_points[:,0,:][flag].astype(int)
            img = deepcopy(cam_img)
            #img[img_points_flag[:,1], img_points_flag[:,0]] = [0, 0 , 255]
            lidar_reflection_flag = lidar_reflection[flag]
            depth_flag = depth[flag]
            lidar_data = np.zeros((2,img.shape[0],img.shape[1]), dtype=np.float32)
            lidar_data[0, img_points_flag[:,1], img_points_flag[:,0]] = depth_flag
            lidar_data[1, img_points_flag[:,1], img_points_flag[:,0]] = lidar_reflection_flag

            img_points_flag = img_points_flag.copy()
            img_points_flag[:, 0] = img_points_flag[:, 0] * args.size / img.shape[1]
            img_points_flag[:, 1] = img_points_flag[:, 1] * args.size / img.shape[0]
            lidar_data = np.zeros((2, args.size, args.size), dtype=np.float32)
            lidar_data[0, img_points_flag[:,1], img_points_flag[:,0]] = depth_flag
            lidar_data[1, img_points_flag[:,1], img_points_flag[:,0]] = lidar_reflection_flag
            img = cv2.resize(img, (args.size, args.size))
            img_projected = deepcopy(img)
            img_projected[img_points_flag[:,1], img_points_flag[:,0]] = [0, 0 , 255]
            
            cv2.imwrite(path.join(dataset_folder, f'{cam_sensor}/{cntr}.jpg'), img)
            cv2.imwrite(path.join(projected_folder, f'{cam_sensor}/{cntr}.jpg'), img_projected)
            np.savez_compressed(path.join(dataset_folder, f'LIDAR/{cam_sensor}/{cntr}.npy'), lidar_data)

        cntr += 1
        current_sample = nusc.get('sample', current_sample['next'])
