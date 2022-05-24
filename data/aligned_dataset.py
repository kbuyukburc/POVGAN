import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
from glob import glob
import cv2

class POVGANDataset(BaseDataset):
    CAMERA_SENSORS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']    
    camera_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot        
        self.lidar_files = []
        self.image_files = []
        self.mask_files = []
        for cam_sensor in self.CAMERA_SENSORS:
            self.lidar_files += sorted(glob(os.path.join(self.root, f'LIDAR/{cam_sensor}/*.npy'))) + sorted(glob(os.path.join(self.root, f'LIDAR/{cam_sensor}/*.npz')))
            self.mask_files += sorted(glob(os.path.join(self.root, f'MASK/{cam_sensor}/*.jpg')))
            self.image_files += sorted(glob(os.path.join(self.root, f'{cam_sensor}/*.jpg')))
        assert len(self.image_files) == len(self.lidar_files), "Dataset corrupted"
        assert not(self.opt.mask) or len(self.mask_files), "Mask files not exist"
        self.dataset_size = len(self.lidar_files) 
        
    def __getitem__(self, index):
        M_tensor = 0
        lidar_path = self.lidar_files[index]
        if lidar_path[-3:] == 'npy':
            points = np.load(lidar_path)
        elif lidar_path[-3:] == 'npz':
            points = np.load(lidar_path)['arr_0']
        else:
            raise "Error"
        if self.opt.mask:
            mask_path = self.mask_files[index]
            mask = cv2.imread(mask_path)[..., 0]
            mask = ((mask == 255) * 5) + 1
            M_tensor = torch.FloatTensor(mask)
        points[0] = points[0] / 255
        points[1] = points[1] / 255
        A_tensor = torch.FloatTensor(points)
        image_path = self.image_files[index]
        img = Image.open(image_path).convert('RGB')
        B_tensor = self.camera_transform(img)
        input_dict = {'label': A_tensor, 'inst': 0, 'image': B_tensor, 
                'feat': 0, 'mask': M_tensor,
                'path': lidar_path, 'image_path': image_path}
        return input_dict
        
    def __len__(self):
        return len(self.image_files) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'POVGANDataset'

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'