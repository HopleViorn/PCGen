import os
import math
import pickle
import json
import trimesh
import torch
from plyfile import PlyData
import numpy as np 
from tqdm import tqdm
import random
from multiprocessing.pool import Pool
from utils import (
    rotate_point_cloud,
    bbox_corners,
    rotate_axis,
    get_bbox,
    pad_repeat,
    pad_zero,
)

# furniture class labels
text2int = {'bathtub':0, 'bed':1, 'bench':2, 'bookshelf':3,'cabinet':4, 'chair':5, 'couch':6, 'lamp':7, 'sofa':8, 'table':9}


def normalize_point_cloud(points):
    """Normalize point cloud to unit sphere"""
    # Center the point cloud
    shifts = (points.max(axis=0) + points.min(axis=0)) / 2
    points = points - shifts
    
    # Scale to unit sphere
    distances = np.linalg.norm(points, axis=1)
    scale = 1 / np.max(distances)
    points *= scale
    
    return points


def resample_point_cloud(points, target_size):
    """Resample point cloud to target size"""
    if len(points) != target_size:
        indices = np.random.choice(len(points), target_size, replace=len(points) < target_size)
        points = points[indices]
    return points

def filter_data(data):
    """ 
    Helper function to check if a brep needs to be included
        in the training data or not 
    """
    data_path, max_face, max_edge, scaled_value, threshold_value, data_class = data
    # Load data 
    with open(data_path, "rb") as tf:
        data = pickle.load(tf)
    _, _, _, _, _, _, _, faceEdge_adj, surf_bbox, edge_bbox, _, _ = data.values()   
    
    skip = False

    # Skip over max size data
    if len(surf_bbox)>max_face:
        skip = True

    for surf_edges in faceEdge_adj:
        if len(surf_edges)>max_edge:
            skip = True 
    
    # Skip surfaces too close to each other
    surf_bbox = surf_bbox * scaled_value  # make bbox difference larger

    _surf_bbox_ = surf_bbox.reshape(len(surf_bbox),2,3)
    non_repeat = _surf_bbox_[:1]
    for bbox in _surf_bbox_:
        diff = np.max(np.max(np.abs(non_repeat - bbox),-1),-1)
        same = diff < threshold_value
        if same.sum()>=1:
            continue # repeat value
        else:
            non_repeat = np.concatenate([non_repeat, bbox[np.newaxis,:,:]],0)
    if len(non_repeat) != len(_surf_bbox_):
        skip = True

    # Skip edges too close to each other
    se_bbox = []
    for adj in faceEdge_adj:
        if len(edge_bbox[adj]) == 0: 
            skip = True
        se_bbox.append(edge_bbox[adj] * scaled_value)

    for bbb in se_bbox:
        _edge_bbox_ = bbb.reshape(len(bbb),2,3)
        non_repeat = _edge_bbox_[:1]
        for bbox in _edge_bbox_:
            diff = np.max(np.max(np.abs(non_repeat - bbox),-1),-1)
            same = diff < threshold_value
            if same.sum()>=1:
                continue # repeat value
            else:
                non_repeat = np.concatenate([non_repeat, bbox[np.newaxis,:,:]],0)
        if len(non_repeat) != len(_edge_bbox_):
            skip = True

    if skip: 
        return None, None 
    else: 
        return data_path, data_class


def load_data(input_data, input_list, validate, args):
    if input_list:
        # Filter data list
        with open(input_list, "rb") as tf:
            if validate:
                data_list = pickle.load(tf)['val']
            else:
                data_list = pickle.load(tf)['train']

        data_paths = []
        data_classes = []
        for uid in data_list:
            try:
                path = os.path.join(input_data, str(math.floor(int(uid.split('.')[0])/10000)).zfill(4), uid)
                class_label = -1  # unconditional generation (abc/deepcad)
            except Exception:
                path = os.path.join(input_data, uid)
                class_label = text2int[uid.split('/')[0]]  # conditional generation (furniture)
            data_paths.append(path)
            data_classes.append(class_label)
    else:
        print("No input list provided, loading from default json split...")
        json_path = '/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/data/filtered.json'
        with open(json_path, 'r') as f:
            data_split = json.load(f)
        
        if validate:
            data_list = data_split['validation']
        else:
            data_list = data_split['train']

        data_paths = []
        for item in data_list:
            sub_folder, file_no = item.split('/')
            pickle_filename = f"{sub_folder}_{file_no}.pkl"
            path = os.path.join(input_data, 'processed/cad_step', pickle_filename)
            data_paths.append(path)
        data_classes = [-1] * len(data_paths)
    
    # Filter data in parallel
    loaded_data = []
    params = zip(data_paths, [args.max_face]*len(data_paths), [args.max_edge]*len(data_paths),
                    [args.bbox_scaled]*len(data_paths), [args.threshold]*len(data_paths), data_classes)
    convert_iter = Pool(os.cpu_count()).imap(filter_data, params)
    for data_path, data_class in tqdm(convert_iter, total=len(data_paths)):
        if data_path is not None:
            if data_class<0: # abc or deepcad
                loaded_data.append(data_path)
            else:   # furniture
                loaded_data.append((data_path,data_class))

    print(f'Processed {len(loaded_data)}/{len(data_paths)}')
    return loaded_data


class SurfData(torch.utils.data.Dataset):
    """ Surface VAE Dataloader """
    def __init__(self, input_data, input_list, validate=False, aug=False):
        self.validate = validate
        self.aug = aug
        self.data = []

        if input_list:
            # Load validation data
            if self.validate:
                print('Loading validation data...')
                with open(input_list, "rb") as tf:
                    data_list = pickle.load(tf)['val']
                
                datas = []
                for uid in data_list:
                    try:
                        path = os.path.join(input_data, str(math.floor(int(uid.split('.')[0])/10000)).zfill(4), uid)
                    except Exception:
                        path = os.path.join(input_data, uid)
                    
                    with open(path, "rb") as tf:
                        data = pickle.load(tf)
                    _, _, surf_uv, _, _, _, _, _, _, _, _, _ = data.values()
                    datas.append(surf_uv)
                self.data = np.vstack(datas)

            # Load training data (deduplicated)
            else:
                print('Loading training data...')
                with open(input_list, "rb") as tf:
                    self.data = pickle.load(tf)
        else:
            print("No input list provided, scanning directory for data...")
            all_paths = []
            for root, _, files in os.walk(input_data):
                for file in files:
                    if file.endswith(".pkl"):
                        all_paths.append(os.path.join(root, file))
            
            datas = []
            for path in all_paths:
                with open(path, "rb") as tf:
                    data = pickle.load(tf)
                _, _, surf_uv, _, _, _, _, _, _, _, _, _ = data.values()
                datas.append(surf_uv)
            self.data = np.vstack(datas)

        print(len(self.data))
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        surf_uv = self.data[index]
        if np.random.rand()>0.5 and self.aug:
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270])
                surf_uv = rotate_point_cloud(surf_uv.reshape(-1, 3), angle, axis).reshape(32, 32, 3)
        return torch.FloatTensor(surf_uv)


class EdgeData(torch.utils.data.Dataset):
    """ Edge VAE Dataloader """
    def __init__(self, input_data, input_list, validate=False, aug=False):
        self.validate = validate
        self.aug = aug
        self.data = []

        if input_list:
            # Load validation data
            if self.validate:
                print('Loading validation data...')
                with open(input_list, "rb") as tf:
                    data_list = pickle.load(tf)['val']

                datas = []
                for uid in tqdm(data_list):
                    try:
                        path = os.path.join(input_data, str(math.floor(int(uid.split('.')[0])/10000)).zfill(4), uid)
                    except Exception:
                        path = os.path.join(input_data, uid)

                    with open(path, "rb") as tf:
                        data = pickle.load(tf)

                    _, _, _, edge_u, _, _, _, _, _, _, _, _ = data.values()
                    datas.append(edge_u)
                self.data = np.vstack(datas)

            # Load training data (deduplicated)
            else:
                print('Loading training data...')
                with open(input_list, "rb") as tf:
                    self.data = pickle.load(tf)
        else:
            print("No input list provided, scanning directory for data...")
            all_paths = []
            for root, _, files in os.walk(input_data):
                for file in files:
                    if file.endswith(".pkl"):
                        all_paths.append(os.path.join(root, file))
            
            datas = []
            for path in all_paths:
                with open(path, "rb") as tf:
                    data = pickle.load(tf)
                _, _, _, edge_u, _, _, _, _, _, _, _, _ = data.values()
                datas.append(edge_u)
            self.data = np.vstack(datas)

        print(len(self.data))
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        edge_u = self.data[index]
        # Data augmentation, randomly rotate 50% of the times
        if np.random.rand()>0.5 and self.aug:
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270])
                edge_u = rotate_point_cloud(edge_u, angle, axis)   
        return torch.FloatTensor(edge_u)
        

class SurfPosData(torch.utils.data.Dataset):
    """ Surface position (3D bbox) Dataloader """
    def __init__(self, input_data, input_list, validate=False, aug=False, args=None): 
        self.max_face = args.max_face
        self.max_edge = args.max_edge
        self.bbox_scaled = args.bbox_scaled
        self.aug = aug
        # Load data 
        self.data = load_data(input_data, input_list, validate, args)
        # Inflate furniture x50 times for training
        if len(self.data)<2000 and not validate:
            self.data = self.data*50
        return      

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load data
        data_class = None
        if isinstance(self.data[index], tuple):
            data_path, data_class = self.data[index]
        else:
            data_path = self.data[index]

        with open(data_path, "rb") as tf:
            data = pickle.load(tf)
        _, _, _, _, _, _, _, _, surf_pos, _, _, _ = data.values()

        # Data augmentation
        random_num = np.random.rand()

        if random_num>0.5 and self.aug:  
            # Get all eight corners
            surfpos_corners = bbox_corners(surf_pos)

            # Random rotation
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270]) 
                surfpos_corners = rotate_axis(surfpos_corners, angle, axis, normalized=True)
            
            # Re-compute the bottom left and top right corners 
            surf_pos = get_bbox(surfpos_corners)
            surf_pos = surf_pos.reshape(len(surf_pos),6)

        # Make bbox range larger
        surf_pos = surf_pos * self.bbox_scaled  

        # Randomly shuffle the sequence
        random_indices = np.random.permutation(surf_pos.shape[0])
        surf_pos = surf_pos[random_indices]
    
        # Padding
        surf_pos = pad_repeat(surf_pos, self.max_face)
        
        # Randomly shuffle the sequence
        random_indices = np.random.permutation(surf_pos.shape[0])
        surf_pos = surf_pos[random_indices]

        if data_class is not None:
            return (
                torch.FloatTensor(surf_pos), 
                torch.LongTensor([data_class+1]) # add 1, class 0 = uncond (furniture)
            )  
        else:
            return torch.FloatTensor(surf_pos) # abc or deepcad
    

class SurfZData(torch.utils.data.Dataset):
    """ Surface latent geometry Dataloader """
    def __init__(self, input_data, input_list, validate=False, aug=False, args=None): 
        self.max_face = args.max_face
        self.max_edge = args.max_edge
        self.bbox_scaled = args.bbox_scaled
        self.aug = aug
        # Load data 
        self.data = load_data(input_data, input_list, validate, args)
        # Inflate furniture x50 times for training
        if len(self.data)<2000 and not validate:
            self.data = self.data*50
        return  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load data
        data_class = None
        if isinstance(self.data[index], tuple):
            data_path, data_class = self.data[index]
        else:
            data_path = self.data[index]

        with open(data_path, "rb") as tf:
            data = pickle.load(tf)
        _, _, surf_ncs, _, _, _, _, _, surf_pos, _, _, _ = data.values()

        # Data augmentation
        random_num = np.random.rand()

        if random_num>0.5 and self.aug:
            # Get all eight corners
            surfpos_corners = bbox_corners(surf_pos)

            # Random rotation
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270]) 
                surfpos_corners = rotate_axis(surfpos_corners, angle, axis, normalized=True)
                surf_ncs = rotate_axis(surf_ncs, angle, axis, normalized=False)
            
            # Re-compute the bottom left and top right corners 
            surf_pos = get_bbox(surfpos_corners)
            surf_pos = surf_pos.reshape(len(surf_pos),6)

        # Make bbox range larger
        surf_pos = surf_pos * self.bbox_scaled  

        # Randomly shuffle the sequence
        random_indices = np.random.permutation(surf_pos.shape[0])
        surf_pos = surf_pos[random_indices]
        surf_ncs = surf_ncs[random_indices]
    
        # Pad data
        surf_pos, surf_mask = pad_zero(surf_pos, self.max_face, return_mask=True)
        surf_ncs = pad_zero(surf_ncs, self.max_face)

        if data_class is not None:
            return (
                torch.FloatTensor(surf_pos),
                torch.FloatTensor(surf_ncs),
                torch.BoolTensor(surf_mask),
                torch.LongTensor([data_class+1]) # add 1, class 0 = uncond (furniture)
            )  
        else:
            return (
                torch.FloatTensor(surf_pos),
                torch.FloatTensor(surf_ncs),
                torch.BoolTensor(surf_mask),
            ) # abc or deepcad
    

class EdgePosData(torch.utils.data.Dataset):
    """ Edge Position (3D bbox) Dataloader """
    def __init__(self, input_data, input_list, validate=False, aug=False, args=None): 
        self.max_face = args.max_face
        self.max_edge = args.max_edge
        self.bbox_scaled = args.bbox_scaled
        self.aug = aug
        self.data = []
        # Load data 
        self.data = load_data(input_data, input_list, validate, args)
        # Inflate furniture x50 times for training
        if len(self.data)<2000 and not validate:
            self.data = self.data*50
        return  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load data
        data_class = None
        if isinstance(self.data[index], tuple):
            data_path, data_class = self.data[index]
        else:
            data_path = self.data[index]

        with open(data_path, "rb") as tf:
            data = pickle.load(tf)
        
        _, _, surf_ncs, _, _, _, _, faceEdge_adj, surf_pos, edge_pos, _, _ = data.values()

        # Data augmentation
        random_num = np.random.rand()
        if random_num > 0.5 and self.aug:
            # Get all eight corners
            surfpos_corners = bbox_corners(surf_pos)
            edgepos_corners = bbox_corners(edge_pos)

            # Random rotation
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270]) 
                surfpos_corners = rotate_axis(surfpos_corners, angle, axis, normalized=True)
                edgepos_corners = rotate_axis(edgepos_corners, angle, axis, normalized=True)
                surf_ncs = rotate_axis(surf_ncs, angle, axis, normalized=False)
            
            # Re-compute the bottom left and top right corners 
            surf_pos = get_bbox(surfpos_corners)
            surf_pos = surf_pos.reshape(len(surf_pos),6)
            edge_pos = get_bbox(edgepos_corners)
            edge_pos = edge_pos.reshape(len(edge_pos),6)

        # Increase bbox value range
        surf_pos = surf_pos * self.bbox_scaled 
        edge_pos = edge_pos * self.bbox_scaled 

        # Mating duplication
        edge_pos_duplicated = []
        for adj in faceEdge_adj:
            edge_pos_duplicated.append(edge_pos[adj])

        # Randomly shuffle the edges per face
        edge_pos_new = []
        for pos in edge_pos_duplicated:
            random_indices = np.random.permutation(pos.shape[0])
            pos = pos[random_indices]
            pos = pad_repeat(pos, self.max_edge) #make sure some values are always repeated
            random_indices = np.random.permutation(pos.shape[0])
            pos = pos[random_indices]
            edge_pos_new.append(pos)
        edge_pos = np.stack(edge_pos_new)

        # Randomly shuffle the face sequence
        random_indices = np.random.permutation(surf_pos.shape[0])
        surf_pos = surf_pos[random_indices]
        edge_pos = edge_pos[random_indices]
        surf_ncs = surf_ncs[random_indices]
    
        # Padding
        surf_pos, surf_mask = pad_zero(surf_pos, self.max_face, return_mask=True)
        surf_ncs = pad_zero(surf_ncs, self.max_face)
        edge_pos = pad_zero(edge_pos, self.max_face)
    
        if data_class is not None:
            return (
                torch.FloatTensor(edge_pos), 
                torch.FloatTensor(surf_ncs), 
                torch.FloatTensor(surf_pos), 
                torch.BoolTensor(surf_mask), 
                torch.LongTensor([data_class+1]) # add 1, class 0 = uncond (furniture)
            )  
        else:
            return (
                torch.FloatTensor(edge_pos), 
                torch.FloatTensor(surf_ncs), 
                torch.FloatTensor(surf_pos), 
                torch.BoolTensor(surf_mask), 
            )# abc or deepcad
    
    

class EdgeZData(torch.utils.data.Dataset):
    """ Edge Latent z Dataloader """
    def __init__(self, input_data, input_list, validate=False, aug=False, args=None): 
        self.max_face = args.max_face
        self.max_edge = args.max_edge
        self.bbox_scaled = args.bbox_scaled
        self.aug = aug
        self.data = []
        # Load data 
        self.data = load_data(input_data, input_list, validate, args)
        # Inflate furniture x50 times for training
        if len(self.data)<2000 and not validate:
            self.data = self.data*50
        return 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load data
        data_class = None
        if isinstance(self.data[index], tuple):
            data_path, data_class = self.data[index]
        else:
            data_path = self.data[index]

        with open(data_path, "rb") as tf:
            data = pickle.load(tf)

        _, _, surf_ncs, edge_ncs, corner_wcs, _, _, faceEdge_adj, surf_pos, edge_pos, _, _ = data.values()

        # Data augmentation
        random_num = np.random.rand()
        if random_num > 0.5 and self.aug:
            # Get all eight corners
            surfpos_corners = bbox_corners(surf_pos)
            edgepos_corners = bbox_corners(edge_pos)

            # Random rotation
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270]) 
                surfpos_corners = rotate_axis(surfpos_corners, angle, axis, normalized=True)
                edgepos_corners = rotate_axis(edgepos_corners, angle, axis, normalized=True)
                corner_wcs = rotate_axis(corner_wcs, angle, axis, normalized=True)
                surf_ncs = rotate_axis(surf_ncs, angle, axis, normalized=False)
                edge_ncs = rotate_axis(edge_ncs, angle, axis, normalized=False)
            
            # Re-compute the bottom left and top right corners 
            surf_pos = get_bbox(surfpos_corners)
            surf_pos = surf_pos.reshape(len(surf_pos),6)
            edge_pos = get_bbox(edgepos_corners)
            edge_pos = edge_pos.reshape(len(edge_pos),6)

        # Increase value range
        surf_pos = surf_pos * self.bbox_scaled 
        edge_pos = edge_pos * self.bbox_scaled 
        corner_wcs = corner_wcs * self.bbox_scaled 

        # Mating duplication
        edge_pos_duplicated = []
        vertex_pos_duplicated = []
        edge_ncs_duplicated = []
        for adj in faceEdge_adj:
            edge_ncs_duplicated.append(edge_ncs[adj])
            edge_pos_duplicated.append(edge_pos[adj])
            corners = corner_wcs[adj]
            corners_sorted = []
            for corner in corners:
                sorted_indices = np.lexsort((corner[:, 2], corner[:, 1], corner[:, 0])) 
                corners_sorted.append(corner[sorted_indices].flatten()) # 1 x 6 corner pos
            corners = np.stack(corners_sorted)
            vertex_pos_duplicated.append(corners)

        # Edge Shuffle and Padding
        edge_pos_new = []
        edge_ncs_new = []
        vert_pos_new = []
        edge_mask = []
        for pos, ncs, vert in zip(edge_pos_duplicated, edge_ncs_duplicated, vertex_pos_duplicated):
            random_indices = np.random.permutation(pos.shape[0])
            pos = pos[random_indices]
            ncs = ncs[random_indices]
            vert = vert[random_indices]

            pos, mask = pad_zero(pos, self.max_edge, return_mask=True)
            ncs = pad_zero(ncs, self.max_edge)
            vert = pad_zero(vert, self.max_edge)
            
            edge_pos_new.append(pos)
            edge_ncs_new.append(ncs)
            edge_mask.append(mask)
            vert_pos_new.append(vert)

        edge_pos = np.stack(edge_pos_new)
        edge_ncs = np.stack(edge_ncs_new)
        edge_mask = np.stack(edge_mask)
        vertex_pos = np.stack(vert_pos_new)

        # Face Shuffle and Padding
        random_indices = np.random.permutation(surf_pos.shape[0])
        surf_pos = surf_pos[random_indices]
        edge_pos = edge_pos[random_indices]
        surf_ncs = surf_ncs[random_indices]
        edge_ncs = edge_ncs[random_indices]
        edge_mask = edge_mask[random_indices]
        vertex_pos = vertex_pos[random_indices]
    
        # Padding
        surf_pos = pad_zero(surf_pos, self.max_face)
        surf_ncs = pad_zero(surf_ncs, self.max_face)
        edge_pos = pad_zero(edge_pos, self.max_face)
        edge_ncs = pad_zero(edge_ncs, self.max_face)
        vertex_pos = pad_zero(vertex_pos, self.max_face)
        padding = np.zeros((self.max_face-len(edge_mask), *edge_mask.shape[1:]))==0
        edge_mask = np.concatenate([edge_mask, padding], 0)

        if data_class is not None:
            return (
                torch.FloatTensor(edge_ncs), 
                torch.FloatTensor(edge_pos), 
                torch.BoolTensor(edge_mask),
                torch.FloatTensor(surf_ncs), 
                torch.FloatTensor(surf_pos), 
                torch.FloatTensor(vertex_pos), 
                torch.LongTensor([data_class+1]) # add 1, class 0 = uncond (furniture)
            )
        else:
            return (
                torch.FloatTensor(edge_ncs), 
                torch.FloatTensor(edge_pos), 
                torch.BoolTensor(edge_mask),
                torch.FloatTensor(surf_ncs), 
                torch.FloatTensor(surf_pos), 
                torch.FloatTensor(vertex_pos),  # uncond deepcad/abc
            )
class CondSurfPosData(SurfPosData):
    """ Surface position (3D bbox) Dataloader with point cloud condition """
    def __init__(self, input_data, input_list, validate=False, aug=False, args=None):
        super().__init__(input_data, input_list, validate, aug, args)
        self.vae_encoder_type = getattr(args, 'vae_encoder_type', 'vecset')
        self.use_precomputed_cond = getattr(args, 'use_precomputed_cond', False)
        self.cond_dir = getattr(args, 'cond_dir', '/home/ljr/Hunyuan3D-2.1/RelatedWork/BrepGen/data/conditions')
        if self.use_precomputed_cond and self.cond_dir is None:
            raise ValueError("cond_dir must be provided when use_precomputed_cond is True")
        return
    
    def __getitem__(self, index):
        # Load data
        data_class = None
        if isinstance(self.data[index], tuple):
            data_path, data_class = self.data[index]
        else:
            data_path = self.data[index]

        with open(data_path, "rb") as tf:
            data = pickle.load(tf)
        _, _, _, _, _, _, _, _, surf_pos, _, _, _ = data.values()

        # Load point cloud embedding or pre-calculated condition
        filename_pkl = os.path.basename(data_path)
        filename_pkl_no_ext = os.path.splitext(filename_pkl)[0]
        parts = filename_pkl_no_ext.split('_')
        if len(parts) >= 2:
            subdir = parts[0]
            filename_base = '_'.join(parts[1:])
        else:
            subdir = os.path.basename(os.path.dirname(data_path))
            filename_base = filename_pkl_no_ext

        # Data augmentation
        random_num = np.random.rand()
        
        # --- Rotation Augmentation ---
        # We decide the rotation first, then load either the rotated PC or the rotated pre-computed condition
        rotation = None
        if random_num > 0.5 and self.aug:
            axis = random.choice(['x', 'y', 'z'])
            angle = random.choice([90, 180, 270])
            rotation = (axis, angle)

            # Apply rotation to surf_pos
            surfpos_corners = bbox_corners(surf_pos)
            surfpos_corners = rotate_axis(surfpos_corners, angle, axis, normalized=True)
            surf_pos = get_bbox(surfpos_corners)
            surf_pos = surf_pos.reshape(len(surf_pos), 6)

        if self.use_precomputed_cond:
            cond_item_dir = os.path.join(self.cond_dir, subdir, filename_base)
            if rotation:
                axis, angle = rotation
                cond_filename = f"rot_{axis}_{angle}.pkl"
            else:
                cond_filename = "no_rot.pkl"
            
            cond_path = os.path.join(cond_item_dir, cond_filename)
            with open(cond_path, 'rb') as f:
                condition = pickle.load(f)
            condition = torch.FloatTensor(condition).squeeze(0) # Remove batch dim from saved data
            point_cloud_or_cond = condition
        else:
            # --- Load Point Cloud (original logic) ---
            if self.vae_encoder_type == 'hy3dshape':
                pc_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(data_path))), 'pc_normal', subdir, filename_base + '.ply')
                plydata = PlyData.read(pc_path)
                vertex_data = plydata['vertex']
                points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
                normals = np.vstack([vertex_data['nx'], vertex_data['ny'], vertex_data['nz']]).T
                points = normalize_point_cloud(points)
                if len(points) != 81920:
                    indices = np.random.choice(len(points), 81920, replace=len(points) < 81920)
                    points = points[indices]
                    normals = normals[indices]
                sharp_edge_labels = np.zeros((points.shape[0], 1), dtype=np.float32)
                point_cloud = np.concatenate([points, normals, sharp_edge_labels], axis=1)
            else: # 'vecset' or default
                pc_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(data_path))), 'pc_cad', subdir, filename_base + '.ply')
                mesh = trimesh.load(pc_path)
                point_cloud = mesh.vertices
                point_cloud = normalize_point_cloud(point_cloud)
                point_cloud = resample_point_cloud(point_cloud, 8192)
            
            # --- Apply rotation to point cloud if augmented ---
            if rotation:
                axis, angle = rotation
                if self.vae_encoder_type == 'hy3dshape':
                    points = point_cloud[:, :3]
                    normals = point_cloud[:, 3:6]
                    rotated_points = rotate_point_cloud(points, angle, axis)
                    rotated_normals = rotate_point_cloud(normals, angle, axis)
                    sharp_edges = point_cloud[:, 6:]
                    point_cloud = np.concatenate([rotated_points, rotated_normals, sharp_edges], axis=1)
                else:
                    point_cloud = rotate_point_cloud(point_cloud, angle, axis)

            point_cloud_or_cond = torch.FloatTensor(point_cloud)

        # Make bbox range larger
        surf_pos = surf_pos * self.bbox_scaled

        # Randomly shuffle the sequence
        random_indices = np.random.permutation(surf_pos.shape[0])
        surf_pos = surf_pos[random_indices]
    
        # Padding
        surf_pos = pad_repeat(surf_pos, self.max_face)
        
        # Randomly shuffle the sequence
        random_indices = np.random.permutation(surf_pos.shape[0])
        surf_pos = surf_pos[random_indices]

        if data_class is not None:
            return (
                torch.FloatTensor(surf_pos),
                torch.LongTensor([data_class+1]), # add 1, class 0 = uncond (furniture)
                point_cloud_or_cond
            )
        else:
            return torch.FloatTensor(surf_pos), point_cloud_or_cond # abc or deepcad


class CondSurfZData(SurfZData):
    """ Surface latent geometry Dataloader with point cloud condition """
    def __init__(self, input_data, input_list, validate=False, aug=False, args=None):
        super().__init__(input_data, input_list, validate, aug, args)
        return

    def __getitem__(self, index):
        # Load data
        data_class = None
        if isinstance(self.data[index], tuple):
            data_path, data_class = self.data[index]
        else:
            data_path = self.data[index]

        with open(data_path, "rb") as tf:
            data = pickle.load(tf)
        _, _, surf_ncs, _, _, _, _, _, surf_pos, _, _, _ = data.values()

        # Load point cloud
        filename_pkl = os.path.basename(data_path)
        filename_pkl_no_ext = os.path.splitext(filename_pkl)[0]
        parts = filename_pkl_no_ext.split('_')
        if len(parts) >= 2:
            subdir = parts[0]
            filename_base = '_'.join(parts[1:])
        else:
            subdir = os.path.basename(os.path.dirname(data_path))
            filename_base = filename_pkl_no_ext
        
        pc_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(data_path))), 'pc_cad', subdir, filename_base + '.ply')
        mesh = trimesh.load(pc_path)
        point_cloud = mesh.vertices
        point_cloud = normalize_point_cloud(point_cloud)
        point_cloud = resample_point_cloud(point_cloud, 8192)
        point_cloud = torch.FloatTensor(point_cloud)

        # Data augmentation
        random_num = np.random.rand()

        if random_num>0.5 and self.aug:
            # Get all eight corners
            surfpos_corners = bbox_corners(surf_pos)

            # Random rotation
            for axis in ['x', 'y', 'z']:
                angle = random.choice([90, 180, 270]) 
                surfpos_corners = rotate_axis(surfpos_corners, angle, axis, normalized=True)
                surf_ncs = rotate_axis(surf_ncs, angle, axis, normalized=False)
                point_cloud = rotate_point_cloud(point_cloud.numpy(), angle, axis)
                point_cloud = torch.FloatTensor(point_cloud)
            
            # Re-compute the bottom left and top right corners 
            surf_pos = get_bbox(surfpos_corners)
            surf_pos = surf_pos.reshape(len(surf_pos),6)

        # Make bbox range larger
        surf_pos = surf_pos * self.bbox_scaled  

        # Randomly shuffle the sequence
        random_indices = np.random.permutation(surf_pos.shape[0])
        surf_pos = surf_pos[random_indices]
        surf_ncs = surf_ncs[random_indices]
    
        # Pad data
        surf_pos, surf_mask = pad_zero(surf_pos, self.max_face, return_mask=True)
        surf_ncs = pad_zero(surf_ncs, self.max_face)

        if data_class is not None:
            return (
                torch.FloatTensor(surf_pos),
                torch.FloatTensor(surf_ncs),
                torch.BoolTensor(surf_mask),
                torch.LongTensor([data_class+1]), # add 1, class 0 = uncond (furniture)
                point_cloud
            )  
        else:
            return (
                torch.FloatTensor(surf_pos),
                torch.FloatTensor(surf_ncs),
                torch.BoolTensor(surf_mask),
                point_cloud
            ) # abc or deepcad
