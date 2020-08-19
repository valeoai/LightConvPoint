
import torch
import numpy as np
import lightconvpoint.nn
import os
import random
from torchvision import transforms
from PIL import Image
import time
from tqdm import *
from plyfile import PlyData, PlyElement
from lightconvpoint.nn import with_indices_computation_rotation

def gauss_clip(mu, sigma, clip):
    v = random.gauss(mu, sigma)
    v = max(min(v, mu + clip * sigma), mu - clip * sigma)
    return v

def uniform(bound):
    return bound * (2 * random.random() - 1)


def scaling_factor(scaling_param, method):
    try:
        scaling_list = list(scaling_param)
        return random.choice(scaling_list)
    except:
        if method == 'g':
            return gauss_clip(1.0, scaling_param, 3)
        elif method == 'u':
            return 1.0 + uniform(scaling_param)

class DatasetTrainVal():

    def __init__ (self, filelist, folder,
                    training=False, 
                    block_size=2,
                    npoints = 4096,
                    iteration_number = None,
                    jitter=0, rgb=True, scaling_param=0,
                    rgb_dropout=False,
                    network_function=None, network_fusion_function=None):

        self.training = training
        self.filelist = filelist
        self.folder = folder
        self.bs = block_size
        self.rgb = rgb
        self.npoints = npoints
        self.iterations = iteration_number
        self.verbose = False
        self.number_of_run = 10
        self.rgb_dropout = rgb_dropout

        # data augmentation at training
        self.jitter = jitter # 0.8 for more
        self.scaling_param = scaling_param

        self.transform = transforms.ColorJitter(
            brightness=jitter,
            contrast=jitter,
            saturation=jitter)

        if network_function is not None:
            self.net = network_function()
        else:
            self.net = None

        if network_fusion_function is not None:
            self.net_fusion = network_fusion_function()
        else:
            self.net_fusion = None

    @with_indices_computation_rotation
    def __getitem__(self, index):

        folder = self.folder
        if self.training or self.iterations is not None:
            index = random.randint(0, len(self.filelist)-1)
            dataset = self.filelist[index]
        else:
            dataset = self.filelist[index//self.number_of_run]

        filename_data = os.path.join(folder, dataset, 'xyzrgb.npy')
        xyzrgb = np.load(filename_data).astype(np.float32)

        # load labels
        filename_labels = os.path.join(folder, dataset, 'label.npy')
        if self.verbose:
            print('{}-Loading {}...'.format(datetime.now(), filename_labels))
        labels = np.load(filename_labels).astype(int).flatten()

        # pick a random point
        pt_id = random.randint(0, xyzrgb.shape[0]-1)
        pt = xyzrgb[pt_id, :3]

        mask_x = np.logical_and(xyzrgb[:,0]<pt[0]+self.bs/2, xyzrgb[:,0]>pt[0]-self.bs/2)
        mask_y = np.logical_and(xyzrgb[:,1]<pt[1]+self.bs/2, xyzrgb[:,1]>pt[1]-self.bs/2)
        mask = np.logical_and(mask_x, mask_y)
        pts = xyzrgb[mask]
        lbs = labels[mask]

        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]
        lbs = lbs[choice]

        # get the colors
        features = pts[:,3:]

        # apply jitter if trainng
        if self.training and self.jitter > 0:
            features = features.astype(np.uint8)
            features = np.array(self.transform( Image.fromarray(np.expand_dims(features, 0)) ))
            features = np.squeeze(features, 0)
        
        features = features.astype(np.float32)
        features = features / 255 - 0.5

        pts = pts[:,:3]

        if self.training and self.scaling_param > 0:
            pts -= pts.mean(0)
            pts *= scaling_factor(self.scaling_param, 'g')
    
        # convert to torch
        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(features).float()
        lbs = torch.from_numpy(lbs).long()

        if (not self.rgb) or (self.training and self.rgb_dropout and random.randint(0,1)):
            fts = torch.ones(fts.shape).float()

        # transpose for lightconvpoint
        pts = pts.transpose(0,1)
        fts = fts.transpose(0,1)

        return_dict = {
            "pts": pts,
            "features": fts,
            "target": lbs,
        }

        return return_dict

    def __len__(self):
        if self.iterations is None:
            return len(self.filelist) * self.number_of_run
        else:
            return self.iterations


class DatasetTest():


    def compute_mask(self, pt, bs):
        # build the mask
        mask_x = np.logical_and(self.xyzrgb[:,0]<=pt[0]+bs/2, self.xyzrgb[:,0]>=pt[0]-bs/2)
        mask_y = np.logical_and(self.xyzrgb[:,1]<=pt[1]+bs/2, self.xyzrgb[:,1]>=pt[1]-bs/2)
        mask = np.logical_and(mask_x, mask_y)
        return mask

    def __init__ (self, filename, folder,
                    block_size=2,
                    npoints = 4096,
                    network_function=None,
                    step=0.5, rgb=False, network_fusion_function=None):

        self.folder = folder
        self.bs = block_size
        self.npoints = npoints
        self.rgb = rgb

        # load data
        self.filename = filename
        filename_data = os.path.join(folder, self.filename, 'xyzrgb.npy')
        self.xyzrgb = np.load(filename_data)

        # load the labels
        filename_labels = os.path.join(folder, self.filename, 'label.npy')
        self.labels = np.load(filename_labels).astype(int).flatten()

        # compute occupation grid
        mini = self.xyzrgb[:,:2].min(0)
        discretized = ((self.xyzrgb[:,:2]-mini).astype(float)/step).astype(int)
        self.pts = np.unique(discretized, axis=0)
        self.pts = self.pts.astype(np.float)*step + mini + step/2

        # compute the masks
        self.choices = []
        self.pts_ref = []
        for index in tqdm(range(self.pts.shape[0]), ncols=80):
            pt_ref = self.pts[index]
            mask = self.compute_mask(pt_ref, self.bs)

            pillar_points_indices = np.where(mask)[0]
            valid_points_indices = pillar_points_indices.copy()

            while(valid_points_indices is not None):
                # print(valid_points_indices.shape[0])
                if valid_points_indices.shape[0] > self.npoints:
                    choice = np.random.choice(valid_points_indices.shape[0], self.npoints, replace=True)
                    mask[valid_points_indices[choice]] = False
                    choice = valid_points_indices[choice]
                    valid_points_indices = np.where(mask)[0]
                else:
                    choice = np.random.choice(pillar_points_indices.shape[0], self.npoints-valid_points_indices.shape[0], replace=True)
                    choice = np.concatenate([valid_points_indices, pillar_points_indices[choice]], axis=0)
                    valid_points_indices = None

                self.choices.append(choice)
                self.pts_ref.append(pt_ref)


        # for fast compute
        if network_function is not None:
            self.net = network_function()
        else:
            self.net = None

        if network_fusion_function is not None:
            self.net_fusion = network_fusion_function()
        else:
            self.net_fusion = None

    @with_indices_computation_rotation
    def __getitem__(self, index):

        choice = self.choices[index]
        pts = self.xyzrgb[choice]
        pt_ref = self.pts_ref[index]
        lbs = self.labels[choice]

        features = pts[:,3:6] / 255 - 0.5
        pts = pts[:,:3].copy()

        # convert to torch
        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(features).float()
        choice = torch.from_numpy(choice).long()

        if not self.rgb:
            fts = torch.ones(fts.shape).float()
        
        # transpose for lightconvpoint
        pts = pts.transpose(0,1)
        fts = fts.transpose(0,1)

        return_dict = {
            "pts": pts,
            "features": fts,
            "target": lbs,
            "pts_ids": choice
        }

        return return_dict

    def __len__(self):
        # return len(self.pts)
        return len(self.choices)