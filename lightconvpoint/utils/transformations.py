import numpy as np
from torchvision import transforms
from PIL import Image
import random

class UnitBallNormalize:

    def __call__(self, points):  # from KPConv code
        pts = points[:,:3]
        pmin = np.min(pts, axis=0)
        pmax = np.max(pts, axis=0)
        pts -= (pmin + pmax) / 2
        scale = np.max(np.linalg.norm(pts, axis=1))
        pts *= 1.0 / scale
        points[:,:3] = pts
        return points


class NormalPerturbation:
    
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, points):
        pts = points[:,:3]
        pts = pts + self.sigma * np.random.normal(size=pts.shape)
        points[:,:3] = pts
        return points


class PillarSelection:

    def __init__(self, pillar_size, infinite_pillar_dim=2):
        self.pillar_size = pillar_size
        self.infinite_pillar_dim = infinite_pillar_dim

    def __call__(self, data, pillar_center=None):

        # data should have shape x,y,z,...
        if pillar_center is None:
            pillar_center = data[random.randint(0, data.shape[0]-1), :3]

        # compute the mask
        mask = None
        for i in range(pillar_center.shape[0]):
            if self.infinite_pillar_dim != i:
                mask_i = np.logical_and(data[:,i]<=pillar_center[i]+self.pillar_size/2,
                                        data[:,i]>=pillar_center[i]-self.pillar_size/2)
                if mask is None:
                    mask = mask_i
                else:
                    mask = np.logical_and(mask, mask_i)

        # apply the mask
        return data[mask]

class RandomSubSample:

    def __init__(self, number_of_points):
        self.n = number_of_points

    def __call__(self, data, return_choice=False):
        choice = np.random.choice(data.shape[0], self.n, replace=(data.shape[0] < self.n))
        if return_choice:
            return data[choice], choice
        else:
            return data[choice]


class FixedSubSample:

    # subsample by taking the k first points
    # this is no random

    def __init__(self, number_of_points):
        self.n = number_of_points

    def __call__(self, data, return_choice=False):
        if return_choice:
            return data[:self.n], np.arange(self.n)
        else:
            return data[:self.n]


class RandomRotation:

    def __init__(self, rotation_axis=2):
        self.rotation_axis=rotation_axis

    def __call__(self, points):

        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)

        if self.rotation_axis==2:
            rotation_matrix = np.array([[cosval, sinval, 0],
                                        [-sinval, cosval, 0],
                                        [0, 0, 1],])
        elif self.rotation_axis==1:
            rotation_matrix = np.array([[cosval, 0, sinval],
                                        [0, 1, 0]
                                        [-sinval, 0, cosval],])
        elif self.rotation_axis==0:
            rotation_matrix = np.array([[1, 0, 0],
                                        [0, cosval, sinval],
                                        [0, -sinval, cosval],])
        return points @ rotation_matrix


class ColorJittering:

    def __init__(self, jitter_value):
        self.transform = transforms.ColorJitter(
            brightness=jitter_value,
            contrast=jitter_value,
            saturation=jitter_value)


    def __call__(self, features):

        # features are considered to belong in [0,1]
        features = features * 255        
        features = features.astype(np.uint8)
        features = np.array(self.transform( Image.fromarray(np.expand_dims(features, 0))))
        features = np.squeeze(features, 0)
        return features.astype(np.float32)/255

class ColorDropout:

    def __init__(self, dropout_value):
        self.dropout_value = dropout_value
    
    def __call__(self, features):

        if np.random.rand() < self.dropout_value:
            return np.ones_like(features)
        return features


class NoColor:
    def __call__(self, features):
        return np.ones_like(features)