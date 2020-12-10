import torch
import lightconvpoint.knn as nearest_neighbors

def knn(points, support_points, K):
    indices = nearest_neighbors.knn(points.cpu().detach(), support_points.cpu().detach(), K)
    if points.is_cuda:
        indices = indices.cuda()
    return indices
