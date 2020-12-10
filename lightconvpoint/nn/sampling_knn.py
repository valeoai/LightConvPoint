import torch
import lightconvpoint.knn as nearest_neighbors


def batched_index_select(input, dim, index):
    index_shape = index.shape
    views = [input.shape[0]] + [
        1 if i != dim else -1 for i in range(1, len(input.shape))
    ]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index).view(input.size(0), index_shape[1], -1)

def search_base(search_function, points, K, ratio=1, dilation=1, support_points=None):

    search_K = K * dilation

    if support_points is None:
        if ratio==1:
            # support points are all points (knn computation)
            support_points = points
            indices = nearest_neighbors.knn(
                points.cpu().detach(), support_points.cpu().detach(), search_K
            )
            if points.is_cuda:
                indices = indices.cuda()
        elif ratio<1 and ratio>0:
            points = points.contiguous()
            support_point_number = max(1,int(points.shape[2] * ratio))
            support_points_ids, indices, _ = search_function(points.cpu().detach(), support_point_number, search_K)

            support_points_ids = support_points_ids.contiguous().long()
            indices = indices.contiguous().long()

            if points.is_cuda:
                indices = indices.cuda()
                support_points_ids = support_points_ids.cuda()
            support_points = batched_index_select(
                points.transpose(1, 2), dim=1, index=support_points_ids
            ).transpose(1, 2)
        else:
            raise ValueError(f"Search ConvPoint - ratio value error {ratio} should be in ]0,1]")
    else:
        # support points are known, only compute the knn
        indices = nearest_neighbors.knn(
            points.cpu().detach(), support_points.cpu().detach(), search_K
        )
        if points.is_cuda:
            indices = indices.cuda()

    if dilation > 1:
        indices = indices[:,:, torch.randperm(indices.size(2))]
        indices = indices[:,:,:K]

    return indices, support_points

def sampling_knn_convpoint(points, K, ratio=1, dilation=1, support_points=None):
    return search_base(nearest_neighbors.sampling_knn_convpoint, points, K, ratio, dilation, support_points)

def sampling_knn_quantized(points, K, ratio=1, dilation=1, support_points=None):
    return search_base(nearest_neighbors.sampling_knn_quantized, points, K, ratio, dilation, support_points)

def sampling_knn_fps(points, K, ratio=1, dilation=1, support_points=None):
    return search_base(nearest_neighbors.sampling_knn_fps, points, K, ratio, dilation, support_points)

def sampling_knn_random(points, K, ratio=1, dilation=1, support_points=None):
    return search_base(nearest_neighbors.sampling_knn_random, points, K, ratio, dilation, support_points)
