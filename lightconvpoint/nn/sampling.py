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

def sampling_base(sampling_function, points, ratio=1, return_support_points=False):

    if ratio==1:
        support_points_ids = torch.arange(points.shape[2], dtype=torch.long, device=points.device)
        support_points_ids = support_points_ids.unsqueeze(0).expand(points.shape[0], points.shape[2])
    elif ratio>0 and ratio<1:
        support_point_number = max(1,int(points.shape[2] * ratio))
        support_points_ids = sampling_function(points.cpu().detach(), support_point_number)
        support_points_ids = support_points_ids.contiguous().long()
        if points.is_cuda:
            support_points_ids = support_points_ids.cuda()
    else:
        raise ValueError(f"Search ConvPoint - ratio value error {ratio} should be in ]0,1]")

    if return_support_points:
        support_points = batched_index_select(points.transpose(1, 2), dim=1, index=support_points_ids).transpose(1, 2)
        return support_points_ids, support_points
    else:
        return support_points_ids

def sampling_quantized(points, ratio=1, return_support_points=False):
    return sampling_base(nearest_neighbors.sampling_quantized, points, ratio, return_support_points)

def sampling_convpoint(points, ratio=1, return_support_points=False):
    raise NotImplementedError
    return sampling_base(nearest_neighbors.quantized_sampling.convpoint_sampling, points, ratio, return_support_points)

def sampling_fps(points, ratio=1, return_support_points=False):
    return sampling_base(nearest_neighbors.sampling_fps, points, ratio, return_support_points)

def sampling_random(points, ratio=1, return_support_points=False):
    return sampling_base(nearest_neighbors.sampling_random, points, ratio, return_support_points)

def sampling_apply_on_data(data, support_point_ids, dim=1):
    if dim==1:
        return batched_index_select(data, dim=dim, index=support_point_ids)
    elif dim==2:
        return batched_index_select(data.transpose(1,2), dim=1, index=support_point_ids).transpose(1,2)
    else:
        raise NotImplementedError