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


class SearchRandom:
    def __init__(self, K, stride=1, npoints=-1):
        self.K = K
        self.stride = stride
        self.npoints = npoints

    def __call__(self, points, support_points):

        if support_points is None and self.stride == 1:
            support_points = points

        if support_points is None:
            # no support points have been given
            points = points.contiguous()
            if self.stride > 1 or self.stride == 1 and self.npoints == -1:
                support_point_number = max(1, int(points.shape[2]) // self.stride)
            else:
                support_point_number = self.npoints
            support_points_ids, indices, _ = nearest_neighbors.random_pick_knn(
                points.cpu().detach(), support_point_number, self.K
            )

            support_points_ids = support_points_ids.contiguous().long()
            indices = indices.contiguous().long()

            if points.is_cuda:
                indices = indices.cuda()
                support_points_ids = support_points_ids.cuda()
            support_points = batched_index_select(
                points.transpose(1, 2), dim=1, index=support_points_ids
            ).transpose(1, 2)

            return indices, support_points
        else:
            # support points are known, only compute the knn
            indices = nearest_neighbors.knn(
                points.cpu().detach(), support_points.cpu().detach(), self.K
            )
            if points.is_cuda:
                indices = indices.cuda()
            return indices, support_points
