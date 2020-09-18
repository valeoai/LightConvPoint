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


class SearchQuantized:
    """Search object for computing support points and neighborhoods with quantized
    support point search.

    Computes the support points and their K-nearest neighbors according to the strategy
    defined in LightConvPoint paper.

    # Arguments
        K: int.
            Size of the neighborhood.
        stride: int.
            Defaults to 1. Reduction factor for computing the number of support points
            (1 all input points are supoprt points).
        npoints: (optional) int.
            Defaults to None. Number of support points to be generated.
            (if used, overrides the stride)

    # Forward arguments
        points: 3-D torch tensor.
            The input points. Dimensions are (B, D, N) with B the batch size, D the
            dimension of the spatial space and N the number of input points.
        support_points: (optional) 3-D torch tensor.
            The support points to project features on. If not provided, use the `search`
            object of the layer to compute them.
            Dimensions are (B, D, N) with B the batch size, D the dimenstion of the
            spatial space and N the number of input points.

    # Returns
        support_points: 3-D torch tensor.
            The support points. If they were provided as an input, return the same
            tensor.
        indices: 3-D torch tensor.
            The indices of the neighboring points with respect to the support points.
            If they were provided as an input, return the same tensor.
    """

    def __init__(self, K, stride=1, npoints=None, dilation=1):
        self.K = K
        self.stride = stride
        self.npoints = npoints
        self.dilation = dilation

    def __call__(self, points, support_points=None):

        search_K = self.K * self.dilation

        if support_points is None and self.stride == 1 and (self.npoints is None):
            support_points = points

        if support_points is None:
            # no support points have been given
            points = points.contiguous()
            if self.stride > 1 or self.stride == 1 and (self.npoints is None):
                support_point_number = max(1, int(points.shape[2]) // self.stride)
            else:
                support_point_number = self.npoints
            support_points_ids, indices, _ = nearest_neighbors.quantized_pick_knn(
                points.cpu().detach(), support_point_number, search_K
            )

            support_points_ids = support_points_ids.contiguous().long()
            indices = indices.contiguous().long()

            if points.is_cuda:
                indices = indices.cuda()
                support_points_ids = support_points_ids.cuda()
            support_points = batched_index_select(
                points.transpose(1, 2), dim=1, index=support_points_ids
            ).transpose(1, 2)

        else:
            # support points are known, only compute the knn
            indices = nearest_neighbors.knn(
                points.cpu().detach(), support_points.cpu().detach(), search_K
            )
            if points.is_cuda:
                indices = indices.cuda()

        if self.dilation > 1:
            indices = indices[:,:, torch.randperm(indices.size(2))]
            indices = indices[:,:,:self.K]

        return indices, support_points
