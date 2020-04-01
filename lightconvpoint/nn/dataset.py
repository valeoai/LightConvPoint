import torch
import functools
import numpy as np


def rotate_point_cloud(batch_data, rotation_matrix=None, inverse=False):
    """ Randomly rotate the point clouds to augument the dataset.

        # Arguments
            batch_data: 3-D array.
                Input batch points. Dimensions (B, N, 3) with B the batchsize, N the
                number of points and 3 the dimension of the spatial space.
            rotation_matrix: 2-D array.
                Defaults to `None`. If provided, the rotation matrix is applied on the
                points.
            inverse: Boolean.
                Defaults to `False`. Apply inverse rotation.

        # Returns
            Batch of rotated points, with same size as input.
    """
    if rotation_matrix is None:  # create the rotation matrix
        rotation_angle_x = np.random.uniform() * 2 * np.pi
        rotation_angle_y = np.random.uniform() * 2 * np.pi
        rotation_angle_z = np.random.uniform() * 2 * np.pi

        cosval = np.cos(rotation_angle_x)
        sinval = np.sin(rotation_angle_x)
        rotation_matrix_x = np.array(
            [[1, 0, 0], [0, cosval, sinval], [0, -sinval, cosval]]
        )
        cosval = np.cos(rotation_angle_y)
        sinval = np.sin(rotation_angle_y)
        rotation_matrix_y = np.array(
            [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        )
        cosval = np.cos(rotation_angle_z)
        sinval = np.sin(rotation_angle_z)
        rotation_matrix_z = np.array(
            [[cosval, sinval, 0], [-sinval, cosval, 0], [0, 0, 1]]
        )
        rotation_matrix = rotation_matrix_x @ rotation_matrix_y @ rotation_matrix_z
    if inverse:
        rotation_matrix = rotation_matrix.transpose()
    return rotation_matrix @ batch_data, rotation_matrix


def with_indices_computation_rotation(func):
    """Computes the indices and support points in the dataset.

    Calls the network instance of the dataset to generate search indices and support
    points to exploit multi-thread data loading.

    # Requires
        net: network instance.
            Network instance in field `net`of the dataset.

    # Returns

        Add `net_indices` and `net_support`to the return dictionary of the dataset.
    """

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # Do something before
        return_dict = func(*args, **kwargs)

        if hasattr(args[0], "net") and args[0].net is not None:

            # random rotation
            pts_r, rotation_matrix = rotate_point_cloud(return_dict["pts"].numpy())
            pts_r = torch.from_numpy(pts_r).float()
            _, indices, indices_pts = args[0].net(None, pts_r.unsqueeze(0))

            for i in range(len(indices)):
                indices[i] = indices[i].squeeze(0)
            for i in range(len(indices_pts)):
                indices_pts[i] = indices_pts[i].squeeze(0)

                # inverse rotation
                pts_tmp, _ = rotate_point_cloud(
                    indices_pts[i].numpy(), rotation_matrix, inverse=True
                )
                indices_pts[i] = torch.from_numpy(pts_tmp).float()

            return_dict["net_indices"] = indices
            return_dict["net_support"] = indices_pts

        # Do something after
        return return_dict

    return wrapper_decorator


def with_indices_computation(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # Do something before
        return_dict = func(*args, **kwargs)

        if hasattr(args[0], "net") and args[0].net is not None:

            # random rotation
            _, indices, indices_pts = args[0].net(None, return_dict["pts"].unsqueeze(0))

            for i in range(len(indices)):
                indices[i] = indices[i].squeeze(0)
            for i in range(len(indices_pts)):
                indices_pts[i] = indices_pts[i].squeeze(0)

            return_dict["net_indices"] = indices
            return_dict["net_support"] = indices_pts

        # Do something after
        return return_dict

    return wrapper_decorator
