import os
import plyfile
import numpy as np
from matplotlib import cm


def save_ply(points, filename, colors=None, normals=None):
    vertex = np.array(
        [tuple(p) for p in points], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
    )
    n = len(vertex)
    desc = vertex.dtype.descr

    if normals is not None:
        vertex_normal = np.array(
            [tuple(n) for n in normals],
            dtype=[("nx", "f4"), ("ny", "f4"), ("nz", "f4")],
        )
        assert len(vertex_normal) == n
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        vertex_color = np.array(
            [tuple(c * 255) for c in colors],
            dtype=[("red", "u1"), ("green", "u1"), ("blue", "u1")],
        )
        assert len(vertex_color) == n
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(n, dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData(
        [plyfile.PlyElement.describe(vertex_all, "vertex")], text=False
    )
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ply.write(filename)


def save_ply_property(points, property, property_max, filename, cmap_name="tab20"):
    point_num = points.shape[0]
    colors = np.full(points.shape, 0.5)
    cmap = cm.get_cmap(cmap_name)
    for point_idx in range(point_num):
        if property[point_idx] == 0:
            colors[point_idx] = np.array([0, 0, 0])
        else:
            colors[point_idx] = cmap(property[point_idx] / property_max)[:3]
    save_ply(points, filename, colors)


def save_ply_batch(points_batch, file_path, points_num=None):
    batch_size = points_batch.shape[0]
    if type(file_path) != list:
        basename = os.path.splitext(file_path)[0]
        ext = ".ply"
    for batch_idx in range(batch_size):
        point_num = (
            points_batch.shape[1] if points_num is None else points_num[batch_idx]
        )
        if type(file_path) == list:
            save_ply(points_batch[batch_idx][:point_num], file_path[batch_idx])
        else:
            save_ply(
                points_batch[batch_idx][:point_num],
                "%s_%04d%s" % (basename, batch_idx, ext),
            )


def save_ply_color_batch(points_batch, colors_batch, file_path, points_num=None):
    batch_size = points_batch.shape[0]
    if type(file_path) != list:
        basename = os.path.splitext(file_path)[0]
        ext = ".ply"
    for batch_idx in range(batch_size):
        point_num = (
            points_batch.shape[1] if points_num is None else points_num[batch_idx]
        )
        if type(file_path) == list:
            save_ply(
                points_batch[batch_idx][:point_num],
                file_path[batch_idx],
                colors_batch[batch_idx][:point_num],
            )
        else:
            save_ply(
                points_batch[batch_idx][:point_num],
                "%s_%04d%s" % (basename, batch_idx, ext),
                colors_batch[batch_idx][:point_num],
            )


def save_ply_property_batch(
    points_batch,
    property_batch,
    file_path,
    points_num=None,
    property_max=None,
    cmap_name="tab20",
):
    batch_size = points_batch.shape[0]
    if type(file_path) != list:
        basename = os.path.splitext(file_path)[0]
        ext = ".ply"
    property_max = np.max(property_batch) if property_max is None else property_max
    for batch_idx in range(batch_size):
        point_num = (
            points_batch.shape[1] if points_num is None else points_num[batch_idx]
        )
        if type(file_path) == list:
            save_ply_property(
                points_batch[batch_idx][:point_num],
                property_batch[batch_idx][:point_num],
                property_max,
                file_path[batch_idx],
                cmap_name,
            )
        else:
            save_ply_property(
                points_batch[batch_idx][:point_num],
                property_batch[batch_idx][:point_num],
                property_max,
                "%s_%04d%s" % (basename, batch_idx, ext),
                cmap_name,
            )


def save_ply_point_with_normal(data_sample, folder):
    for idx, sample in enumerate(data_sample):
        filename_pts = os.path.join(folder, f"{idx:08d}.ply")
        save_ply(sample[..., :3], filename_pts, normals=sample[..., 3:])
