import numpy as np
import os
from tqdm import tqdm
import argparse
from sklearn.metrics import confusion_matrix
import yaml

# torch imports
import torch
import torch.utils.data

import lightconvpoint.utils.data_utils as data_utils
import lightconvpoint.utils.metrics as metrics
from semantic8_dataset import DatasetTest as Dataset
from lightconvpoint.utils import get_network
from lightconvpoint.knn import knn


def batched_index_select(input, dim, index):
        index_shape = index.shape
        views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index).view(input.size(0), -1, index_shape[1], index_shape[2])

def nearest_correspondance(pts_src, pts_dest, data_src, K=1):
    pts_src = pts_src.unsqueeze(0).cpu().clone()
    pts_dest = pts_dest.unsqueeze(0).cpu().clone()
    indices = knn(pts_src, pts_dest, K)
    if K==1:
        indices = indices[0, :, 0]
        data_dest = data_src.transpose(0,1)[indices].transpose(0,1)
    else:
        data_dest = batched_index_select(data_src.unsqueeze(0).cpu(), 2, indices)
        data_dest = data_dest.mean(3)[0]
    return data_dest

def main(_config):

    print(_config)

    savedir_root = _config['training']['savedir']
    device = torch.device(_config['misc']['device'])
    rootdir = os.path.join(_config['dataset']['datasetdir'], _config['dataset']['dataset'], 'test_voxel_0_05m/pointcloud')
    
    # create the filelits (train / val) according to area
    print("Create filelist...", end="")
    print("Semantic3D filelist...", end="", flush="True")
    filelist_test = [
        "birdfountain_station1_xyz_intensity_rgb_voxels.npy",
        "castleblatten_station1_intensity_rgb_voxels.npy",
        "castleblatten_station5_xyz_intensity_rgb_voxels.npy",
        "marketplacefeldkirch_station1_intensity_rgb_voxels.npy",
        "marketplacefeldkirch_station4_intensity_rgb_voxels.npy",
        "marketplacefeldkirch_station7_intensity_rgb_voxels.npy",
        "sg27_station10_intensity_rgb_voxels.npy",
        "sg27_station3_intensity_rgb_voxels.npy",
        "sg27_station6_intensity_rgb_voxels.npy",
        "sg27_station8_intensity_rgb_voxels.npy",
        "sg28_station2_intensity_rgb_voxels.npy",
        "sg28_station5_xyz_intensity_rgb_voxels.npy",
        "stgallencathedral_station1_intensity_rgb_voxels.npy",
        "stgallencathedral_station3_intensity_rgb_voxels.npy",
        "stgallencathedral_station6_intensity_rgb_voxels.npy",
    ]
    print(f"done, {len(filelist_test)} test files")

    N_CLASSES = 8

    # create the network
    print("Creating the network...", end="", flush=True)
    def network_function():
        return get_network(
            _config["network"]["model"],
            in_channels=3,
            out_channels=N_CLASSES,
            backend_conv=_config["network"]["backend_conv"],
            backend_search=_config["network"]["backend_search"],
            config=_config,
            loadSubModelWeights = False
        )
    net = network_function()
    net.load_state_dict(torch.load(os.path.join(savedir_root, "checkpoint.pth"))["state_dict"])
    net.to(device)
    net.eval()
    print("Done")


    for filename in filelist_test:
        print(f"### {filename}")

        # create the dataloader
        ds = Dataset(filename, rootdir,
                block_size=_config['dataset']['pillar_size'], 
                npoints=_config['dataset']['npoints'],
                rgb=_config['training']['rgb'],
                step=_config['test']['step'],
                network_function=network_function)
        
        test_loader = torch.utils.data.DataLoader(ds, batch_size=_config['test']['batchsize'],
                shuffle=False, num_workers=_config['misc']['threads'])

        scores = np.zeros((ds.xyzrgb.shape[0], N_CLASSES))
        count = np.zeros(ds.xyzrgb.shape[0])

        # iterate over the dataloader
        t = tqdm(test_loader, ncols=100, desc=filename)
        with torch.no_grad():
            for data in t:

                pts = data['pts'].to(device)
                features = data['features'].to(device)
                pts_ids = data['pts_ids']
                net_ids = data["net_indices"]
                net_pts = data["net_support"]
                for i in range(len(net_ids)):
                    net_ids[i] = net_ids[i].to(device)
                for i in range(len(net_pts)):
                    net_pts[i] = net_pts[i].to(device)
                
                outputs = net(features, pts, indices=net_ids, support_points=net_pts)
                outputs_np = outputs.transpose(1,2).cpu().detach().numpy().reshape((-1, N_CLASSES))
                scores[pts_ids.numpy().ravel()] += outputs_np
                count[pts_ids.numpy().ravel()] += 1

        # get the original points
        original_points = ds.xyzrgb[:,:3]

        # compute the mask of points seen at prediction time
        mask = (count > 0)
        seen_scores = scores[mask] / count[mask][:,None]
        seen_points = original_points[mask]

        # project the scores on the original points
        scores = nearest_correspondance(
                    torch.from_numpy(seen_points).float().transpose(0,1), 
                    torch.from_numpy(original_points).float().transpose(0,1), 
                    torch.from_numpy(seen_scores).float().transpose(0,1), K=1).transpose(0,1).numpy()

        original_preds = scores.argmax(1)

        # save the results
        step = _config["test"]["step"]
        os.makedirs(os.path.join(savedir_root, f"results_{step}"), exist_ok=True)
        save_fname = os.path.join(savedir_root, f"results_{step}", filename)
        np.savetxt(save_fname,original_preds,fmt='%d')

        # save the points
        if _config['test']['savepts']:
            os.makedirs(os.path.join(savedir_root, f"results_{step}_pts"), exist_ok=True)
            save_fname = os.path.join(savedir_root, f"results_{step}_pts", filename+".txt")
            xyzrgb = np.concatenate([original_points, np.expand_dims(original_preds,1)], axis=1)
            np.savetxt(save_fname, xyzrgb, fmt=['%.4f','%.4f','%.4f','%d'])



if __name__ == "__main__":

    # get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to config file in savedir")
    parser.add_argument("--step", "-s", default=2, help="Path to config file in savedir")
    args = parser.parse_args()

    # update the base directory
    # makes it possible to move the directory
    # without editing the config file
    savedir = os.path.dirname(args.config)

    # load the configuration
    config = yaml.load(open(args.config))
    config["training"]["savedir"] = savedir

    config["test"]["step"] = args.step

    # call the main function
    main(config)