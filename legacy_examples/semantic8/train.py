# other imports
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import h5py

# torch imports
import torch
import torch.nn.functional as F
import torch.utils.data

from semantic8_dataset import DatasetTrainVal as Dataset
import lightconvpoint.utils.metrics as metrics
from lightconvpoint.utils import get_network

# SACRED
from sacred import Experiment
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.config import save_config_file

SETTINGS.CAPTURE_MODE = "sys"  # for tqdm
ex = Experiment("Semantic8")
ex.captured_out_filter = apply_backspaces_and_linefeeds  # for tqdm
ex.add_config("semantic8.yaml")
######



class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# wrap blue / green
def wblue(str):
    return bcolors.OKBLUE+str+bcolors.ENDC
def wgreen(str):
    return bcolors.OKGREEN+str+bcolors.ENDC


@ex.automain
def main(_run, _config):

    print(_config)

    savedir_root = _config['training']['savedir']
    device = torch.device(_config['misc']['device'])

    # save the config file
    os.makedirs(savedir_root, exist_ok=True)
    save_config_file(eval(str(_config)), os.path.join(
        savedir_root, "config.yaml"))
    
    # create the path to data
    rootdir = os.path.join(_config['dataset']['datasetdir'], _config['dataset']['dataset'], 'train_voxel_0_05m/pointcloud')

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
            config=_config
        )
    net = network_function()
    net.to(device)
    print("Done")

    filelist_train=[
            "bildstein_station1_xyz_intensity_rgb_voxels.npy",
            "bildstein_station3_xyz_intensity_rgb_voxels.npy",
            "bildstein_station5_xyz_intensity_rgb_voxels.npy",
            "domfountain_station1_xyz_intensity_rgb_voxels.npy",
            "domfountain_station2_xyz_intensity_rgb_voxels.npy",
            "domfountain_station3_xyz_intensity_rgb_voxels.npy",
            "neugasse_station1_xyz_intensity_rgb_voxels.npy",
            "sg27_station1_intensity_rgb_voxels.npy",
            "sg27_station2_intensity_rgb_voxels.npy",
            "sg27_station4_intensity_rgb_voxels.npy",
            "sg27_station5_intensity_rgb_voxels.npy",
            "sg27_station9_intensity_rgb_voxels.npy",
            "sg28_station4_intensity_rgb_voxels.npy",
            "untermaederbrunnen_station1_xyz_intensity_rgb_voxels.npy",
            "untermaederbrunnen_station3_xyz_intensity_rgb_voxels.npy",
        ]
    filelist_val=[]

    print("Creating dataloader and optimizer...", end="", flush=True)
    ds = Dataset(filelist_train, rootdir,
                             training=True, block_size=_config['dataset']['pillar_size'],
                             npoints=_config['dataset']['npoints'],
                             iteration_number=_config['training']['batchsize']*_config['training']['epoch_iter'],
                             jitter=_config['training']['jitter'],
                             rgb_dropout=_config['training']['rgb_dropout'], 
                             rgb=_config['training']['rgb'], network_function=network_function)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=_config['training']['batchsize'], shuffle=True,
                                        num_workers=_config['misc']['threads']
                                        )

    if len(filelist_val) > 0:
        ds_val = Dataset(filelist_test, rootdir,
                                training=False, block_size=_config['dataset']['pillar_size'],
                                npoints=_config['dataset']['npoints'],
                                iteration_number=_config['training']['batchsize']*100,
                                rgb=_config['training']['rgb'],
                                network_function=network_function)
        test_loader = torch.utils.data.DataLoader(ds_val, batch_size=_config['training']['batchsize'], shuffle=False,
                                            num_workers=_config['misc']['threads']
                                            )
    print("Done")


    print("Creating optimizer...", end="", flush=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=_config['training']['lr_start'])
    print("done")


    print("Weights")
    if _config['training']['weights']: # computed on the train set
        weights = torch.Tensor([ 0.7772,  0.7216,  0.4977,  2.9913,  0.3884,  4.2342,  9.2966, 15.1820])
    else:
        weights = torch.ones(N_CLASSES).float()
    weights=weights.to(device)
    print("Done")


    # iterate over epochs
    for epoch in range(0, _config['training']['epoch_nbr']):

        #######
        # training
        net.train()

        count=0

        train_loss = 0
        cm = np.zeros((N_CLASSES, N_CLASSES))
        t = tqdm(train_loader, ncols=100, desc="Epoch {}".format(epoch), disable=_config['misc']['disable_tqdm'])
        for data in t:

            pts = data['pts'].to(device)
            features = data['features'].to(device)
            seg = data['target'].to(device)
            net_ids = data["net_indices"]
            net_pts = data["net_support"]
            for i in range(len(net_ids)):
                net_ids[i] = net_ids[i].to(device)
            for i in range(len(net_pts)):
                net_pts[i] = net_pts[i].to(device)

            optimizer.zero_grad()
            outputs = net(features, pts, indices=net_ids, support_points=net_pts)
            loss =  F.cross_entropy(outputs, seg, weight=weights)
            loss.backward()
            optimizer.step()

            output_np = np.argmax(outputs.cpu().detach().numpy(), axis=1).copy()
            target_np = seg.cpu().numpy().copy()

            cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES)))
            cm += cm_

            oa = f"{metrics.stats_overall_accuracy(cm):.5f}"
            aa = f"{metrics.stats_accuracy_per_class(cm)[0]:.5f}"
            iou = f"{metrics.stats_iou_per_class(cm)[0]:.5f}"

            train_loss += loss.detach().cpu().item()

            t.set_postfix(OA=wblue(oa), AA=wblue(aa), IOU=wblue(iou), LOSS=wblue(f"{train_loss/cm.sum():.4e}"))

        ######
        ## validation
        if len(filelist_val) > 0:
            net.eval()
            cm_test = np.zeros((N_CLASSES, N_CLASSES))
            test_loss = 0
            t = tqdm(test_loader, ncols=80, desc="  Test epoch {}".format(epoch), disable=_config['misc']['disable_tqdm'])
            with torch.no_grad():
                for data in t:

                    pts = data['pts'].to(device)
                    features = data['features'].to(device)
                    seg = data['target'].to(device)
                    net_ids = data["net_indices"]
                    net_pts = data["net_support"]
                    for i in range(len(net_ids)):
                        net_ids[i] = net_ids[i].to(device)
                    for i in range(len(net_pts)):
                        net_pts[i] = net_pts[i].to(device)
                    
                    outputs = net(features, pts, indices=net_ids, support_points=net_pts)
                    loss =  F.cross_entropy(outputs, seg)

                    output_np = np.argmax(outputs.cpu().detach().numpy(), axis=1).copy()
                    target_np = seg.cpu().numpy().copy()

                    cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES)))
                    cm_test += cm_

                    oa_val = f"{metrics.stats_overall_accuracy(cm_test):.5f}"
                    aa_val = f"{metrics.stats_accuracy_per_class(cm_test)[0]:.5f}"
                    iou_val = f"{metrics.stats_iou_per_class(cm_test)[0]:.5f}"

                    test_loss += loss.detach().cpu().item()

                    t.set_postfix(OA=wgreen(oa_val), AA=wgreen(aa_val), IOU=wgreen(iou_val), LOSS=wgreen(f"{test_loss/cm_test.sum():.4e}"))

        # create the root folder
        os.makedirs(savedir_root, exist_ok=True)

        # save the checkpoint
        torch.save({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, os.path.join(savedir_root, "checkpoint.pth"))

        # write the logs
        logs = open(os.path.join(savedir_root, "logs.txt"), "a+")
        logs.write(f"{epoch} {oa} {aa} {iou}")
        if len(filelist_val)>0:
            logs.write(" {oa_val} {aa_val} {iou_val}")
        logs.write("\n")
        logs.close()

        # log train values
        _run.log_scalar("trainOA", oa, epoch)
        _run.log_scalar("trainAA", aa, epoch)
        _run.log_scalar("trainIoU", iou, epoch)
        if len(filelist_val) > 0:
            _run.log_scalar("testOA", oa_val, epoch)
            _run.log_scalar("testAA", aa_val, epoch)
            _run.log_scalar("testAIoU", iou_val, epoch)
