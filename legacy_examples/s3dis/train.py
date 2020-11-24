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

from s3dis_dataset import DatasetTrainVal as Dataset
import lightconvpoint.utils.metrics as metrics
from lightconvpoint.utils import get_network

# SACRED
from sacred import Experiment
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.config import save_config_file

SETTINGS.CAPTURE_MODE = "sys"  # for tqdm
ex = Experiment("S3DIS")
ex.captured_out_filter = apply_backspaces_and_linefeeds  # for tqdm
ex.add_config("s3dis.yaml")
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
    rootdir = os.path.join(_config['dataset']['datasetdir'], _config['dataset']['dataset'])

    N_CLASSES = 13

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

    # create the filelits (train / val) according to area
    print("Create filelist...", end="")
    filelist_train = []
    filelist_test = []
    for area_idx in range(1 ,7):
        folder = os.path.join(rootdir, f"Area_{area_idx}")
        datasets = [os.path.join(f"Area_{area_idx}", dataset) for dataset in os.listdir(folder)]
        if area_idx == _config['dataset']['area']:
            filelist_test = filelist_test + datasets
        else:
            filelist_train = filelist_train + datasets
    filelist_train.sort()
    filelist_test.sort()
    print(f"done, {len(filelist_train)} train files, {len(filelist_test)} test files")


    print("Creating dataloader and optimizer...", end="", flush=True)
    ds = Dataset(filelist_train, rootdir,
                             training=True, block_size=_config['dataset']['pillar_size'],
                             npoints=_config['dataset']['npoints'],
                             iteration_number=_config['training']['batchsize']*_config['training']['epoch_iter'],
                             jitter=_config['training']['jitter'],
                             scaling_param=_config['training']['scaling_param'],
                             rgb_dropout=_config['training']['rgb_dropout'], 
                             rgb=_config['training']['rgb'], network_function=network_function)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=_config['training']['batchsize'], shuffle=True,
                                        num_workers=_config['misc']['threads']
                                        )

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
        if _config['area']==1:
            weights = torch.Tensor([0.7615,  0.3969,  0.4546,  0.2727,  6.7376,  4.1650,  1.6270,  3.2547,
                                2.3042,  2.1289, 17.7709,  1.1333,  6.7996])
        elif _config['area']==2:
            weights = torch.Tensor([ 0.7366,  0.4071,  0.4866,  0.2736,  4.0031,  3.3682,  1.6507,  2.5912,
                                2.0347,  3.0115, 17.2155,  1.1268,  5.9607])
        elif _config['area']==3:
            weights = torch.Tensor([0.7499,  0.3991,  0.4636,  0.2758,  4.4585,  3.7786,  1.6039,  2.9821,
                                2.2443,  2.1931, 20.1374,  1.2197,  6.2980])
        elif _config['area']==4:
            weights = torch.Tensor([0.7543,  0.3921,  0.4622,  0.2818,  3.8026,  3.8313,  1.7192,  3.0418,
                                2.1892,  2.1827, 19.7227,  1.2032,  5.5455])
        elif _config['area']==5:
            weights = torch.Tensor([0.7045,  0.4006,  0.4644,  0.2815,  3.1686,  3.6080,  1.4001,  3.6230,
                                2.3671,  1.8859, 15.7542,  1.6276,  6.0848])
        elif _config['area']==6:
            weights = torch.Tensor([0.7508,  0.3955,  0.4576,  0.2720,  5.9368,  4.1264,  1.6474,  3.0501,
                                2.5304,  2.2307, 18.0194,  1.1336,  6.5966])
        else:
            raise Exception('Unknown area')
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
        logs.write(f"{epoch} {oa} {aa} {iou} {oa_val} {aa_val} {iou_val}\n")
        logs.close()

        # log train values
        _run.log_scalar("trainOA", oa, epoch)
        _run.log_scalar("trainAA", aa, epoch)
        _run.log_scalar("trainIoU", iou, epoch)
        _run.log_scalar("testOA", oa_val, epoch)
        _run.log_scalar("testAA", aa_val, epoch)
        _run.log_scalar("testAIoU", iou_val, epoch)
