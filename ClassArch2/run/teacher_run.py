import os
import sys
import shutil
import h5py
import numpy as np
import argparse
import math
from distutils.dir_util import copy_tree

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms

from ClassArch2.models.sonet_cls import SONet_cls as SONet
from ClassArch2.data.ModelNet40Loader import ModelNet40_SONet
import ClassArch2.data.data_utils as d_utils
from ClassArch2.data.data_sampling import initDir

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

test_transforms = transforms.Compose([
    d_utils.PointcloudToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for cls training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--visdom-port", type=int, default=8097)
    parser.add_argument("--visdom", action="store_true")

    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id: e.g. 0, 1, 2. -1 is no GPU')

    parser.add_argument('--dataset', type=str, default='modelnet', help='modelnet / shrec / shapenet')
    parser.add_argument('--dataroot', default='../data/modelnet40-normal_numpy/', help='path to images & laser point clouds')
    parser.add_argument('--classes', type=int, default=40, help='ModelNet40 or ModelNet10')
    parser.add_argument('--name', type=str, default='train', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--input_pc_num', type=int, default=5000, help='# of input points')
    parser.add_argument('--surface_normal', type=bool, default=True, help='use surface normal in the pc input')
    parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')

    parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
    parser.add_argument('--display_id', type=int, default=200, help='window id of the web display')

    parser.add_argument('--feature_num', type=int, default=1024, help='length of encoded feature')
    parser.add_argument('--activation', type=str, default='relu', help='activation function: relu, elu')
    parser.add_argument('--normalization', type=str, default='batch', help='normalization function: batch, instance')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.7, help='probability of an element to be zeroed')
    parser.add_argument('--node_num', type=int, default=64, help='som node number')
    parser.add_argument('--k', type=int, default=3, help='k nearest neighbor')
    parser.add_argument('--pretrain', type=str, default=None, help='pre-trained encoder dict path')
    parser.add_argument('--pretrain_lr_ratio', type=float, default=1, help='learning rate ratio between pretrained encoder and classifier')

    parser.add_argument('--som_k', type=int, default=9, help='k nearest neighbor of SOM nodes searching on SOM nodes')
    parser.add_argument('--som_k_type', type=str, default='avg', help='avg / center')

    parser.add_argument('--random_pc_dropout_lower_limit', type=float, default=1, help='keep ratio lower limit')
    parser.add_argument('--bn_momentum', type=float, default=0.1, help='normalization momentum, typically 0.1. Equal to (1-m) in TF')
    parser.add_argument('--bn_momentum_decay_step', type=int, default=None, help='BN momentum decay step. e.g, 0.5->0.01.')
    parser.add_argument('--bn_momentum_decay', type=float, default=0.6, help='BN momentum decay step. e.g, 0.5->0.01.')


    parser.add_argument('--rot_horizontal', type=bool, default=False, help='Rotation augmentation around vertical axis.')
    parser.add_argument('--rot_perturbation', type=bool, default=False, help='Small rotation augmentation around 3 axis.')
    parser.add_argument('--translation_perturbation', type=bool, default=False, help='Small translation augmentation around 3 axis.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device('cuda:%d' % (args.gpu_id) if torch.cuda.is_available() else 'cpu')


    with open(os.path.join(args.dataroot, 'shape_names.txt'), 'r') as f:
        labels = [str.rstrip() for str in f.readlines()]

    with open(os.path.join(args.dataroot, 'unlabeled_files.txt'), 'r') as f:
        files = [str.rstrip() for str in f.readlines()]
    
    som_folder_list = []
    for i in range(3, 12):
        som_folder_list.append("%dx%d_som_nodes" %(i, i))

    copy_tree(os.path.join(args.dataroot, "train"), os.path.join(args.dataroot, "train_stu"))
    train_stu_file = open(os.path.join(args.dataroot, 'train_stu_files.txt'), 'w')
    # initDir(os.path.join(args.dataroot, "pseudo_labeled"), labels, som_folder_list)

    torch.multiprocessing.freeze_support()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device=device)
    ngpus_per_node = torch.cuda.device_count()
    target_folder = './data/'

    ds_test = ModelNet40_SONet(args.dataroot, 'unlabeled', args)
    
    ds_test_loader = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.nThreads,
        pin_memory=True,
    )
    
    model = SONet(args)
    model.encoder.load_state_dict(torch.load('../train/checkpoints/sonet_enc_best.pth'))

    model.encoder.eval()
    model.classifier.eval()

    print('LABELING...')
    data_cnt = 0
    for i, data in enumerate(ds_test_loader):
        input_pc, input_sn, input_label, input_node, input_node_knn_I = data
        model.set_input_inference(input_pc, input_sn, input_node, input_node_knn_I)
        model.feature = model.encoder(model.input_pc, model.input_sn, model.input_node, model.input_node_knn_I)
        # model.feature = model.feature.to(device)
        pred = model.classifier(model.feature)
        _, predicted_idx = torch.max(pred.data, dim=1, keepdim=False)
        predicted_idx = predicted_idx.cpu().data.numpy()
        # print(str(input_label) + "   :   " + str(predicted_idx))

        for pred in predicted_idx:
            folder = files[data_cnt][0:-5]
            file_name = files[data_cnt]
            start_path = os.path.join(args.dataroot, "unlabeled", folder, file_name + '.npy')
            dest_path = os.path.join(args.dataroot, "train_stu", labels[pred], file_name + '.npy')
            shutil.copyfile(start_path, dest_path)
            for som_nodes_folder in som_folder_list:
                start_path = os.path.join(args.dataroot, "unlabeled", som_nodes_folder, folder, file_name + '.npy')
                dest_path = os.path.join(args.dataroot, "train_stu", som_nodes_folder, labels[pred], file_name + '.npy')
                shutil.copyfile(start_path, dest_path)
            train_stu_file.write(file_name)
            data_cnt += 1

    train_stu_file.close()
    print(data_cnt)

    print('COMBINED PSEUDO-LABELED DATA WITH TRAIN DATA...')