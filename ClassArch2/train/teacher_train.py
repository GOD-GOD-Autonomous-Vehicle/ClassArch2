from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import etw_pytorch_utils as pt_utils
import pprint
import os.path as osp
import os
import argparse
import time

from ClassArch2.utils.sonet_visualizer import Visualizer
from ClassArch2.models.sonet_cls import SONet_cls as SONet
from ClassArch2.data.ModelNet40Loader import ModelNet40_SONet
from ClassArch2.data.ModelNet40Loader import ModelNet40
import ClassArch2.data.data_utils as d_utils

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for cls training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")

    parser.add_argument("--visdom-port", type=int, default=8097)
    parser.add_argument("--visdom", action="store_true")

    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id: e.g. 0, 1, 2. -1 is no GPU')

    parser.add_argument('--dataset', type=str, default='modelnet', help='modelnet / shrec / shapenet')
    parser.add_argument('--dataroot', default='../data/modelnet40-normal_numpy', help='path to images & laser point clouds')
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

if __name__ == "__main__":
    args = parse_args()
    args.device = torch.device('cuda:%d' % (args.gpu_id) if torch.cuda.is_available() else 'cpu')
    
    transforms = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
        ]
    )

    test_set = ModelNet40_SONet(args.dataroot, 'test', args, transforms)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.nThreads,
        pin_memory=True,
    )
    
    train_set = ModelNet40_SONet(args.dataroot, 'train', args, transforms)
    dataset_size = len(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.nThreads,
        pin_memory=True,
    )

    model = SONet(args)

    if args.pretrain is not None:
        model.encoder.load_state_dict(torch.load(args.pretrain))
    ############################# automation for ModelNet10 / 40 configuration ####################
    if args.classes == 10:
        args.dropout = args.dropout + 0.1
    ############################# automation for ModelNet10 / 40 configuration ####################

    visualizer = Visualizer(args)

    best_accuracy = 0
    for epoch in range(args.epochs):

        epoch_iter = 0
        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            epoch_iter += args.batch_size

            input_pc, input_sn, input_label, input_node, input_node_knn_I = data
            model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)

            model.optimize(epoch=epoch)

            if i % 200 == 0:
                # print/plot errors
                t = (time.time() - iter_start_time) / args.batch_size

                errors = model.get_current_errors()

                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, args, errors)

                # print(model.autoencoder.encoder.feature)
                # visuals = model.get_current_visuals()
                # visualizer.display_current_results(visuals, epoch, i)

        # test network
        if epoch >= 0 and epoch%1==0:
            batch_amount = 0
            model.test_loss.data.zero_()
            model.test_accuracy.data.zero_()
            for i, data in enumerate(test_loader):
                input_pc, input_sn, input_label, input_node, input_node_knn_I = data
                model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
                model.test_model()

                batch_amount += input_label.size()[0]

                # # accumulate loss
                model.test_loss += model.loss.detach() * input_label.size()[0]

                # # accumulate accuracy
                _, predicted_idx = torch.max(model.score.data, dim=1, keepdim=False)
                correct_mask = torch.eq(predicted_idx, model.input_label).float()
                test_accuracy = torch.mean(correct_mask).cpu()
                model.test_accuracy += test_accuracy * input_label.size()[0]

            model.test_loss /= batch_amount
            model.test_accuracy /= batch_amount
            if model.test_accuracy.item() > best_accuracy:
                best_accuracy = model.test_accuracy.item()
                model.save_network2(model.encoder, 'sonet_enc_best.pth', args.gpu_id)
                model.save_network2(model.classifier, 'sonet_cls_best.pth', args.gpu_id)

            print('Tested network. So far best: %f' % best_accuracy)

            # save network
            saving_acc_threshold = 0.918
            if model.test_accuracy.item() > saving_acc_threshold:
                print("Saving network...")
                model.save_network(model.encoder, 'encoder', '%d_%f' % (epoch, model.test_accuracy.item()), args.gpu_id)
                model.save_network(model.classifier, 'classifier', '%d_%f' % (epoch, model.test_accuracy.item()), args.gpu_id)

        # learning rate decay
        if args.classes == 10:
            lr_decay_step = 40
        else:
            lr_decay_step = 20
        if epoch%lr_decay_step==0 and epoch > 0:
            model.update_learning_rate(0.5)
        # batch normalization momentum decay:
        next_epoch = epoch + 1
        if (args.bn_momentum_decay_step is not None) and (next_epoch >= 1) and (
                next_epoch % args.bn_momentum_decay_step == 0):
            current_bn_momentum = args.bn_momentum * (
            args.bn_momentum_decay ** (next_epoch // args.bn_momentum_decay_step))
            print('BN momentum updated to: %f' % current_bn_momentum)

        # save network
        # if epoch%20==0 and epoch>0:
        #     print("Saving network...")
        #     model.save_network(model.classifier, 'cls', '%d' % epoch, opt.gpu_id)

