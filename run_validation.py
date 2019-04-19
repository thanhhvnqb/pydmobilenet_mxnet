from __future__ import division

import os, sys, argparse, time, logging, random, math
import argparse

import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = "0"
class Model(object):
    def __init__(self, name, folder, net_name, epoch=0):
        self.name = name
        self.folder = folder
        self.net_name = net_name
        self.epoch = epoch
        self.val = 0

    def __repr__(self):
        return str(self)

    def __str__(self):
     return "%30s | %30s | %30s | %3d" % (self.name, self.folder, self.net_name, self.epoch)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run validation of PydMobileNet, MobileNet, ResNet on CIFAR10 and CIFAR100 dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset', default=0, type=int, choices=[0, 1, 2],
                        help="The validation dataset. 0: all | 1: CIFAR10 | 2: CIFAR100")
    parser.add_argument('-g', '--gpu', default=-1, type=int,
                        help="GPU used in validation. -1: cpu.")
    parser.add_argument('--model-dir', default="models/",
                        help="The directory that contains downloaded tar files")
    parser.add_argument('-m', '--model', default=0, type=int, choices=[0, 1, 2, 3, 4],
                        help="Models to be evaluated. 0: all | 1: ResNet | 2: MobileNet | 3: PydMobileNet-Add | 4: PydMobileNet-Concat")
    parser.add_argument('-l', '--layer', default=0, type=int, choices=[0, 29, 56],
                        help="#Layer of evaluated models. 0 = [29, 56].")
    parser.add_argument('--print-net', action='store_true',
                        help="If print sumamary information of network.")
    args = parser.parse_args()
    return args


def test(net, ctx, val_data):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)
    return metric.get()


def get_list_models(args):
    models_lst = list()
    if args.layer == 0:
        layers = [29, 56]
    else:
        layers = [args.layer]
    models = list()
    if args.model == 0:
        models = [1, 2, 3, 4]
    else:
        models = [args.model]
    for layer in layers:
        for model in models:
            if model == 1:
                name = "ResNet" + str(layer)
                folder = args.model_dir + "/resnet/"
                net_name = "resnet" + str(layer)
                models_lst.append(Model(name, folder, net_name))
            elif model == 2:
                name = "MobileNet" + str(layer)
                folder = args.model_dir + "/mobilenet/"
                net_name = "mobilenet" + str(layer)
                for alpha in ['_0.5', '_1', '_1.5']:
                    models_lst.append(Model(name + alpha, folder, net_name + alpha))
            elif model == 3:
                name = "PydMobileNet" + str(layer)
                folder = args.model_dir + "/pydmobilenet_add/"
                net_name = "pydmobilenet" + str(layer)
                for alpha in ['_0.25', '_0.5', '_0.75', '_1']:
                    models_lst.append(Model(name + alpha + "-Add", folder, net_name + alpha + "_plus"))
            elif model == 4:
                name = "PydMobileNet" + str(layer)
                folder = args.model_dir + "/pydmobilenet_concat/"
                net_name = "pydmobilenet" + str(layer)
                for alpha in ['_0.25', '_0.5', '_0.75']:
                    models_lst.append(Model(name + alpha + "-Concat", folder, net_name + alpha + "_concat"))

    return models_lst

def run_validation(args, models_lst):
    transform_test = transforms.Compose([
    #     transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    if args.gpu == -1:
        ctx = [mx.cpu(0)]
    else:
        ctx = [mx.gpu(args.gpu)]


    # Number of data loader workers
    num_workers = 8
    # Calculate effective total batch size
    batch_size = 128
    if args.dataset == 0:
        datasets = ['cifar10_', 'cifar100_']
        val_datas = [gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(root='cifar10',train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers), gluon.data.DataLoader(
            gluon.data.vision.CIFAR100(root='cifar100',train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)]
    elif args.dataset == 1:
        datasets = ['cifar10_']
        val_datas = [gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(root='cifar10',train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)]
    elif args.dataset == 2:
        datasets = ['cifar100_']
        val_datas = [gluon.data.DataLoader(
            gluon.data.vision.CIFAR100(root='cifar100',train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)]
    print_str = ""
    for (data, val_data) in zip(datasets, val_datas):
        print_str += '*' * 45 + "\n"
        print_str += "Dataset: " + data[:-1].upper() + "\n"
        print_str += '%30s | %10s' % ('Model', 'Val error') + "\n"
        print_str += '-' * 45 + "\n"
        for model in models_lst:
            net = gluon.nn.SymbolBlock.imports("%s/%s-symbol.json" % (model.folder, data + model.net_name), ['data'], '%s/%s-%04d.params' % (model.folder, data + model.net_name, model.epoch), ctx=ctx)
            net.hybridize()
            net.forward(mx.nd.ones((1, 3, 32, 32), ctx=ctx[0]))
            if args.print_net:
                shape = {}
                shape['data'] = (1, 3, 32, 32)
                mx.visualization.print_summary(net(mx.sym.Variable('data')), shape)
            name, val_acc = test(net, ctx, val_data)
            print_str += "%30s | %10.2f" % (model.name, (1 - val_acc) * 100) + "\n"
        print_str += "\n"
    print(print_str)



def main():
    args = parse_args()
    models_lst = get_list_models(args)
    # print("\n".join([str(model) for model in models_lst]))
    run_validation(args, models_lst)


if __name__ == '__main__':
    main()
