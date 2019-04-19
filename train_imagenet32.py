import argparse, time, logging, os, math
from datetime import datetime

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxboard import SummaryWriter

from gluoncv.data import imagenet
from gluoncv.data import transforms as gcv_transforms
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRScheduler

from mxnet_utils import *
from data import ImageNet32
import pydnet

context = [mx.gpu(0)]
net_name = 'pydnet_0_25'

get_pydnet = getattr(pydnet, net_name)
work_dir = 'out/20190111_%s/' % net_name
saved_dir = work_dir + '/params/'
makedirs(work_dir + "/params/")
logsw = SummaryWriter(logdir=work_dir + "/logs/")

# CLI
parser = argparse.ArgumentParser(
    description='Train a model for image classification.')
parser.add_argument(
    '--data-dir',
    type=str,
    default='~/.mxnet/datasets/imagenet',
    help='training and validation pictures to use.')
parser.add_argument(
    '--rec-train',
    type=str,
    default='~/.mxnet/datasets/imagenet/rec/train.rec',
    help='the training data')
parser.add_argument(
    '--rec-train-idx',
    type=str,
    default='~/.mxnet/datasets/imagenet/rec/train.idx',
    help='the index of training data')
parser.add_argument(
    '--rec-val',
    type=str,
    default='~/.mxnet/datasets/imagenet/rec/val.rec',
    help='the validation data')
parser.add_argument(
    '--rec-val-idx',
    type=str,
    default='~/.mxnet/datasets/imagenet/rec/val.idx',
    help='the index of validation data')
parser.add_argument(
    '--use-rec',
    action='store_true',
    help='use image record iter for data input. default is false.')
parser.add_argument(
    '--batch-size',
    type=int,
    default=32,
    help='training batch size per device (CPU/GPU).')
parser.add_argument(
    '--dtype',
    type=str,
    default='float32',
    help='data type for training. default is float32')
parser.add_argument(
    '-j',
    '--num-data-workers',
    dest='num_workers',
    default=4,
    type=int,
    help='number of preprocessing workers')
parser.add_argument(
    '--num-epochs', type=int, default=3, help='number of training epochs.')
parser.add_argument(
    '--lr', type=float, default=0.1, help='learning rate. default is 0.1.')
parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='momentum value for optimizer, default is 0.9.')
parser.add_argument(
    '--wd',
    type=float,
    default=0.0001,
    help='weight decay rate. default is 0.0001.')
parser.add_argument(
    '--lr-mode',
    type=str,
    default='step',
    help='learning rate scheduler mode. options are step, poly and cosine.')
parser.add_argument(
    '--lr-decay',
    type=float,
    default=0.1,
    help='decay rate of learning rate. default is 0.1.')
parser.add_argument(
    '--lr-decay-period',
    type=int,
    default=0,
    help='interval for periodic learning rate decays. default is 0 to disable.'
)
parser.add_argument(
    '--lr-decay-epoch',
    type=str,
    default='40,60',
    help='epoches at which learning rate decays. default is 40,60.')
parser.add_argument(
    '--warmup-lr',
    type=float,
    default=0.0,
    help='starting warmup learning rate. default is 0.0.')
parser.add_argument(
    '--warmup-epochs', type=int, default=0, help='number of warmup epochs.')
parser.add_argument(
    '--last-gamma',
    action='store_true',
    help='whether to init gamma of the last BN layer in each bottleneck to 0.')
parser.add_argument(
    '--mode',
    type=str,
    help=
    'mode in which to train the model. options are symbolic, imperative, hybrid'
)
parser.add_argument(
    '--input-size',
    type=int,
    default=32,
    help='size of the input image size. default is 32')
parser.add_argument(
    '--crop-ratio',
    type=float,
    default=0.875,
    help='Crop ratio during validation. default is 0.875')
parser.add_argument(
    '--use-pretrained',
    action='store_true',
    help='enable using pretrained model from gluon.')
parser.add_argument(
    '--use_se',
    action='store_true',
    help='use SE layers or not in resnext. default is false.')
parser.add_argument(
    '--mixup',
    action='store_true',
    help='whether train the model with mix-up. default is false.')
parser.add_argument(
    '--mixup-alpha',
    type=float,
    default=0.2,
    help='beta distribution parameter for mixup sampling, default is 0.2.')
parser.add_argument(
    '--mixup-off-epoch',
    type=int,
    default=0,
    help='how many last epochs to train without mixup, default is 0.')
parser.add_argument(
    '--label-smoothing',
    action='store_true',
    help='use label smoothing or not in training. default is false.')
parser.add_argument(
    '--no-wd',
    action='store_true',
    help=
    'whether to remove weight decay on bias, and beta/gamma for batchnorm layers.'
)
parser.add_argument(
    '--batch-norm',
    action='store_true',
    help='enable batch normalization or not in vgg. default is false.')
parser.add_argument(
    '--save-frequency',
    type=int,
    default=10,
    help='frequency of model saving.')
parser.add_argument(
    '--save-dir',
    type=str,
    default=saved_dir,
    help='directory of saved models')
parser.add_argument(
    '--resume-epoch',
    type=int,
    default=0,
    help='epoch to resume training from.')
parser.add_argument(
    '--resume-params',
    type=str,
    default='',
    help='path of parameters to load from.')
parser.add_argument(
    '--resume-states',
    type=str,
    default='',
    help='path of trainer state to load from.')
parser.add_argument(
    '--log-interval',
    type=int,
    default=50,
    help='Number of batches to wait before logging.')
parser.add_argument(
    '--logging-file',
    type=str,
    default=work_dir + 'train_imagenet.log',
    help='name of training log file')
opt = parser.parse_args()
opt.num_gpus = len(context)
filehandler = logging.FileHandler(opt.logging_file)
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

logger.info(opt)

batch_size = opt.batch_size
classes = 1000
num_training_samples = 1281167

num_gpus = len(context)
batch_size *= max(1, num_gpus)
# context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
num_workers = opt.num_workers

lr_decay = opt.lr_decay
lr_decay_period = opt.lr_decay_period
if opt.lr_decay_period > 0:
    lr_decay_epoch = list(
        range(lr_decay_period, opt.num_epochs, lr_decay_period))
else:
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
num_batches = num_training_samples // batch_size
lr_scheduler = LRScheduler(
    mode=opt.lr_mode,
    baselr=opt.lr,
    niters=num_batches,
    nepochs=opt.num_epochs,
    step=lr_decay_epoch,
    step_factor=opt.lr_decay,
    power=2,
    warmup_epochs=opt.warmup_epochs)

kwargs = {'ctx': context, 'pretrained': opt.use_pretrained, 'classes': classes}

if opt.last_gamma:
    kwargs['last_gamma'] = True

optimizer = 'nag'
optimizer_params = {
    'wd': opt.wd,
    'momentum': opt.momentum,
    'lr_scheduler': lr_scheduler
}
if opt.dtype != 'float32':
    optimizer_params['multi_precision'] = True

net = get_pydnet(prefix=net_name, classes=classes)

net.cast(opt.dtype)
if opt.resume_params is not '':
    net.load_parameters(opt.resume_params, ctx=context)


def get_data_loader(data_dir, batch_size, num_workers):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    jitter_param = 0.4
    lighting_param = 0.1
    input_size = opt.input_size
    crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size / crop_ratio))

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(
            batch[1], ctx_list=ctx, batch_axis=0)
        return data, label

    transform_train = transforms.Compose([
        gcv_transforms.RandomCrop(input_size, pad=4),
        transforms.RandomFlipLeftRight(),
        # transforms.RandomColorJitter(
        #     brightness=jitter_param,
        #     contrast=jitter_param,
        #     saturation=jitter_param),
        # transforms.RandomLighting(lighting_param),
        transforms.ToTensor(), normalize
    ])
    transform_test = transforms.Compose([
        # transforms.Resize(resize, keep_ratio=True),
        # transforms.CenterCrop(input_size),
        transforms.Resize(input_size),
        transforms.ToTensor(), normalize
    ])

    train_data = gluon.data.DataLoader(
        ImageNet32(data_dir, train=True).transform_first(transform_train),
        batch_size=batch_size,
        shuffle=True,
        last_batch='discard',
        num_workers=num_workers)
    val_data = gluon.data.DataLoader(
        ImageNet32(data_dir, train=False).transform_first(transform_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    return train_data, val_data, batch_fn


train_data, val_data, batch_fn = get_data_loader(opt.data_dir, batch_size,
                                                 num_workers)

if opt.mixup:
    train_metric = mx.metric.RMSE()
else:
    train_metric = mx.metric.Accuracy()
acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)

save_frequency = opt.save_frequency
if opt.save_dir and save_frequency:
    save_dir = opt.save_dir
    makedirs(save_dir)
else:
    save_dir = ''
    save_frequency = 0


def mixup_transform(label, classes, lam=1, eta=0.0):
    if isinstance(label, nd.NDArray):
        label = [label]
    res = []
    for l in label:
        y1 = l.one_hot(
            classes, on_value=1 - eta + eta / classes, off_value=eta / classes)
        y2 = l[::-1].one_hot(
            classes, on_value=1 - eta + eta / classes, off_value=eta / classes)
        res.append(lam * y1 + (1 - lam) * y2)
    return res


def smooth(label, classes, eta=0.1):
    if isinstance(label, nd.NDArray):
        label = [label]
    smoothed = []
    for l in label:
        res = l.one_hot(
            classes, on_value=1 - eta + eta / classes, off_value=eta / classes)
        smoothed.append(res)
    return smoothed


def test(ctx, val_data):
    if opt.use_rec:
        val_data.reset()
    acc_top1.reset()
    acc_top5.reset()
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, ctx)
        outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return (1 - top1, 1 - top5)


def train(ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    max_step = num_batches
    if opt.resume_epoch == 0:
        net.initialize(mx.init.MSRAPrelu(), ctx=ctx)
        if opt.mode == 'hybrid':
            net.hybridize(static_alloc=True, static_shape=True)
            gstep = 0
            shape = {}
            net.forward(
                mx.nd.ones((2, 3, opt.input_size, opt.input_size),
                           ctx=ctx[0],
                           dtype=opt.dtype))
            shape['data'] = (2, 3, opt.input_size, opt.input_size)
            mx.viz.print_summary(net(mx.sym.Variable('data')), shape=shape)
            logsw.add_graph(net)
    else:
        if opt.mode == 'hybrid':
            net.hybridize(static_alloc=True, static_shape=True)
        # net = gluon.nn.SymbolBlock.imports(saved_dir + "/%s-symbol.json" % net_name, ['data'],
        #                          saved_dir + "/%s-%04d.params" % (net_name, opt.resume_epoch - 1), ctx=ctx)
        net.collect_params().load(saved_dir + "/%s-%04d.params" %
                                  (net_name, opt.resume_epoch), ctx=ctx)
        gstep = max_step * (opt.resume_epoch - 1)

    if opt.no_wd:
        for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0

    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

    if opt.label_smoothing or opt.mixup:
        L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
    else:
        L = gluon.loss.SoftmaxCrossEntropyLoss()

    best_val_score = 1

    for epoch in range(opt.resume_epoch, opt.num_epochs):
        now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        print("[Epoch %d] - Start run: %s" % (epoch + 1, now))
        progbar = Progbar(
            target=max_step, prefix='Train - ', stateful_metrics=[])
        tic = time.time()
        train_metric.reset()
        btic = time.time()
        dtic = time.time()
        for i, batch in enumerate(train_data):
            # print("Data time:", time.time() - dtic)
            # dtic = time.time()
            gstep += 1
            data, label = batch_fn(batch, ctx)

            if opt.mixup:
                lam = np.random.beta(opt.mixup_alpha, opt.mixup_alpha)
                if epoch >= opt.num_epochs - opt.mixup_off_epoch:
                    lam = 1
                data = [lam * X + (1 - lam) * X[::-1] for X in data]

                if opt.label_smoothing:
                    eta = 0.1
                else:
                    eta = 0.0
                label = mixup_transform(label, classes, lam, eta)

            elif opt.label_smoothing:
                hard_label = label
                label = smooth(label, classes)
            # print("Mixup time:", time.time() - dtic)
            # dtic = time.time()
            with ag.record():
                outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
                loss = [
                    L(yhat, y.astype(opt.dtype, copy=False))
                    for yhat, y in zip(outputs, label)
                ]
            # print("Forward time:", time.time() - dtic)
            # dtic = time.time()
            for l in loss:
                if np.isnan(l.asnumpy()).any():
                    print("Nan in Loss")
                    print(loss)
                    print(outputs)
                    print(label)
                    sys.exit()
                l.backward()
            lr_scheduler.update(i, epoch)
            trainer.step(batch_size)
            # print("Backward time:", time.time() - dtic)
            # dtic = time.time()
            # Update metrics
            batch_loss = sum([l.sum().asscalar() for l in loss])
            batch_loss /= (len(loss) * batch_size)
            logsw.add_scalar(
                'loss_batch', {net_name + '_train_loss': batch_loss},
                global_step=gstep)
            progbar.update(i + 1)
            # print("Calculate loss time:", time.time() - dtic)
            # dtic = time.time()
            if opt.mixup:
                output_softmax = [nd.SoftmaxActivation(out.astype('float32', copy=False)) \
                                  for out in outputs]
                train_metric.update(label, output_softmax)
            else:
                if opt.label_smoothing:
                    train_metric.update(hard_label, outputs)
                else:
                    train_metric.update(label, outputs)
            # print("Update metric time:", time.time() - dtic)
            # dtic = time.time()
            if opt.log_interval and not (i + 1) % opt.log_interval:
                train_metric_name, train_metric_score = train_metric.get()
                logger.info(
                    '\nEpoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f\tlr=%f'
                    % (epoch + 1, i + 1, batch_size * opt.log_interval /
                       (time.time() - btic), train_metric_name,
                       train_metric_score, trainer.learning_rate))
                btic = time.time()
            # print("Logging time:", time.time() - dtic)
            # dtic = time.time()
            for arg in net.collect_params().values():
                if np.isnan(arg._reduce().asnumpy()).any():
                    print("NaN in parameters")
                    net.export(
                        saved_dir + '/%s_nan_parameter' % (net_name),
                        epoch=epoch)
                    sys.exit()
            net.export(saved_dir + '/%s' % (net_name), epoch=epoch)
        train_metric_name, train_metric_score = train_metric.get()
        throughput = int(batch_size * i / (time.time() - tic))

        err_top1_val, err_top5_val = test(ctx, val_data)

        logger.info('[Epoch %d] training: %s=%f' %
                    (epoch + 1, train_metric_name, train_metric_score))
        logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f' %
                    (epoch + 1, throughput, time.time() - tic))
        logger.info('[Epoch %d] validation: err-top1=%f err-top5=%f' %
                    (epoch + 1, err_top1_val, err_top5_val))
        net.export(saved_dir + '/%s' % (net_name), epoch=epoch)
        if err_top1_val < best_val_score:
            best_val_score = err_top1_val
            net.export(
                saved_dir + '/%s-best-%.4f' % (net_name, best_val_score),
                epoch=epoch)


def main():
    train(context)


if __name__ == '__main__':
    main()