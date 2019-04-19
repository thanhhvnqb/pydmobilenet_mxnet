import mxnet as mx
import numpy as np
from mxnet.gluon import nn, Parameter, HybridBlock
from mxnet.gluon.contrib.nn import HybridConcurrent, Identity

class ResPydBlock(HybridBlock):
    def __init__(self, multiplier, stage, ordinal, num_features, stride, dropout, **kwargs):
        super(ResPydBlock, self).__init__(prefix='stg%d/blk%d/' % (stage, ordinal), **kwargs)
        with self.name_scope():
            use_bias = False
            dw_channels = np.int(num_features * multiplier)
            self.btl_in = nn.HybridSequential(prefix='btl_in/')
            with self.btl_in.name_scope():
                self.btl_in.add(nn.BatchNorm())
                self.btl_in.add(nn.Activation('relu'))
                self.btl_in.add(nn.Conv2D(dw_channels, kernel_size=1, use_bias=use_bias))
                if dropout:
                    self.btl_in.add(nn.Dropout(dropout))
                self.btl_in.add(nn.BatchNorm())
                self.btl_in.add(nn.Activation('relu'))

            self.conv3 = nn.HybridSequential(prefix='conv3/')
            with self.conv3.name_scope():
                self.conv3.add(nn.Conv2D(dw_channels, kernel_size=3, strides=stride, padding=1, groups=dw_channels, use_bias=use_bias))
                if dropout:
                    self.conv3.add(nn.Dropout(dropout))
#                 self.conv3.add(nn.BatchNorm())

            self.conv5 = nn.HybridSequential(prefix='conv5/')
            with self.conv5.name_scope():
                self.conv5.add(nn.Conv2D(dw_channels, kernel_size=5, strides=stride, padding=2, groups=dw_channels, use_bias=use_bias))
                if dropout:
                    self.conv5.add(nn.Dropout(dropout))
#                 self.conv5.add(nn.BatchNorm())

            self.conv7 = nn.HybridSequential(prefix='conv7/')
            with self.conv7.name_scope():
                self.conv7.add(nn.Conv2D(dw_channels, kernel_size=7, strides=stride, padding=3, groups=dw_channels, use_bias=use_bias))
                if dropout:
                    self.conv7.add(nn.Dropout(dropout))
#                 self.conv7.add(nn.BatchNorm())

            self.btl_out = nn.HybridSequential(prefix='btl_out/')
            with self.btl_out.name_scope():
                self.btl_out.add(nn.BatchNorm())
                self.btl_out.add(nn.Activation('relu'))
                self.btl_out.add(nn.Conv2D(num_features, kernel_size=1, use_bias=use_bias))
                if dropout:
                    self.btl_out.add(nn.Dropout(dropout))
                self.btl_out.add(nn.BatchNorm())

            if stride > 1:
                self.downsample = nn.HybridSequential(prefix='')
#                 self.downsample.add(nn.BatchNorm())
#                 self.downsample.add(nn.Activation('relu'))
                self.downsample.add(nn.Conv2D(num_features, kernel_size=1, strides=stride, padding=0,
                                          use_bias=use_bias))
                self.downsample.add(nn.BatchNorm())
                # self.alpha = self.params.get("alpha", shape=(1, 1), init = mx.init.One(), lr_mult=0)
            else:
                self.downsample = None
                # self.alpha = self.params.get("alpha", shape=(1, 1), init = mx.init.One())

    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        if self.downsample:

            residual = self.downsample(x)
            # residual = F.broadcast_mul(*[alpha, residual])
        else:
            residual = x
            # residual = F.broadcast_mul(*[alpha, x])
        x = self.btl_in(x)
        conv3 = self.conv3(x)
        conv5 = self.conv5(x)
        conv7 = self.conv7(x)
        out = F.concat(*[conv3, conv5, conv7], dim=1)
#         out = conv3 + conv5 + conv7
        out = self.btl_out(out)
        out = out + residual
        return out


class PydNet(HybridBlock):
    def __init__(self, prefix, multiplier, num_init_fts, num_final_fts, cells, channels, dropout=0, classes=10, **kwargs):
        super(PydNet, self).__init__(prefix=prefix, **kwargs)
        with self.name_scope():
            assert len(cells) == len(channels), "Cells and channels should be same"
            self.conv0 = nn.HybridSequential('conv0/')
            with self.conv0.name_scope():
                self.conv0.add(nn.BatchNorm(scale=False, center=False))
                self.conv0.add(nn.Conv2D(num_init_fts, kernel_size=3,
                                            strides=1, padding=1, use_bias=False))
            # Add gated cnn cell
            self.cells = nn.HybridSequential()
            for stage in range(len(cells)):
                for cell in range(cells[stage]):
                    stride = 2 if cell == 0 and stage > 0 else 1
                    self.cells.add(ResPydBlock(multiplier, stage, cell, channels[stage], stride, dropout))

            self.output = nn.HybridSequential('classifier/')
            with self.output.name_scope():
                self.output.add(nn.BatchNorm())
                self.output.add(nn.Activation('relu'))
                self.output.add(nn.Conv2D(num_final_fts, kernel_size=1,
                                            strides=1, padding=0, use_bias=True))
                self.output.add(nn.GlobalAvgPool2D())
                self.output.add(nn.Dense(classes))

    def hybrid_forward(self, F, x):
        x = self.conv0(x)
        x = self.cells(x)
        x = self.output(x)
        return x

# Specification
msmobilenet_spec = {56: (64, [6, 6, 6], [64, 128, 256], 1280),
             } # 20: 3; 56: 6; 110: 18
dropout = 0

def get_pydnet(prefix, num_layers, multiplier, classes=1000, **kwargs):
    num_init_fts, cells, channels, num_final_fts = msmobilenet_spec[num_layers]
    net = PydNet(prefix, multiplier, num_init_fts, num_final_fts, cells, channels, dropout, classes, **kwargs)
    return net

def pydnet_0_25(prefix='pydnet_0.25', classes=10, **kwargs):
    return get_pydnet(prefix + '/', 56, 0.25, classes, **kwargs)

def pydnet_0_5(prefix='pydnet_0.5', classes=10, **kwargs):
    return get_pydnet(prefix + '/', 56, 0.5, classes, **kwargs)

def pydnet_0_75(prefix='pydnet_0.75', classes=10, **kwargs):
    return get_pydnet(prefix + '/', 56, 0.75, classes, **kwargs)

# classes = 1000
# ctx = [mx.cpu(0)]
# net = pydnet_0_75(prefix='pydnet_0.75', classes=classes)
# net.hybridize()

# net.initialize(mx.init.MSRAPrelu(), ctx=ctx)
# out = net.forward(mx.nd.ones((2, 4, 224, 224), ctx=ctx[0], dtype='float32'))
# print(out.shape)
# shape = {}
# shape['data'] = (2, 4, 224, 224)
# mx.viz.print_summary(net(mx.sym.Variable('data')), shape=shape)
