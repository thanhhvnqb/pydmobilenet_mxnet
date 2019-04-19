import os
import pickle
from mxnet import nd
from mxnet.gluon.data.vision import CIFAR10
import numpy as np


class ImageNet32(CIFAR10):
    """ImageNet32 image classification dataset

    Each sample is an image (in 3D NDArray) with shape (32, 32, 3).

    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/cifar100
        Path to temp folder for storing data.
    fine_label : bool, default False
        Whether to load the fine-grained (100 classes) or coarse-grained (20 super-classes) labels.
    train : bool, default True
        Whether to load the training or testing set.
    transform : function, default None
        A user defined callback that transforms each sample. For example::

            transform=lambda data, label: (data.astype(np.float32)/255, label)

    """

    def __init__(self,
                 root=os.path.join('datasets', 'imagenet32'),
                 fine_label=False,
                 train=True,
                 transform=None):
        self._train = train
        self._train_data = [
            'train_data_batch_1', 'train_data_batch_2', 'train_data_batch_3',
            'train_data_batch_4', 'train_data_batch_5', 'train_data_batch_6',
            'train_data_batch_7', 'train_data_batch_8', 'train_data_batch_9',
            'train_data_batch_10'
        ]
        self._test_data = ['val_data']
        self._fine_label = fine_label
        self._namespace = 'ImageNet32'
        super(CIFAR10, self).__init__(root, transform)  # pylint: disable=bad-super-call

    def _read_batch(self, filename):
        img_size = 32
        with open(filename, 'rb') as fin:
            data = pickle.load(fin)
        x = data['data']
        y = data['labels']
        y = np.array(y) - 1
        img_size2 = img_size * img_size
        x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2],
                    x[:, 2 * img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3))
        return x, y.astype(np.int32)

    def _get_data(self):
        if self._train:
            data_files = self._train_data
        else:
            data_files = self._test_data
        data, label = zip(*(self._read_batch(os.path.join(self._root, name))
                            for name in data_files))
        data = np.concatenate(data)
        label = np.concatenate(label)

        self._data = nd.array(data, dtype=data.dtype)
        self._label = label


class ImageNet64(CIFAR10):
    """ImageNet64 image classification dataset

    Each sample is an image (in 3D NDArray) with shape (32, 32, 3).

    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/cifar100
        Path to temp folder for storing data.
    fine_label : bool, default False
        Whether to load the fine-grained (100 classes) or coarse-grained (20 super-classes) labels.
    train : bool, default True
        Whether to load the training or testing set.
    transform : function, default None
        A user defined callback that transforms each sample. For example::

            transform=lambda data, label: (data.astype(np.float32)/255, label)

    """

    def __init__(self,
                 root=os.path.join('datasets', 'ImageNet64'),
                 fine_label=False,
                 train=True,
                 transform=None):
        self._train = train
        self._train_data = [
            'train_data_batch_1', 'train_data_batch_2', 'train_data_batch_3',
            'train_data_batch_4', 'train_data_batch_5', 'train_data_batch_6',
            'train_data_batch_7', 'train_data_batch_8', 'train_data_batch_9',
            'train_data_batch_10'
        ]
        self._test_data = ['val_data']
        self._fine_label = fine_label
        self._namespace = 'ImageNet64'
        super(CIFAR10, self).__init__(root, transform)  # pylint: disable=bad-super-call

    def __getitem__(self, idx):
        if self._transform is not None:
            return self._transform(nd.array(self._data[idx]), self._label[idx])
        return self._data[idx], self._label[idx]

    def _read_batch(self, filename):
        img_size = 64
        with open(filename, 'rb') as fin:
            data = pickle.load(fin)
        x = data['data']
        y = data['labels']
        y = np.array(y) - 1
        img_size2 = img_size * img_size
        x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2],
                    x[:, 2 * img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3))
        return x, y.astype(np.int32)

    def _get_data(self):
        if self._train:
            data_files = self._train_data
        else:
            data_files = self._test_data
        data, label = zip(*(self._read_batch(os.path.join(self._root, name))
                            for name in data_files))
        data = np.concatenate(data)
        label = np.concatenate(label)

        # self._data = nd.array(data, dtype=data.dtype)
        self._data = data
        self._label = label

    def transform_first(self, fn, lazy=True):
        """Returns a new dataset with the first element of each sample
        transformed by the transformer function `fn`.

        This is useful, for example, when you only want to transform data
        while keeping label as is.

        Parameters
        ----------
        fn : callable
            A transformer function that takes the first elemtn of a sample
            as input and returns the transformed element.
        lazy : bool, default True
            If False, transforms all samples at once. Otherwise,
            transforms each sample on demand. Note that if `fn`
            is stochastic, you must set lazy to True or you will
            get the same result on all epochs.

        Returns
        -------
        Dataset
            The transformed dataset.
        """
        def base_fn(x, *args):
            if args:
                return (fn(nd.array(x)),) + args
            return fn(nd.array(x))
        return self.transform(base_fn, lazy)
