import torchnet as tnt
import numpy as np
import torch
from torchvision.datasets.mnist import MNIST

from torchvision import transforms


class MNISTLoader(object):
    def __init__(self, batch_size, num_processes, debug=False):
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.intialize_visdom_samples()
        self.debug = debug

    def intialize_visdom_samples(self):

        ds = MNIST(root='./', download=True, train=False) 
        data = getattr(ds, 'test_data')

        # stupid hack to get a sample from every class
        self.visdom_data = torch.zeros(10, data.shape[1], data.shape[2])
        self.visdom_labels = np.arange(0, 10)

        labels = getattr(ds, 'test_labels').numpy()
        for l in self.visdom_labels:
            b = labels == l  
            idx = np.nonzero(b)[0][0]
            self.visdom_data[l - 1] = data[idx]

        self.visdom_data= self.visdom_data.float() / 255
        self.visdom_data.unsqueeze_(1)
        
    def get_debug_iter(self, train):

        ds = MNIST(root='./', download=True, train=train)
        data = getattr(ds, 'train_data' if train else 'test_data')

        labels = getattr(ds, 'train_labels' if train else 'test_labels').numpy()
        b = labels == 5  
        idx = np.nonzero(b)[0]

        data = data[idx]
        data.unsqueeze_(1)

        tds = tnt.dataset.TensorDataset([data])

        return tds.parallel(batch_size=self.batch_size, num_workers=self.num_processes, shuffle=train)

    def get_iterator(self, train):

        if self.debug:
            return self.get_debug_iter(train)

        ds = MNIST(root='./', download=True, train=train)
        
        data = getattr(ds, 'train_data' if train else 'test_data')

        data = data.float() / 255
        data.unsqueeze_(1)

        tds = tnt.dataset.TensorDataset(data)
        return tds.parallel(batch_size=self.batch_size, num_workers=self.num_processes, shuffle=train)


from torchvision.datasets.folder import default_loader
from torchvision.datasets import ImageFolder




# The whole dataset is moved to RAM
from tqdm import tqdm

import torch
import os


class ChairsLoader(object):
    def __init__(self, batch_size, num_processes, debug=False, root_dir='data/chairs/', grayscale=False, train_test_ratio=0.7):
        self.batch_size = batch_size
        self.grayscale = grayscale
        self.num_processes = num_processes
        self.debug = debug
        self.root_dir = root_dir
        self._intialize_visdom_samples()

    def _intialize_visdom_samples(self):
        data, _ = torch.load(os.path.join(self.root_dir, 'test.pt'))
        if self.grayscale:
            from tools import rgb_tensors_grayscale
            data = rgb_tensors_grayscale(data)

        self.visdom_data = torch.stack([data[idx] for idx in range(10)])

    def get_iterator(self, train):

        if train:
            data, _ = torch.load(os.path.join(self.root_dir, 'training.pt'))
        else:
            data, _ = torch.load(os.path.join(self.root_dir, 'test.pt'))

        if self.debug:
            data = data[:1000]

        if self.grayscale:
            from tools import rgb_tensors_grayscale
            data = rgb_tensors_grayscale(data)

        ds = tnt.dataset.TensorDataset(data)

        return ds.parallel(batch_size=self.batch_size, num_workers=self.num_processes, shuffle=train)


if __name__ == '__main__':

    l = ChairsLoader(40, 4)
    l.get_iterator(True)