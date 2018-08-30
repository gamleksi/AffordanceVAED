import torchnet as tnt
import numpy as np
import torch
from torchvision.datasets.mnist import MNIST
import os


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



class ImageLoader(object):

    def __init__(self, batch_size, num_processes, root_dir,  debug=False, grayscale=False):
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

class AffordanceLoader(ImageLoader):

    def __init__(self, batch_size, num_processes, root_dir,  debug=False):

        self.batch_size = batch_size
        self.num_processes = num_processes
        self.debug = debug
        self.root_dir = root_dir
        self._intialize_visdom_samples()

    def _intialize_visdom_samples(self):

        images, affordances, _ = torch.load(os.path.join(self.root_dir, 'test.pt'))
        num_samples = 20

        self.visdom_data = (images[:num_samples], affordances[:num_samples])

    def get_iterator(self, train):

        if train:
            images, affordances, _ = torch.load(os.path.join(self.root_dir, 'training.pt'))
        else:
            images, affordances, _ = torch.load(os.path.join(self.root_dir, 'test.pt'))

        if self.debug:
            images = images[:100]
            affordances = affordances[:100]

        ds = tnt.dataset.TensorDataset([images, affordances])

        return ds.parallel(batch_size=self.batch_size, num_workers=self.num_processes, shuffle=train)

if __name__ == '__main__':

    loader = AffordanceLoader(50, 8, 'data/affordances/full_64', debug=True)
    loader.get_iterator(True)
