import sys, os
sys.path.insert(0, './libs/tnt/')

import torchnet as tnt   
import torch
import numpy as np
import torch
from torchvision.datasets.mnist import MNIST

from torchvision import transforms
from PIL import Image

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
        self.visdom_labels = np.arange(0,10)

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

from torchvision.datasets import ImageFolder

 


class ChairsLoader(object):
    def __init__(self, batch_size, num_processes, debug=False, root_dir='data/chairs/rendered_chairs/', grayscale=False, train_test_ratio=0.7): 
        self.batch_size = batch_size
        self.grayscale = grayscale
        self.num_processes = num_processes
        self.debug = debug
        self._initialize_dataset(root_dir, train_test_ratio)
        self._intialize_visdom_samples()

    def _intialize_visdom_samples(self):

        self.dataset.select('test')
        self.visdom_data = torch.stack([self.dataset.__getitem__(idx) for idx in range(10)])

    def _initialize_dataset(self, root_dir, train_test_ratio):

        data = ImageFolder(root=root_dir)
        path_list = [path[0] for path in data.imgs] 

        if self.debug:
            path_list = path_list[:40]
        
        ds = tnt.dataset.ListDataset(path_list, load=self._load_data)
        ds = ds.shuffle()
        self.dataset = ds.split({'train': 0.75, 'test': 0.25})

    def get_iterator(self, train):

        if train:
            self.dataset.select('train')
        else:
            self.dataset.select('test')

        data_loader = self.dataset.parallel(batch_size=self.batch_size, num_workers=self.num_processes, shuffle=train)

        return data_loader 

    def _load_data(self, path):

        image = Image.open(path) 

        if self.grayscale:
            grayscale = transforms.Grayscale(num_output_channels=1)
            image = grayscale(image)

        transform_content = transforms.Compose([
            transforms.CenterCrop(64),
            transforms.ToTensor(),

        ])

        transformed = transform_content(image)

        return transformed

if __name__ == '__main__':
    l = ChairsLoader(40, 4)