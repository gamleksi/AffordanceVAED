import numpy as np
from torchvision.datasets.mnist import MNIST
import torchnet as tnt
import torch

class MNISTLoader(object):
    def __init__(self, batch_size, num_processes, debug=False): 
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.intialize_visdom_samples()
        self.debug = debug

    def intialize_visdom_samples(self):
        ds = MNIST(root='./', download=True, train=False) 
        data = getattr(ds, 'test_data')
        labels = getattr(ds, 'test_labels').numpy()
        self.visdom_data = np.zeros([9,data.shape[1], data.shape[2]])
        self.visdom_labels = np.arange(1,10)
        
        for l in self.visdom_labels:
            b = labels == l  
            idx = np.nonzero(b)[0][0]
            self.visdom_data[l - 1] = data[idx]
        
        self.visdom_data = torch.from_numpy(self.visdom_data)
        
    def get_debug_iter(self, train):

        ds = MNIST(root='./', download=True, train=train) 
        data = getattr(ds, 'train_data' if train else 'test_data')

        labels = getattr(ds, 'train_labels' if train else 'test_labels').numpy()

        b = labels == 5  
        idx = np.nonzero(b)[0]

        data = data[idx]

        tds = tnt.dataset.TensorDataset([data])

        return tds.parallel(batch_size=self.batch_size, num_workers=self.num_processes, shuffle=train)

        
    def get_iterator(self, train):

        if self.debug:
            return self.get_debug_iter(train) 

        ds = MNIST(root='./', download=True, train=train) 

        data = getattr(ds, 'train_data' if train else 'test_data')

        tds = tnt.dataset.TensorDataset([data])

        return tds.parallel(batch_size=self.batch_size, num_workers=self.num_processes, shuffle=train)
    
if __name__ == '__main__':
    MNISTLoader(10, 1)