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

from torchvision import transforms
from PIL import Image
from tools import load_format_paths
import torch.utils.data as data
from scipy.io import loadmat


class UMDFolder(data.Dataset):

    def __init__(self, data_path, include_depth, debug=False):

        max_idx = 100 if debug else None

        self.include_depth = include_depth

        image_paths, classes = load_format_paths(data_path, '.jpg')
        self.images = image_paths[:max_idx]
        self.classes = classes[:max_idx]

        affordance_paths, _ = load_format_paths(data_path, '_label.mat')
        self.affordances = affordance_paths[:max_idx]

        if self.include_depth:
            depth_paths, _ = load_format_paths(data_path, '.png')
            self.depths = depth_paths[:max_idx]

    def __getitem__(self, index):

        img_path = self.images[index]
        img = self.loader(img_path)
        img = self.image_transform(img)

        if self.include_depth:
            depth_path = self.depths[index]
            depth = self.loader(depth_path)
            depth = self.depth_transform(depth)

            sample = self.create_sample(img, depth)
        else:
            sample = img

        affordance_path = self.affordances[index]
        mat_dict = list(loadmat(affordance_path).values())
        mat = np.array(mat_dict[3])
        mat = self.affordance_transform(mat)

        mat_tensor = np.zeros((7, mat.shape[0], mat.shape[1]))

        for idx in range(7):
            label = idx + 1
            indices = mat == int(label)
            mat_tensor[idx, indices] = 1.

        target = torch.from_numpy(mat_tensor)
        target = target.float()

        return sample, target

    def loader(self, path):
        image = Image.open(path)
        image.load()

        return image

    def create_sample(self, img, depth):
        return torch.cat([img, depth])

    def image_transform(self, image):

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        transform_content = transforms.Compose([transforms.CenterCrop((240, 320)), transforms.ToTensor()])
        transformed = transform_content(image)

        return transformed

    def depth_transform(self, image):

        transform_content = transforms.Compose([transforms.CenterCrop((240, 320)), transforms.ToTensor()])
        transformed = transform_content(image)
        return transformed.float() / (3626)

    def affordance_transform(self, mat):

        h, w = mat.shape

        startx = h // 2 - (240//2)
        starty = w // 2 - (320//2)

        return mat[startx:startx+240, starty:starty+320]

    def __len__(self):
        return len(self.images)


class AffordanceLoader(ImageLoader):

    def __init__(self, batch_size, num_processes, root_dir, depth,  debug=False):

        self.batch_size = batch_size
        self.num_processes = num_processes
        self.debug = debug
        self.root_dir = root_dir
        self.depth = depth
        dataset = UMDFolder('part-affordance-dataset/tools', self.depth, debug=debug)
        train_size = int(dataset.__len__() * 0.7)
        test_size = dataset.__len__() - train_size
        trainset, testset = torch.utils.data.random_split(dataset, (train_size, test_size))
        self.trainset = trainset
        self.testset = testset

    def get_iterator(self, train):

        if train:
            dataset = self.trainset

        else:
            dataset = self.testset

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=train, num_workers=self.num_processes)

#    def get_iterator(self, train):
#
#        if train:
#            images, affordances, _ = torch.load(os.path.join(self.root_dir, 'training.pt'))
#        else:
#            images, affordances, _ = torch.load(os.path.join(self.root_dir, 'test.pt'))
#
#        if not(self.depth):
#            images = images[:, :3]
#
#        if self.debug:
#            images = images[:100]
#            affordances = affordances[:100]
#
#        ds = tnt.dataset.TensorDataset([images, affordances])
#
        #return ds.parallel(batch_size=self.batch_size, num_workers=self.num_processes, shuffle=train)

if __name__ == '__main__':

    loader = AffordanceLoader(50, 8, 'full_64', True, debug=True)
    loader.get_iterator(True)
    loader.testset.__getitem__(0)
