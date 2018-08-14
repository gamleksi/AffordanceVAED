import os
from PIL import Image
import numpy as np
from os import listdir
import torch

from torchvision import transforms
import torch.utils.data as data
from tools import save_affordance_pair


class BlenderFolder(data.Dataset):

    def __init__(self, root_path, debug=False):

        self.root_path = root_path
        self.images = self.list_file_paths(self.root_path, 'images')
        self.depths = self.list_file_paths(self.root_path, 'depths')
        self.affordances = self.list_file_paths(self.root_path, 'affordances')

        if debug:
            self.images = self.images[:100]
            self.depths = self.depths[:100]
            self.affordances = self.affordances[:100]

    def __getitem__(self, index):

        img_path = self.images[index]
        img = self.loader(img_path)
        img = self.image_transform(img)

        depth_path = self.depths[index]
        depth = self.loader(depth_path)
        depth = self.depth_transform(depth)

        sample = self.create_sample(img, depth)

        affordance_path = self.affordances[index]
        affordance = self.loader(affordance_path)

        target = self.affordance_transform(affordance)

        return sample, target

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root_path)
        print(fmt_str)

    def create_sample(self, img, depth):
        return torch.cat([img, depth])

    def list_file_paths(self, root_path, folder_name):

        folder_path = os.path.join(root_path, folder_name)
        files = listdir(folder_path)
        files = sorted(files)
        file_paths = [os.path.join(folder_path, file_name)  for file_name in files]

        return np.array(file_paths)

    def loader(self, path):
        image = Image.open(path)
        image.load()
        return image

    def affordance_transform(self, image):

        image = np.array(image)
        affordance_tensor = np.zeros((2, image.shape[0], image.shape[1]))

        color_label= [188, 0] # 188: hole, 0: grasp

        for idx, label  in enumerate(color_label):

            indices = image[:, :, 0] == label
            affordance_tensor[idx, indices] = 1.

        affordance_torch = torch.from_numpy(affordance_tensor)
        affordance_torch = affordance_torch.float()

        return affordance_torch

    def image_transform(self, image):

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        transform_content = transforms.Compose([transforms.CenterCrop((image.size[1], image.size[0])),
                                                     transforms.ToTensor()])
        transformed = transform_content(image)
        return transformed

    def depth_transform(self, image):

        image = image.getchannel(0)

        transform_content = transforms.Compose([transforms.CenterCrop((image.size[1], image.size[0])),
                                             transforms.ToTensor()
                                             ])

        transformed = transform_content(image)

        return transformed.float()

    def generate_examples(self, num_examples=10, folder_name='examples'):

        for idx in range(num_examples):
            sample, target = self.__getitem__(idx)
            img = sample[:3]
            depth = torch.unsqueeze(sample[3], 0)

            save_affordance_pair(img, target, depth,
                                 save_file=os.path.join(self.root_path, folder_name, 'pair_example_{}.jpg'.format(idx)))


class BlenderLoader(object):

    def __init__(self, batch_size, num_processes, debug=False, num_visdon_samples=20):

        self.batch_size = batch_size
        self.num_processes = num_processes

        dataset = BlenderFolder('/opt/data/table_dataset/dataset', debug=debug)
        train_size = int(dataset.__len__() * 0.7)
        test_size = dataset.__len__() - train_size
        trainset, testset = torch.utils.data.random_split(dataset, (train_size, test_size))
        self.trainset = trainset
        self.testset = testset
        self._intialize_visdom_samples(num_visdon_samples)

    def _intialize_visdom_samples(self, num_samples):

        sample, affordance = self.testset.__getitem__(0)

        samples = torch.empty(num_samples, sample.shape[0], sample.shape[1], sample.shape[2])
        affordances = torch.empty(num_samples, affordance.shape[0], affordance.shape[1], affordance.shape[2])

        samples[0] = sample
        affordances[0] = affordance

        for i in range(1, num_samples):
            sample, affordance = self.testset.__getitem__(i)
            samples[i] = sample
            affordances[i] = affordance

        self.visdom_data = (samples, affordances)


    def get_iterator(self, train):

        if train:
            dataset = self.trainset

        else:
            dataset = self.testset

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=train, num_workers=self.num_processes)


if __name__ == '__main__':

    #loader = BlenderLoader(40, 30, debug=True)
    dataset = BlenderFolder('/opt/data/table_dataset/dataset', debug=True)

