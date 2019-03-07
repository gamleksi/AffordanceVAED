import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torch.utils.data as data
from scipy.io import loadmat
from torchvision.datasets.folder import find_classes, make_dataset


def load_format_paths(folder_path, extension):

    classes, class_to_idx = find_classes(folder_path)
    samples = make_dataset(folder_path, class_to_idx, [extension])
    paths = np.array([s[0] for s in samples])
    classes = np.array([int(s[1]) for s in samples])

    return paths, classes


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


class UMDLoader(object):

    def __init__(self, batch_size, num_processes, data_path, depth,  debug=False):

        self.batch_size = batch_size
        self.num_processes = num_processes
        dataset = UMDFolder(data_path, depth, debug=debug)
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


if __name__ == '__main__':

    loader = UMDLoader(50, 8, 'full_64', True, debug=True)
    loader.get_iterator(True)
    loader.testset.__getitem__(0)
