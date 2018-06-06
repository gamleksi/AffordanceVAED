import os
from PIL import Image
import numpy as np

from torchvision.datasets import ImageFolder
from torchvision import transforms

from tqdm import tqdm
import torch
from torch import multiprocessing
from torch.multiprocessing import Pool


def load_and_process(path):

    image = Image.open(path)
    image.load()

    transform_content = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    transformed = transform_content(image)
    return transformed


def wrapper_loader(i, path):
    return i, load_and_process(path)


def multiloader(paths):
    # multiprocessing
    sharing_strategy = 'file_system'
    multiprocessing.set_sharing_strategy(sharing_strategy)
    pool = Pool(16)

    N = len(paths)


    pbar = tqdm(total=N)
    images = torch.empty(len(paths), 3, 64, 64)

    def update(ans):
        images[ans[0]] = ans[1]
        pbar.update()

    def error_cl(err):
        print(err)

    for idx in range(N):
        path = paths[idx]
        pool.apply_async(wrapper_loader, args=(idx, path), callback=update, error_callback=error_cl)

    pool.close()
    pool.join()
    pool.close()

    return images



def images_to_torch_package(folder_path, save_path, train_test_ratio=0.7, resize=None):

    np.random.seed(3)

    data = ImageFolder(root=folder_path)
    data.imgs = data.imgs[:int(len(data.imgs))]
    path_list = np.array([path[0] for path in data.imgs])
    label_list = np.array([path[1] for path in data.imgs])

    # shuffle
    indices = np.arange(len(path_list))
    np.random.shuffle(indices)

    # divide
    m_idx = int(len(path_list) * train_test_ratio)

    train_indices = indices[:m_idx]
    test_indices = indices[m_idx:]

    train_paths = path_list[train_indices]
    train_labels = torch.from_numpy(label_list[train_indices])

    test_paths = path_list[test_indices]
    test_labels = torch.from_numpy(label_list[test_indices])

    path_lists = [train_paths, test_paths]
    labels = [train_labels, test_labels]
    path_names = ['training.pt', 'test.pt']

    for idx, paths in enumerate(path_lists):

        print('Processing {}-data...'.format(path_names[idx]))

        images = multiloader(paths)

        torch.save((images, labels[idx]), os.path.join(save_path, path_names[idx]))


def rgb_tensors_grayscale(tensors):

    tensors = (tensors[:, 0] * 299 / 1000) + (tensors[:, 1] * 587 / 1000) + (tensors[:, 2] * 114 / 1000)
    tensors.unsqueeze_(1)
    return tensors


if __name__ == '__main__':

    data = images_to_torch_package('data/chairs/rendered_chairs/', 'data/chairs/')
