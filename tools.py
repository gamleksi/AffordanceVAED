import os
from PIL import Image
import numpy as np
import pickle
from torchvision.datasets.folder import find_classes, make_dataset

from torchvision import transforms

from tqdm import tqdm
import torch
from torch import multiprocessing
from torch.multiprocessing import Pool


def load_and_process(path, resize):

    image = Image.open(path)
    image.load()

    transform_content = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])

    transformed = transform_content(image)
    return transformed


def wrapper_loader(i, path, resize):
    return i, load_and_process(path, resize)


def multiloader(paths, resize):
    # multiprocessing
    sharing_strategy = 'file_system'
    multiprocessing.set_sharing_strategy(sharing_strategy)
    pool = Pool(16)

    N = len(paths)


    pbar = tqdm(total=N)
    images = torch.empty(len(paths), 3, resize[0], resize[1])

    def update(ans):
        images[ans[0]] = ans[1]
        pbar.update()

    def error_cl(err):
        print(err)

    for idx in range(N):
        path = paths[idx]
        pool.apply_async(wrapper_loader, args=(idx, path, resize), callback=update, error_callback=error_cl)

    pool.close()
    pool.join()
    pool.close()

    return images

# IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def images_to_torch_package(folder_path, save_path, extension, train_test_ratio=0.7, resize=(64, 64), num_samples=None):

    np.random.seed(3)

    classes, class_to_idx = find_classes(folder_path)
    samples = make_dataset(folder_path, class_to_idx, [extension])


    path_list = np.array([s[0] for s in samples])
    label_list = np.array([int(s[1]) for s in samples])

    if num_samples is not None:
        path_list = path_list[:num_samples]
        label_list = label_list[:num_samples]

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

        images = multiloader(paths, resize)

        torch.save((images, labels[idx]), os.path.join(save_path, path_names[idx]))

    with open(os.path.join(save_path, 'class_to_idx.pkl'), 'wb') as f:
        pickle.dump(class_to_idx, f, pickle.HIGHEST_PROTOCOL)




def rgb_tensors_grayscale(tensors):

    tensors = (tensors[:, 0] * 299 / 1000) + (tensors[:, 1] * 587 / 1000) + (tensors[:, 2] * 114 / 1000)
    tensors.unsqueeze_(1)
    return tensors


if __name__ == '__main__':

    images_to_torch_package('/home/aleksi/hacks/thesis/code/gibson/data/affordances/part-affordance-dataset/tools',
            '/home/aleksi/hacks/thesis/code/gibson/data/affordances', '.jpg', num_samples=None, train_file='training_affordances.pt',
                            test_file='test_affordances.pt')
