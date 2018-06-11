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
from scipy.io import loadmat

from skimage import transform

'''
  1 - 'grasp': red (255, 0, 0)
  2 - 'cut':   yellow (255, 255, 0)
  3 - 'scoop'  green(0, 255, 0)
  4 - 'contain' turquoise (0, 255, 255)
  5 - 'pound' blue (0, 0, 255)
  6 - 'support': pink (255, 0, 255)
  7 - 'wrap-grasp': white (255, 255, 255)
  Background: black (0, 0, 0)
'''

AFFORDANCE_RGB = {
    1: (255, 0, 0),
    2: (255, 255, 0),
    3: (0, 255, 0),
    4: (0, 255, 255),
    5: (0, 0, 255),
    6: (255, 0, 255),
    7: (255, 255, 255),
    }


def affordance_to_image(affordance_matrix):

    if torch.is_tensor(affordance_matrix):
        affordance_matrix = affordance_matrix.detach().numpy()

    image = np.zeros((affordance_matrix.shape[1], affordance_matrix.shape[2], 3))

    for idx in range(7):

        indices = affordance_matrix[idx] > 0
        label = idx + 1
        color = np.array(AFFORDANCE_RGB[label])
        image[indices] = color

    return image

def save_affordance_pair(image, affordance_matrix, save_file='testi.jpg'):

    if torch.is_tensor(image):
        image = image.detach().numpy()

    if image.shape[0] > 3:
        image = image[0:3]

    image = np.transpose(image, (1, 2, 0)) * 255

    affordance_matrix = affordance_to_image(affordance_matrix)

    imgs_comb = np.hstack((image, affordance_matrix))
    imgs_comb = imgs_comb.astype('uint8')

    imgs_comb = Image.fromarray(imgs_comb)
    imgs_comb.save(save_file)



def mat_process(path, resize=None):

    mat_dict = list(loadmat(path).values())
    mat = np.array(mat_dict[3])

    mat_tensor = np.zeros((7, mat.shape[0], mat.shape[1]))

    for idx in range(7):
        label = idx + 1
        indices = mat == int(label)
        mat_tensor[idx, indices] = 1.

    mat_torch = torch.from_numpy(mat_tensor)
    mat_torch = mat_torch.float()

    return mat_torch


def image_process(path, resize=None):

    image = Image.open(path)
    image.load()

    if resize is None:
         transform_content = transforms.ToTensor()
    else:
        transform_content = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])

    transformed = transform_content(image)
    return transformed


def orignal_image_size(path):

    image = Image.open(path)
    image.load()

    return (image.size[1], image.size[0])


def wrapper_loader(i, path, resize, load_function):
    return i, load_function(path, resize)


def multiloader(paths, load_function, img_dim, resize=None):
    # multiprocessing
    sharing_strategy = 'file_system'
    multiprocessing.set_sharing_strategy(sharing_strategy)
    pool = Pool(1)

    N = len(paths)

    pbar = tqdm(total=N)

    if resize is None:
        images = torch.empty(len(paths), img_dim[0], img_dim[1], img_dim[2])

    else:
        images = torch.empty(len(paths), img_dim[0], resize[0], resize[1])

    def update(ans):
        images[ans[0]] = ans[1]
        pbar.update()

    def error_cl(err):
        print(err)

    for idx in range(N):
        path = paths[idx]
        pool.apply_async(wrapper_loader, args=(idx, path, resize, load_function), callback=update, error_callback=error_cl)

    pool.close()
    pool.join()
    pool.close()

    return images

# IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def images_to_torch_package(folder_path, save_path, extension, train_test_ratio=0.7, resize=(64, 64), num_samples=None,
                            train_file='training.pt', test_file='test.pt', load_function=image_process, img_dim=3):

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
    path_names = [train_file, test_file]

    for idx, paths in enumerate(path_lists):

        print('Processing {}-data...'.format(path_names[idx]))

        images = multiloader(paths, resize, load_function, img_dim)

        torch.save((images, labels[idx]), os.path.join(save_path, path_names[idx]))

    with open(os.path.join(save_path, 'class_to_idx.pkl'), 'wb') as f:
        pickle.dump(class_to_idx, f, pickle.HIGHEST_PROTOCOL)


def build_affordance_dataset(folder_path, save_path, train_test_ratio=0.7, num_samples=None,
                            train_file='training.pt', test_file='test.pt', resize=None):

    # Load Paths

    classes, class_to_idx = find_classes(folder_path)
    image_samples = make_dataset(folder_path, class_to_idx, ['.jpg'])

    image_paths = np.array([s[0] for s in image_samples])
    classes = np.array([int(s[1]) for s in image_samples])

    affordance_paths = make_dataset(folder_path, class_to_idx, ['_label.mat'])
    affordance_paths = np.array([s[0] for s in affordance_paths])




    assert(len(affordance_paths) == len(image_paths))

    # shuffle
    np.random.seed(3)
    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)

    if num_samples is not None:
        indices = indices[:num_samples]

    # divide
    m_idx = int(len(indices) * train_test_ratio)

    train_indices = indices[:m_idx]
    test_indices = indices[m_idx:]

    dataset_names = [train_file, test_file]

    img_frame_size = orignal_image_size(image_paths[0])
    img_dim = (3, img_frame_size[0], img_frame_size[1])
    affordance_dim = (7, img_frame_size[0], img_frame_size[1])


    for idx, indices in enumerate([train_indices, test_indices]):
        images = multiloader(image_paths[indices], image_process, img_dim, resize=resize)
        affordances = multiloader(affordance_paths[indices], mat_process, affordance_dim, resize=resize)

        for idx in range(50):
            save_affordance_pair(images[idx], affordances[idx],
                                 save_file='/home/aleksi/hacks/thesis/code/gibson/data/affordances/examples/pair_example_{}.jpg'.format(idx))

        break;




def rgb_tensors_grayscale(tensors):

    tensors = (tensors[:, 0] * 299 / 1000) + (tensors[:, 1] * 587 / 1000) + (tensors[:, 2] * 114 / 1000)
    tensors.unsqueeze_(1)
    return tensors


if __name__ == '__main__':
    build_affordance_dataset('/home/aleksi/hacks/thesis/code/gibson/data/affordances/part-affordance-dataset/tools',
                            '/home/aleksi/hacks/thesis/code/gibson/data/affordances', num_samples=100)

#    images_to_torch_package('/home/aleksi/hacks/thesis/code/gibson/data/affordances/part-affordance-dataset/tools',
#            '/home/aleksi/hacks/thesis/code/gibson/data/affordances', '_label.mat',  num_samples=100,
#                            train_file='training_affordances.pt', test_file='test_affordances.pt', load_function=mat_process,
#                            resize=(480, 640), img_dim=9)
