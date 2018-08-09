import os
from os import listdir
import torch
import numpy as np
import torch
from tools import multiloader, orignal_image_size, image_process, depth_rgba_process, save_affordance_pair
from PIL import Image


def affordance_process(path, dim, resize=None):

    image = Image.open(path)
    image.load()
    image = np.array(image)

#  TODO
#    mat = crop_center(mat, crop_dim[1], crop_dim[2])
#
#    if resize is not None:
#        mat = frequency_drop(mat, resize)

    affordance_tensor = np.zeros(dim)

    color_label= [188, 0] # 188: hole, 0: grasp

    for idx, label  in enumerate(color_label):

        indices = image[:, :, 0] == label
        affordance_tensor[idx, indices] = 1.

    affordance_torch = torch.from_numpy(affordance_tensor)
    affordance_torch = affordance_torch.float()

    return affordance_torch


def list_file_paths(root_path, folder_name):

    folder_path = os.path.join(root_path, folder_name)
    files = listdir(folder_path)
    files = sorted(files)
    file_paths = [os.path.join(folder_path, file_name)  for file_name in files]

    return np.array(file_paths)


def build_blender_dataset(root_path, save_path, train_test_ratio=0.7, debug_samples=None,
                            train_file='training.pt', test_file='test.pt'):

    # Load Paths
    image_paths = list_file_paths(root_path, 'images')
    depth_paths = list_file_paths(root_path, 'depths')
    affordance_paths = list_file_paths(root_path, 'affordances')

    assert(len(affordance_paths) == len(image_paths))

    # shuffle
    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)

    if debug_samples is not None:
        indices = indices[:debug_samples]

    # divide
    m_idx = int(len(indices) * train_test_ratio)

    train_indices = indices[:m_idx]
    test_indices = indices[m_idx:]

    crop_frame = orignal_image_size(image_paths[0])
    img_dim = (3, crop_frame[0], crop_frame[1])
    depth_dim = (1, crop_frame[0], crop_frame[1])
    affordance_dim = (2, crop_frame[0], crop_frame[1])

    path_names = [train_file, test_file]

#    for dataset_idx, indices in enumerate([train_indices, test_indices]):
#
#        images = torch.zeros(len(indices), img_dim[0], img_dim[1], img_dim[2])
#        affordances = torch.zeros(len(indices), affordance_dim[0], affordance_dim[1], affordance_dim[2])
#        depths = torch.zeros(len(indices), depth_dim[0], depth_dim[1], depth_dim[2])
#
#        for i, idx in enumerate(indices):
#
#            images[i] =  image_process(image_paths[idx], img_dim)
#            affordances[i] = affordance_process(affordance_paths[idx], affordance_dim)
#            depths[i] = depth_rgba_process(depth_paths[idx], depth_dim)
#
#            input = torch.cat([images, depths], dim=1)

    for idx, indices in enumerate([train_indices, test_indices]):
        images = multiloader(image_paths[indices], image_process, img_dim)
        depths = multiloader(depth_paths[indices], depth_rgba_process, depth_dim)
        input = torch.cat([images, depths], dim=1)

        affordances = multiloader(affordance_paths[indices], affordance_process, affordance_dim)
        torch.save((input, affordances), path_names[idx])  #os.path.join(save_path, path_names[idx]))

    for idx in range(30):
        save_affordance_pair(images[idx], affordances[idx], depths[idx],
                             save_file=os.path.join(root_path, 'examples', 'pair_example_{}.jpg'.format(idx)))


if __name__ == '__main__':

    build_blender_dataset('/opt/data/table_dataset/dataset', '/opt/data/table_dataset/dataset/processed')
