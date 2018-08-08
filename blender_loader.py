import os
import torch
import numpy as np
import torch
from tools import multiloader, orignal_image_size, depth_image_process, image_process


def affordance_process(path, dim, resize=None):

    image = Image.open(path)
    image.load()

#  TODO
#    mat = crop_center(mat, crop_dim[1], crop_dim[2])
#
#    if resize is not None:
#        mat = frequency_drop(mat, resize)

    affordance_tensor = np.zeros(dim.shape)

    color_label= [188, 0] # 188: hole, 0: grasp

    for idx, label  in color_label:

        indices = affordance_tensor[:, :, 0] == label
        affordance_tensor[idx, indices] = 1.

    affordance_torch = torch.from_numpy(affordance_tensor)
    affordance_torch = affordance_torch.float()

    return affordance_torch


def list_file_paths(root_path, folder_name):

    folder_path = os.path.join(root_path, folder_name)
    files = listdir(folder_path)
    file_paths = [os.path.join(folder_path, file_name)  for file_name in files]

    return file_paths


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
    affordance_dim = (2, crop_frame[0], crop_frame[1])

    path_names = [train_file, test_file]

    for idx, indices in enumerate([train_indices, test_indices]):
        images = multiloader(image_paths[indices], image_process, img_dim, resize=resize)
        depth_images = multiloader(depth_paths[indices], depth_image_process, (1, crop_frame[0], crop_frame[1]), resize=resize)
        input = torch.cat([images, depth_images], dim=1)

        affordances = multiloader(affordance_paths[indices], affordance_process, affordance_dim, resize=resize)
        torch.save((input, affordances), os.path.join(save_path, path_names[idx]))

    for idx in range(50):
        save_affordance_pair(images[idx], affordances[idx], depth_images[idx],
                             save_file=os.path.join(root_path, 'examples', 'pair_example_{}.jpg'.format(idx)))


if __name__ == '__main__':

    build_blender_dataset('/opt/data/table_dataset/dataset', '/opt/data/table_dataset/dataset/processed', debug_samples=50)