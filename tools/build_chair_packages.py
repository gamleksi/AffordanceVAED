import os

import numpy as np
from sklearn.externals.joblib import Parallel, delayed
import scipy.io as sio
from skimage import io
import torch

def initialize_image_paths(data_file, root_path, data_folder):
    data_path = os.path.join(root_path, data_folder, data_file)
    chair_frame = sio.loadmat(data_path)
    return data_path, chair_frame

def main(data_file, root_path, data_folder):

    data_path, chair_frame = initialize_image_paths(data_file, root_path, data_folder)

    def folder(idx):
        return chair_frame['folder_names'][0][idx][0]

    def file(idx):
        return chair_frame['instance_names'][0][idx][0]

    num_categories = chair_frame['folder_names'].shape[1]
    num_img_per_item = chair_frame['instance_names'].shape[1]

    num_items = num_categories * num_img_per_item

    def image_path(idx):

        folder_idx = int(idx / num_img_per_item)
        folder_path = os.path.join(root_path, data_folder, folder(folder_idx), 'renders')

        file_idx = idx - folder_idx * num_img_per_item
        file_path = os.path.join(folder_path, file(file_idx))

        return file_path, folder_idx

    def get_img_path(idx):

        img_name, item_label = image_path(idx)
        image = io.imread(img_name)

        sample = {'image': image, 'label': item_label}

        return sample

    num_train_samples = int(0.7 * num_items)
    num_test_samples = num_items - num_train_samples

    arr = np.arange(num_items)
    np.random.shuffle(arr)

    train_indices = arr[:num_train_samples]
    test_indices = arr[num_train_samples:]

    def images_labels(indices, n):
        images = []
        labels = np.zeros(n, dtype=int)
        for i, idx in enumerate(indices):
            img_name, item_label = image_path(idx)
            images.append(img_name)
            labels[i] = item_label

        return images, labels


    train_img_paths, train_labels = images_labels(train_indices, num_train_samples)
    test_img_paths, test_labels = images_labels(test_indices, num_test_samples)

    print('processing images...')


    def files_to_package(image_paths, labels, package_name, steps):

        def read_images(image_paths_list):
            images = Parallel(n_jobs=16, verbose=5)(
                delayed(io.imread)(p) for p in image_paths_list
            )
            return images

        with open(os.path.join(root_path, package_name), 'ab') as f:

            h = int(len(image_paths) / steps)

            for s in range(2): 
                c_i = s * h
                if s < (steps - 1): 
                    n_i = c_i + h 
                else:
                    n_i = -1
                imgs = read_images(image_paths[c_i:n_i])
                import ipdb; ipdb.set_trace()
                print('{}: saving images...'.format(s + 1))
                data_set = (imgs, labels[c_i:n_i])
                torch.save(data_set, f)

    steps = 400 
    files_to_package(train_img_paths, train_labels, 'training.pt', steps)

    import ipdb; ipdb.set_trace()

    files_to_package(test_img_paths, test_labels, 'test.pt', steps)


if __name__ == '__main__':
    main('all_chair_names.mat', '../data/chairs', 'rendered_chairs')
