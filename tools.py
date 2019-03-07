import os
from PIL import Image
import numpy as np
import torch

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

DEPTH_MAX = 3626



def model_name_search(folder_path):

  for file in os.listdir(folder_path):
        if file.endswith(".tar"):
          splitted = file.split('.pth', 1)

  return splitted[0]

# -> H, W,
def affordance_to_image(affordance_matrix):

    if torch.is_tensor(affordance_matrix):
        affordance_matrix = affordance_matrix.detach().cpu().numpy()

    image = np.zeros((affordance_matrix.shape[1], affordance_matrix.shape[2], 3))

    for idx in range(2):

        indices = affordance_matrix[idx] > 0
        label = idx + 1
        color = np.array(AFFORDANCE_RGB[label])
        image[indices] = color

    return image


def affordance_to_array(affordance_matrix):


    if torch.is_tensor(affordance_matrix):
        affordance_matrix = affordance_matrix.detach().cpu().numpy()

    max_table = np.argmax(affordance_matrix, 0).reshape(-1)
    max_values = np.max(affordance_matrix, 0).reshape(-1) # 0 or 1

    image = np.zeros((3, affordance_matrix.shape[1] * affordance_matrix.shape[2]))

    for idx, max_label in enumerate(max_table):

        if max_values[idx] > 0.5:

            max_label = int(max_label) + 1

            color = np.array(AFFORDANCE_RGB[max_label])
            image[0, idx] = color[0]
            image[1, idx] = color[1]
            image[2, idx] = color[2]

    return image.reshape((3, affordance_matrix.shape[1], affordance_matrix.shape[2]))


def affordance_layers_to_array(affordance_matrix):

    if torch.is_tensor(affordance_matrix):
        affordance_matrix = affordance_matrix.detach().cpu().numpy()

    grayscale_layers = np.array([np.stack((affordance_matrix[idx] ,) * 3) * 255 for idx in range(affordance_matrix.shape[0])])

    return grayscale_layers


def save_affordance_pair(image, affordance_matrix, depth_image, save_file='testi.jpg'):

    if torch.is_tensor(image):
        image = image.detach().numpy()
        depth_image = depth_image.detach().numpy()

    image = np.transpose(image, (1, 2, 0)) * 255

    depth_image = np.stack((depth_image[0],)*3, -1) * 255

    affordance_matrix = affordance_to_image(affordance_matrix)

    imgs_comb = np.hstack([image, depth_image, affordance_matrix])
    imgs_comb = imgs_comb.astype('uint8')

    imgs_comb = Image.fromarray(imgs_comb)
    imgs_comb.save(save_file)


def save_arguments(args, save_path):

    args = vars(args)
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)
    file = open(os.path.join(save_path, "arguments.txt"), 'w')
    lines = [item[0] + " " + str(item[1]) + "\n" for item in args.items()]
    file.writelines(lines)
