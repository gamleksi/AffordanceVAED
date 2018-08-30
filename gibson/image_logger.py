import matplotlib.pyplot as plt
from torchnet.logger import VisdomLogger
import numpy as np
import os
plt.switch_backend('agg')

class VisdomWrapper(object):

    def __init__(self, port):
        self.port = port
        self.env = env

    def plot_image_list(images, number_of_rows, file_name, main_title, sub_titles=None):
        sample_logger = VisdomLogger('images', port=self.port, nrow=number_of_rows, env=self.env, opts={'title': title})
        sample_logger.log(affordances)


class MatplotLogger(object):

    def __init__(self, model_folder, train, figsize = (30, 20), save_folder=None, root_path='.'):

        self.figure_size = figsize
        if save_folder is None:
            if train:
                save_folder = 'train_results'
            else:
                save_folder = 'eval'

        self.save_path = os.path.join(root_path, 'log', model_folder, save_folder)


    def plot_image_list(self, images, number_of_rows, file_name, main_title, sub_titles = None):

        if images.shape[1] <= 3:
            images = np.transpose(images, (0, 2, 3, 1)) / 255 # N, H, W, 3

        assert(images.shape[0] % number_of_rows == 0)

        number_of_columns = int(images.shape[0] / number_of_rows)

        fig, axeslist = plt.subplots(ncols=number_of_columns, nrows=number_of_rows, figsize=self.figure_size)
        #fig.canvas.set_window_title(main_title)

        for idx in range(len(images)):
            axeslist.ravel()[idx].imshow(images[idx], cmap=plt.jet())
            if sub_titles is not None:
                axeslist.ravel()[idx].set_title(sub_titles[idx], fontsize = 5.0)
            axeslist.ravel()[idx].set_axis_off()

        plt.tight_layout()


        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        plt.savefig(os.path.join(self.save_path, '{}.png'.format(file_name)))
        plt.close(fig)
