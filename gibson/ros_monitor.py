import os
import torch
from tools import model_name_search, affordance_to_array, affordance_layers_to_array
from models.blender_model import Decoder, Encoder
from models.simple_model import AffordanceVAE
from image_logger import MatplotLogger
from blender_dataset import KinectImageHandler
import numpy as np

ABSOLUTE_DIR = os.path.dirname(os.path.abspath(__file__))

class ROSVAE(object):

    def __init__(self, model_folder, latent_dim, include_depth, log_folder='ROS_RESULTS', root_path=ABSOLUTE_DIR):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if device.type == 'cuda':
            print('GPU works!')
        else:
            print('YOU ARE NOT USING GPU')

        image_channels = 4 if include_depth else 3

        encoder = Encoder(latent_dim, image_channels)
        decoder = Decoder(latent_dim, 2)
        self.model = AffordanceVAE(encoder, decoder, device).to(device)
        self.load_parameters(model_folder, root_path)
        self.logger = MatplotLogger(model_folder, False, save_folder=log_folder, figsize=(20, 10), root_path=root_path)
        self.kinect_handler = KinectImageHandler()

    def load_parameters(self, folder, root_path):
        model_path = os.path.join(root_path, 'log', folder)
        model_name = model_name_search(model_path)
        path = os.path.join(model_path, '{}.pth.tar'.format(model_name))
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def _process_samples(self, sample):
        image = self.kinect_handler.image_transform(sample)
        return torch.unsqueeze(image, 0)

    def _get_result(self, sample):
        recons = self.model.reconstruct(sample)
        return recons

    def get_result_pair(self, sample, title):

        sample = self._process_samples(sample)
        recon = self._get_result(sample)
        recon = recon[0]
        image = sample[0].cpu().detach().numpy() * 255

        reconstruction = affordance_to_array(recon)
        affordance_layers = affordance_layers_to_array(recon)

        samples = np.stack((image, reconstruction, affordance_layers[0], affordance_layers[1]))

        self.logger.plot_image_list(samples, 1, title, title)

def debug():

    evaluator = ROSVAE('rgb_test', 10, False, log_folder='ROS_RESULTS')
    from PIL import Image
    KINECT_SAMPLES = '/home/aleksi/hacks/vae_ws/real_images/images'
    sample = Image.open(os.path.join(KINECT_SAMPLES, '001_image.png'))
    evaluator.get_result_pair(sample, 'test')

if __name__  == '__main__':
    debug()
