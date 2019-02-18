import os
import torch
from tools import model_name_search # , affordance_to_array, affordance_layers_to_array
from models.blender_model import Decoder, Encoder
from models.simple_model import AffordanceVAE

from torchvision import transforms

#    def get_result_pair(self, sample, title):
#
#        sample = self._process_samples(sample)
#        recon = self._get_result(sample)
#        recon = recon[0]
#        image = sample[0].cpu().detach().numpy() * 255
#
#        reconstruction = affordance_to_array(recon)
#        affordance_layers = affordance_layers_to_array(recon)

#        samples = np.stack((image, reconstruction, affordance_layers[0], affordance_layers[1]))
#
        #self.logger.plot_image_list(samples, 1, title, title)

# Blender W: 320, H: 160
# Kinect W: 640, H: 480

class RosPerceptionVAE(object):

    def __init__(self, model_dir, latent_dim):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if device.type == 'cuda':
            print('GPU works for Perception!')
        else:
            print('Perception is not using GPU')

        encoder = Encoder(latent_dim, 3)
        decoder = Decoder(latent_dim, 2)
        self.model = AffordanceVAE(encoder, decoder, device).to(device)
        self.load_parameters(model_dir)

    def load_parameters(self, model_dir):

        model_name = model_name_search(model_dir)
        path = os.path.join(model_dir, '{}.pth.tar'.format(model_name))

        model_state = torch.load(path, map_location='cuda:0')
        self.model.load_state_dict(model_state)
        self.model.eval()

    def process_image(self, image):
        sample = transforms.Compose([transforms.CenterCrop((320, 640)), transforms.Resize((160, 320)), transforms.ToTensor()])(image)
        return torch.unsqueeze(sample, 0)

    def get_latent(self, sample):

        sample = self.process_image(sample)
        latent, _ = self.model.latent_distribution(sample)

        return latent

    def reconstruct(self, sample):
        sample = self.process_image(sample)
        return self.model.reconstruct(sample)[0], sample[0]


#   def debug():
#
#    evaluator = ROSVAE('rgb_test', 10, False, log_folder='ROS_RESULTS')
#    from PIL import Image
#    KINECT_SAMPLES = '/home/aleksi/hacks/vae_ws/real_images/images'
#    sample = Image.open(os.path.join(KINECT_SAMPLES, '001_image.png'))
#    evaluator.get_result_pair(sample, 'test')

if __name__  == '__main__':
    RosPerceptionVAE('log/rgb_test', 10)
