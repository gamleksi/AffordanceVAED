import os
import torch
from tools import model_name_search
from affordance_vaed import Decoder, Encoder, AffordanceVAED
from torchvision import transforms


class RosPerceptionVAE(object):

    def __init__(self, model_dir, latent_dim):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if device.type == 'cuda':
            print('GPU works for Perception!')
        else:
            print('Perception is not using GPU')

        encoder = Encoder(latent_dim, 3)
        decoder = Decoder(latent_dim, 2)
        self.model = AffordanceVAED(encoder, decoder, device).to(device)
        self.load_parameters(model_dir)

    def load_parameters(self, model_dir):

        model_name = model_name_search(model_dir)
        path = os.path.join(model_dir, '{}.pth.tar'.format(model_name))
        self.model.load_state_dict(torch.load(path))
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


