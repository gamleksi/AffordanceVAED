from tools import affordance_to_array, affordance_layers_to_array
import numpy as np
import torch
from torchnet.logger import VisdomLogger

class AffordanceVisualizer(object):

    def __init__(self, model, loader, logger, latent_dim):


        self.model = model
        self.dataloader = loader
        self.logger = logger

        self.latent_dim = latent_dim

    def get_latent_dim(self):
        return self.latent_dim

    def _get_results(self, samples):

        assert(len(samples.shape) == 4)
        recons = self.model.reconstruct(samples)
        return recons

    def _decode_latent_neighbors(self, mu, num_samples, step_size):

        zdim = mu.shape[0]
        if num_samples % 2 == 0:
            num_samples += 1

        coefs = torch.linspace(-1, 1, num_samples).to(self.model.device) * step_size

        latent_samples = torch.zeros(zdim, num_samples, zdim).to(self.model.device)
        latent_samples[:, :] = mu

        for i in range(zdim):

            latent_samples[i, :, i] = latent_samples[i, :, i] + coefs

        sub_titles = []

        for i in range(zdim):
            for j in coefs:
                sub_titles.append('Variable {}, change : {}'.format(i +1 , j))

        latent_samples = latent_samples.view(num_samples * zdim, zdim)

        return self.model.decoder(latent_samples), sub_titles

    def _decode_variable_neighbors(self, mu, num_samples, step_size, modified_latent_idx):

        zdim = mu.shape[0]

        coefs = torch.linspace(-1, 1, num_samples).to(self.model.device) * step_size

        latent_samples = torch.zeros(num_samples, zdim).to(self.model.device)
        latent_samples[:] = mu

        sub_titles = []

        for i, coef in enumerate(coefs):

            latent_samples[i, modified_latent_idx] += coef
            sub_titles.append('Change: {}'.format(coef))

        return self.model.decoder(latent_samples), sub_titles


    def _latent_transformation(self, sample1, sample2, num_samples):

        mu1, _ = self.model.latent_distribution(sample1)
        mu2, _ = self.model.latent_distribution(sample2)
        latent_dim = mu1.shape[0]

        assert(num_samples >= 3)

        unit_vector = (mu2 - mu1) / (num_samples + 1)

        latent_samples = torch.ones(num_samples + 2, latent_dim).to(self.model.device) * mu1[0]
        sub_titles = []
        for i in range(0, num_samples + 2):
            latent_samples[i] += i  * unit_vector[0]

            if i == 0:
                sub_titles.append('Decoded 1')
            elif i == num_samples + 1:
                sub_titles.append('Decoded 2')
            else:
                sub_titles.append('Latent Step: {}'.format(i))

        decoded_affordances = self.model.decoder(latent_samples)
        return decoded_affordances, sub_titles

    def _latent_dimensional_transformation(self, sample1, sample2, num_samples, latent_idx):

        mu1, _ = self.model.latent_distribution(sample1)
        mu2, _ = self.model.latent_distribution(sample2)
        latent_dim = mu1.shape[0]

        assert(num_samples >= 3)

        unit_change = (mu2[0, latent_idx] - mu1[0, latent_idx]) / (num_samples + 1)

        latent_samples = torch.ones(num_samples + 2, latent_dim).to(self.model.device) * mu1[0]
        sub_titles = []

        for i in range(0, num_samples + 2):
            latent_samples[i, latent_dim - 1] += i  * unit_change

            if i == 0:
                sub_titles.append('Decoded 1')
            elif i == num_samples + 1:
                sub_titles.append('Decoded 2')
                latent_samples[i] = mu2[0]
            else:
                sub_titles.append('Step: {}'.format(i))
                latent_samples[i, latent_idx] += i  * unit_change

        decoded_affordances = self.model.decoder(latent_samples)

        return decoded_affordances, sub_titles

    def latent_distribution_of_sample(self, sample_idx, file_name, step_size=10, num_samples=10):

        if file_name is None:
            file_name = 'latent distribution of {} with step size {}'.format(sample_idx + 1, step_size)

        sample, _ = self.dataloader.get(sample_idx)
        mu, _ = self.model.latent_distribution(sample)

        affordances, sub_titles = self._decode_latent_neighbors(mu[0], num_samples, step_size)

        affordances = np.array([affordance_to_array(affordances[idx]) for idx in range(affordances.shape[0])])

        self.logger.plot_image_list(affordances, mu.shape[1], file_name, file_name, sub_titles)

    def latent_distribution_of_zero(self, save_file, step_size=3, num_samples = 15):

        zdim = self.get_latent_dim()

        affordances, sub_titles = self._decode_latent_neighbors(torch.zeros(zdim).to(self.model.device), num_samples, step_size)

        affordances = np.array([affordance_to_array(affordances[idx]) for idx in range(affordances.shape[0])])

        self.logger.plot_image_list(affordances, zdim, save_file, save_file, sub_titles)

    def list_of_latent_distribution_samples(self, index_list, file_names, step_size=10, num_samples=10):

        assert(len(index_list) == len(file_names))

        for i, sample_idx in enumerate(index_list):
            self.latent_distribution_of_sample(sample_idx, file_names[i], step_size=step_size, num_samples=num_samples)

    def get_result_pair(self, list_idx, file_name):

        num_samples = len(list_idx)
        samples, affordances = self.dataloader.get_samples(list_idx)
        recons = self._get_results(samples)

        images = samples[:, :3].cpu().detach().numpy() * 255

        reconstructions = np.array([affordance_to_array(recons[idx]) for idx in range(num_samples)])

        affordance_layers = np.array([affordance_layers_to_array(recons[idx]) for idx in range(num_samples)])
        affordance_layers = np.transpose(affordance_layers, (1, 0, 2, 3, 4))
        affordance_layers = [layer for layer in affordance_layers]

        if affordances is not None:
            affordances = np.array([affordance_to_array(affordances[idx]) for idx in range(num_samples)])
            samples = np.column_stack([images, affordances, reconstructions] + affordance_layers)
            samples = samples.reshape((len(affordance_layers) + 3) * num_samples, images.shape[1], images.shape[2], images.shape[3])
        else:
            samples = np.column_stack([images, reconstructions] + affordance_layers)
            samples = samples.reshape((len(affordance_layers) + 2) * num_samples, images.shape[1], images.shape[2], images.shape[3])

        self.logger.plot_image_list(samples, num_samples, file_name, file_name)

    def get_result(self, list_idx, file_name):

        num_samples = len(list_idx)
        samples, affordances = self.dataloader.get_samples(list_idx)
        recons = self._get_results(samples)

        images = samples[:, :3].cpu().detach().numpy() * 255

        reconstructions = np.array([affordance_to_array(recons[idx]) for idx in range(num_samples)])

        samples = np.concatenate([images, reconstructions])
        #samples = samples.reshape((len(affordance_layers) + 2) * num_samples, images.shape[1], images.shape[2], images.shape[3])

        self.logger.plot_image_list(samples, num_samples, file_name, file_name)

    def dimensional_neighbors_of_zero_area(self, latent_dim, file_name, step_size=5, num_samples = 100):
        assert(latent_dim > 0)

        latent_idx = latent_dim - 1
        zdim = self.get_latent_dim()
        affordances, sub_titles = self._decode_variable_neighbors(torch.zeros(zdim).to(self.model.device), num_samples, step_size, latent_idx)

        affordances = np.array([affordance_to_array(affordances[idx]) for idx in range(affordances.shape[0])])

        self.logger.plot_image_list(affordances, zdim, file_name, sub_titles)

    def transform_of_samples(self, sample_id1, sample_id2, file_name):

        num_samples = 100
        assert(sample_id1 > 0 and sample_id2 > 0)

        sample_idx1 = sample_id1 - 1
        sample_idx2 = sample_id2 - 1
        sample1, _ = self.dataloader.get(sample_idx1)
        sample2, _ = self.dataloader.get(sample_idx2)
        affordances, sub_titles = self._latent_transformation(sample1, sample2, num_samples)

        affordances = np.array([affordance_to_array(affordances[idx]) for idx in range(affordances.shape[0])])
        sample1 = sample1[:, :3].cpu().detach().numpy() * 255.
        sample2 = sample2[:, :3].cpu().detach().numpy() * 255.

        images = np.concatenate((sample1, affordances, sample2), 0)

        sub_titles = ['sample 1'] + sub_titles + ['sample 2']
        self.logger.plot_image_list(images, 8, file_name, sub_titles)

    def dimensional_transform_of_samples(self, sample_id1, sample_id2, file_name, num_samples = 10):

        assert(sample_id1 > 0 and sample_id2 > 0)

        sample_idx1 = sample_id1 - 1
        sample_idx2 = sample_id2 - 1
        sample1, _ = self.dataloader.get(sample_idx1)
        sample2, _ = self.dataloader.get(sample_idx2)

        zdim = self.get_latent_dim()
        images = np.zeros((zdim, num_samples + 4, 3, sample1.shape[2], sample1.shape[3]))
        sub_titles = []

        for latent_idx in range(zdim):

            affordances, dim_titles = self._latent_dimensional_transformation(sample1, sample2, num_samples, latent_idx)

            affordances = np.array([affordance_to_array(affordances[idx]) for idx in range(affordances.shape[0])])

            images[latent_idx] = np.concatenate((sample1[:, :3].cpu().detach().numpy() * 255., affordances, sample2[:, :3].cpu().detach().numpy() * 255.), 0)

            sub_titles = sub_titles + ['sample 1'] + dim_titles + ['sample 2']

        self.logger.plot_image_list(np.concatenate(images, 0), zdim, file_name, file_name, sub_titles)

from blender_dataset import BlenderEvaluationLoader
from image_logger import MatplotLogger
import os

class AffordanceDemonstrator(AffordanceVisualizer):

    def __init__(self, model, folder, model_name, latent_dim, include_depth, logger=None,
                 loader=None):
        self.model = model
        self.model_name = model_name
        self.latent_dim = latent_dim
        self.load_parameters(folder, model_name)
        self.model_name = model_name
        if loader is None:
            self.dataloader = BlenderEvaluationLoader(include_depth)
        else:
            self.dataloader = loader
        if logger is None:
            self.logger = MatplotLogger(folder, False)
        else:
            self.logger = logger

    def load_parameters(self, folder, model_name):
        path = os.path.join('log/{}'.format(folder), '{}.pth.tar'.format(model_name))
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
