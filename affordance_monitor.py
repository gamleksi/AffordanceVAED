from tools import affordance_to_array, affordance_layers_to_array
import numpy as np
import torch
from torchnet.logger import VisdomLogger
from monitor import Trainer
import pathlib

class AffordanceTrainer(Trainer):

    def __init__(self, dataloader, model, visdom_title, save_folder=None, save_name=None, log=False, visdom=True, server='localhost',
                 port=8097, env='samples'):

        super(AffordanceTrainer, self).__init__(dataloader, model, save_folder=save_folder, save_name=save_name,
                                                log=log, visdom=visdom, server=server, port=port, visdom_title=visdom_title)
        self.env = env

    def get_results(self, samples):

        assert(len(samples.shape) == 4)
        recons = self.model.reconstruct(samples)
        return recons

    def generate_visdom_samples(self, samples=None, affordances=None, title=None, env=None):

        if env is None: # TODO
            env = self.env

        if samples is None:
            samples, affordances = self.visdom_samples

        if title is None:
            title = 'Epoch: {}'.format(self.epoch_iter)

        recons = self.get_results(samples)
        num_samples = samples.shape[0]

        images = samples.cpu().detach().numpy() * 255
        images = images[:, 0:3]

        affordances = np.array([affordance_to_array(affordances[idx]) for idx in range(num_samples)])

        reconstructions = np.array([affordance_to_array(recons[idx]) for idx in range(num_samples)])

        affordance_layers = np.array([affordance_layers_to_array(recons[idx]) for idx in range(num_samples)])
        affordance_layers = np.transpose(affordance_layers, (1, 0, 2, 3, 4))
        affordance_layers = [layer for layer in affordance_layers]

        samples = np.column_stack([images, affordances, reconstructions] + affordance_layers)

        sample_logger = VisdomLogger('images', port=self.port, nrow=len(affordance_layers) + 3, env=env,
                opts={'title': title})

        samples = samples.reshape((len(affordance_layers) + 3) * num_samples, images.shape[1], images.shape[2], images.shape[3])

        sample_logger.log(samples)


    def generate_latent_samples(self, sample=None, step_size=1, num_samples=21, title=None, env=None):

        if env is None: # TODO
            env = self.env

        if title is None:
            title = 'Epoch: {}'.format(self.epoch_iter)

        if sample is None:
            sample = self.visdom_samples[0][3]
            sample.unsqueeze_(0)

        sample_logger = VisdomLogger('images', port=self.port, nrow=num_samples, env=env, opts={'title': title})

        mu, _ = self.model.latent_distribution(sample)

        affordances, sub_titles = self.decode_latent_neighbors(mu[0], num_samples, step_size)

        affordances = np.array([affordance_to_array(affordances[idx]) for idx in range(affordances.shape[0])])

        sample_logger.log(affordances)
        # self.logger.plot_image_list(affordances, mu.shape[1], 'testi', title, sub_titles)


from blender_dataset import BlenderEvaluationLoader
from image_logger import MatplotLogger


class AffordanceDemonstrator(AffordanceTrainer):

    def __init__(self, model, folder, model_name, latent_dim, include_depth, server='localhost',
                 port=8097):

        self.model = model
        self.model_name = model_name
        self.latent_dim = latent_dim
        self.load_parameters(folder, model_name)
        self.model_name = model_name
        self.dataloader = BlenderEvaluationLoader(include_depth)
        self.port = port
        self.logger = MatplotLogger(folder, False)

    def get_latent_dim(self):
        return self.latent_dim

    def load_parameters(self, folder, model_name):
        Path = pathlib.Path('./log/{}'.format(folder)).joinpath('{}.pth.tar'.format(model_name))
        self.model.load_state_dict(torch.load(Path))
        self.model.eval()

    def latent_distribution_of_sample(self, sample_idx, title=None, step_size=10, num_samples=10):

        if title is None:
            title = 'latent distribution of {} with step size {}'.format(sample_idx + 1, step_size)

        sample, _ = self.dataloader.get(sample_idx)
        self.generate_latent_samples(sample=sample, title=title, step_size=step_size, env=self.model_name, num_samples=num_samples)

    def list_of_latent_distribution_samples(self, index_list, step_size=10, num_samples=10):

        samples, _ = self.dataloader.get_samples(index_list)

        for sample_idx in index_list:
           self.latent_distribution_of_sample(sample_idx, step_size=step_size, num_samples=num_samples)

    def get_result_pair(self, list_idx, title=None):

        num_samples = len(list_idx)
        samples, affordances = self.dataloader.get_samples(list_idx)
        recons = self.get_results(samples)

        images = samples[:, :3].cpu().detach().numpy() * 255

        affordances = np.array([affordance_to_array(affordances[idx]) for idx in range(num_samples)])

        reconstructions = np.array([affordance_to_array(recons[idx]) for idx in range(num_samples)])

        affordance_layers = np.array([affordance_layers_to_array(recons[idx]) for idx in range(num_samples)])
        affordance_layers = np.transpose(affordance_layers, (1, 0, 2, 3, 4))
        affordance_layers = [layer for layer in affordance_layers]

        samples = np.column_stack([images, affordances, reconstructions] + affordance_layers)

        samples = samples.reshape((len(affordance_layers) + 3) * num_samples, images.shape[1], images.shape[2], images.shape[3])

        self.logger.plot_image_list(samples, num_samples, 'aaaaa', 'aaaaa')


    def neighbors_of_zero_latent(self, step_size=3, num_samples = 15):

        title = 'zero_latents_{}'.format(step_size)
        zdim = self.get_latent_dim()

        affordances, sub_titles = self.decode_latent_neighbors(torch.zeros(zdim), num_samples, step_size)

        affordances = np.array([affordance_to_array(affordances[idx]) for idx in range(affordances.shape[0])])

        self.logger.plot_image_list(affordances, zdim, title, title, sub_titles)


    def neighbors_of_zero_latent_variable(self, latent_id, step_size=5, num_samples = 100):
        assert(latent_id > 0)
        latent_idx = latent_id - 1
        title = 'laten_id: {}, step_size: {}'.format(latent_id , step_size)
        zdim = self.get_latent_dim()
        affordances, sub_titles = self.decode_variable_neighbors(torch.zeros(zdim), num_samples, step_size, latent_idx)

        affordances = np.array([affordance_to_array(affordances[idx]) for idx in range(affordances.shape[0])])
        self.logger.plot_image_list(affordances, zdim, title, title, sub_titles)

    def transform_of_samples(self, sample_id1, sample_id2, num_samples = 100):

        assert(sample_id1 > 0 and sample_id2 > 0)

        sample_idx1 = sample_id1 - 1
        sample_idx2 = sample_id2 - 1
        sample1, _ = self.dataloader.get(sample_idx1)
        sample2, _ = self.dataloader.get(sample_idx2)
        affordances, sub_titles = self.latent_transformation(sample1, sample2, num_samples)

        affordances = np.array([affordance_to_array(affordances[idx]) for idx in range(affordances.shape[0])])
        sample1 = sample1[:, :3].cpu().detach().numpy() * 255.
        sample2 = sample2[:, :3].cpu().detach().numpy() * 255.

        images = np.concatenate((sample1, affordances, sample2), 0)

        title = 'sample1: {}, sample2: {}'.format(sample_id1 , sample_id2)
        sub_titles = ['sample 1'] + sub_titles + ['sample 2']
        self.logger.plot_image_list(images, 8, title, title, sub_titles)

    def dimensional_transform_of_samples(self, sample_id1, sample_id2, num_samples = 10):

        assert(sample_id1 > 0 and sample_id2 > 0)

        sample_idx1 = sample_id1 - 1
        sample_idx2 = sample_id2 - 1
        sample1, _ = self.dataloader.get(sample_idx1)
        sample2, _ = self.dataloader.get(sample_idx2)

        zdim = self.get_latent_dim()
        images = np.zeros((zdim, num_samples + 4, 3, sample1.shape[2], sample1.shape[3]))
        sub_titles = []

        for latent_idx in range(zdim):

            affordances, dim_titles = self.latent_dimensional_transformation(sample1, sample2, num_samples, latent_idx)

            affordances = np.array([affordance_to_array(affordances[idx]) for idx in range(affordances.shape[0])])

            images[latent_idx] = np.concatenate((sample1[:, :3].cpu().detach().numpy() * 255., affordances, sample2[:, :3].cpu().detach().numpy() * 255.), 0)

            title = 'dimensional_transform sample1: {}, sample2: {}'.format(sample_id1 , sample_id2)
            sub_titles = sub_titles + ['sample 1'] + dim_titles + ['sample 2']

        self.logger.plot_image_list(np.concatenate(images, 0), zdim, title, title, sub_titles)

#    def neighbor_channels(self, title='zero latent development', step_size=10, num_samples=10):
#
#        sample_logger = VisdomLogger('images', port=self.port, nrow=num_samples, env=self.model_name, opts={'title': title})
#        zdim = self.get_latent_dim()
#
#        recons = self.decode_latent_neighbors(torch.zeros(zdim), num_samples, step_size)
#
#        n = zdim * num_samples
#
#        affordance_layers = np.array([affordance_layers_to_array(recons[idx]) for idx in range(n)])
#        affordance_layers = np.transpose(affordance_layers, (1, 0, 2, 3, 4))
#
#        affordance_layers = [layer for layer in affordance_layers]
#
#        built_affordances = np.array([affordance_to_array(recons[idx]) for idx in range(n)])
#
#        samples = np.column_stack(built_affordances + affordance_layers)
#
#        samples = samples.reshape(samples.shape[0] * num_samples, 3, recons.shape[2], recons.shape[3])
#        sample_logger.log(samples)
#
#
