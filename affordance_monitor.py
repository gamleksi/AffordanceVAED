from tools import affordance_to_array, affordance_layers_to_array
import numpy as np
import torch
from torchnet.logger import VisdomLogger
from monitor import Trainer
import pathlib

class AffordanceTrainer(Trainer):

    def __init__(self, dataloader, model, save_folder=None, save_name=None, log=False, visdom=True, server='localhost',
                 port=8097, visdom_title="Affordance_logger"):

        super(AffordanceTrainer, self).__init__(dataloader, model, save_folder=save_folder, save_name=save_name,
                                                log=log, visdom=visdom, server=server, port=port, visdom_title=visdom_title)

    def get_results(self, samples):

        assert(len(samples.shape) == 4)
        recons = self.model.evaluate(samples)
        return recons

    def generate_visdom_samples(self, samples=None, affordances=None, title=None, env='samples'):

        if samples == None:
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

        samples = samples.reshape((len(affordance_layers) + 3) * n, images.shape[1], images.shape[2], images.shape[3])

        sample_logger.log(samples)


    def generate_latent_samples(self, sample=None, step_size=1, num_samples=21, title=None, env='samples'):

        if title is None:
            title = 'Epoch: {}'.format(self.epoch_iter)

        if sample is None:
            sample = self.visdom_samples[0][3]

        sample_logger = VisdomLogger('images', port=self.port, nrow=num_samples, env=env, opts={'title': title})

        mu, _ = self.model.latent_distribution(sample)

        affordances = self.decode_latent_neighbors(mu[0], num_samples, step_size)

        affordances = np.array([affordance_to_array(affordances[idx]) for idx in range(affordances.shape[0])])

        sample_logger.log(affordances)


from blender_dataset import BlenderEvaluationLoader

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

    def get_latent_dim(self):
        return self.latent_dim

    def load_parameters(self, folder, model_name):
        Path = pathlib.Path('./log/{}'.format(folder)).joinpath('{}.pth.tar'.format(model_name))
        self.model.load_state_dict(torch.load(Path))
        self.model.eval()

    def latent_distribution_of_sample(self, sample_idx, title=None, step_size=10):

        if title is None:
            title = 'latent distribution of {} with step size {}'.format(sample_idx + 1, step_size)

        sample, _ = self.dataloader.get(sample_idx)
        self.generate_latent_samples(sample=sample, title=title, step_size=step_size, env=self.model_name)

    def list_of_latent_distribution_samples(self, index_list, step_size=10):

        samples, _ = self.dataloader.get_samples(index_list)

        for sample_idx in index_list:
           self.latent_distribution_of_sample(sample_idx, step_size=step_size)

    def get_result_pair(self, idx, title=None):

        if title is None:
            title = 'Result of sample {}'.format(idx + 1)

        sample, affordance = self.dataloader.get(idx)
        self.generate_visdom_samples(samples=sample, affordances=affordance, title=title, env=self.model_name)


    def neighbors_of_zero_latent(self, step_size=10, num_samples = 20):

        title = 'zero latent space. step size {}'.format(step_size)
        sample_logger = VisdomLogger('images', port=self.port, nrow=num_samples, env=self.model_name, opts={'title': title})
        zdim = self.get_latent_dim()

        affordances = self.decode_latent_neighbors(torch.zeros(zdim), num_samples, step_size)

        affordances = np.array([affordance_to_array(affordances[idx]) for idx in range(affordances.shape[0])])

        sample_logger.log(affordances)


    def neighbor_channels(self, title='zero latent development', step_size=10, num_samples=10):

        sample_logger = VisdomLogger('images', port=self.port, nrow=num_samples, env=self.model_name, opts={'title': title})
        zdim = self.get_latent_dim()

        recons = self.decode_latent_neighbors(torch.zeros(zdim), num_samples, step_size)

        n = zdim * num_samples

        affordance_layers = np.array([affordance_layers_to_array(recons[idx]) for idx in range(n)])
        affordance_layers = np.transpose(affordance_layers, (1, 0, 2, 3, 4))

        affordance_layers = [layer for layer in affordance_layers]

        built_affordances = np.array([affordance_to_array(recons[idx]) for idx in range(n)])

        samples = np.column_stack(built_affordances + affordance_layers)

        samples = samples.reshape(samples.shape[0] * 8, 3, recons.shape[2], recons.shape[3])
        sample_logger.log(samples)



#    def (self, env, title='zero latent development', step_size=10, num_samples = 20):
#
#        sample_logger = VisdomLogger('images', port=self.port, nrow=7, env=env, opts={'title': title})
#
#        # dirty hack: TODO
#        mu, _ = self.model.latent_distribution(self.visdom_samples[0][0])
#        zdim = mu.shape[1]
#        n = zdim * num_samples
#
#        affordances = self.latent_vectors(torch.zeros(zdim), num_samples, step_size)
#        affordances = np.array([affordance_to_array(affordances[idx]) for idx in range(affordances.shape[0])])
#
#        sample_logger.log(affordances)