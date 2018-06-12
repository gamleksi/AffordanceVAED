import numpy as np
import torch

import torchnet as tnt
from torchnet.engine import Engine
from torchnet.logger import MeterLogger, VisdomLogger

from tqdm import tqdm
import csv
import pathlib
import os


class Trainer(Engine):
    def __init__(self, dataloader, save_folder=None, save_name=None, log=False, visdom=True, server='localhost', port=8097, visdom_title="mnist_meterlogger"):
        super(Trainer, self).__init__()

        self.get_iterator = dataloader.get_iterator
        self.meter_loss = tnt.meter.AverageValueMeter()
        self.initialize_engine()

        self.log_data = log

        if self.log_data:
            assert(save_folder is not None and save_name is not None)
            self.initilize_log(save_folder, save_name)
            self.best_loss = np.inf
        else:
            assert(save_folder is None and save_name is None)

        self.visdom = visdom
        if self.visdom:
            self.mlog = MeterLogger(server=server, port=port, title=visdom_title)
            self.port = port
            self.visdom_samples = dataloader.visdom_data
            self.epoch_iter = 0
            self.sample_images_step = 4
            
    def initialize_engine(self):
        self.hooks['on_sample'] = self.on_sample
        self.hooks['on_forward'] = self.on_forward
        self.hooks['on_start_epoch'] = self.on_start_epoch
        self.hooks['on_end_epoch'] = self.on_end_epoch


    def train(self, model, num_epoch, optimizer):
        self.model = model
        super(Trainer, self).train(self.model.evaluate, self.get_iterator(True), maxepoch=num_epoch, optimizer=optimizer)
    
    def reset_meters(self):
        self.meter_loss.reset()

    def on_sample(self, state):
        state['sample'] = (state['sample'], state['train'])

    def on_forward(self, state):
        loss = state['loss']
        self.meter_loss.add(loss.item())
        if self.visdom:
            self.mlog.update_loss(loss, meter='loss')

    def on_start_epoch(self, state):
        self.reset_meters()
        if self.visdom:
            self.mlog.timer.reset()

        state['iterator'] = tqdm(state['iterator'])
    
    def on_end_epoch(self, state):
        
        train_loss = self.meter_loss.value()[0]
        self.reset_meters()
        if self.visdom:
            self.mlog.print_meter(mode="Train", iepoch=state['epoch'])
            self.mlog.reset_meter(mode="Train", iepoch=state['epoch'])
        

            self.test(self.model.evaluate, self.get_iterator(False))
            val_loss = self.meter_loss.value()[0]

            if self.log_data:

                self.log_csv(train_loss, val_loss, val_loss < self.best_loss)

                if val_loss < self.best_loss:
                    self.save_model()
                    self.best_loss = val_loss
        if self.visdom:
            self.mlog.print_meter(mode="Test", iepoch=state['epoch'])
            self.mlog.reset_meter(mode="Test", iepoch=state['epoch'])

            if self.epoch_iter % self.sample_images_step == 0:

                self.generate_visdom_samples()
                self.generate_latent_samples()

            self.epoch_iter += 1


    def initilize_log(self, save_folder, save_name):

        self.log_path = pathlib.Path('./log/{}'.format(save_folder))
        # remove a suggested folder or name another
        assert(not(self.log_path.exists())) 
        self.log_path.mkdir(parents=True)
        self.csv_path = self.log_path.joinpath('log_{}.csv'.format(save_name))
        self.model_path = self.log_path.joinpath('{}.pth.tar'.format(save_name))

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def log_csv(self, train_loss, val_loss, improved):
        
        fieldnames = ['train_loss', 'val_loss', 'improved']
        fields=[train_loss, val_loss, int(improved)]

        file_exists = os.path.isfile(self.csv_path)

        with open(self.csv_path, 'a') as f:
            writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
            if not(file_exists):
                writer.writeheader()
            row = {}
            for i, name in enumerate(fieldnames):
                row[name] = fields[i] 

            writer.writerow(row)
        
    def generate_visdom_samples(self, samples):
        samples = self.visdom_samples

        # builds a new visdom block for every image 
        sample_logger = VisdomLogger('images', port=self.port, nrow=2, env='samples', opts={'title': 'Epoch: {}'.format(self.epoch_iter)})

        state = (samples, False)
        _, recons = self.model.evaluate(state)

        n = recons.shape[0]

        input = samples.cpu().detach().numpy() * 255
        output = recons.cpu().detach().numpy() * 255

        samples = np.column_stack((input, output)).reshape(n * 2, samples.shape[1], samples.shape[2], samples.shape[3])

        sample_logger.log(samples)
    
    def generate_latent_samples(self, sample):

        samples = self.visdom_samples[0]

        num_samples = 20
        sample_logger = VisdomLogger('images', port=self.port, nrow=num_samples, env='samples', opts={'title': 'Epoch: {}'.format(self.epoch_iter)})

        mu, logvar = self.model.latent_distribution(sample)

        stds = torch.exp(0.5*logvar)
        zdim = stds.shape[1]

        latent_samples = torch.zeros(zdim, num_samples, zdim).to(self.model.device)
        latent_samples[:, :] = mu

        coefs = torch.linspace(-1, 1, num_samples).to(self.model.device) * 10

        for i in range(zdim):

            c_mu = mu[0][i] 
            c_stds = stds[0][i]
            z_samples = c_stds.mul(coefs).add(c_mu)
            latent_samples[i, :, i] = z_samples

        latent_samples = latent_samples.view(num_samples * zdim, zdim)
        images = self.model.decoder(latent_samples)

        sample_logger.log(images.cpu().detach().numpy())

from tools import affordance_to_array

class AffordanceTrainer(Trainer):

    def __init__(self, dataloader, save_folder=None, save_name=None, log=False, visdom=True, server='localhost',
                 port=8097, visdom_title="Affordance_logger"):

        super(AffordanceTrainer, self).__init__(dataloader, save_folder=save_folder, save_name=save_name,
                                                log=log, visdom=visdom, server=server, port=port, visdom_title=visdom_title)

    def generate_visdom_samples(self):

        samples = self.visdom_samples

        # builds a new visdom block for every image
        sample_logger = VisdomLogger('images', port=self.port, nrow=3, env='samples', opts={'title': 'Epoch: {}'.format(self.epoch_iter)})

        state = (samples, False)
        _, recons = self.model.evaluate(state)

        n = recons.shape[0]

        images = samples[0].cpu().detach().numpy() * 255
        images = images[:, 0:3] # np.transpose(, (0, 2, 3, 1))

        affordances = np.array([affordance_to_array(samples[1][idx]) for idx in range(n)])
        built_affordances = np.array([affordance_to_array(recons[idx]) for idx in range(n)])

        samples = np.column_stack((images, affordances, built_affordances))
        samples = samples.reshape(n * 3, images.shape[1], images.shape[2], images.shape[3])

        sample_logger.log(samples)

    def generate_latent_samples(self):

        sample = self.visdom_samples[0][0]

        num_samples = 20
        sample_logger = VisdomLogger('images', port=self.port, nrow=num_samples, env='samples', opts={'title': 'Epoch: {}'.format(self.epoch_iter)})

        mu, logvar = self.model.latent_distribution(sample)

        stds = torch.exp(0.5*logvar)
        zdim = stds.shape[1]

        latent_samples = torch.zeros(zdim, num_samples, zdim).to(self.model.device)
        latent_samples[:, :] = mu

        coefs = torch.linspace(-1, 1, num_samples).to(self.model.device) * 10

        for i in range(zdim):

            c_mu = mu[0][i]
            c_stds = stds[0][i]
            z_samples = c_stds.mul(coefs).add(c_mu)
            latent_samples[i, :, i] = z_samples

        latent_samples = latent_samples.view(num_samples * zdim, zdim)
        affordances = self.model.decoder(latent_samples)
        affordances = np.array([affordance_to_array(affordances[idx]) for idx in range(num_samples * zdim)])

        sample_logger.log(affordances)

class Demonstrator(Trainer):

    def __init__(self,  folder, model_name, model, data_loader, visdom_title='training_results'):
        super(Demonstrator, self).__init__(data_loader, visdom_title=visdom_title, visdom=True)
        self.model = model
        self.load_parameters(folder, model_name)

    def initialize_engine(self):
        self.hooks['on_sample'] = self.on_sample
        self.hooks['on_forward'] = self.on_forward

    def load_parameters(self, folder, model_name):
        Path = pathlib.Path('./log/{}'.format(folder)).joinpath('{}.pth.tar'.format(model_name)) 
        self.model.load_state_dict(torch.load(Path))
        self.model.eval()

    def evaluate(self):
        self.test(self.model.evaluate, self.get_iterator(False))
        val_loss = self.meter_loss.value()[0]

        print('Testing loss: %.4f' % (val_loss))

        self.generate_visdom_samples(self.visdom_samples)
        self.generate_latent_samples(self.visdom_samples[0])
