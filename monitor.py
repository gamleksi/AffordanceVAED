import numpy as np
import torch
import sys
# sys.path.insert(0, '../../libs/')
import torchnet as tnt 
from torchnet.engine import Engine
from torchnet.logger import MeterLogger, VisdomLogger

from tqdm import tqdm
import csv
import pathlib
import os
from random import sample  
from torch.autograd import Variable


class Trainer(Engine):
    def __init__(self, dataloader, save_folder, save_name, visdom=True, log=True, server='localhost', port=8097, validation_step=1):
        super(Trainer, self).__init__()

        self.get_iterator = dataloader.get_iterator
        self.visdom_samples = dataloader.visdom_data

        self.meter_loss = tnt.meter.AverageValueMeter()
        self.epoch_iter = 0
        self.validation_step = validation_step
        self.best_loss = np.inf 
        self.log_data = log
        if self.log_data: 
            self.initilize_log(save_folder, save_name)
        self.visdom = visdom
        if self.visdom:
            self.mlog = MeterLogger(server=server, port=port, title="mnist_meterlogger")
            self.port = port
            
        self.initialize_engine()

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
        state['sample'].append(state['train'])

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
        
        self.epoch_iter += 1

        if self.epoch_iter % self.validation_step == 0:
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
                self.generate_visdom_samples(self.visdom_samples)
                self.generate_latent_samples(self.visdom_samples[2])
        

    def initilize_log(self, save_folder, save_name):

        self.log_path = pathlib.Path('./log/{}'.format(save_folder)) 

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
        
    def form_grayscale_images(self, recons, num_samples, img_shape):
        output = recons.detach().numpy()
        output = output.reshape(num_samples, img_shape[0],img_shape[1]) * 255
        output = output[:, np.newaxis]
        return output


    def generate_visdom_samples(self, samples):

        _, recons = self.model.evaluate([samples, False])

        sample_logger = VisdomLogger('images', port=self.port, nrow=2, env='samples', opts={'title': 'Epoch: {}'.format(self.epoch_iter)})

        # TODO: RBB data

        n = recons.shape[0]
        input = samples.numpy()
        input = input[:,np.newaxis]

        output = self.form_grayscale_images(recons, n, (samples.shape[-2], samples.shape[-1]))
        samples = np.column_stack((input,output)).reshape(n*2, 1, input.shape[-2],input.shape[-1])

        sample_logger.log(samples)
    
    def generate_latent_samples(self, sample):

        num_samples = 10
        img_dim = (sample.shape[1], sample.shape[1])

        sample_logger = VisdomLogger('images', port=self.port, nrow=10, env='samples', opts={'title': 'Epoch: {}'.format(self.epoch_iter)})

        mu, logvar = self.model.latent_values(sample)

        stds = torch.exp(0.5*logvar)
        zdim = stds.shape[1]

        latent_samples = torch.zeros(zdim, num_samples, zdim)
        latent_samples[:,:] = mu 

        coefs = torch.linspace(-1, 1, num_samples) * 100  

        for i in range(zdim):

            c_mu = mu[0][i] 
            c_stds = stds[0][i]
            z_samples = c_stds.mul(coefs).add(c_mu)
            latent_samples[i, :, i] = z_samples

        latent_samples = latent_samples.view(num_samples * zdim, zdim)
        images = self.model.decoder(latent_samples)
        images = self.form_grayscale_images(images, num_samples * zdim, img_dim)

        sample_logger.log(images)

class Demonstrator(Engine):

    def __init__(self, model, folder, model_name):
        super(Demonstrator, self).__init__()
        self.model = model
        self.load_parameters(folder, model_name)
        self.initialize_engine()

    def initialize_engine(self):
        self.hooks['on_forward'] = self.on_forward
    
    def load_parameters(self, folder, model_name):
        Path = pathlib.Path('./log/{}'.format(folder)).joinpath('{}.pth.tar'.format(model_name)) 
        self.model.load_state_dict(torch.load(Path))
        self.model.eval()

    def on_forward(self, state):
        self.meter_loss.add(state['loss'].item())

    def on_sample(self, state):
        state['sample'].append(state['train'])
    
    def evaluate(self, get_iterator):
        self.meter_loss = tnt.meter.AverageValueMeter()
        self.test(self.model.evaluate, get_iterator(False))
        val_loss = self.meter_loss.value()[0]
        print('Testing loss: %.4f' % (val_loss))