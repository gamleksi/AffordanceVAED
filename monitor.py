import csv
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import torch
import torchnet as tnt
from torchnet.engine import Engine
from tools import affordance_to_array, affordance_layers_to_array


class Saver(object):

    def __init__(self, save_path):
        self.save_path = save_path

        self.train_losses = []
        self.val_losses = []

        self.bce_losses = []
        self.bce_vals = []

        self.kld_losses = []
        self.kld_vals = []

    def log_csv(self, train_loss, val_loss, bce_loss, bce_val, kld_train, kld_val, improved):

        fieldnames = ['train_loss', 'val_loss', 'bce_loss', 'bce_val', 'kld_train', 'kld_val', 'improved']
        fields = [train_loss, val_loss, bce_loss, bce_val, kld_train, kld_val, int(improved)]
        csv_path = os.path.join(self.save_path, 'log.csv')
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, 'a') as f:
            writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
            if not(file_exists):
                writer.writeheader()
            row = {}
            for i, name in enumerate(fieldnames):
                row[name] = fields[i]

            writer.writerow(row)

    def save_model(self, model, epoch):

        model_path = os.path.join(self.save_path, 'model_epoch_{}.pth.tar'.format(epoch))
        torch.save(model.state_dict(), model_path)

    def update_losses(self, train_loss, val_loss):
        self.train_losses.append(np.log(train_loss))
        self.val_losses.append(np.log(val_loss))
        steps = range(1, len(self.train_losses) + 1)
        plt.figure()
        plt.plot(steps, self.train_losses, 'r', label='Train')
        plt.plot(steps, self.val_losses, 'b', label='Validation')
        plt.title('Average Loss (in log scale)')
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'log_loss.png'))
        plt.close()

    def update_bces(self, bce_train, bce_val):

        self.bce_losses.append(np.log(bce_train))
        self.bce_vals.append(np.log(bce_val))

        steps = range(1, len(self.bce_losses) + 1)
        plt.figure()
        plt.plot(steps, self.bce_losses, 'r', label='Train')
        plt.plot(steps, self.bce_vals, 'b', label='Validation')

        plt.title('KLD (in log scale)')
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'log_klds.png'))
        plt.close()


    def update_klds(self, kld_train, kld_val):

        self.kld_losses.append(np.log(kld_train))
        self.kld_vals.append(np.log(kld_val))

        steps = range(1, len(self.kld_losses) + 1)
        plt.figure()
        plt.plot(steps, self.kld_losses, 'r', label='Train')
        plt.plot(steps, self.kld_vals, 'b', label='Validation')

        plt.title('KLD (in log scale)')
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'log_klds.png'))
        plt.close()

    def get_result_pair(self, samples, affordances, recons, epoch):

        num_samples = samples.shape[0]

        images = samples[:, :3].cpu().detach().numpy()
        reconstructions = np.array([affordance_to_array(recons[idx]) for idx in range(num_samples)])

        affordance_layers = np.array([affordance_layers_to_array(recons[idx]) for idx in range(num_samples)])
        affordance_layers = np.transpose(affordance_layers, (1, 0, 2, 3, 4))
        affordance_layers = [layer for layer in affordance_layers]

        if affordances is not None:
            affordances = np.array([affordance_to_array(affordances[idx]) for idx in range(num_samples)])
            samples = np.column_stack([images, affordances, reconstructions] + affordance_layers)
            num_cols = len(affordance_layers) + 3
            samples = samples.reshape(num_cols * num_samples, images.shape[1], images.shape[2], images.shape[3])
        else:
            samples = np.column_stack([images, reconstructions] + affordance_layers)
            num_cols = len(affordance_layers) + 3
            samples = samples.reshape(num_cols * num_samples, images.shape[1], images.shape[2], images.shape[3])

        sample_path = os.path.join(self.save_path, '{}_epoch'.format(epoch))
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)

        num_rows = 6
        num_images = int(num_samples / num_rows)
        iter = 0

        for b in range(num_images):

            fig, axeslist = plt.subplots(ncols=num_cols, nrows=num_rows)

            for i in range(num_rows):

                for j in range(num_cols):
                    img = samples[b * num_rows * num_cols + i * num_cols + j].transpose(1, 2, 0)
                    axeslist[i][j].imshow(img)
                    axeslist[i][j].set_axis_off()

            plt.savefig(os.path.join(sample_path, 'sample_{}.png'.format(iter)))
            plt.tight_layout()
            plt.close(fig)
            iter += 1


class Trainer(Engine):

    def __init__(self, dataloader, model, log_path=None):
        super(Trainer, self).__init__()

        # TODO latent_dim
        self.get_iterator = dataloader.get_iterator
        self.meter_loss = tnt.meter.AverageValueMeter()
        self.meter_loss = tnt.meter.AverageValueMeter()
        self.kld = tnt.meter.AverageValueMeter()
        self.bce = tnt.meter.AverageValueMeter()

        self.initialize_engine()

        self.model = model
        self.log_path = log_path

        if self.log_path is not None:
            self.saver = Saver(log_path)
            self.best_loss = np.inf

        self.epoch_iter = 0
        self.sample_images_step = 1

    def initialize_engine(self):
        self.hooks['on_sample'] = self.on_sample
        self.hooks['on_forward'] = self.on_forward
        self.hooks['on_start_epoch'] = self.on_start_epoch
        self.hooks['on_end_epoch'] = self.on_end_epoch

    def train(self, num_epoch, optimizer):
        super(Trainer, self).train(self.model.evaluate, self.get_iterator(True), maxepoch=num_epoch, optimizer=optimizer)

    def reset_meters(self):
        self.meter_loss.reset()

    def on_sample(self, state):
        state['sample'] = (state['sample'], state['train'])

    def on_forward(self, state):
        loss = state['loss']
        self.meter_loss.add(loss.item())
        BCE, KLD, input_samples, affordances, reconstruction_samples = state['output']
        self.meter_loss.add(loss.item())
        self.bce.add(BCE.item())
        self.kld.add(KLD.item())
        self.input_samples = input_samples
        self.affordances_samples = affordances
        self.reconstruction_samples = reconstruction_samples

    def on_start_epoch(self, state):
        self.model.train(True)
        self.reset_meters()

    def reset_meters(self):
        self.meter_loss.reset()
        self.kld.reset()
        self.bce.reset()

    def on_end_epoch(self, state):

        print('EPOCH', self.epoch_iter)

        train_loss = self.meter_loss.value()[0]
        bce_train = self.bce.value()[0]
        kld_train = self.kld.value()[0]

        self.reset_meters()
        self.test(self.model.evaluate, self.get_iterator(False))

        val_loss = self.meter_loss.value()[0]  # type: float
        bce_val = self.bce.value()[0] # type: float
        kld_val = self.kld.value()[0] # type: float

        print("Loss train: {}, val: {}".format(train_loss, val_loss))
        print("BCE: train: {}, val: {}".format(bce_train, bce_val))
        print("KLD: train: {}, val: {}".format(kld_train, kld_val))

        if self.log_path is not None:

            self.saver.log_csv(train_loss, val_loss, bce_train, bce_val, kld_train, kld_val, val_loss < self.best_loss)

            self.saver.update_losses(train_loss, val_loss)
            self.saver.update_bces(bce_train, bce_val)
            self.saver.update_klds(kld_train, kld_val)

            if val_loss < self.best_loss:

                self.saver.save_model(self.model, self.epoch_iter)
                self.best_loss = val_loss

            self.saver.get_result_pair(self.input_samples[:18], self.affordances_samples[:18], self.reconstruction_samples[:18], self.epoch_iter)

        self.epoch_iter += 1


class Demonstrator(Trainer):

    def __init__(self,  folder, model_name, model, data_loader, visdom_title='training_results'):
        super(Demonstrator, self).__init__(data_loader, visdom_title=visdom_title, visdom=True)
        self.model = model
        self.load_parameters(folder, model_name)

    def initialize_engine(self):
        self.hooks['on_sample'] = self.on_sample
        self.hooks['on_forward'] = self.on_forward

    def load_parameters(self, folder, model_name):
        Path = os.path.join('log/{}'.format(folder), '{}.pth.tar'.format(model_name))
        self.model.load_state_dict(torch.load(Path))
        self.model.eval()

    def evaluate(self):
        self.test(self.model.evaluate, self.get_iterator(False))
        val_loss = self.meter_loss.value()[0]

        print('Testing loss: %.4f' % (val_loss))

#        self.generate_visdom_samples(self.visdom_samples)
#        self.generate_latent_samples(self.visdom_samples[0])



