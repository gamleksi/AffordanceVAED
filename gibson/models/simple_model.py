import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, img_width, img_height, img_channels, z_dim, hidden_dim):
        super(Decoder, self).__init__()

        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels

        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, self.img_width * self.img_height)

        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):

        hidden = self.softplus(self.fc1(z))
        loc_img = self.sigmoid(self.fc21(hidden))

        return loc_img.view(-1, self.img_channels, self.img_width, self.img_width)

class Encoder(nn.Module):
    def __init__(self, img_width, img_height, z_dim, hidden_dim):
        super(Encoder, self).__init__()
        # setup the three linear transformations used
        self.img_width = img_width
        self.img_height = img_height

        self.fc1 = nn.Linear(self.img_width * self.img_height, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.view(-1, self.img_width * self.img_height)
        hidden = self.softplus(self.fc1(x))
        z_loc = self.fc21(hidden)
        z_scale = self.fc22(hidden)
        return z_loc, z_scale

from torch.autograd import Variable
from torch.nn import functional as F

class VAE(nn.Module):

    def __init__(self, encoder, decoder, device, beta=1):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

    def set_mode(self, train):
        if train:
            self.train()
        else:
            self.eval()

    def _forward(self, x, train):
        self.set_mode(train)
        mu, logvar  = self.encoder(x)
        z = self._reparameterize(mu, logvar, train)
        return self.decoder(z), mu, logvar

    def _reparameterize(self, mu, logvar, train):
        if train:
            std = torch.exp(0.5*logvar, device=self.device)
            eps = torch.randn_like(std, device=self.device)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def evaluate(self, state):
        # state includes batch samples and a train / test flag
        # samples should be tensors and processed in loader function.

        x = Variable(state[0].to(self.device))
        train = state[1]
        x_recon,  mu, log_var = self._forward(x, train)

        BCE = F.binary_cross_entropy(x_recon, x, size_average=False)

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + self.beta * KLD, x_recon

    def latent_distribution(self, sample):

        #  self.set_mode(False)
        x = Variable(sample.to(self.device))
        mu, logvar = self.encoder(x)

        return mu, logvar

    def reconstruct(self, sample):
        x = Variable(sample).to(self.device)
        recon,  _, _ = self._forward(x, False)

        return recon


class AffordanceVAE(VAE):

    def __init__(self, encoder, decoder, device, beta=1):
        super(AffordanceVAE, self).__init__(encoder, decoder, device, beta)

    def evaluate(self, state):

        x = Variable(state[0][0].to(self.device))
        affordances = Variable(state[0][1].to(self.device))
        train = state[1]


        # chooses whether it is in training mode or eval mode.

        affordance_recons,  mu, log_var = self._forward(x, train)

        BCE = F.binary_cross_entropy(affordance_recons, affordances, size_average=False)

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return BCE + self.beta * KLD, affordance_recons


class AffordanceCapacityVAE(VAE):

    def __init__(self, encoder, decoder, device, gamma=300, capacity_limit=100, capacity_change_duration=1000):
        super(AffordanceCapacityVAE, self).__init__(encoder, decoder, device)

        self.gamma = gamma
        self.capacity_limit = capacity_limit
        self.capacity_change_duration = capacity_change_duration
        self.step = 0

    def _encoding_capacity(self):
        if self.step > self.capacity_change_duration:
            return self.capacity_limit
        else:
            return self.capacity_limit * (self.step / self.capacity_change_duration)

    def evaluate(self, state):

        x = Variable(state[0][0].to(self.device))
        affordances = Variable(state[0][1].to(self.device))
        train = state[1]

        # chooses whether it is in training mode or eval mode.
        affordance_recons,  mu, log_var = self._forward(x, train)

        BCE = F.binary_cross_entropy(affordance_recons, affordances, size_average=False)

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        C = self._encoding_capacity()
        if train:
            self.step += 1

        return BCE + self.gamma * torch.abs(KLD - C), affordance_recons
