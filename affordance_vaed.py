import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class Decoder(nn.Module):

    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(latent_dim, 1536)

        self.cnn1 = nn.ConvTranspose2d(64, 64, kernel_size=(4,4),  stride=2)
        self.cnn2 = nn.ConvTranspose2d(64, 32, kernel_size=(4,4),  stride=2)
        self.cnn3 = nn.ConvTranspose2d(32, 32, kernel_size=(4,4),  stride=2)
        self.cnn4 = nn.ConvTranspose2d(32, 32, kernel_size=(4,4),  stride=2, output_padding=1)
        self.cnn5 = nn.ConvTranspose2d(32, output_dim, kernel_size=(4,4),  stride=2)

        self.bn1 = nn.BatchNorm1d(1536)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(32)

        self.sigmoid = nn.Sigmoid()
        self.relu = F.relu

    def forward(self, z):

        x = self.relu(self.bn1(self.fc(z)))
        x = x.view(-1, 64, 3, 8)
        x = self.relu(self.bn2(self.cnn1(x)))
        x = self.relu(self.bn3(self.cnn2(x)))
        x = self.relu(self.bn4(self.cnn3(x)))
        x = self.relu(self.bn5(self.cnn4(x)))
        x = self.sigmoid(self.cnn5(x))
        return x


class Encoder(nn.Module):

    def __init__(self, latent_dim, input_depth):
        super(Encoder, self).__init__()

        self.cnn1 = nn.Conv2d(input_depth, 32, kernel_size=(4,4), stride=2)
        self.cnn2 = nn.Conv2d(32, 32, kernel_size=(4,4),  stride=2)
        self.cnn3 = nn.Conv2d(32, 32, kernel_size=(4,4),  stride=2)
        self.cnn4 = nn.Conv2d(32, 64, kernel_size=(4,4),  stride=2)
        self.cnn5 = nn.Conv2d(64, 64, kernel_size=(4,4),  stride=2)
        self.fc21 = nn.Linear(1536, latent_dim)
        self.fc22 = nn.Linear(1536, latent_dim)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)

        self.relu = F.relu

    def forward(self, x):

        x = self.relu(self.bn1(self.cnn1(x)))
        x = self.relu(self.bn2(self.cnn2(x)))
        x = self.relu(self.bn3(self.cnn3(x)))
        x = self.relu(self.bn4(self.cnn4(x)))
        x = self.relu(self.bn5(self.cnn5(x)))

        x = x.view(-1, 1536)

        z_loc = self.fc21(x)
        z_scale = self.fc22(x)

        return z_loc, z_scale


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
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
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


class AffordanceVAED(VAE):

    def __init__(self, encoder, decoder, device, beta=1):
        super(AffordanceVAED, self).__init__(encoder, decoder, device, beta)

    def evaluate(self, state):

        x = Variable(state[0][0].to(self.device))
        affordances = state[0][1].to(self.device)
        train = state[1]

        # chooses whether it is in training mode or eval mode.
        affordance_recons,  mu, log_var = self._forward(x, train)

        BCE = F.binary_cross_entropy(affordance_recons, affordances, size_average=False)

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return BCE + self.beta * KLD, (BCE, KLD, x, affordances, affordance_recons)
