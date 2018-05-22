import torch
import torch.nn as nn
import torch.distributions as dists

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Decoder, self).__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 784)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        loc_img = self.sigmoid(self.fc21(hidden)) 
        return loc_img

class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim):
        super(Encoder, self).__init__()
        # setup the three linear transformations used
        self.input_dim = input_dim 
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.view(x.shape[0], self.input_dim)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        z_loc = self.fc21(hidden)
        z_scale = self.fc22(hidden)
        return z_loc, z_scale

from torch.autograd import Variable
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, encoder, decoder, num_samples=1):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder 
        self.num_samples = num_samples
    
    def forward(self, x, train):
        mu, logvar  = self.encoder(x)
        z = self.reparameterize(mu, logvar, train) 
        return self.decoder(z), mu, logvar
    
    def reparameterize(self, mu, logvar, train):
        if train:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def evaluate(self, sample):

        x = Variable(sample[0].float() / 255)
        train = sample[1]
        x_recon,  mu, log_var = self.forward(x, train)

        BCE = F.binary_cross_entropy(x_recon, x.view(-1, self.encoder.input_dim), size_average=False) # )
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD, x_recon 