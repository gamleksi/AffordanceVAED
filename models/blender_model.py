import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, running_stats=True, momentum=0.1):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(latent_dim, 2304)

        self.cnn1 = nn.ConvTranspose2d(64, 64, kernel_size=(3,3), stride=2)
        self.cnn2 = nn.ConvTranspose2d(64, 32, kernel_size=(3,3),  stride=2)
        self.cnn3 = nn.ConvTranspose2d(32, 32, kernel_size=(3,3),  stride=2) # output_padding to fix encoder decoder problem 31 != 30
        self.cnn4 = nn.ConvTranspose2d(32, 32, kernel_size=(3,3),  stride=2, output_padding=1) # output_padding to fix encoder decoder problem 31 != 30
        self.cnn5 = nn.ConvTranspose2d(32, output_dim, kernel_size=(3,3), stride=2)

        self.bn1 = nn.BatchNorm1d(2304, track_running_stats=running_stats, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(64, track_running_stats=running_stats, momentum=momentum)
        self.bn3 = nn.BatchNorm2d(32, track_running_stats=running_stats, momentum=momentum)
        self.bn4 = nn.BatchNorm2d(32, track_running_stats=running_stats, momentum=momentum)
        self.bn5 = nn.BatchNorm2d(32, track_running_stats=running_stats, momentum=momentum)

        self.sigmoid = nn.Sigmoid()
        self.relu = F.relu

    def forward(self, z):

        x = self.relu(self.bn1(self.fc(z)))
        x = x.view(-1, 64, 4, 9)
        x = self.relu(self.bn2(self.cnn1(x)))
        x = self.relu(self.bn3(self.cnn2(x)))
        x = self.relu(self.bn4(self.cnn3(x)))
        x = self.relu(self.bn5(self.cnn4(x)))
        x = self.sigmoid(self.cnn5(x))
        return x[:, :, :160, :320]

class Encoder(nn.Module):
    def __init__(self, latent_dim, input_depth, running_stats=True, momentum=0.1):
        super(Encoder, self).__init__()

        self.cnn1 = nn.Conv2d(input_depth, 32, kernel_size=(3,3),stride=2)
        self.cnn2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)
        self.cnn3 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2)
        self.cnn4 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=2)
        self.cnn5 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=2)
        self.fc21 = nn.Linear(2304, latent_dim)
        self.fc22 = nn.Linear(2304, latent_dim)

        self.bn1 = nn.BatchNorm2d(32, track_running_stats=running_stats, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(32, track_running_stats=running_stats, momentum=momentum)
        self.bn3 = nn.BatchNorm2d(32, track_running_stats=running_stats, momentum=momentum)
        self.bn4 = nn.BatchNorm2d(64, track_running_stats=running_stats, momentum=momentum)
        self.bn5 = nn.BatchNorm2d(64, track_running_stats=running_stats, momentum=momentum)

        self.relu = F.relu

    def forward(self, x):

        x = self.relu(self.bn1(self.cnn1(x)))
        x = self.relu(self.bn2(self.cnn2(x)))
        x = self.relu(self.bn3(self.cnn3(x)))
        x = self.relu(self.bn4(self.cnn4(x)))
        x = self.relu(self.bn5(self.cnn5(x)))
        x = x.view(-1, 2304)

        z_loc = self.fc21(x)
        z_scale = self.fc22(x)

        return z_loc, z_scale

if __name__ == '__main__':

    import torch

    device = torch.device('cuda')

    encoder = Encoder(10, 3)
    encoder.to(device)
    decoder = Decoder(10, 2)
    decoder.to(device)

    sample = torch.ones([4, 3, 160, 320]).to(device)

    mu, _ = encoder(sample)
    result = decoder(mu)
    print(sample.shape)
    print(result.shape)



