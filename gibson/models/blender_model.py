import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, running_stats=True, momentum=0.1):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(latent_dim, 1536)

        self.cnn1 = nn.ConvTranspose2d(64, 64, kernel_size=(4,4),  stride=2)
        self.cnn2 = nn.ConvTranspose2d(64, 32, kernel_size=(4,4),  stride=2)
        self.cnn3 = nn.ConvTranspose2d(32, 32, kernel_size=(4,4),  stride=2) # output_padding to fix encoder decoder problem 31 != 30
        self.cnn4 = nn.ConvTranspose2d(32, 32, kernel_size=(4,4),  stride=2, output_padding=1) # output_padding to fix encoder decoder problem 31 != 30
        self.cnn5 = nn.ConvTranspose2d(32, output_dim, kernel_size=(4,4),  stride=2)

        self.bn1 = nn.BatchNorm1d(1536, track_running_stats=running_stats, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(64, track_running_stats=running_stats, momentum=momentum)
        self.bn3 = nn.BatchNorm2d(32, track_running_stats=running_stats, momentum=momentum)
        self.bn4 = nn.BatchNorm2d(32, track_running_stats=running_stats, momentum=momentum)
        self.bn5 = nn.BatchNorm2d(32, track_running_stats=running_stats, momentum=momentum)

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
    def __init__(self, latent_dim, input_depth, running_stats=True, momentum=0.1):
        super(Encoder, self).__init__()

        self.cnn1 = nn.Conv2d(input_depth, 32, kernel_size=(4,4), stride=2)
        self.cnn2 = nn.Conv2d(32, 32, kernel_size=(4,4),  stride=2)
        self.cnn3 = nn.Conv2d(32, 32, kernel_size=(4,4),  stride=2)
        self.cnn4 = nn.Conv2d(32, 64, kernel_size=(4,4),  stride=2)
        self.cnn5 = nn.Conv2d(64, 64, kernel_size=(4,4),  stride=2)
        self.fc21 = nn.Linear(1536, latent_dim)
        self.fc22 = nn.Linear(1536, latent_dim)

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

        x = x.view(-1, 1536)

        z_loc = self.fc21(x)
        z_scale = self.fc22(x)

        return z_loc, z_scale

if __name__ == '__main__':

    from blender_dataset import BlenderFolder
    dataset = BlenderFolder('/opt/data/table_dataset/dataset', debug=True)
    sample1, target = dataset.__getitem__(0)
    sample2, target = dataset.__getitem__(1)

    import torch
    sample = torch.cat([sample1.unsqueeze(0), sample2.unsqueeze(0)], 0)

    encoder = Encoder(10, 4)
    decoder = Decoder(10, 2)

    z, _ = encoder.forward(sample)
    sample = decoder.forward(z)


