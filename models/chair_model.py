import torch.nn as nn
import torch.nn.functional as F


# From: https://openreview.net/pdf?id=Sy2fzU9gl
class Decoder(nn.Module):
    def __init__(self, z_dim, image_depth):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(z_dim, 256)

        self.cnn1 = nn.ConvTranspose2d(64, 64, kernel_size=(4,4),  stride=2)
        self.cnn2 = nn.ConvTranspose2d(64, 32, kernel_size=(4,4),  stride=2)
        self.cnn3 = nn.ConvTranspose2d(32, 32, kernel_size=(4,4),  stride=2, output_padding=1) # output_padding to fix encoder decoder problem 31 != 30
        self.cnn4 = nn.ConvTranspose2d(32, image_depth, kernel_size=(4,4),  stride=2)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn34 = nn.BatchNorm2d(32)

        self.sigmoid = nn.Sigmoid()
        self.relu = F.relu

    def forward(self, z):

        x = self.bn1(self.relu(self.fc(z)))
        x = x.view(-1, 64, 2, 2)
        x = self.bn2(self.relu(self.cnn1(x)))
        x = self.bn34(self.relu(self.cnn2(x)))
        x = self.bn34(self.relu(self.cnn3(x)))
        x = self.sigmoid(self.cnn4(x))
        return x 

class Encoder(nn.Module):
    def __init__(self, z_dim, input_depth):
        super(Encoder, self).__init__()

        self.cnn1 = nn.Conv2d(input_depth, 32, kernel_size=(4,4), stride=2)
        self.cnn2 = nn.Conv2d(32, 32, kernel_size=(4,4),  stride=2)
        self.cnn3 = nn.Conv2d(32, 64, kernel_size=(4,4),  stride=2)
        self.cnn4 = nn.Conv2d(64, 64, kernel_size=(4,4),  stride=2)
        self.fc21 = nn.Linear(256, z_dim)
        self.fc22 = nn.Linear(256, z_dim)

        self.bn12 = nn.BatchNorm2d(32)
        self.bn23 = nn.BatchNorm2d(64)

        self.relu = F.relu 

    def forward(self, x):

        x = self.bn12(self.relu(self.cnn1(x)))
        x = self.bn12(self.relu(self.cnn2(x)))
        x = self.bn23(self.relu(self.cnn3(x)))
        x = self.bn23(self.relu(self.cnn4(x)))

        x = x.view(-1, 256)

        z_loc = self.fc21(x)
        z_scale = self.fc22(x)

        return z_loc, z_scale