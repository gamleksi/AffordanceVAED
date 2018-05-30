import os

import sys
sys.path.insert(0, './libs/tnt/')

import torch.optim as optim
from loaders import MNISTLoader 

from models.simple_model import Decoder, Encoder, VAE 

from monitor import Trainer, Demonstrator 
# from torch.nn.init import kaiming_normal

   
def main():

    # Run options
    LEARNING_RATE = 1.0e-3
    USE_CUDA = False 

    # Run only for a single iteration for testing
    NUM_EPOCHS = 50 
    BATCH_SIZE = 256
    NUM_PROCESSES = 1 

    # Data and model
    z_dim = 3 
    hidden_dim = 400 
    img_channels = 1
    img_height = 28
    img_width = 28

    encoder = Encoder(img_width, img_height, z_dim, hidden_dim)
    decoder = Decoder(img_width, img_height, img_channels, z_dim, hidden_dim)

    model = VAE(encoder, decoder, beta=3)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    dataloader = MNISTLoader(BATCH_SIZE, NUM_PROCESSES, debug=False)

    trainer = Trainer(dataloader, 'simple', 'fc_model', log=False)
    trainer.train(model, NUM_EPOCHS, optimizer)

#    encoder = Encoder(input_dim, z_dim, hidden_dim)
#    decoder = Decoder(z_dim, hidden_dim)
#    model = VAE(encoder, decoder)
#
#    dataloader = MNISTLoader(BATCH_SIZE, NUM_EPOCHS)
#    demo = Demonstrator(model, 'test12', 'model1')
#    demo.evaluate(dataloader.get_iterator)

if __name__ == '__main__':
    main()