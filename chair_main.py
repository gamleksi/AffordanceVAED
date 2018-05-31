import os

import sys
sys.path.insert(0, './libs/tnt/')

import torch.optim as optim
from loaders import ChairsLoader 

from models.chair_model import Decoder, Encoder 
from models.simple_model import VAE 

from monitor import Trainer, Demonstrator 
# from torch.nn.init import kaiming_normal
  
def main():

    # Run options
    LEARNING_RATE = 1.0e-3
    USE_CUDA = False 

    # Run only for a single iteration for testing
    NUM_EPOCHS = 50 
    BATCH_SIZE = 10
    NUM_PROCESSES = 8 

    # Data and model
    z_dim = 4 
    depth = 3
    hidden_dim = 400 
    input_dim = 784 
    grayscale = True 

    if depth > 1:
        grayscale = False 

    encoder = Encoder(z_dim, depth)
    decoder = Decoder(z_dim, depth)
    model = VAE(encoder, decoder, beta=3)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    dataloader = ChairsLoader(BATCH_SIZE, NUM_PROCESSES, grayscale=grayscale, debug=True)

    trainer = Trainer(dataloader, 'simple', 'fc_model', log=False, visdom=True)
    trainer.train(model, NUM_EPOCHS, optimizer)

if __name__ == '__main__':
    main()