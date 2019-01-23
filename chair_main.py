import torch
import torch.optim as optim
from loaders import ImageLoader

from models.chair_model import Decoder, Encoder 
from models.simple_model import VAE 

from monitor import Trainer, Demonstrator 
# from torch.nn.init import kaiming_normal
  
def main():

    # Run options
    LEARNING_RATE = 1.0e-3
    use_cuda =  torch.cuda.is_available()

    if use_cuda:
        print('GPU works!')
    else:
        for i in range(10):
            print('YOU ARE NOT USING GPU')

    device = torch.device('cuda' if use_cuda else 'cpu')

    # Run only for a single iteration for testing
    NUM_EPOCHS = 1000
    BATCH_SIZE = 256
    NUM_PROCESSES = 16

    # Data and model
    z_dim = 20
    depth = 3
    hidden_dim = 400 
    input_dim = 784

    grayscale = True
    if depth > 1:
        grayscale = False

    encoder = Encoder(z_dim, depth)
    decoder = Decoder(z_dim, depth)
    model = VAE(encoder, decoder, device, beta=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    dataloader = ImageLoader(BATCH_SIZE, NUM_PROCESSES, grayscale=grayscale, debug=True)

    # trainer = Trainer(dataloader, save_folder='chair_5', save_name='chair_cnn_beta_5_rgb_dims_20', log=True, visdom=True)
    trainer = Trainer(dataloader, visdom=True)
    trainer.train(model, NUM_EPOCHS, optimizer)


if __name__ == '__main__':
    main()