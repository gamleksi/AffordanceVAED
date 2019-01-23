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
    use_cuda = torch.cuda.is_available()

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
    z_dim = 10
    depth = 3

    grayscale = True
    if depth > 1:
        grayscale = False

    encoder = Encoder(z_dim, depth)
    decoder = Decoder(z_dim, depth)
    model = VAE(encoder, decoder, device, beta=5).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    dataloader = ImageLoader(BATCH_SIZE, NUM_PROCESSES, 'data/affordances', grayscale=grayscale, debug=False)

    trainer = Trainer(dataloader, save_folder='umd_1', save_name='umd_1_simple_cnn_beta_4_dim_10', log=True, visdom=True)
    # trainer = Trainer(dataloader, visdom=True)
    trainer.train(model, NUM_EPOCHS, optimizer)


if __name__ == '__main__':
    main()