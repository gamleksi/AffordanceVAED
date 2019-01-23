import torch
import torch.optim as optim
from loaders import ChairsLoader

from models.chair_model import Decoder, Encoder
from models.simple_model import VAE

from monitor import Demonstrator


def main():
    # Run options

    use_cuda = torch.cuda.is_available()

    if use_cuda == 'cuda':
        print('GPU works!')
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Run only for a single iteration for testing
    NUM_EPOCHS = 300
    BATCH_SIZE = 256
    NUM_PROCESSES = 16

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
    model = VAE(encoder, decoder, device).to(device)

    dataloader = ChairsLoader(BATCH_SIZE, NUM_PROCESSES, grayscale=grayscale, debug=False)

    demonstrator = Demonstrator('chair_2', 'chair_cnn_beta_5_rgb', model, dataloader)
    demonstrator.evaluate()


if __name__ == '__main__':
    main()