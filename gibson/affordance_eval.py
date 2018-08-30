import torch
import torch.optim as optim
from loaders import AffordanceLoader

from models.chair_model import Decoder, Encoder
from models.simple_model import AffordanceVAE

from monitor import AffordanceDemonstrator

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
    NUM_PROCESSES = 17

    # Data and model
    z_dim = 20

    encoder = Encoder(z_dim, 4)
    decoder = Decoder(z_dim, 7)
    model = AffordanceVAE(encoder, decoder, device, beta=5).to(device)
    # model.load('log/affordance_1/affordance_1_simple_cnn_beta_5_dim_20.pth.tar')

    dataloader = AffordanceLoader(BATCH_SIZE, NUM_PROCESSES, 'data/affordances/full_64', debug=False)

    # trainer = AffordanceTrainer(dataloader, save_folder='affordance_1', save_name='affordance_1_simple_cnn_beta_5_dim_20', log=True, visdom=True)

    return AffordanceDemonstrator(dataloader, model, 'affordance_2', 'affordance_2_simple_cnn_beta_5_dim_20')
    # trainer.latent_distribution_of_visdom_samples()




if __name__ == '__main__':
    monitor = main()