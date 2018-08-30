import torch
import torch.optim as optim
from loaders import AffordanceLoader

from models.chair_model import Decoder, Encoder
from models.simple_model import AffordanceVAE

from monitor import AffordanceTrainer

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

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    dataloader = AffordanceLoader(BATCH_SIZE, NUM_PROCESSES, 'data/affordances/full_64', debug=False)

    trainer = AffordanceTrainer(dataloader, model, save_folder='affordance_3', save_name='affordance_3_simple_cnn_beta_5_dim_20',
                                log=True, visdom=True)

    trainer.train(NUM_EPOCHS, optimizer)


if __name__ == '__main__':
    main()