import torch
import torch.optim as optim
from blender_dataset import BlenderLoader

from models.blender_model import Decoder, Encoder
from models.simple_model import AffordanceVAE

from monitor import AffordanceTrainer
import argparse

parser = argparse.ArgumentParser(description='Variational Autoencoder for blender data')
parser.add_argument('--lr', default=1.0e-3, type=float, help='Learning rate')
parser.add_argument('--latent_size', default=10, type=int, help='Number of latent variables')
parser.add_argument('--num_epoch', default=10000, type=int, help='Number of epochs')
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_workers', default=18, type=int)
parser.add_argument('--beta', default=4, type=int)
parser.add_argument('--folder_name', default='blender_vae', type=str)
parser.add_argument('--visdom', default=True, type=bool)
parser.add_argument('--log', default=True, type=bool)

args = parser.parse_args()

LEARNING_RATE = args.lr
NUM_LATENT_VARIABLES = args.latent_size
folder_name = args.folder_name
beta = args.beta
file_name = '{}_beta_{}_latent_{}'.format(folder_name, beta, NUM_LATENT_VARIABLES)
NUM_EPOCHS = args.num_epoch
BATCH_SIZE = args.batch_size
NUM_PROCESSES = args.num_workers

debug = args.debug
visdom = args.visdom

if debug:
    log = False
    file_name = None
    folder_name = None
else:
    log = args.log

# TODO include possiblity not to include depth


def main():

    AFFORDANCE_SIZE = 2

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print('GPU works!')
    else:
        for i in range(10):
            print('YOU ARE NOT USING GPU')

    device = torch.device('cuda' if use_cuda else 'cpu')

    encoder = Encoder(NUM_LATENT_VARIABLES, 4)
    decoder = Decoder(NUM_LATENT_VARIABLES, AFFORDANCE_SIZE)
    model = AffordanceVAE(encoder, decoder, device, beta=beta).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    dataloader = BlenderLoader(BATCH_SIZE, NUM_PROCESSES, debug=args.debug)

    trainer = AffordanceTrainer(dataloader, model, save_folder=folder_name, save_name=file_name,
                                log=log, visdom=visdom)

    trainer.train(NUM_EPOCHS, optimizer)


if __name__ == '__main__':
    main()