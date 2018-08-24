import torch
import torch.optim as optim
from blender_dataset import BlenderLoader

from models.blender_model import Decoder, Encoder
from models.simple_model import AffordanceVAE, AffordanceCapacityVAE

from monitor import Trainer
import argparse

parser = argparse.ArgumentParser(description='Variational Autoencoder for blender data')
parser.add_argument('--lr', default=1.0e-3, type=float, help='Learning rate')
parser.add_argument('--latent_size', default=10, type=int, help='Number of latent variables')
parser.add_argument('--num_epoch', default=10000, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_workers', default=18, type=int)
parser.add_argument('--beta', default=4, type=int)

parser.add_argument('--gamma', default=300, type=int)
parser.add_argument('--capacity_limit', default=30, type=int)
parser.add_argument('--capacity_change_duration', default=60000, type=int)

parser.add_argument('--capacity', dest='capacity', action='store_true')
parser.add_argument('--no-capacity', dest='capacity', action='store_false')
parser.set_defaults(debug=False)

parser.add_argument('--folder_name', default='blender_vae', type=str)
parser.add_argument('--visdom_title', default=None, type=str)
parser.add_argument('--env', default=None, type=str)

parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

parser.add_argument('--visdom', dest='visdom', action='store_true')
parser.add_argument('--no-visdom', dest='visdom', action='store_false')
parser.set_defaults(visdom=True)

parser.add_argument('--log', dest='log', action='store_true')
parser.add_argument('--no-log', dest='log', action='store_false')
parser.set_defaults(log=True)

parser.add_argument('--depth', dest='depth', action='store_true')
parser.add_argument('--no-depth', dest='depth', action='store_false')
parser.set_defaults(depth=True)

args = parser.parse_args()

LEARNING_RATE = args.lr
NUM_LATENT_VARIABLES = args.latent_size
folder_name = args.folder_name
beta = args.beta

use_capacity = args.capacity
gamma = args.gamma
capacity_limit = args.capacity_limit
capacity_change_duration = args.capacity_change_duration

if use_capacity:
    file_name = 'model_g_{}_lim_{}_dur_{}_l_{}_lr_{}'.format(gamma, capacity_limit, capacity_change_duration, NUM_LATENT_VARIABLES, LEARNING_RATE)
else:
    file_name = 'model_b_{}_l_{}_lr_{}'.format(beta, NUM_LATENT_VARIABLES. LEARNING_RATE)
NUM_EPOCHS = args.num_epoch
BATCH_SIZE = args.batch_size
NUM_PROCESSES = args.num_workers
debug = args.debug
visdom = args.visdom

if args.visdom_title is None:
    visdom_title = file_name + 'logger'
else:
    visdom_title = args.visdom_title

if args.env is None:
    env = file_name
else:
    env = args.visdom_title

include_depth = args.depth

if debug:
    log = False
    file_name = None
    folder_name = None
else:
    log = args.log


def main():

    AFFORDANCE_SIZE = 2

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print('GPU works!')
    else:
        for i in range(10):
            print('YOU ARE NOT USING GPU')

    device = torch.device('cuda' if use_cuda else 'cpu')

    image_channels = 4 if include_depth else 3

    encoder = Encoder(NUM_LATENT_VARIABLES, image_channels)
    decoder = Decoder(NUM_LATENT_VARIABLES, AFFORDANCE_SIZE)

    if use_capacity:
        model = AffordanceCapacityVAE(encoder, decoder, device).to(device)
    else:
        model = AffordanceVAE(encoder, decoder, device, beta=beta).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    dataloader = BlenderLoader(BATCH_SIZE, NUM_PROCESSES, include_depth, debug=args.debug)

    trainer = Trainer(dataloader, model, NUM_LATENT_VARIABLES, save_folder=folder_name, save_name=file_name, log=log, visdom=visdom, visdom_title=visdom_title)
    trainer.train(NUM_EPOCHS, optimizer)


if __name__ == '__main__':
    main()
