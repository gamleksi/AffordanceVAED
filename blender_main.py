import os
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
parser.add_argument('--num-epoch', default=10000, type=int, help='Number of epochs')
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--num-workers', default=18, type=int)
parser.add_argument('--beta', default=4, type=int)

# parser.add_argument('--gamma', default=300, type=int)
# parser.add_argument('--capacity-limit', default=30, type=int)
# parser.add_argument('--capacity-change_duration', default=60000, type=int)

parser.add_argument('--capacity', dest='capacity', action='store_true')
parser.add_argument('--no-capacity', dest='capacity', action='store_false')
parser.set_defaults(capacity=False)

parser.add_argument('--folder-name', default='blender_vae', type=str)

parser.add_argument('--debug', dest='debug', action='store_true')
parser.set_defaults(debug=False)

parser.add_argument('--no-log', dest='log', action='store_false')
parser.set_defaults(log=True)

parser.add_argument('--depth', dest='depth', action='store_true')
parser.set_defaults(depth=False)

def save_args(args, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("TODO")


NUM_AFFORDANCES = 2
DATA_PATHS = ['/opt/data/table_dataset/two_cups_dataset', '/opt/data/table_dataset/two_cups_dataset']
# DATA_PATHS = ['debug_ds/debug_samples']

LOG_PATH = 'perception_results'


def save_arguments(args, save_path):

    args = vars(args)
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)
    file = open(os.path.join(save_path, "arguments.txt"), 'w')
    lines = [item[0] + " " + str(item[1]) + "\n" for item in args.items()]
    file.writelines(lines)

def main(args):

    save_path = os.path.join(LOG_PATH, args.folder_name)

    save_arguments(args, save_path)

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print('GPU works!')
    else:
        print('YOU ARE NOT USING GPU')

    device = torch.device('cuda' if use_cuda else 'cpu')

    image_channels = 4 if args.depth else 3

    encoder = Encoder(args.latent_size, image_channels)
    decoder = Decoder(args.latent_size, NUM_AFFORDANCES)

    if args.capacity:
        model = AffordanceCapacityVAE(encoder, decoder, device).to(device)
    else:
        model = AffordanceVAE(encoder, decoder, device, beta=args.beta).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    dataloader = BlenderLoader(args.batch_size, args.num_workers, args.depth, DATA_PATHS, debug=args.debug)
    trainer = Trainer(dataloader, model, log_path=save_path)
    trainer.train(args.num_epoch, optimizer)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
