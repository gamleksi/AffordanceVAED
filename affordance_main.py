import os
import argparse
import torch
import torch.optim as optim
from loaders import AffordanceLoader

from models.blender_model import Decoder, Encoder
from models.simple_model import AffordanceVAE

from monitor import Trainer

parser = argparse.ArgumentParser(description='Variational Autoencoder for blender data')
parser.add_argument('--lr', default=1.0e-3, type=float, help='Learning rate')
parser.add_argument('--latent-size', default=10, type=int, help='Number of latent variables')
parser.add_argument('--num-epoch', default=10000, type=int, help='Number of epochs')
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--num-workers', default=18, type=int)
parser.add_argument('--beta', default=4, type=int)
parser.add_argument('--folder-name', default='affordance_vae', type=str)
parser.add_argument('--debug', dest='debug', action='store_true')
parser.set_defaults(debug=False)

parser.add_argument('--no-log', dest='log', action='store_false')
parser.set_defaults(log=True)

parser.add_argument('--depth', dest='depth', action='store_true')
parser.set_defaults(depth=False)


def save_arguments(args, save_path):

    args = vars(args)
    if not(os.path.exists(save_path)):
        os.makedirs(save_path)
    file = open(os.path.join(save_path, "arguments.txt"), 'w')
    lines = [item[0] + " " + str(item[1]) + "\n" for item in args.items()]
    file.writelines(lines)


LOG_PATH = 'umd_results'
DATA_PATHS = 'umd_dataset'

def main(args):
    # Run options
    use_cuda = torch.cuda.is_available()

    save_path = os.path.join('umd_results', args.folder_name)

    save_arguments(args, save_path)

    if use_cuda:
        print('GPU works!')
    else:
        for i in range(10):
            print('YOU ARE NOT USING GPU')

    device = torch.device('cuda' if use_cuda else 'cpu')

    # Data and model

    input_channels = 4 if args.depth else 3

    encoder = Encoder(args.latent_size, input_channels)
    decoder = Decoder(args.latent_size, 7)
    model = AffordanceVAE(encoder, decoder, device, beta=args.beta).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    dataloader = AffordanceLoader(args.batch_size, args.num_workers, DATA_PATHS, args.depth, debug=False)

    trainer = Trainer(dataloader, model, log_path=save_path)

    trainer.train(args.num_epoch, optimizer)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)