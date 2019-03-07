import os
import torch
import torch.optim as optim
from blender_loader import BlenderLoader

from affordance_vaed import Decoder, Encoder, AffordanceVAED
from affordance_trainer import Trainer

from tools import save_arguments

import argparse

parser = argparse.ArgumentParser(description='VAED for Blender dataset')
parser.add_argument('--lr', default=1.0e-3, type=float, help='Learning rate')
parser.add_argument('--latent-size', default=20, type=int, help='Number of latent dimensions')
parser.add_argument('--num-epoch', default=10000, type=int, help='Number of epochs')
parser.add_argument('--batch-size', default=256, type=int, help='Batch size')
parser.add_argument('--num-workers', default=18, type=int, help='Num processes utilized')
parser.add_argument('--beta', default=4, type=float, help='Beta coefficient for KL-divergence')
parser.add_argument('--model-name', default='vaed_test', type=str, help='Results are saved to a given folder name')
parser.add_argument('--data-path', default='blender_dataset', type=str, help='Dataset loaded to')
parser.add_argument('--debug', dest='debug', action='store_true')
parser.set_defaults(debug=False)
parser.add_argument('--depth', dest='depth', action='store_true', help='Include depth')
parser.set_defaults(depth=False)
NUM_AFFORDANCES = 2
LOG_PATH = 'perception_results'


def main(args):

    save_path = os.path.join(LOG_PATH, args.model_name)

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

    model = AffordanceVAED(encoder, decoder, device, beta=args.beta).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    dataloader = BlenderLoader(args.batch_size, args.num_workers, args.depth, [args.data_path], debug=args.debug)

    trainer = Trainer(dataloader, model, log_path=save_path)
    trainer.train(args.num_epoch, optimizer)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
