import os
import argparse
import torch
import torch.optim as optim

from affordance_vaed import AffordanceVAED, UMDEncoder, UMDDecoder
from umd_loader import UMDLoader
from affordance_trainer import Trainer
from tools import save_arguments


parser = argparse.ArgumentParser(description='VAED for UMD dataset')

parser.add_argument('--lr', default=1.0e-3, type=float, help='Learning rate')
parser.add_argument('--latent-size', default=20, type=int, help='Number of latent dimensions')
parser.add_argument('--num-epoch', default=10000, type=int, help='Number of epochs')
parser.add_argument('--batch-size', default=256, type=int, help='Batch size')
parser.add_argument('--num-workers', default=18, type=int, help='Num processes utilized')
parser.add_argument('--beta', default=4, type=float, help='Beta coefficient for KL-divergence')
parser.add_argument('--model-name', default='vaed_test', type=str, help='Results are saved to a given folder name')
parser.add_argument('--data-path', default='umd_dataset', type=str, help='Dataset loaded to')
parser.add_argument('--debug', dest='debug', action='store_true')
parser.set_defaults(debug=False)
parser.add_argument('--depth', dest='depth', action='store_true')
parser.set_defaults(depth=False)


LOG_PATH = 'umd_results'


def main(args):
    # Run options
    use_cuda = torch.cuda.is_available()

    save_path = os.path.join(LOG_PATH, args.model_name)

    save_arguments(args, save_path)

    if use_cuda:
        print('GPU works!')
    else:
        print('YOU ARE NOT USING GPU!')

    device = torch.device('cuda' if use_cuda else 'cpu')

    input_channels = 4 if args.depth else 3

    encoder = UMDEncoder(args.latent_size, input_channels)
    decoder = UMDDecoder(args.latent_size, 7)

    model = AffordanceVAED(encoder, decoder, device, beta=args.beta).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    dataloader = UMDLoader(args.batch_size, args.num_workers, args.data_path, args.depth, debug=False)

    trainer = Trainer(dataloader, model, log_path=save_path)

    trainer.train(args.num_epoch, optimizer)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)