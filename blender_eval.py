import torch

from models.blender_model import Decoder, Encoder
from models.simple_model import AffordanceVAE

from affordance_monitor import AffordanceDemonstrator
import argparse

parser = argparse.ArgumentParser(description='Variational Autoencoder for blender data')
parser.add_argument('--latent_size', default=10, type=int, help='Number of latent variables')
parser.add_argument('--beta', default=4, type=int)
parser.add_argument('--folder_name', type=str)
parser.add_argument('--file_name', type=str)

parser.add_argument('--depth', dest='depth', action='store_true')
parser.add_argument('--no-depth', dest='depth', action='store_false')
parser.set_defaults(depth=True)

parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

args = parser.parse_args()

NUM_LATENT_VARIABLES = args.latent_size
beta = args.beta

folder_name = args.folder_name
assert(folder_name is not None)

file_name = args.file_name
if file_name is None:
    file_name = '{}_beta_{}_latent_{}'.format(folder_name, beta, NUM_LATENT_VARIABLES)

debug = args.debug
include_depth = args.depth

def main():
    # Run options
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print('GPU works!')
    else:
        for i in range(10):
            print('YOU ARE NOT USING GPU')

    device = torch.device('cuda' if use_cuda else 'cpu')

    image_channels = 4 if include_depth else 3

    AFFORDANCE_SIZE = 2

    encoder = Encoder(NUM_LATENT_VARIABLES, image_channels)
    decoder = Decoder(NUM_LATENT_VARIABLES, AFFORDANCE_SIZE)
    model = AffordanceVAE(encoder, decoder, device, beta=beta).to(device)

    return AffordanceDemonstrator(model, folder_name, file_name, NUM_LATENT_VARIABLES, include_depth)

if __name__ == '__main__':
    import ipdb
    evaluator = main()
    #evaluator.list_of_latent_distribution_samples([0, 1, 2], step_size=2, num_samples=8)
    evaluator.get_result_pair(0)
    #ipdb.set_trace()
    #evaluator.get_result_pair(1)
    #ipdb.set_trace()
    #evaluator.get_result_pair(2)
    #evaluator.neighbors_of_zero_latent(step_size=3, num_samples=2, padding=2)
    #evaluator.neighbor_channels()

