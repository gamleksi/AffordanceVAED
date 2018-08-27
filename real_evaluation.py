import torch

from models.blender_model import Decoder, Encoder
from models.simple_model import AffordanceVAE

from affordance_monitor import AffordanceDemonstrator
from blender_dataset import KinectEvaluationLoader
import argparse
from image_logger import MatplotLogger

from tools import model_name_search
import os


parser = argparse.ArgumentParser(description='Variational Autoencoder for blender data')
parser.add_argument('--latent_size', default=10, type=int, help='Number of latent variables')
parser.add_argument('--beta', default=4, type=int)
parser.add_argument('--folder_name', default='blender_vae_beta_4_v_1', type=str)
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
    file_name = model_name_search(os.path.join('log', folder_name))

debug = args.debug
include_depth = args.depth

def main():
    # Run options
    use_cuda = False # torch.cuda.is_available()

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

    kinect_loader = KinectEvaluationLoader(include_depth)

    return AffordanceDemonstrator(model, folder_name, file_name, NUM_LATENT_VARIABLES, include_depth, loader=kinect_loader, logger=MatplotLogger(folder_name, False, save_folder='real_image_results'))

if __name__ == '__main__':
    import numpy as np
    evaluator = main()
    samples = np.arange(0, 75)

    evaluator.get_result_pair(samples, 'result')
    for s in samples:
        evaluator.get_result_pair([s], 'result {}'.format(s + 1))
#    file_names = ['latent_distribution_sample_{}_step_size_{}'.format(sample, step_size) for sample in samples]
#    evaluator.list_of_latent_distribution_samples(samples, file_names, step_size=step_size, num_samples=15)
#    evaluator.dimensional_transform_of_samples(4, 5, 'dimensional_transform_of_samples_{}-{}_samples'.format(4, 5))
#    evaluator.dimensional_transform_of_samples(10, 12, 'dimensional_transform_of_samples_{}-{}_samples'.format(10, 12))
#    evaluator.get_result_pair([4,5,10, 12], 'result_pairs')
#    evaluator.latent_distribution_of_zero('latent_distribution_of_zero',step_size=1, num_samples=15)
#    evaluator.transform_of_samples(59, 60, 'transform_of_samples_{}-{}_samples'.format(59, 60))
