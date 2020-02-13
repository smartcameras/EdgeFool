import copy
import argparse

from train_base import *

from module import DeepGuidedFilter

parser = argparse.ArgumentParser(description='Train FCNN to Generate Enhanced Adversarial Images ')
parser.add_argument('--adv_model', type=str,  help='adversarial model')
parser.add_argument('--path_imgs', required=True, nargs='+', type=str,
            help='a list of image paths, or a directory name')
parser.add_argument('--path_smooth', required=True, nargs='+', type=str,
            help='a list of smoothed image paths, or a directory name')
args = parser.parse_args()


def forward(imgs,gt, config):
	x_hr= imgs
	gt_hr=gt
	return config.model(x_hr, x_hr)

dataset_path  = args.path_imgs[0] 
dataset_smooth_path = args.path_smooth[0] 


# List of the name of all the images in the dataset_path
image_list =  [f for f in listdir(dataset_path) if isfile(join(dataset_path,f))]
NumImg=len(image_list)


# Configuration
config = copy.deepcopy(default_config)
config.N_EPOCH = 100
# model
config.model = DeepGuidedFilter()
config.forward = forward
config.clip = 0.01   


# Run the attack for each image
for idx in tqdm(range(NumImg)):
	run(config, dataset_path, dataset_smooth_path, image_list, idx, args.adv_model)


