import os
import time
import copy
import torch
from os.path import join,isfile
from tqdm import tqdm
from os import listdir
import random
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import autograd
from utils import Config
from dataset import SuDataset
from vis_utils import VisUtils

import numpy as np
import cv2
import torchvision.transforms as T

from torchvision import models
from torch.nn import functional as F

from misc_function import processImage, detail_enhance_lab, recreate_image, PreidictLabel, AdvLoss

default_config = Config(
	N_START = 0,
	N_EPOCH = None,
	FINE_SIZE = -1,
	#################### CONSTANT #####################
	IMG = None,
	SAVE = 'checkpoints',
	BATCH = 1l,
	GPU = 0,
	LR = 0.001,
	# clip
	clip = None,
	# model
	model = None,
	# forward
	forward = None,
	# img size
	exceed_limit = None,
	# vis
	vis = None
)


def run(config, dataset_path, dataset_smooth_path, image_list, idx, adv_model):

    # Ceate a directory for saving the trained models
	save_path = config.SAVE
	path = os.path.join(save_path, 'snapshots')
	if not os.path.isdir(path):
		os.makedirs(path)

	# Create a directory for saving adversarial images
	adv_path =	'../EnhancedAdvImgsfor_{}/'.format(adv_model)
	if not os.path.isdir(adv_path):
		os.makedirs(adv_path)

	
	# Smoothing loss function
	criterion = nn.MSELoss()

	
	# Using GPU
	if config.GPU >= 0:
		with torch.cuda.device(config.GPU):
			config.model.cuda()
			criterion.cuda()
	
	
	# Setup optimizer
	optimizer = optim.Adam(config.model.parameters(), lr=config.LR)

	

	# Load the classifier for attacking 
	if adv_model ==  'resnet18':
		classifier = models.resnet18(pretrained=True)
	elif adv_model ==  'resnet50':
		classifier = models.resnet50(pretrained=True)
	elif adv_model ==  'alexnet':
		classifier = models.alexnet(pretrained=True)
	classifier.cuda()
	classifier.eval()

	# Freeze the parameters of the classifeir under attack to not be updated
	for param in classifier.parameters():
		param.requires_grad = False



	 	
	# The name of the chosen image
	img_name = image_list[idx].split('/')[-1]
		

	
	# Pre-processing the original image and ground truth L_0 smooth image
	x= processImage(dataset_path,img_name)		
	gt_smooth = processImage(dataset_smooth_path,img_name)
	

	# Prediction of the original image using the classifier chosen for attacking
	class_x, prob_class_x, prob_x, logit_x, target_class = PreidictLabel(x, classifier)


	# Initilize number of misclassification and maximum number of iterations for updating FCNN using total_loss
	misclassified = 0	
	maxIters = 5000


	for it in range(maxIters): 
		t = time.time()
		
		
		with autograd.detect_anomaly():
			# Smooth images
			x_smooth= config.forward(x,gt_smooth, config)
			
			# Enhance adversarial image
			enh = detail_enhance_lab(x,x_smooth)
	   

			# Prediction of the adversarial image using the classifier chosen for attacking
			class_enh, prob_class_enh, prob_enh, logit_enh, _ = PreidictLabel(enh.permute(2,0,1).unsqueeze(dim=0), classifier)

			
			# Computing smoothing and adversarial losses
			loss1 = criterion(x_smooth, gt_smooth)
			loss2 = AdvLoss(logit_enh, class_x, is_targeted=False)

			
			# Combining the smoothing and adversarial losses
			loss = 10*loss1 + loss2


			# backward
			optimizer.zero_grad()
			loss.backward()
			if config.clip is not None:
				torch.nn.utils.clip_grad_norm(config.model.parameters(), config.clip)
			optimizer.step()
			
			
			
			# Save the adversarial image when the classifier is fooled and smoothing loss is less than a threshold
			if (class_x != class_enh): 
				misclassified=1
				cv2.imwrite('{}{}'.format(adv_path,img_name), recreate_image(enh))
                if (loss1< 0.0005):
					break


	# Save the FCNN
	torch.save(config.model.state_dict(), os.path.join(save_path, 'snapshots', '{}_latest.pth'.format(adv_model)))
