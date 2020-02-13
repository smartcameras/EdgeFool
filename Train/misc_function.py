from rgb_lab_formulation_pytorch import *
import cv2
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
import copy


def processImage(dataset_path,img_name):

		x = cv2.imread(dataset_path+img_name, 1)/255.0
		# Have RGB images
		x = x[:, :, (2, 1, 0)]
		x = x.transpose(2, 0, 1)  # Convert array to C,W,H
		x = torch.from_numpy(x).float()
		# Add one more channel to the beginning. Tensor shape = 1,3,224,224
		x.unsqueeze_(0)
		# Convert to Pytorch variable
		x = Variable(x.cuda())
		return x

def detail_enhance_lab(img, smooth_img):
	#mean = [0.485, 0.456, 0.406]
	#std = [0.229, 0.224, 0.225]

	val0 = 15
	val2 = 1
	exposure = 1.0
	saturation = 1.0
	gamma = 1.0
	
	#for c in range(3):
	#	img[:,:,c] = img[:,:,c] * std[c]
	#	img[:,:,c] = img[:,:,c] + mean[c]
	#img[img > 1] = 1
	#img[img < 0] = 0


	# convert 1,C,W,H --> W,H,C
	img = img.squeeze().permute(1,2,0)#(2,1,0)
	smooth_img = smooth_img.squeeze().permute(1,2,0)

	# Convert image and smooth_img from rgb to lab
	img_lab=rgb_to_lab(img)	
	smooth_img_lab=rgb_to_lab(smooth_img)
		# do the enhancement	
	img_l, img_a, img_b =torch.unbind(img_lab,dim=2)
	smooth_l, smooth_a, smooth_b =torch.unbind(smooth_img_lab,dim=2)
	diff = my_sig((img_l-smooth_l)/100.0,val0)*100.0
	base = (my_sig((exposure*smooth_l-56.0)/100.0,val2)*100.0)+56.0
	res = base + diff
	img_l = res
	img_a = img_a * saturation
	img_b = img_b * saturation
	img_lab = torch.stack([img_l, img_a, img_b], dim=2)
	
	L_chan, a_chan, b_chan = preprocess_lab(img_lab)
	img_lab = deprocess_lab(L_chan, a_chan, b_chan)
	#img = color.lab2rgb(img_lab)
	img_final = lab_to_rgb(img_lab)

	#img_final = (img_final - mean) / std

	return img_final
def my_sig(x,a):
	
	# Applies a sigmoid function on the data x in [0-1] range. Then rescales
	# the result so 0.5 will be mapped to itself.

	# Apply Sigmoid
	y = 1./(1+torch.exp(-a*x)) - 0.5

	# Re-scale
	y05 = 1./(1+torch.exp(-torch.tensor(a*0.5,dtype=torch.float32))) - 0.5
	y = y*(0.5/y05)

	return y 


def recreate_image(im_as_var):
	
	if im_as_var.shape[0] == 1:
		recreated_im = copy.copy(im_as_var.cpu().data.numpy()[0]).transpose(1,2,0)
	else:	
		recreated_im = copy.copy(im_as_var.cpu().data.numpy())
	recreated_im[recreated_im > 1] = 1
	recreated_im[recreated_im < 0] = 0
	recreated_im = np.round(recreated_im * 255)
	# Convert RBG to GBR
	recreated_im = recreated_im[..., ::-1]
	return recreated_im	



def PreidictLabel(x, classifier):
 
		mean = torch.zeros(x.shape).float().cuda()
		mean[:,0,:,:]=0.485
		mean[:,1,:,:]=0.456
		mean[:,2,:,:]=0.406

		std = torch.zeros(x.shape).float().cuda()
		std[:,0,:,:]=0.229
		std[:,1,:,:]=0.224
		std[:,2,:,:]=0.225


		# Standarise
		x = (x - mean) / std
  
		logit_x = classifier.forward(x)
		h_x = F.softmax(logit_x).data.squeeze()
		probs_x, idx_x = h_x.sort(0, True)
		class_x = idx_x[0]
		class_x_prob = probs_x[0]
		###############
		target_class = idx_x[1]
		return class_x, class_x_prob, probs_x, logit_x,target_class










def AdvLoss(logits, target, is_targeted, num_classes=1000, kappa=0):
	# inputs to the softmax function are called logits.
	# https://arxiv.org/pdf/1608.04644.pdf
	target_one_hot = torch.eye(num_classes).type(logits.type())[target.long()]

	# workaround here.
	# subtract large value from target class to find other max value
	# https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
	real = torch.sum(target_one_hot*logits, 1)
	other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
	kappa = torch.zeros_like(other).fill_(kappa)

	if is_targeted:
		return torch.sum(torch.max(other-real, kappa))
	return torch.sum(torch.max(real-other, kappa))
