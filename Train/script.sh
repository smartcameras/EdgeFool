#!/bin/bash

clear

# Classifier that is under attack
ADVMODEl=(resnet50)

# Path to the directory that contains images
PATH_IMGS=(../Dataset/)

# Path to the directory that contains smoothed images
PATH_SMOOTHS=(../SmoothImgs/)

for adv_model in "${ADVMODEl[@]}"
do
		        echo Attacking $adv_model via EdgeFool
		        python -W ignore train_hr.py --adv_model=$adv_model --path_imgs=$PATH_IMGS --path_smooth=$PATH_SMOOTHS
done
