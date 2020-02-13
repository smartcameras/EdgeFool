#!/bin/bash


#PATH_IMGS=(/jmain01/home/JAD007/txk02/axs14-txk02/ICASSP19/dataset/2Ali/original/Original/Dataset_resized/test/)
PATH_IMGS=(../Dataset/)
clear
python -W ignore L0_serial.py --path_imgs=$PATH_IMGS

