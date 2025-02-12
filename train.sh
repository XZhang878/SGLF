#!/usr/bin/env bash

#office
##a2w
#python train_image.py --test_interval 300 --epoch 9000 --use_seed True --torch_seed 893561550778924240 --torch_cuda_seed 1756436598284239 --left_weight 1 --right_weight 1 --mdd_weight 0.05 --entropic_weight 0 --log_name a2w.txt --s_dset_path ./data/office/amazon_list.txt --t_dset_path ./data/office/webcam_list.txt


#image_clef
#i2c
python train_SAR.py  