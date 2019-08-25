# An Internal Learning Approach to Video Inpainting
<strong>Haotian zhang</strong>, Long Mai, Ning Xu, Zhaowen Wang, John Collomosse, Hailin Jin 

International Conference on Computer Vision (ICCV) 2019. 

[Paper](#) | [Youtube](https://www.youtube.com/watch?v=_tVhCWGN7s4)
<img src="https://github.com/Haotianz94/IL_video_inpainting/blob/master/img/rollerblade.gif"/>

## Install
The code has been tested on pytorch 1.0.0 with python 3.5 and cuda 9.0. Please refer to [requirements.txt](https://github.com/Haotianz94/IL_video_inpainting/blob/master/requirements.txt) fore details. Alternatively, you can build a docker image using provided [Dockerfile](https://github.com/Haotianz94/IL_video_inpainting/blob/master/Dockerfile).

<strong>Warning!</strong> We have noticed that the optimization may not converge on some GPUs when using pytorch==0.4.0. We have observed the issue on Titan V and Tesla V100. Therefore, we highly recommend upgrading your pytorch version above 1.0.0 to avoid the issue if you are training on these GPUs. 



## Usage
We provide two ways to test our video inpainting approach. Please first download the demo data from [here](https://drive.google.com/open?id=1MJDCjj1aIUbW0OK9UnewhXlkKX9zllQd) into `data/` and download the pretrained model weights for PWC-Net from [here](https://drive.google.com/open?id=1vyoQFBz--DEkUq-0gucbWYrVbfYTOZfz) into `pretrained_models/`. (The model was originally trained by Simon Niklaus from [here](https://github.com/sniklaus/pytorch-pwc)).

* To run our demo, please run the following command:
```
python3 train.py --train_mode DIP-Vid-Flow --video_path data/bmx-trees.avi --mask_path data/bmx-trees_mask.avi --resize 192 384 --res_dir result/DIP_Vid_Flow
```

* Alternatively, you can run through our demo step by step using the provided jupyter notebook [demo.ipynb](https://github.com/Haotianz94/IL_video_inpainting/blob/master/demo.ipynb)


## Citation
```
```

## References
The implementation of our network architecture is mostly borrowed from the Deep Image Prior [repo](https://github.com/DmitryUlyanov/deep-image-prior). The implementation of the PWC-Net is borrowed from this [repo](https://github.com/sniklaus/pytorch-pwc). Should you be making use of this work, please make sure to adhere to the licensing terms of the original authors.
