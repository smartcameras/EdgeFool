# EdgeFool

This is the official repository of [EDGEFOOL: AN ADVERSARIAL IMAGE ENHANCEMENT FILTER](https://arxiv.org/pdf/1910.12227.pdf), a work published in the Proc. of the 45th IEEE International Conference on Acoustics, Speech, and Signal Processing (<b>ICASSP</b>), Barcelona, Spain, May 4-8, 2020.<br>


<b>Example of results</b>

| Original Image | Adversarial Image | Original Image | Adversarial Image |
|---|---|---|---|
| ![Original Image](https://github.com/smartcameras/EdgeFool/blob/master/EdgeFoolExamples/ILSVRC2012_val_00000328.png) | ![Adversarial Image](https://github.com/smartcameras/EdgeFool/blob/master/EdgeFoolExamples/ILSVRC2012_val_00000328_EdgeFool.png) |![Original Image](https://github.com/smartcameras/EdgeFool/blob/master/EdgeFoolExamples/ILSVRC2012_val_00030569.png) | ![Adversarial Image](https://github.com/smartcameras/EdgeFool/blob/master/EdgeFoolExamples/ILSVRC2012_val_00030569_EdgeFool.png) |
| ![Original Image](https://github.com/smartcameras/EdgeFool/blob/master/Dataset/ILSVRC2012_val_00002437.png) | ![Adversarial Image](https://github.com/smartcameras/EdgeFool/blob/master/EdgeFoolExamples/ILSVRC2012_val_00002437_EdgeFool.png) |![Original Image](https://github.com/smartcameras/EdgeFool/blob/master/EdgeFoolExamples/Places365_val_00000702.png) | ![Adversarial Image](https://github.com/smartcameras/EdgeFool/blob/master/EdgeFoolExamples/Places365_val_00000702_EdgeFool.png) |


## Setup
1. Create [conda](https://docs.conda.io/en/latest/miniconda.html) virtual-environment
   ```
    module load python2/anaconda
    conda create --name EdgeFool python=2.7.15
   ```
2. Activate conda environment
   ```
    source activate EdgeFool
   ```
3. Download source code from GitHub
   ```
    git clone https://github.com/AliShahin/EdgeFool.git 
   ```
4. Install requirements
   ```
    pip install -r requirements.txt
   ```


## Description
The code first locates all the images in Dataset folder and then generates the enhanced adversarial images in two steps: 
1. Image smoothing with l0 smoothing filters
2. Generate the enhanced adversarial images after training of a Fully Convolutional Neural Network  


### Image Smoothing 

Image smoothing is performed with the Python implementation of [Image Smoothing via L0 Gradient Minimization](http://www.cse.cuhk.edu.hk/~leojia/papers/L0smooth_Siggraph_Asia2011.pdf) provided by [Kevin Zhang](https://github.com/kjzhang/kzhang-cs205-l0-smoothing), as follows: 

1. Go to Smoothing directory
   ```
   cd Smoothing
   ```
2. Smooth the original images
   ```
   bash script.sh
   ```
3. The l0 smoothed images will be saved in the SmoothImgs directory (within the 'root' directory) with the same name as their corresponding original images

### Generate the enhanced adversarial images

A Fully Convolutional Neural Network (FCNN) is first trained end-to-end with a multi-task loss function which includes smoothing and adversarial losses. The architecture of the FCNN is instantiated from [Fast Image Processing with Fully-Convolutional Networks](https://arxiv.org/pdf/1709.00643.pdf) implemented in PyTorch by [Wu Huikai](https://github.com/wuhuikai/DeepGuidedFilter/tree/master/ImageProcessing/DeepGuidedFilteringNetwork). We enhance the image details of the L image channel only, after conversion to the Lab colour space without changing the colours of the image. In order to do this, we provided a differentiable PyTorch implementation of RGB-to-Lab and Lab-to-RGB. The enhanced adversarial images are then generated


1. Go to Train directory
   ```
   cd Train
   ```
2. In the script.sh set the paths of
(i) directory of the original images,
(ii) directory of the smoothed images, and
(iii) classifier under attack. The current implementation supports three classifiers Resnet18, Resnet50 and Alexnet, however other classifiers can be employed by changing the lines (80-88) in train_base.py.
3. Generate the enhanced adversarial images 
   ```
   bash script.sh
   ```
4. The enhanced adversarial images are saved in the EnhancedAdvImgsfor_{classifier} (within the 'root' directory) with the same name as their corresponding original images


## Authors
* [Ali Shahin Shamsabadi](mailto:a.shahinshamsabadi@qmul.ac.uk)
* [Changjae Oh](mailto:c.oh@qmul.ac.uk)
* [Andrea Cavallaro](mailto:a.cavallaro@qmul.ac.uk)


## References
If you use our code, please cite the following paper:

      @InProceedings{shamsabadi2020edgefool,
        title = {EdgeFool: An Adversarial Image Enhancement Filter},
        author = {Shamsabadi, Ali Shahin and Oh, Changjae and Cavallaro, Andrea},
        booktitle = {Proceedings of the 45th IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
        year = {2020},
        address = {Barcelona, Spain},
        month = May
      }
## License
The content of this project itself is licensed under the [Creative Commons Non-Commercial (CC BY-NC)](https://creativecommons.org/licenses/by-nc/2.0/uk/legalcode).
