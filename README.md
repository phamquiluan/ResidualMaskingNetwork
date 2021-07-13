# Facial Expression Recognition using Residual Masking Network, in PyTorch

[![pypi package](https://img.shields.io/badge/version-v3.0.2-blue)](https://pypi.org/project/rmn)
[![circleci](https://circleci.com/gh/phamquiluan/ResidualMaskingNetwork.svg?style=shield&circle-token=3ca7f15b6bd362b625bec536d57c45fe5ef6f2c9)](https://app.circleci.com/pipelines/github/phamquiluan/ResidualMaskingNetwork)
[![style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/phamquiluan/residualmaskingnetwork)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/challenges-in-representation-learning-a/facial-expression-recognition-on-fer2013)](https://paperswithcode.com/sota/facial-expression-recognition-on-fer2013?p=challenges-in-representation-learning-a)

A [PyTorch](http://pytorch.org/) implementation of my thesis with the same name.


<p align="center">
<img width=700 src= "https://user-images.githubusercontent.com/24642166/104806861-14b29880-580d-11eb-814b-a074847e35fe.png"/>
</p>

# Inference:

## Approach 1: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZEbJ6rJuGZ6UzmOFE1XCzSgacTUJiN-H?usp=sharing)


1. Install from pip
```bash
pip install rmn

# or 

git clone git@github.com:phamquiluan/ResidualMaskingNetwork.git
cd ResidualMaskingNetwork
pip install -e .
```

2. Run infenrece by the following Python scripts
```python
from rmn import RMN
m = RMN()
m.video_demo()

# or
image = cv2.imread("some-image-path.png")
results = m.detect_emotion_for_single_frame(image)
print(results)
image = m.draw(image, results)
cv2.imwrite("output.png", image)
```

<p align="center">
<img width=500 src= "https://user-images.githubusercontent.com/24642166/117097030-d4176480-ad94-11eb-8c65-097a62ede067.png"/>
</p>



## Approach 2:
- Model file: [download](https://drive.google.com/open?id=1_6CzlKRS9ksxlo0TjqIGXMzQE4I83tE0) (this checkpoint is trained on VEMO dataset, locate it at ```./saved/checkpoints/``` directory)
- Download 2 files: [prototxt](https://drive.google.com/open?id=1ANVPx3JM4EcJVZOstV_kEO1Jcv74mBu5), and [res10_300x300_ssd](https://drive.google.com/open?id=1Iy_3I_mWGhBA63W0IK8tRrUuvr-WrGQ2) for face detection OpenCV. Locate at current directory or checking file path with ```ssd_infer.py``` file.

```Shell
python ssd_infer.py
```

<p align="center">
<img width=500 src= "https://user-images.githubusercontent.com/24642166/72135777-da244d80-33b9-11ea-90ee-706b25c0a5a9.png"/>
</p>




### Table of Contents
- <a href='#recent_update'>Recent Update</a>
- <a href='#benchmarking_fer2013'>Benchmarking on FER2013</a>
- <a href='#benchmarking_imagenet'>Benchmarking on ImageNet</a>
- <a href='#install'>Installation</a>
- <a href='#datasets'>Download datasets</a>
- <a href='#train_fer'>Training on FER2013</a>
- <a href='#train_imagenet'>Training on ImageNet</a>
- <a href='#eval'>Evaluation results</a>
- <a href='#docs'>Download dissertation and slide</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

<p id="recent_update"></p>


## Recent Update
 - [05/05/2021] Release ver 2, add colab
 - [27/02/2021] Add paper
 - [14/01/2021] Packaging Project and publish `rmn` on Pypi
 - [27/02/2020] Update Tensorboard visualizations and Overleaf source
 - [22/02/2020] Test-time augmentation implementation.
 - [21/02/2020] Imagenet training code and trained weights released.
 - [21/02/2020] Imagenet evaluation results released.
 - [10/01/2020] Checking demo stuff and training procedure works on another machine
 - [09/01/2020] First time upload

<p id="benchmarking_fer2013"></p>


## Benchmarking on FER2013

We benchmark our code thoroughly on two datasets: FER2013 and VEMO. Below are the results and trained weights:


Model | Accuracy |
---------|--------|
[VGG19](https://drive.google.com/open?id=1FPkwhmel0AiGK3UtYiWCHPi5CYkF7BRc) | 70.80
[EfficientNet\_b2b](https://drive.google.com/open?id=1pEyupTGQPoX1gj0NoJQUHnK5-mxB8NcS) | 70.80
[Googlenet](https://drive.google.com/open?id=1LvxAxDmnTuXgYoqBj41qTdCRCSzaWIJr) | 71.97
[Resnet34](https://drive.google.com/open?id=1iuTkqApioWe_IBPQ7gQHticrVlPA-xz_) | 72.42
[Inception\_v3](https://drive.google.com/open?id=17mapZKWYMdxGTrbrAbRpfgniT5onmQXO) | 72.72
[Bam\_Resnet50](https://drive.google.com/open?id=1K_gyarekwIxQMA_fEPJMApgqo3mYaM0H) | 73.14
[Densenet121](https://drive.google.com/open?id=1f8wUtQj-UatrZtCnkJFcB--X2eJS1m_N) | 73.16
[Resnet152](https://drive.google.com/open?id=1LBaHaVtu8uKiNsoTN7wl5Pg5ywh-QxRW) | 73.22
[Cbam\_Resnet50](https://drive.google.com/open?id=1i9zk8sGXiixkQGTA1txBxSuew6z_c86T) | 73.39
[ResMaskingNet](https://drive.google.com/open?id=1_ASpv0QNxknMFI75gwuVWi8FeeuMoGYy) | 74.14
[ResMaskingNet + 6](https://drive.google.com/open?id=1y28VHzJcgBpW0Qn_K0XVVd-hxG4feIHG) | 76.82



Results in VEMO dataset could be found in my thesis or slide (attached below)

<p id="benchmarking_imagenet"></p>

 
## Benchmarking on ImageNet 

We also benchmark our model on ImageNet dataset.


Model | Top-1 Accuracy | Top-5 Accuracy |
---------|--------|--------|
[Resnet34](https://drive.google.com/open?id=16lErBAk7K3WswKP0wyE9S0dNrr7AF6wd) | 72.59 | 90.92
[CBAM Resnet34](https://drive.google.com/open?id=16lErBAk7K3WswKP0wyE9S0dNrr7AF6wd) | 73.77 | 91.72
[ResidualMaskingNetwork](https://drive.google.com/open?id=1myjp4_XL8mNJlAbz0TFjYKUc7B0N64eb) | 74.16 | 91.91


<p id="install"></p>
 

## Installation
- Install [PyTorch](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository and install package [prerequisites](#prerequisites) below.
- Then download the dataset by following the [instructions](#datasets) below.


### prerequisites

* Python 3.6+
* [requirements.txt](requirements.txt)


<p id="datasets"></p>


## Datasets

- [FER2013 Dataset](https://drive.google.com/open?id=18ovcnZBsPvwXXFVAqczACe9zciO_1q6J) (locate it in ```saved/data/fer2013``` like ```saved/data/fer2013/train.csv```)
- [ImageNet 1K Dataset](http://image-net.org/download-images) (ensure it can be loaded by torchvision.datasets.Imagenet)



<p id="train_fer"></p>


## Training on FER2013

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LbDiAs5xOmhwaoKtJepaK_oVU_IkgLM8?usp=sharing)

- To train network, you need to specify model name and other hyperparameters in config file (located at configs/\*) then ensure it is loaded in main file, then run training procedure by simply run main file, for example:

```Shell
python main_fer.py  # Example for fer2013_config.json file
```

- The best checkpoints will chosen at term of best validation accuracy, located at ```saved/checkpoints```
- The TensorBoard training logs are located at ```saved/logs```, to open it, use ```tensorboard --logdir saved/logs/```


<p align="center">
<img width=900 src= "https://user-images.githubusercontent.com/24642166/75408653-fddf2b00-5948-11ea-981f-3d95478d5708.png"/>
</p>

- By default, it will train `alexnet` model, you can switch to another model by edit `configs/fer2013\_config.json` file (to `resnet18` or `cbam\_resnet50` or my network `resmasking\_dropout1`.


<p id="train_imagenet"></p>


## Training on Imagenet dataset

To perform training resnet34 on 4 V100 GPUs on a single machine:

```Shell
python ./main_imagenet.py -a resnet34 --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 
```

<p id="eval"></p>


## Evaluation

For student, who takes care of font family of confusion matrix and would like to write things in LaTeX, below is an example for generating a striking confusion matrix. 

(Read [this article](https://matplotlib.org/3.1.1/tutorials/text/usetex.html) for more information, there will be some bugs if you blindly run the code without reading).

```Shell
python cm_cbam.py 
```

<p align="center">
<img width=600 src= "https://user-images.githubusercontent.com/24642166/104806916-81c62e00-580d-11eb-8dcd-c5759e5d48ae.png"/>
</p>


## Ensemble method

I used no-weighted sum avarage ensemble method to fusing 7 different models together, to reproduce results, you need to do some steps:

1. Download all needed trained weights and located on ```./saved/checkpoints/``` directory.  Link to download can be found on Benchmarking section.
2. Edit file ```gen_results``` and run it to generate result offline for **each** model.
3. Run ```gen_ensemble.py``` file to generate accuracy for example methods.



<p id="docs"></p>


## Dissertation and Slide
- [Dissertation PDF (in Vietnamese)](https://drive.google.com/open?id=1HxqvQSZRf-3ashGtZ5o9OABdhmdjS64a)
- [Dissertation Overleaf Source](https://www.overleaf.com/read/qdyhnzjmbscd)
- [Presentation slide PDF (in English) with full appendix](https://drive.google.com/open?id=19zweCDX8Vz4jgwJ6cBWr5x_iQPvahsQg)
- [Presentation slide Overleaf Source](https://www.overleaf.com/read/vxdhjvhvgwdn)
- [Paper](docs/paper.pdf)


## TODO
We have accumulated the following to-do list, which we hope to complete in the near future
- Still to come:
  * [x] Upload all models and training code.
  * [x] Test time augmentation.
  * [x] GPU-Parallel.
  * [x] Pretrained model.
  * [x] Demo and inference code.
  * [x] Imagenet trained and pretrained weights.
  * [ ] GradCAM visualization and Pooling method for visualize activations.
  * [ ] Centerloss Visualizations.
  

<p id="author"></p>


## Authors

* [**Luan Pham**](https://github.com/phamquiluan)
* [**Tuan Anh Tran**](https://github.com/phamquiluan)

***Note:*** Unfortunately, I am currently join a full-time job and research on another topic, so I'll do my best to keep things up to date, but no guarantees.  That being said, thanks to everyone for your continued help and feedback as it is really appreciated. I will try to address everything as soon as possible.


<p id="references"></p>


## References
- Same as in dissertation.


## Citation

L. Pham, H. Vu, T. A. Tran, "Facial Expression Recognition Using Residual Masking Network", IEEE 25th International Conference on Pattern Recognition, 2020, 4513-4519. Milan -Italia.


```
@inproceedings{luanresmaskingnet2020,
  title={Facial Expression Recognition using Residual Masking Network},
  author={Luan, Pham and Huynh, Vu and Tuan Anh, Tran},
  booktitle={IEEE 25th International Conference on Pattern Recognition},
  pages={4513--4519},
  year={2020}
}
```

