# PydMobileNet: Deep Residual Networks with Pyramid Depthwise Separable Convolution
## Dependencies
1. Mxnet >= 1.3.0. To install the mxnet, follow the instruction in https://mxnet.incubator.apache.org/install/index.html?platform=Linux|language=Python|processor=GPU
2. Numpy. By command: 'pip install numpy'
3. Gluoncv. By command: 'pip install gluoncv'

## Main results
### Network architecture (CIFAR)

| Group | Output size |Networks |
| --- | :-: | :-: |
| Image | 32x32x3 | |
| Convolution | 32x32x32 |3x3 Convolution  |
| Stage 1 | 32x32x32 | Residual block x6 |
| Stage 2 | 16x16x64 | Residual block x6 |
| Stage 3 | 8x8x128 | Residual block x6 |
| Pooling | 1x1x128 | 8x8 Global Average Pooling |
| Classifier | 1x1x10/100 | 10/100D Fully-Connected |

### CIFAR
- Number of channels: {32, 32, 64, 128, 128}

| Model | Depth | #Params | FLOPs | CIFAR-10 | CIFAR-100 |
| --- | :-: | :-: | :-: | :-: | :-: |
| ResNet-29-0.5 | 29 | 0.221M | 29M | 6.97 | 19.62 |
| | | | | | | |
| MobileNet-29-0.5 | 29 | 0.079M | 12M | 8.63 | 22.59 |
| MobileNet-29-1 | 29 | 0.142M | 22M | 7.09 | 19.40 |
| MobileNet-29-1.5 | 29 | 0.206M | 32M | 6.56 | 18.09 |
| | | | | | | |
| PydMobileNet-Add-29-0.25 | 29 | 0.060M | 10M | 9.43 | 21.96 |
| PydMobileNet-Add-29-0.5 | 29 | 0.104M | 18M | 7.29 | 20.26 |
| PydMobileNet-Add-29-0.75 | 29 | 0.148M | 26M | 6.52 | 17.95 |
| PydMobileNet-Add-29-1 | 29 | 0.193M | 34M | 6.00 | 17.54 |
| | | | | | | |
| PydMobileNet-Concat-29-0.25 | 29 | 0.092M | 14M | 7.33 | 21.04 |
| PydMobileNet-Concat-29-0.5 | 29 | 0.170M | 27M | 5.71 | 17.27 |
| PydMobileNet-Concat-29-0.75 | 29 | 0.247M | 39M | 5.68 | 16.28 |
| | | | | | | |
| | | | | | | |
| ResNet-56-0.5 | 56 | 0.435M | 60M | 5.76 | 17.60 |
| | | | | | | |
| MobileNet-56-0.5 | 56 | 0.151M | 23M | 6.75 | 18.56 |
| MobileNet-56-1 | 56 | 0.283M | 43M | 6.02 | 17.15 |
| MobileNet-56-1.5 | 56 | 0.416M | 63M | 5.29 | 16.58 |
| | | | | | | |
| PydMobileNet-Add-56-0.25 | 56 | 0.109M | 19M | 7.38 | 20.41 |
| PydMobileNet-Add-56-0.5 | 56 | 0.200M | 36M | 6.19 | 17.36 |
| PydMobileNet-Add-56-0.75 | 56 | 0.292M | 52M | 5.55 | 16.58 |
| PydMobileNet-Add-56-1 | 56 | 0.382M | 69M | 4.98 | 16.23 |
| | | | | | | |
| PydMobileNet-Concat-56-0.25 | 56 | 0.175M | 28M | 6.23 | 17.85 |
| PydMobileNet-Concat-56-0.5 | 56 | 0.332M | 53M | 5.24 | 15.67 |
| PydMobileNet-Concat-56-0.75 | 56 | 0.489M | 79M | 4.72 | 14.60 |

Notes: #Params are for CIFAR-10 dataset. For CIFAR-100: #Params + (128 + 1) * 100 - (128 + 1) * 10
### ImageNet32
- Number of channels: {64, 64, 128, 256, 1280}

| Model | Depth | #Params | Top-1 | Top-5 |
| --- | :-: | :-: | :-: | :-: |
| WRN-28-2 | 28 | 1.6M | 56.92 | 30.92 |
| WRN-28-5 | 28 | 9.5M | 45.36 | 21.36 |
| WRN-28-10 | 28 | 37.1M | 40.96 | 18.87 |
| | | | | | |
| PydMobileNet-Concat-56-0.25 | 56 | 2.23M | 48.18 | 23.70 |
| PydMobileNet-Concat-56-0.5 | 56 | 2.80M | 44.70 | 20.58 |
| PydMobileNet-Concat-56-0.75 | 56 | 3.37M | 41.99 | 18.59 |

## Run validation
- To validate all models, run 'python run_validation.py'
- To see all configuration, run 'python run_validation.py --help'
- Note: program needs internet connection to download the CIFAR-10 and CIFAR-100 dataset from mxnet website at the first running.

## Citation
Please cite these papers in your publications if it helps your research:

    @ARTICLE{thanh2018arxiv,
      author    = {Van{-}Thanh, Hoang and Kang{-}Hyun, Jo},
      title     = {PydMobileNet: Improved Version of MobileNets with Pyramid Depthwise Separable Convolution},
      journal   = {CoRR},
      volume    = {abs/1811.07083},
      year      = {2018},
      url       = {http://arxiv.org/abs/1811.07083},
      archivePrefix = {arXiv},
      eprint    = {1811.07083},
    }
