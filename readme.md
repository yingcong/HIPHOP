# HIPHOP

## Overview

This is the matlab implementation of the HIPHOP feature. HIPHOP feature is a person re-identification feature propopsed in [”Person Re-Identification by Camera Correlation Aware Feature Augmentation"][http://ieeexplore.ieee.org/document/7849147/]. Note that this implementation requires [caffe][http://caffe.berkeleyvision.org] and its matlab interface.

## Files

| Files              | Description                              |
| ------------------ | ---------------------------------------- |
| demo.m             | The entrance script. Run it for demonstration. |
| HIPHOP.m           | The HIPHOP feature extraction.           |
| Alexnet.proptotxt  | The Alexnet protext file. Can be downloaded [here][https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet]. |
| Alexnet.caffemodel | The weights of the Alexnet model.        |

## How to use

"demo.m" is the entrance script. Please run it with matlab to see the pipeline of our system. You can use "matlabpool" for parallel computing.

Note that if the dimension is too large for computation, you can use PCA for dimensional reduction.

## Citation

Ying-Cong Chen, Xiatian Zhu,Wei-Shi Zheng, and Jian-Huang Lai. Person Re-Identification by Camera Correlation Aware Feature Augmentation. IEEETransactions on Pattern Analysis and Machine Intelligence (DOI: 10.1109/TPAMI.2017.2666805)