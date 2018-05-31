# OctSceneScan

Welcome to my side project for 3D Scene handling. Recently 3D conv have been dominating many tasks such as medical imaging and 3D object classification. However, this are generally very limited in input dimensions because the data grows to the 3rd power with resolution. Scan complete handles this by creating 8 different nets for 3 levels (24 total) where each is trained separately. This causes the net to be very big and does not permit end to end training. 

There are a few papers that handle this using the sparsity in 3D such as octnet. Genereally these require knowledge of the sparsity in advance (example 3D point classification) and is not computed on the fly as is needed for signed distance function completion where the location of the level set is not known in advance (it’s the target of the net)
Malik used dynamic computation on graphs (in Torch) for 3D object generation from images. This work is my exploration, trying to use PyTorch simple dynamic graphs to denoise/ complete SDF efficiently.

## Method
Goal is to denois the input TSDF. As a hello world I generate a TSDF from random ellipsoids in a 32^3 grid. This is small enough to run a full 3D conv net (no hyrarchy, no dynamic branching…)

## Hyrarchal architecture
TBD

## occupancy classification
Every voxel block in every layer should classify is the surface of the scene intersects this voxel. It should use the downsampled input TSDF and the previouse feature vectures. If P(surface) is small enough finner layers should not be called. 

TBD

## References
TODO format this shit

@article{Wang2017,
abstract = {Fig. 1. An illustration of our octree-based convolutional neural network (O-CNN). Our method represents the input shape with an octree and feeds the averaged normal vectors stored in the finest leaf octants to the CNN as input. All the CNN operations are efficiently executed on the GPU and the resulting features are stored in the octree structure. Numbers inside the blue dashed square denote the depth of the octants involved in computation. We present O-CNN, an Octree-based Convolutional Neural Network (CNN) for 3D shape analysis. Built upon the octree representation of 3D shapes, our method takes the average normal vectors of a 3D model sampled in the finest leaf octants as input and performs 3D CNN operations on the octants occupied by the 3D shape surface. We design a novel octree data structure to efficiently store the octant information and CNN features into the graphics memory and execute the entire O-CNN training and evaluation on the GPU. O-CNN supports various CNN structures and works for 3D shapes in different representations. By restraining the computations on the octants occupied by 3D surfaces, the memory and computational costs of the O-CNN grow quadratically as the depth of the octree increases, which makes the 3D CNN feasible for high-resolution 3D models. We compare the performance of the O-CNN with other existing 3D CNN solutions and demonstrate the efficiency and efficacy of O-CNN in three shape analysis tasks, including object classification, shape retrieval, and shape segmentation.},
author = {Wang, Peng-shuai and Wang, Peng-Shuai and Liu, Yang and Guo, Yu-Xiao and Sun, Chun-Yu and Tong, Xin},
doi = {10.1145/3072959.3073608},
file = {::},
journal = {ACM Trans. Graph. Article},
number = {11},
title = {{O-CNN: Octree-based Convolutional Neural Networks for 3D Shape Analysis}},
volume = {36},
year = {2017}
}
@article{Riegler,
abstract = {We present OctNet, a representation for deep learning with sparse 3D data. In contrast to existing models, our rep-resentation enables 3D convolutional networks which are both deep and high resolution. Towards this goal, we ex-ploit the sparsity in the input data to hierarchically parti-tion the space using a set of unbalanced octrees where each leaf node stores a pooled feature representation. This al-lows to focus memory allocation and computation to the relevant dense regions and enables deeper networks without compromising resolution. We demonstrate the utility of our OctNet representation by analyzing the impact of resolution on several 3D tasks including 3D object classification, ori-entation estimation and point cloud labeling.},
author = {Riegler, Gernot and {Osman Ulusoy}, Ali and Geiger, Andreas},
file = {::},
title = {{OctNet: Learning Deep 3D Representations at High Resolutions}},
url = {https://arxiv.org/pdf/1611.05009.pdf}
}
@article{Dai,
abstract = {3D scans of indoor environments suffer from sensor occlusions, leaving 3D reconstructions with highly incomplete 3D geometry (left). We propose a novel data-driven approach based on fully-convolutional neural networks that transforms incomplete signed distance functions (SDFs) into complete meshes at unprecedented spatial extents (middle). In addition to scene completion, our approach infers semantic class labels even for previously missing geometry (right). Our approach outperforms existing approaches both in terms of completion and semantic labeling accuracy by a significant margin. Abstract We introduce ScanComplete, a novel data-driven ap-proach for taking an incomplete 3D scan of a scene as input and predicting a complete 3D model along with per-voxel semantic labels. The key contribution of our method is its ability to handle large scenes with varying spatial extent, managing the cubic growth in data size as scene size in-creases. To this end, we devise a fully-convolutional gen-erative 3D CNN model whose filter kernels are invariant to the overall scene size. The model can be trained on scene subvolumes but deployed on arbitrarily large scenes at test time. In addition, we propose a coarse-to-fine inference strategy in order to produce high-resolution output while also leveraging large input context sizes. In an extensive series of experiments, we carefully evaluate different model design choices, considering both deterministic and proba-bilistic models for completion and semantic inference. Our results show that we outperform other methods not only in the size of the environments handled and processing effi-ciency, but also with regard to completion quality and se-mantic segmentation performance by a significant margin.},
author = {Dai, Angela and Ritchie, Daniel and Bokeloh, Martin and Reed, Scott and Sturm, J{\"{u}}rgen and Nie{\ss}ner, Matthias},
file = {::},
title = {{ScanComplete: Large-Scale Scene Completion and Semantic Segmentation for 3D Scans}}
}
@article{Hane,
abstract = {—Recently, Convolutional Neural Networks have shown promising results for 3D geometry prediction. They can make predictions from very little input data such as a single color image. A major limitation of such approaches is that they only predict a coarse resolution voxel grid, which does not capture the surface of the objects well. We propose a general framework, called hierarchical surface prediction (HSP), which facilitates prediction of high resolution voxel grids. The main insight is that it is sufficient to predict high resolution voxels around the predicted surfaces. The exterior and interior of the objects can be represented with coarse resolution voxels. Our approach is not dependent on a specific input type. We show results for geometry prediction from color images, depth images and shape completion from partial voxel grids. Our analysis shows that our high resolution predictions are more accurate than low resolution predictions.},
author = {H{\"{a}}ne, Christian and Tulsiani, Shubham and Malik, Jitendra},
file = {::},
title = {{Hierarchical Surface Prediction for 3D Object Reconstruction}}
}
@article{Facebook,
abstract = {Convolutional networks are the de-facto standard for an-alyzing spatio-temporal data such as images, videos, and 3D shapes. Whilst some of this data is naturally dense (e.g., photos), many other data sources are inherently sparse. Ex-amples include 3D point clouds that were obtained using a LiDAR scanner or RGB-D camera. Standard " dense " implementations of convolutional networks are very ineffi-cient when applied on such sparse data. We introduce new sparse convolutional operations that are designed to pro-cess spatially-sparse data more efficiently, and use them to develop spatially-sparse convolutional networks. We demonstrate the strong performance of the resulting mod-els, called submanifold sparse convolutional networks (SS-CNs), on two tasks involving semantic segmentation of 3D point clouds. In particular, our models outperform all prior state-of-the-art on the test set of a recent semantic segmen-tation competition.},
author = {Facebook, Benjamin Graham and Research, Ai and Engelcke, Martin and {Van Der Maaten}, Laurens},
file = {::},
title = {{3D Semantic Segmentation with Submanifold Sparse Convolutional Networks}}
}


 
