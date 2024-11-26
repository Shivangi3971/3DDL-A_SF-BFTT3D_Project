#!/bin/bash



# Reproduce results from given pre-trained source model, pointnet
python3 run_BFTT3D_mn40.py --pth pointnet



# -------------------------------------------------------------------------------------------------



# Reproduce results after trained from scratch of source models pointnet, dgcnn, curvenet on modelnet40

python3 run_BFTT3D_mn40.py --pth pointnet_trained

python3 run_BFTT3D_mn40_dgcnn.py --pth dgcnn

python3 run_BFTT3D_mn40_curvenet.py --pth curvenet



# -------------------------------------------------------------------------------------------------


# Source Free BFTT3D with prototypes using target labels
python3 run_SF-BFTT3D_TL.py --pth pointnet


# -------------------------------------------------------------------------------------------------


# Source Free BFTT3D through entropy-based approaches

#Using low entropy features from source model (pointnet) to create prototypes
python3 proposed_prototype_from_source_model.py --pth pointnet

#Using low entropy features from non-parametric network (point-nn) to create prototypes
python3 proposed_prototype_from_non-parametric.py --pth pointnet


# -------------------------------------------------------------------------------------------------


# Source Free BFTT3D through pseudo-labeling approaches

python3 run_SF-BFTT3D_PL_herd_true.py --pth pointnet
python3 run_SF-BFTT3D_PL_herd_false.py --pth pointnet


# -------------------------------------------------------------------------------------------------


# Source Free BFTT3D through clustering-based approaches

#K-Means Clustering
python3 run_SF-BFTT3D_kmeans_clustering.py --pth pointnet

#Spectral Clustering
python3 run_SF-BFTT3D_spectral_clustering.py --pth pointnet

#FFT based high-entropy  Clustering
python3 run_SF-BFTT3D_fft-based-kmeans_clustering.py --pth pointnet


