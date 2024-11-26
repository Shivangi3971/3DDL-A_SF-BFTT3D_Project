# 3DDL-A_SF-BFTT3D_Project

Installation 


Create a python virtual environment using pip and install dependencies after activating it:

##### Use Cuda 11.1 and torch 1.8.2

```
pip install -r requirements.txt
```

If this fails, please try to install the environment given in this repository https://github.com/abie-e/BFTT3D


## Dataset

### Folder structure


```
.

├── ...
├── datasets
├── data                    
│      ├── modelnet40_ply_hdf5_2048          
│      ├── modelnet40_c         
│      └── ...                
├── checkpoint                   
│      └── ... 
└── ....
```

Please download the following ModelNet40 and ModelNet40C datasets of source data and target data with Corruption respectively for the prototype memory. 
##### [Download ModelNet40](https://drive.google.com/drive/folders/1H3UOF1268UIK3z_FkNcBZfauuDbOYLNY?usp=sharing)


Checkpoint:
[checkpoint](https://drive.google.com/drive/folders/1nOvmsCR_7SMOoUeimD8YYGxW033gCiEP?usp=sharing)  is uploaded here.

```
pointnet.pth - trained pointnet model given in BFTT3D repository.

pointnet_trained.pth - trained pointnet model trained by our group.
curvenet.t7 - trained curvenet model trained by our group.
dgcnn.t7 - trained dgcnn model trained by our group.
```

## Run 3DDL-A_SF-BFTT3D_Project files

Just run below command to create all tables given in the report:

``` 
bash run.sh
```

Reference:
1. [Backpropagation-free Network for 3D Test-time Adaptation (BFTT3D)](https://github.com/abie-e/BFTT3D)
2. [PointNet]
3. [DGCNN]
4. [Curvenet]


#### Code belongs to Group-9 (Shivangi Rai and Kunal Jangid)


