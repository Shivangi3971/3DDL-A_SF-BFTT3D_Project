import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import models as models
import numpy as np
from datasets.data_mn40 import ModelNet40,ModelNet40C
from utils.TCA import TCA
from utils.JDA import JDA
from utils.setup import *
from utils_nn import * 
from models import Point_NN, PointNet, DGCNN, CurveNet, build
import cv2
import torch.jit
from copy import deepcopy
import pc_utils as pc_utl
    
    
def model_builder(config):
    model = build.build_model_from_cfg(config)
    return model
    
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_bz', type=int, default=24)
    parser.add_argument('--points', type=int, default=1024)
    parser.add_argument('--stages', type=int, default=3)
    parser.add_argument('--dim', type=int, default=72)
    parser.add_argument('--k', type=int, default=120)
    parser.add_argument('--alpha', type=int, default=1000)
    parser.add_argument('--beta', type=int, default=100)
    parser.add_argument('--gamma', type=int, default=205)
    parser.add_argument('--pth', type=str, default="pointnet")
    args = parser.parse_args()
    return args
 
       
@torch.no_grad()
def main():

    print('==> Loading args..')
    args = get_arguments()
    print(args)
    torch.manual_seed(666)

    print('==> Preparing model..')
    point_nn = Point_NN(input_points=args.points, num_stages=args.stages,
                        embed_dim=args.dim, k_neighbors=args.k,
                        alpha=args.alpha, beta=args.beta).cuda()
    point_nn.eval()
    print('==> Preparing data..')
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.points), num_workers=8, batch_size=args.load_bz, shuffle=False, drop_last=False)
                                             
    print('==> Constructing Memory Bank..')
    feature_memory, label_memory = [], []

    if True:
        device = 'cuda'
        if True:    
            # load pointnet, curvnet, dgcnn, pct model        
            print('==> Loading pretrained adaptation model')
            source_model_path = f"checkpoint/{args.pth}.pth"
            if args.pth == "pointnet":
                source_model = PointNet("mn40")
            elif args.pth == "dgcnn":
                source_model = DGCNN("mn40")
            elif args.pth == "curvenet":
                source_model = CurveNet("mn40")
            checkpoint = torch.load(source_model_path)
            # checkpoint key edit
            for key in list(checkpoint.keys()):
                checkpoint[key.replace('module.', 'module.model.')] = checkpoint.pop(key)
            try:
                try:
                    source_model.load_state_dict(checkpoint)
                except:
                    source_model.load_state_dict(checkpoint['model_state'])
            except:
                print("WARNING: using dataparallel to load data")
                source_model = nn.DataParallel(source_model)
                try:
                    source_model.load_state_dict(checkpoint)
                except:
                    source_model.load_state_dict(checkpoint['model_state'])
            source_model.to(device)

    # # coral_project, tca_project, gfk_project, jda_project= CORAL(), TCA(), GFK(), JDA()
        
        
    # feature_memory /= feature_memory.norm(dim=-1, keepdim=True)
    print('==> Saving Test Point Cloud Features..')

    # corrupted modelnet40c and iterate through all corrutptions
    error_list, error_list_source, error_list_mixed, error_list_LAME = [], [], [], []
    
    test_c = ["lidar"]
    test_corruptions = ['uniform','gaussian','background','impulse', 'upsampling', 'distortion_rbf',
                        'distortion_rbf_inv', 'density', 'density_inc', 'shear', 'rotation', 'cutout', 'distortion', 'occlusion', 'lidar']

    # test_corruptions = ['background']
    

    for i in test_corruptions:
        test_loader = DataLoader(ModelNet40C(split='test', corruption=i), num_workers=8, batch_size=args.load_bz, shuffle=False, drop_last=True)

        tta_model = setup_BFTT3D(source_model)
        tta_model.eval()

        # label_memory= []
        feature_memory = []
        
        label_domain_list, test_features = [], []
        logits_domain_list, logits_source_list, = [], []

        for points_cpu, labels in tqdm(test_loader):
            points = points_cpu.cuda()
            points = points.permute(0, 2, 1)

            logits_source = source_model(points).detach()
            logits_source_list.append(logits_source)

            # Pass through the Non-Parametric Encoder
            test_features.append(point_nn(points))
            labels = labels.cuda()
            label_domain_list.append(labels)
            
            
            feature_memory.append(point_nn(points))
          
        # Feature Memory
        feature_memory = torch.cat(feature_memory, dim=0)
        feature_memory /= feature_memory.norm(dim=-1, keepdim=True)


        
        feature_memory_prototypes = protoype_cal_kmeans_refined_herding(feature_memory, num_clusters=40, herding= True)
       
        # Refine prototypes
        feature_memory = nearest_neighbor_refinement(feature_memory, feature_memory_prototypes, k=40)
        
        feature_memory.to(device)
        
        
        
        

        # Feature
        test_features = torch.cat(test_features, dim=0)
        test_features /= test_features.norm(dim=-1, keepdim=True)

        logits_source_list = torch.cat(logits_source_list)
        label_domain_list = torch.cat(label_domain_list)
        
        #  sperate batch test
        bz = 128
        n_batches = math.ceil(test_features.shape[0] / bz)

        for counter in range(n_batches):
            test_feature_curr = test_features[counter*bz : (counter+1)*bz]
            logits_source_curr = logits_source_list[counter*bz : (counter+1)*bz]
            # Subspace learning
            if True:
                feature_memory_aligned, test_features_aligned = feature_shift(feature_memory,test_feature_curr)
                feature_memory_aligned = feature_memory_aligned.permute(1, 0)
            else:
                feature_memory_aligned = feature_memory.permute(1, 0)
                test_features_aligned = test_feature_curr
                feature_memory_aligned = feature_memory_aligned.permute(1, 0)

            Sim = test_features_aligned @ feature_memory_aligned
            # print('==> Starting Predicition..')
            logits = (-args.gamma * (1 - Sim)).exp() # @ label_memory
            # Generate logits based on similarity
    
            s_entropy = softmax_entropy(logits_source_curr).mean(0)
            nn_entropy = softmax_entropy(logits).mean(0)

            # Label Integrate
            p = 1 - (s_entropy / (s_entropy + nn_entropy))
            f_logits = (1 - p) * logits + p * logits_source_curr
            logits_domain_list.append(f_logits)   
        logits_domain_list = torch.cat(logits_domain_list)
        domain_acc = cls_acc(logits_domain_list, label_domain_list)
        source_acc = cls_acc(logits_source_list, label_domain_list)
        error_list_mixed.append(100 - domain_acc)
        error_list_source.append(100 - source_acc)
        print(f"Source's {i} classification error: {100 - source_acc:.2f}.")
        print(f"BFTT3D's {i} classification error: {100 - domain_acc:.2f}.")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        
    # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(f"Mean classification error source: {sum(error_list_source) / len(error_list_source):.2f}.")
    print(f"Mean classification error mixed: {sum(error_list_mixed) / len(error_list_mixed):.2f}.")




def nearest_neighbor_refinement(features, prototypes, k=5):
    """
    Refine prototypes by reassigning them based on the nearest neighbors.

    Args:
        features (torch.Tensor): Tensor of shape (num_samples, feature_dim), representing the feature space.
        prototypes (torch.Tensor): Tensor of shape (num_clusters, feature_dim), representing initial prototypes.
        k (int): Number of nearest neighbors to consider for refinement.

    Returns:
        torch.Tensor: Refined prototypes of shape (num_clusters, feature_dim).
    """
    # Calculate pairwise distances between features and prototypes
    distances = torch.cdist(features, prototypes)  # Shape: (num_samples, num_clusters)
    
    # Find the top-k nearest neighbors for each prototype
    nearest_indices = distances.topk(k, largest=False).indices  # Shape: (num_samples, k)
    
    # Refine each prototype based on the mean of its nearest neighbors
    refined_prototypes = []
    for idx in range(prototypes.size(0)):  # Iterate over each prototype
        cluster_features = features[nearest_indices[:, idx]]  # Extract nearest neighbors for the cluster
        refined_prototypes.append(cluster_features.mean(dim=0))  # Compute the mean of nearest neighbors
    
    # Stack the refined prototypes into a tensor
    return torch.stack(refined_prototypes)






def protoype_cal_kmeans_refined_herding(features, num_clusters, herding=True):
    features_cpu = features.cpu().numpy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(features_cpu)
    prototypes = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(features.device)

    if herding:
        cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.long).to(features.device)
        refined_prototypes = []
        for cluster_id in range(num_clusters):
            cluster_indices = (cluster_labels == cluster_id).nonzero(as_tuple=True)[0]
            cluster_features = features[cluster_indices]

            if cluster_features.size(0) == 0:  # Handle empty clusters
                refined_prototypes.append(prototypes[cluster_id].unsqueeze(0))
                continue

            # Apply herding for refinement
            mean_feature = cluster_features.mean(dim=0, keepdim=True)
            weights = cluster_features.norm(dim=1, keepdim=True)  # Weight by norms
            weighted_mean = (cluster_features * weights).sum(dim=0) / weights.sum()
            refined_prototypes.append(weighted_mean.unsqueeze(0))
        
        prototypes = torch.cat(refined_prototypes, dim=0)

    return prototypes



# def protoype_cal_kmeans_refined(features, num_clusters, herding=True):
#     features_cpu = features.cpu().numpy()
#     kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(features_cpu)
#     prototypes = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(features.device)

#     if herding:
#         cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.long).to(features.device)
#         refined_prototypes = []
#         for cluster_id in range(num_clusters):
#             cluster_indices = (cluster_labels == cluster_id).nonzero(as_tuple=True)[0]
#             cluster_features = features[cluster_indices]
#             weights = cluster_features.norm(dim=1, keepdim=True)
#             weighted_mean = (cluster_features * weights).sum(dim=0) / weights.sum()
#             refined_prototypes.append(weighted_mean.unsqueeze(0))
#         prototypes = torch.cat(refined_prototypes, dim=0)

#     return prototypes




# def protoype_cal_kmeans(features, num_clusters, herding=True):
#     # Perform clustering on the feature space
#     features_cpu = features.cpu().numpy()
#     kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(features_cpu)
    
#     # Cluster centroids as prototypes
#     prototypes = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(features.device)
    
#     if herding:
#         # Use the herding algorithm to refine prototypes
#         cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.long).to(features.device)
#         refined_prototypes = []
#         for cluster_id in range(num_clusters):
#             cluster_indices = (cluster_labels == cluster_id).nonzero(as_tuple=True)[0]
#             cluster_features = features[cluster_indices]
#             mean_feature = cluster_features.mean(dim=0, keepdim=True)
#             distances = (cluster_features - mean_feature).norm(dim=1)
#             top_indices = distances.topk(max(1, cluster_features.size(0) // 4), largest=False).indices
#             refined_prototypes.append(cluster_features[top_indices].mean(dim=0, keepdim=True))
#         prototypes = torch.cat(refined_prototypes, dim=0)
    
#     return prototypes




def generate_logits_from_similarity(similarity, gamma=1.0):
    """
    Convert similarity to logits by scaling with an exponential function.
    
    Args:
        similarity (torch.Tensor): The similarity matrix (num_samples, num_prototypes).
        gamma (float): Scaling factor (higher values make the model more confident).
    
    Returns:
        torch.Tensor: The logits (num_samples, num_prototypes).
    """
    logits = torch.exp(gamma * similarity)  # Apply exponential scaling
    return logits


def load_model_state(model, model_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)

def feature_shift(feature_memory, test_features):
    # feature_memory_shift = coral_project.fit(feature_memory.cpu().numpy(), test_features.cpu().numpy())
    # feature_memory_shift = torch.tensor(feature_memory_shift, dtype=torch.float).cuda()
    # feature_memory_shift = feature_memory_shift.permute(1, 0)
    # feature_memory_shift, test_features_shift = jda_project.fit(feature_memory.cpu().numpy(), test_features.cpu().numpy())
    
    tca_project = TCA()
    feature_memory_shift, test_features_shift = tca_project.fit(feature_memory.cpu().numpy(), test_features.cpu().numpy())
    
    feature_memory_shift = torch.tensor(feature_memory_shift, dtype=torch.float).cuda()
    test_features_shift = torch.tensor(test_features_shift, dtype=torch.float).cuda()
    
    feature_memory_shift /= feature_memory_shift.norm(dim=-1, keepdim=True)
    test_features_shift /= test_features_shift.norm(dim=-1, keepdim=True)
    return feature_memory_shift, test_features_shift 

def copy_model_and_optimizer(model):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    return model_state
    
@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def mean_square_distance_batch(batch_vector1, vector2):

    
    squared_differences = (batch_vector1 - vector2)**2
    msd = torch.mean(squared_differences, dim=1)
    return msd
    
def protoype_cal(features, label, class_num, herding = True):
    
    label = label.to(torch.int64)

    prototye_list, label_memory_list = [], []
    for i in range(class_num):
        idx = torch.squeeze((label==i))
        
        # if idx.sum() == 0:
        #     print(f"Warning: No samples found for class {i}. Skipping.")
        #     continue

        if idx.sum().item()== 0: #idx.shape[0] == 0:  # Skip empty classes
            print(f"Warning: No samples found for class {i}. Skipping.")
     
        mean_emb = features[idx].mean(0).unsqueeze(0)
        if herding:
            class_emb = features[idx]
            # print("class_emb.shape[0]: ",class_emb.shape[0])
            k = int(class_emb.shape[0]/4)
            # k = max(1, min(int(class_emb.shape[0] / 4), class_emb.shape[0]))
            _, closese_emb_index = torch.topk(mean_square_distance_batch(class_emb, torch.squeeze(mean_emb)), k, largest=False)
            prototye_list.append(class_emb[closese_emb_index])
            label_memory_list.append(torch.ones(k)*i)
        else:
            prototye_list.append(mean_emb)
            label_memory_list.append((torch.tensor(i).unsqueeze(0)))
    prototye_list = torch.cat(prototye_list, dim=0)
    # Label Memory
    label_memory_list = torch.cat(label_memory_list, dim=0).type(torch.LongTensor)
    label_memory_list = F.one_hot(label_memory_list).squeeze().float()
    return prototye_list, label_memory_list

if __name__ == '__main__':
    main()
