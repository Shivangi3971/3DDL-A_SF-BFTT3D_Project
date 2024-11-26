import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import math
from datasets.data_mn40 import ModelNet40C
from utils.TCA import TCA
# from models import PointNet, DGCNN, CurveNet
from models import Point_NN, PointNet, DGCNN, CurveNet, build, Point_PN_scan
from utils_nn import * 

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
    args = get_arguments()
    torch.manual_seed(666)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    print('==> Preparing model..')
    point_nn = Point_NN(input_points=args.points, num_stages=args.stages,
                        embed_dim=args.dim, k_neighbors=args.k,
                        alpha=args.alpha, beta=args.beta).cuda()
    point_nn.eval()

    if True:    
        # load pointnet, curvnet, dgcnn, pct model        
        print('==> Loading pretrained adaptation model')
        source_model_path = f"checkpoint/{args.pth}.pth"
        if args.pth == "pointnet":
            source_model = PointNet("mn40")
            # source_mode = Point_PN_scan("mn40")
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
    source_model.eval()

    
    # corrupted modelnet40c and iterate through all corrutptions
    error_list, error_list_source, error_list_mixed, error_list_LAME = [], [], [], []
    
    test_c = ["lidar"]
    test_corruptions = ['uniform','gaussian','background','impulse', 'upsampling', 'distortion_rbf',
                        'distortion_rbf_inv', 'density', 'density_inc', 'shear', 'rotation', 'cutout', 'distortion', 'occlusion', 'lidar']
  

    for i in test_corruptions:
        test_loader = DataLoader(ModelNet40C(split='test', corruption=i), num_workers=8, batch_size=args.load_bz, shuffle=False, drop_last=True)
            
        # Separate high- and low-entropy features
        low_entropy_features, high_entropy_features, label_memory, high_label_memory = [], [], [], []

        feature_memory = []
        logits_domain_list, logits_source_list, = [], []
        label_domain_list = []
        logits_nn_list =[]
        
        nn_label_memory = []
        
        th  = 2
        for points, labels in tqdm(test_loader):
            points = points.cuda()
            points = points.permute(0, 2, 1)
            
            features_nn = point_nn(points).detach()
            
            # Define classifier layer separately in main
            classifier_layer = nn.Linear(features_nn.shape[1] , 40).cuda()
            
            # Pass extracted features through the classifier layer to get logits
            logits_nn = classifier_layer(features_nn)
        
            logits_nn_list.append(logits_nn)
            
            labels = labels.cuda()
            label_domain_list.append(labels)
            nn_label_memory.append(labels)

            # Get logits and calculate entropy
            logits_source = source_model(points).detach()
            # logits = point_nn(points).detach()
            entropy = softmax_entropy(logits_source)

            
            low_entropy_idx = entropy < entropy.mean()
            high_entropy_idx = ~low_entropy_idx
            
            pseudo_labels = nn.Softmax(dim=1)(logits_source).argmax(dim=1) #logits_source.argmax(dim=1)  # The predicted class is the one with the max logit
            labels =pseudo_labels
            
            
            low_entropy_features.append(logits_source[low_entropy_idx])
            high_entropy_features.append(logits_source[high_entropy_idx])
            label_memory.append(labels[low_entropy_idx].cuda())
            high_label_memory.append(labels[high_entropy_idx].cuda())

            feature_memory.append(logits_source)
            
         
            logits_source_list.append(logits_source)
            
            
        
        # # Feature Memory
        feature_memory = torch.cat(feature_memory, dim=0)
        feature_memory /= feature_memory.norm(dim=-1, keepdim=True)


        label_memory_ys = torch.cat(label_memory)#, dim=0)
        label_memory = F.one_hot(label_memory_ys, num_classes=feature_memory.shape[1]).float()
        
        low_label_memory = label_memory
        
        low_entropy_features = torch.cat(low_entropy_features, dim=0)
        low_entropy_features /= low_entropy_features.norm(dim=-1, keepdim=True)
     
        label_memory_ys = label_memory_ys.unsqueeze(-1)
        
        prototypes, label_memory = prototype_cal(low_entropy_features.cuda(), label_memory_ys.cuda(), feature_memory.shape[1])
        

        high_label_memory_ys = torch.cat(high_label_memory, dim=0)
        high_label_memory = F.one_hot(high_label_memory_ys, num_classes=feature_memory.shape[1]).float()

        
        # Align High-Entropy Features
        high_entropy_features = torch.cat(high_entropy_features)
        high_entropy_features /= high_entropy_features.norm(dim=-1, keepdim=True)


        # Align Non-parametric Features
        logits_nn_list = torch.cat(logits_nn_list)
        logits_nn_list /= logits_nn_list.norm(dim=-1, keepdim=True)
        
        
        nn_label_memory_ys = torch.cat(nn_label_memory, dim=0)
        nn_label_memory = F.one_hot(nn_label_memory_ys, num_classes=feature_memory.shape[1]).float()
            
        
        logits_source_list = torch.cat(logits_source_list)
        label_domain_list = torch.cat(label_domain_list)

        
        
        #  sperate batch test
        bz = 1
        n_batches = math.ceil(high_entropy_features.shape[0] / bz)

        for counter in range(n_batches):
            test_feature_curr = high_entropy_features[counter*bz : (counter+1)*bz]
            logits_nn_curr = logits_source_list[counter*bz : (counter+1)*bz]
            
            test_feature_curr_nn = logits_nn_list[counter*bz : (counter+1)*bz]
            
            # Subspace learning
            if True:
                #alignment of high entropy features (test_feature_curr) using prototypes build from low entropy features for source model
                feature_memory_aligned, test_features_aligned = feature_shift(prototypes,test_feature_curr)
                feature_memory_aligned = feature_memory_aligned.permute(1, 0)
                
                #alignment of non-parametric features (test_feature_curr) using prototypes build from low entropy features for non-parametric model
                feature_memory_aligned_nn, test_features_aligned_nn = feature_shift(prototypes,test_feature_curr_nn)
                feature_memory_aligned_nn = feature_memory_aligned_nn.permute(1, 0)
                
            else:
                feature_memory_aligned = feature_memory.permute(1, 0)
                test_features_aligned = test_feature_curr
                feature_memory_aligned = feature_memory_aligned.permute(1, 0)
            
                    
            label_memory = label_memory.cuda()
            
            Sim = test_features_aligned @ feature_memory_aligned
            
            Sim_nn = test_features_aligned_nn @ feature_memory_aligned_nn
            

            
            high_label_memory = high_label_memory.cuda()
            
            high_label_memory = high_label_memory[:Sim.shape[1]] if high_label_memory.shape[0] > Sim.shape[1] else high_label_memory            
            
            nn_label_memory = nn_label_memory.cuda()
            nn_label_memory = nn_label_memory[:Sim_nn.shape[1]] if nn_label_memory.shape[0] > Sim_nn.shape[1] else nn_label_memory

            logits = (-args.gamma * (1 - Sim)).exp() @ high_label_memory

            logits_nn_sim = (-args.gamma * (1 - Sim_nn)).exp() @ nn_label_memory
            
            
            # Adjust the shape of logits to match logits_nn_curr if the final batch size is smaller than bz
            if logits.shape[0] != logits_nn_curr.shape[0]:
                min_batch_size = min(logits.shape[0], logits_nn_curr.shape[0], logits_nn_sim.shape[0])
                logits = logits[:min_batch_size]
                logits_nn_curr = logits_nn_curr[:min_batch_size]
                logits_nn_sim = logits_nn_sim[:min_batch_size]
            
              
            nn_entropy = softmax_entropy(logits_nn_sim).mean(0)  #  source
            s_entropy = softmax_entropy(logits).mean(0)    #non-parametric model
 
            # Label Integrate
            p = 1 - (nn_entropy / (nn_entropy + s_entropy))


            f_logits = (1 - p) * logits + p * logits_nn_sim
            
            logits_domain_list.append(f_logits)  

        
    
        logits_domain_list = torch.cat(logits_domain_list)
        logits_domain_list = torch.cat((logits_domain_list, low_label_memory), dim=0)        
        
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
    

def prototype_cal(features, label, class_num, herding = True):
    prototye_list, label_memory_list = [], []
    for i in range(class_num):
        idx = torch.squeeze((label==i))
        
        
         # Check if idx selects any elements
        if idx.sum() == 0:
            print(f"No elements found for class {i}, skipping...")
            continue  # Skip to the next class if no elements are found
        
        
        mean_emb = features[idx].mean(0).unsqueeze(0)
        if herding:
            class_emb = features[idx]
            # k = int(class_emb.shape[0]/4)
            k = max(1, min(int(class_emb.shape[0] / 4), class_emb.shape[0]))  # Ensure k is within bounds
            # k = min(int(class_emb.shape[0] / 4), class_emb.shape[0] - 1) if class_emb.shape[0] > 1 else 1
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




def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
  
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def softmax_entropy_rarranged(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(0)

def feature_shift(feature_memory, test_features):
    tca_project = TCA()
    feature_memory_shift, test_features_shift = tca_project.fit(feature_memory.cpu().numpy(), test_features.cpu().numpy())
    feature_memory_shift = torch.tensor(feature_memory_shift, dtype=torch.float).cuda()
    test_features_shift = torch.tensor(test_features_shift, dtype=torch.float).cuda()
    feature_memory_shift /= feature_memory_shift.norm(dim=-1, keepdim=True)
    test_features_shift /= test_features_shift.norm(dim=-1, keepdim=True)
    return feature_memory_shift, test_features_shift

def mean_square_distance_batch(batch_vector1, vector2):

    # Ensure batch_vector1 is 2D (N, D)
    if batch_vector1.dim() == 1:
        batch_vector1 = batch_vector1.unsqueeze(1)
    
    # Ensure vector2 is 2D and matches batch_vector1's second dimension
    if vector2.dim() == 0:
        vector2 = vector2.unsqueeze(0).expand_as(batch_vector1)

    squared_differences = (batch_vector1 - vector2) ** 2
    msd = torch.mean(squared_differences, dim=1)
    return msd


if __name__ == '__main__':
    main()
