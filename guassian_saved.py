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

def conv_p(logits, num_class, lambda_calib):
    alpha_t = torch.exp(logits) + lambda_calib / num_class
    total_alpha_t = torch.sum(alpha_t, dim=1, keepdim=True)
    expected_p = alpha_t / total_alpha_t
    return expected_p

def protoype_cal(features, label, class_num, herding=True):
    label = label.to(torch.int64)
    prototype_list, label_memory_list = [], []
    for i in range(class_num):
        idx = torch.squeeze(label == i)
        if idx.sum().item() == 0:
            print(f"Warning: No samples found for class {i}. Skipping.")
            continue
        mean_emb = features[idx].mean(0).unsqueeze(0)
        if herding:
            class_emb = features[idx]
            k = max(1, min(int(class_emb.shape[0] / 4), class_emb.shape[0]))
            _, closest_emb_index = torch.topk(
                ((class_emb - mean_emb) ** 2).sum(dim=1), k, largest=False
            )
            prototype_list.append(class_emb[closest_emb_index])
            label_memory_list.append(torch.ones(k) * i)
        else:
            prototype_list.append(mean_emb)
            label_memory_list.append(torch.tensor(i).unsqueeze(0))
    prototype_list = torch.cat(prototype_list, dim=0)
    label_memory_list = F.one_hot(
        torch.cat(label_memory_list, dim=0).long(), num_classes=class_num
    ).float()
    return prototype_list, label_memory_list

@torch.no_grad()
def main():
    args = get_arguments()
    print(args)
    torch.manual_seed(666)

    point_nn = Point_NN(
        input_points=args.points,
        num_stages=args.stages,
        embed_dim=args.dim,
        k_neighbors=args.k,
        alpha=args.alpha,
        beta=args.beta,
    ).cuda()
    point_nn.eval()

    print('==> Loading pretrained adaptation model')
    source_model_path = f"checkpoint/{args.pth}.pth"
    device = 'cuda'
    if args.pth == "pointnet":
        source_model = PointNet("mn40")
    elif args.pth == "dgcnn":
        source_model = DGCNN("mn40")
    elif args.pth == "curvenet":
        source_model = CurveNet("mn40")

    checkpoint = torch.load(source_model_path)
    source_model.load_state_dict(checkpoint, strict=False)
    source_model.to(device)

    test_corruptions = [
        'gaussian', 'uniform', 'background', 'impulse', 'upsampling', 'distortion_rbf',
        'distortion_rbf_inv', 'density', 'density_inc', 'shear', 'rotation',
        'cutout', 'distortion', 'occlusion', 'lidar'
    ]

    gaussian_prototypes = None
    gaussian_labels = None

    for corruption in test_corruptions:
        print(f"Processing corruption: {corruption}")
        test_loader = DataLoader(
            ModelNet40C(split='test', corruption=corruption),
            num_workers=8, batch_size=args.load_bz, shuffle=False, drop_last=True
        )

        feature_memory, label_memory = [], []

        for points_cpu, labels in tqdm(test_loader):
            points = points_cpu.cuda()
            points = points.permute(0, 2, 1)

            logits_source = source_model(points).detach()
            pseudo_labels = conv_p(logits_source, num_class=40, lambda_calib=40).argmax(dim=1)

            feature_memory.append(point_nn(points))
            label_memory.append(pseudo_labels)

        label_memory_ys = torch.cat(label_memory, dim=0)
        label_memory = F.one_hot(label_memory_ys, num_classes=40).float()

        feature_memory = torch.cat(feature_memory, dim=0)
        feature_memory /= feature_memory.norm(dim=-1, keepdim=True)

        feature_memory, label_memory = protoype_cal(feature_memory, label_memory_ys, 40, herding=True)

        if corruption == 'gaussian':
            print("Saving Gaussian prototypes.")
            gaussian_prototypes = feature_memory.clone()
            gaussian_labels = label_memory.clone()
            continue

        if gaussian_prototypes is not None:
            for class_id in range(40):
                if not (label_memory_ys == class_id).any():
                    print(f"Class {class_id} missing in {corruption}. Using Gaussian prototype.")
                    feature_memory = torch.cat((feature_memory, gaussian_prototypes[class_id].unsqueeze(0)), dim=0)
                    label_memory = torch.cat((label_memory, gaussian_labels[class_id].unsqueeze(0)), dim=0)

        torch.save(
            {'feature_memory': feature_memory, 'label_memory': label_memory},
            f"prototypes_{corruption}.pth"
        )
        print(f"Saved prototypes for {corruption} corruption.")

if __name__ == '__main__':
    main()
