import argparse
import datetime
import json
import os
from pathlib import Path
import time
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
import util.misc as misc
import util.datasets as mydataset

import models_BrainPIPS
import math
import sys
import torch.nn as nn
from typing import Iterable
import pickle

import util.misc as misc
import util.lr_sched as lr_sched
from sklearn import preprocessing
from sklearn.manifold import TSNE

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def get_args_parser():
    parser = argparse.ArgumentParser('MAE validation', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eval_mode', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory of the dataset (e.g., sch_data)')
    parser.add_argument('--csv', type=str, required=True,
                        help='CSV file path (e.g., final2.csv)')
    parser.add_argument('--deviated_root', type=str, required=True,
                        help='Directory for deviated flow matrices (e.g., Flow)')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_net', default=7, type=int)
    parser.add_argument('--rho_0', type=float, default=0.4,
                        help='Initial value of rho for dynamic masking')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--model', default='mae_BNTF_base', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--debug', default=1, type=int)
    parser.add_argument('--all', default=0, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--dist_on_itp', action='store_true')
    return parser

# Method Deviated Subnetwork Aggregation
class Classifier2(nn.Module):
    def __init__(self):
        super(Classifier2, self).__init__()
        self.fc1 = nn.Linear(100, 8)
        self.relu = nn.LeakyReLU()
        self.net = nn.Sequential(
            nn.Linear(8 * 100, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU()
            
        )
        self.deviated = nn.Sequential(
            nn.Linear(49, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU()
        )
        self.clf = nn.Sequential(
            nn.Linear(256, 32),
            nn.LeakyReLU(), 
            nn.Dropout(0.3),
            nn.Linear(32, 2) 
        )

    def forward(self, x,deviated):
        batch_size, _, _ = x.shape
        features = self.fc1(x)
        features = self.relu(features)
        features = features.reshape((batch_size, -1))
        output1 = self.net(features)
        K = int(deviated.shape[-1] ** 0.5)  # e.g. 49 -> 7
        deviated = deviated - torch.diag_embed(torch.diagonal(deviated, dim1=1, dim2=2))
        deviated = deviated.view(batch_size, -1)
        output2 = self.deviated(deviated)
        output = output1 + output2
        output = self.clf(output)
        return output


def train_epoch(model, classifier, criterion, optimizer, train_loader, device, log_file, epoch):
    model.train()
    classifier.train()
    total_loss = 0.0
    correct_train = 0
    total_train = 0
    TP_train = TN_train = FN_train = FP_train = 0

    for data, flow,targets in train_loader:
        data,flow,targets = data.to(device), flow.to(device),targets.to(device)
        optimizer.zero_grad()
        features = model(imgs = data, pretrain=False)
        outputs = classifier(features,flow)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_train += targets.size(0)
        correct_train += predicted.eq(targets).sum().item()

        TP_train += ((predicted == 1) & (targets == 1)).sum().item()
        TN_train += ((predicted == 0) & (targets == 0)).sum().item()
        FN_train += ((predicted == 0) & (targets == 1)).sum().item()
        FP_train += ((predicted == 1) & (targets == 0)).sum().item()

    train_acc = 100. * correct_train / total_train
    sen_train = TP_train / (TP_train + FN_train) if (TP_train + FN_train) != 0 else 0
    spe_train = TN_train / (TN_train + FP_train) if (TN_train + FP_train) != 0 else 0

    with open(log_file, 'a') as f:
        f.write(f'Train Epoch: {epoch + 1}, Train Loss: {total_loss / len(train_loader):.6f}, '
                f'Train Acc: {train_acc:.3f}%, Train Sen: {sen_train:.3f}, Train Spe: {spe_train:.3f}\n')

    return total_loss / len(train_loader), train_acc, sen_train, spe_train


def validate_epoch(model, classifier, criterion, val_loader, device, log_file, epoch, mode=0):
    model.eval()
    classifier.eval()
    val_loss = 0.0
    correct_val = total_val = 0
    TP_val = TN_val = FN_val = FP_val = 0
    all_probs, all_targets = [], []

    with torch.no_grad():
        for data,flow, targets in val_loader:
            data,flow,targets = data.to(device),flow.to(device),targets.to(device)
            features = model(imgs = data, pretrain=False)
            outputs = classifier(features,flow)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = outputs.max(1)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            total_val += targets.size(0)
            correct_val += predicted.eq(targets).sum().item()
            TP_val += ((predicted == 1) & (targets == 1)).sum().item()
            TN_val += ((predicted == 0) & (targets == 0)).sum().item()
            FN_val += ((predicted == 0) & (targets == 1)).sum().item()
            FP_val += ((predicted == 1) & (targets == 0)).sum().item()

    val_acc = 100. * correct_val / total_val
    sen_val = TP_val / (TP_val + FN_val) if (TP_val + FN_val) != 0 else 0
    spe_val = TN_val / (TN_val + FP_val) if (TN_val + FP_val) != 0 else 0
    auc_val = roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0

    with open(log_file, 'a') as f:
        label = 'Validation' if mode == 0 else 'Test'
        f.write(f'{label} Epoch: {epoch + 1}, {label} Loss: {val_loss / len(val_loader):.6f}, '
                f'{label} Acc: {val_acc:.3f}%, {label} Sen: {sen_val:.3f}, '
                f'{label} Spe: {spe_val:.3f}, {label} AUC: {auc_val:.3f}\n')

    return val_loss / len(val_loader), val_acc, sen_val, spe_val, auc_val


def main(args):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset_train = mydataset.Task4Data(root=args.root,csv=args.csv,deviated_root=args.deviated_root,mode=0, fold=args.fold)
    dataset_val = mydataset.Task4Data(root=args.root,csv=args.csv,deviated_root=args.deviated_root,mode=1, fold=args.fold)
    dataset_test = mydataset.Task4Data(root=args.root,csv=args.csv,deviated_root=args.deviated_root,mode=2, fold=args.fold)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    rho_str = f"rho{args.rho_0:.2f}"
    args.output_dir = os.path.join(args.output_dir, f'fold{args.fold}', rho_str)
    log_file = os.path.join(args.output_dir, 'log.txt')
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model = models_BrainPIPS.__dict__[args.model](num_network=args.num_net, rho_0=args.rho_0)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print("Resumed model weights from:", args.resume)

    classifier = Classifier2()
    model.cuda()
    classifier.cuda()

    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_val_auc = 0.0  

    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_acc, train_sen, train_spe = train_epoch(model, classifier, criterion, optimizer, train_loader, device, log_file, epoch)
        val_loss, val_acc, val_sen, val_spe, val_auc = validate_epoch(model, classifier, criterion, val_loader, device, log_file, epoch)

        print(f'Epoch: [{epoch + 1}/{args.epochs}] Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.3f}% | Train Sen: {train_sen:.3f} | Train Spe: {train_spe:.3f} | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.3f}% | Val Sen: {val_sen:.3f} | Val Spe: {val_spe:.3f} | Val AUC: {val_auc:.3f}')

        if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_auc > best_val_auc):
            best_val_acc = val_acc
            best_val_auc = val_auc  # NEW
            test_loss, test_acc, test_sen, test_spe, test_auc = validate_epoch(model, classifier, criterion, test_loader, device, log_file, epoch, mode=1)

            print(f'Test Loss: {test_loss:.6f} | Test Acc: {test_acc:.3f}% | Test Sen: {test_sen:.3f} | Test Spe: {test_spe:.3f} | Test AUC: {test_auc:.3f}')
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{epoch + 1}.pth')
            torch.save({
                'model': model.state_dict(),
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
                'best_val_acc': best_val_acc,
                'test_acc': test_acc,
                'test_sen': test_sen,
                'test_spe': test_spe,
                'test_auc': test_auc
            }, checkpoint_path)

            with open(log_file, 'a') as f:
                f.write(f'Epoch: {epoch + 1}, Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.3f}, Test Sen: {test_sen:.3f}, Test Spe: {test_spe:.3f}, Test AUC: {test_auc:.3f}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Classifier Training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
