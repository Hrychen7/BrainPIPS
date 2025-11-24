import os
from torch.utils import data
import numpy as np
import nibabel as nib
import random
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import warnings
from nilearn.connectome import ConnectivityMeasure
from sklearn.utils  import shuffle
import torch
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")

class Task2Data(data.Dataset):
    def __init__(self, root=None, csv=None, mode=0, fold=0):
        self.root = root
        df = pd.read_csv(csv)

        if mode == 0:
            df = df[(df[f'fold_{fold}'] == 0) | (df[f'fold_{fold}'] == 3)]
        elif mode == 1:
            df = df[(df[f'fold_{fold}'] == 1) | (df[f'fold_{fold}'] == 4)]
        else:
            df = df[(df[f'fold_{fold}'] == 2) | (df[f'fold_{fold}'] == 5)]

        self.names = list(df['new_name'])
        self.lbls = list(df['dx'])
        self.correlation_measure = ConnectivityMeasure(kind='correlation')
        print(f"[Task2Data] Mode {mode} | Fold {fold} | Loaded {len(self.names)} samples")

    def __getitem__(self, index):
        name = self.names[index].replace('adhd', 'sch_adhd') + '.npy'
        label = self.lbls[index]
        img = np.load(os.path.join(self.root, name))  # shape: (100, T)
        FC = self.correlation_measure.fit_transform([img.T])[0]
        FC[np.isnan(FC)] = 0

        return FC, label

    def __len__(self):
        return len(self.names)
class Task4Data(data.Dataset):
    def __init__(self, root=None, csv=None,deviated_root = None,mode=0, fold=0):
        self.root = root
        self.deviated_root = deviated_root
        
        df = pd.read_csv(csv)
        if mode == 0:
            df = df[(df[f'fold_{fold}'] == 0) | (df[f'fold_{fold}'] == 3)]
        elif mode == 1:
            df = df[(df[f'fold_{fold}'] == 1) | (df[f'fold_{fold}'] == 4)]
        else:
            df = df[(df[f'fold_{fold}'] == 2) | (df[f'fold_{fold}'] == 5)]

        self.names = list(df['new_name'])
        self.lbls = list(df['dx'])
        self.correlation_measure = ConnectivityMeasure(kind='correlation')
        print(f"[Task4Data] Mode {mode} | Fold {fold} | Loaded {len(self.names)} samples")

    def __getitem__(self, index):
        name = self.names[index].replace('adhd', 'sch_adhd') + '.npy'
        label = self.lbls[index]
        img = np.load(os.path.join(self.root, name))  # shape: (100, T)
        flow  = np.load(os.path.join(self.deviated_root, name))
        FC = self.correlation_measure.fit_transform([img.T])[0]
        FC[np.isnan(FC)] = 0

        return FC, flow,label

    def __len__(self):
        return len(self.names)

