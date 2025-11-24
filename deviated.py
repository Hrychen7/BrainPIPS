import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from nilearn.connectome import ConnectivityMeasure
from tqdm import tqdm
import argparse
from models_BrainPIPS import mae_BNTF_base 

# Method Calculate Deviation Strength
class FullTask2Data(Dataset):
    def __init__(self, root, csv):
        self.root = root
        df = pd.read_csv(csv)
        self.names = list(df['new_name'])  
        self.correlation = ConnectivityMeasure(kind='correlation')
        print(f"[FullTask2Data] Loaded {len(self.names)} samples")

    def __getitem__(self, idx):
        name = self.names[idx].replace('adhd', 'sch_adhd') + '.npy'
        path = os.path.join(self.root, name)
        data = np.load(path)  # shape: (100, T)
        fc = self.correlation.fit_transform([data.T])[0]
        fc[np.isnan(fc)] = 0
        return torch.tensor(fc, dtype=torch.float32), name

    def __len__(self):
        return len(self.names)


@torch.no_grad()
def extract_q_deviated(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = FullTask2Data(root=args.root, csv=args.csv)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = mae_BNTF_base(num_network=args.num_network)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    deviated_dir = os.path.join(args.save_dir, 'Flow')
    os.makedirs(deviated_dir, exist_ok=True)
    
    for x, name in tqdm(loader):
        x = x.to(device)
        deviated = model(imgs=x, pretrain=False,deviated=True)
        deviated = deviated.squeeze(0).cpu().numpy()   # shape: [K, K]
        base_name = name[0].replace('.npy', '')
        np.save(os.path.join(deviated_dir, base_name + '.npy'), deviated)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='')
    parser.add_argument('--csv', default='')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--save_dir', default='')  
    parser.add_argument('--num_network', default=7, type=int)
    args = parser.parse_args()

    extract_q_deviated(args)
