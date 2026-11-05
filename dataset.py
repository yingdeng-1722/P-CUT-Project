import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

class OCTMuTMaskDataset(Dataset):
    def __init__(self, oct_dir, mu_t_dir, mask_dir, he_dir, crop_size=256, load_size=286):
        exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
        self.crop_size = crop_size
        self.load_size = load_size

        mu_t_index = {f[:-len("_mu_t_filled.npy")]: os.path.join(mu_t_dir, f) 
                      for f in os.listdir(mu_t_dir) if f.endswith("_mu_t_filled.npy")}
        mask_index = {f[:-len("_mask.npy")]: os.path.join(mask_dir, f) 
                      for f in os.listdir(mask_dir) if f.endswith("_mask.npy")}

        oct_files = sorted([f for f in os.listdir(oct_dir) if f.lower().endswith(exts)])
        self.pairs = []
        
        for f in oct_files:
            base = os.path.splitext(f)[0]
            oct_path = os.path.join(oct_dir, f)
            prefix_mu = base if base in mu_t_index else None
            if not prefix_mu:
                candidates = [p for p in mu_t_index if base.startswith(p) or p.startswith(base)]
                if candidates: prefix_mu = max(candidates, key=len)
            if not prefix_mu: continue

            prefix_mask = base if base in mask_index else None
            if not prefix_mask:
                candidates = [p for p in mask_index if base.startswith(p) or p.startswith(base)]
                if candidates: prefix_mask = max(candidates, key=len)
            if not prefix_mask: continue

            self.pairs.append((oct_path, mu_t_index[prefix_mu], mask_index[prefix_mask]))

        self.he_files = sorted([os.path.join(he_dir, f) for f in os.listdir(he_dir) if f.lower().endswith(exts)])
        print(f"[Data] Found {len(self.pairs)} OCT+mu_t+mask pairs, {len(self.he_files)} H&E images.")

    def __len__(self):
        return max(len(self.pairs), len(self.he_files))

    def _sync_transform_3ch(self, oct_pil, mu_t_pil, mask_pil):
        oct_pil = TF.resize(oct_pil, self.load_size, interpolation=Image.BICUBIC)
        mu_t_pil = TF.resize(mu_t_pil, self.load_size, interpolation=Image.BICUBIC)
        mask_pil = TF.resize(mask_pil, self.load_size, interpolation=Image.NEAREST)

        i, j, h, w = transforms.RandomCrop.get_params(oct_pil, output_size=(self.crop_size, self.crop_size))
        oct_pil, mu_t_pil, mask_pil = TF.crop(oct_pil, i, j, h, w), TF.crop(mu_t_pil, i, j, h, w), TF.crop(mask_pil, i, j, h, w)

        if np.random.rand() > 0.5:
            oct_pil, mu_t_pil, mask_pil = TF.hflip(oct_pil), TF.hflip(mu_t_pil), TF.hflip(mask_pil)
        return oct_pil, mu_t_pil, mask_pil

    def __getitem__(self, index):
        oct_path, mu_t_path, mask_path = self.pairs[index % len(self.pairs)]
        oct_pil = Image.open(oct_path).convert('L')
        
        mu_t_np = np.load(mu_t_path).astype(np.float32)
        mu_t_pil = Image.fromarray((mu_t_np * 255).astype(np.uint8), mode='L')
        
        mask_np = np.load(mask_path).astype(np.uint8)
        mask_bin = (mask_np > 0).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_bin, mode='L')

        oct_pil, mu_t_pil, mask_pil = self._sync_transform_3ch(oct_pil, mu_t_pil, mask_pil)

        oct_t = (TF.to_tensor(oct_pil) - 0.5) / 0.5
        mu_t_t = TF.to_tensor(mu_t_pil) * 2.0 - 1.0
        mask_t = TF.to_tensor(mask_pil) * 2.0 - 1.0
        A = torch.cat([oct_t, mu_t_t, mask_t], dim=0)

        he_path = self.he_files[np.random.randint(len(self.he_files))]
        he_pil = Image.open(he_path).convert('RGB')
        he_pil = TF.resize(he_pil, self.load_size, interpolation=Image.BICUBIC)
        i, j, h, w = transforms.RandomCrop.get_params(he_pil, output_size=(self.crop_size, self.crop_size))
        he_pil = TF.crop(he_pil, i, j, h, w)
        if np.random.rand() > 0.5:
            he_pil = TF.hflip(he_pil)
        B = (TF.to_tensor(he_pil) - 0.5) / 0.5

        return {'A': A, 'B': B, 'A_path': oct_path, 'B_path': he_path}