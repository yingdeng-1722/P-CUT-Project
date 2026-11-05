import os
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from models import ResnetGenerator, get_norm_layer

def inference(oct_path, mu_t_path, mask_path, weights_path, output_path, device):
    # 1. Load Model
    netG = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, norm_layer=get_norm_layer('instance')).to(device)
    netG.load_state_dict(torch.load(weights_path, map_location=device))
    netG.eval()

    # 2. Load Data
    oct_pil = Image.open(oct_path).convert('L')
    oct_t = (TF.to_tensor(oct_pil) - 0.5) / 0.5

    mu_t_np = np.load(mu_t_path).astype(np.float32)
    mu_t_t = (torch.from_numpy(mu_t_np).unsqueeze(0) * 2.0) - 1.0

    mask_np = np.load(mask_path).astype(np.uint8)
    mask_bin = (mask_np > 0).astype(np.float32)
    mask_t = (torch.from_numpy(mask_bin).unsqueeze(0) * 2.0) - 1.0

    A = torch.cat([oct_t, mu_t_t, mask_t], dim=0).unsqueeze(0).to(device)

    # 3. Inference
    with torch.no_grad():
        fake_B = netG(A)

    # 4. Save
    fake_B = fake_B.squeeze(0).cpu()
    fake_B = (fake_B + 1.0) / 2.0 # [-1, 1] -> [0, 1]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    TF.to_pil_image(fake_B).save(output_path)
    print(f"✅ VHE image saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="P-CUT Inference")
    parser.add_argument('--oct', type=str, required=True, help='Path to OCT image')
    parser.add_argument('--mu_t', type=str, required=True, help='Path to mu_t npy file')
    parser.add_argument('--mask', type=str, required=True, help='Path to mask npy file')
    parser.add_argument('--weights', type=str, required=True, help='Path to netG.pth weights')
    parser.add_argument('--output', type=str, default='./results/vhe_output.png', help='Output path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference(args.oct, args.mu_t, args.mask, args.weights, args.output, device)