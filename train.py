import os
import argparse
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from models import ResnetGenerator, PatchDiscriminator, PatchNCELoss, get_norm_layer
from dataset import OCTMuTMaskDataset

class CUTModel:
    def __init__(self, cfg):
        self.cfg = cfg
        norm_layer = get_norm_layer(cfg.norm)
        self.netG = ResnetGenerator(input_nc=3, output_nc=3, ngf=cfg.ngf, norm_layer=norm_layer).to(cfg.device)
        self.netD = PatchDiscriminator(input_nc=3, ndf=cfg.ndf, norm_layer=norm_layer).to(cfg.device)
        self.proj_heads = nn.ModuleList([
            nn.Conv2d(in_ch, 256, kernel_size=1) for _, in_ch in self.netG.nce_layer_specs
        ]).to(cfg.device)

        self.criterionNCE = PatchNCELoss(nce_T=cfg.nce_T, num_patches=cfg.num_patches)
        self.criterionGAN = nn.MSELoss()

        params_G = itertools.chain(self.netG.parameters(), self.proj_heads.parameters())
        self.optimizer_G = torch.optim.Adam(params_G, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
        
        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda e: 1.0 - max(0, e - cfg.n_epochs) / float(cfg.n_epochs_decay + 1))
        self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=lambda e: 1.0 - max(0, e - cfg.n_epochs) / float(cfg.n_epochs_decay + 1))

    def set_input(self, data):
        self.real_A = data['A'].to(self.cfg.device)
        self.real_B = data['B'].to(self.cfg.device)

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))
        pred_fake = self.netD(self.fake_B.detach())
        loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))
        self.loss_D = 0.5 * (loss_D_real + loss_D_fake)
        self.loss_D.backward()

    def calculate_NCE_loss(self, src, tgt):
        tgt_gray = tgt.mean(dim=1, keepdim=True)
        tgt_3ch = torch.cat([tgt_gray, src[:, 1:2, :, :], src[:, 2:3, :, :]], dim=1)
        feat_q = [head(f) for head, f in zip(self.proj_heads, self.netG.encode(tgt_3ch))]
        feat_k = [head(f) for head, f in zip(self.proj_heads, self.netG.encode(src))]
        loss_NCE = sum(self.criterionNCE(q, k) for q, k in zip(feat_q, feat_k)) / len(feat_q)
        return loss_NCE

    def backward_G(self):
        pred_fake = self.netD(self.fake_B)
        loss_GAN = self.criterionGAN(pred_fake, torch.ones_like(pred_fake))
        loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        self.loss_G = loss_GAN + self.cfg.lambda_NCE * loss_NCE
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def update_lr(self):
        self.scheduler_G.step()
        self.scheduler_D.step()

def main():
    parser = argparse.ArgumentParser(description="P-CUT Training")
    parser.add_argument('--oct_dir', type=str, required=True, help='Path to OCT images')
    parser.add_argument('--mu_t_dir', type=str, required=True, help='Path to mu_t npy files')
    parser.add_argument('--mask_dir', type=str, required=True, help='Path to mask npy files')
    parser.add_argument('--he_dir', type=str, required=True, help='Path to real H&E images')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    class Config: pass
    cfg = Config()
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.crop_size, cfg.load_size = 256, 286
    cfg.n_epochs, cfg.n_epochs_decay = args.epochs, 100
    cfg.lr, cfg.beta1, cfg.beta2 = 0.0002, 0.5, 0.999
    cfg.lambda_NCE, cfg.nce_T, cfg.num_patches = 1.0, 0.07, 256
    cfg.ngf, cfg.ndf, cfg.norm = 64, 64, 'instance'
    cfg.save_epoch_freq = 10

    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)

    dataset = OCTMuTMaskDataset(args.oct_dir, args.mu_t_dir, args.mask_dir, args.he_dir, cfg.crop_size, cfg.load_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)

    model = CUTModel(cfg)
    print(f"[Device] {cfg.device}")

    for epoch in range(cfg.n_epochs + cfg.n_epochs_decay):
        model.netG.train()
        model.netD.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.n_epochs + cfg.n_epochs_decay}")
        for data in pbar:
            model.set_input(data)
            model.optimize_parameters()
            pbar.set_postfix({'G': f'{model.loss_G.item():.3f}', 'D': f'{model.loss_D.item():.3f}'})
        model.update_lr()

        if (epoch + 1) % cfg.save_epoch_freq == 0 or epoch == 0:
            model.netG.eval()
            with torch.no_grad():
                fake_B = model.netG(model.real_A)
            sample = torch.cat([
                model.real_A[:, 0:1, :, :].repeat(1, 3, 1, 1),
                model.real_A[:, 1:2, :, :].repeat(1, 3, 1, 1),
                model.real_A[:, 2:3, :, :].repeat(1, 3, 1, 1),
                fake_B, model.real_B], dim=0)
            save_image(sample * 0.5 + 0.5, os.path.join(args.output_dir, 'samples', f'epoch_{epoch+1}.png'), nrow=args.batch_size)
            torch.save(model.netG.state_dict(), os.path.join(args.output_dir, 'checkpoints', f'netG_epoch_{epoch+1}.pth'))
            torch.save(model.netD.state_dict(), os.path.join(args.output_dir, 'checkpoints', f'netD_epoch_{epoch+1}.pth'))
            print(f"[Saved] Epoch {epoch+1} models saved.")

if __name__ == '__main__':
    main()