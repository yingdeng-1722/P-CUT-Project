import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout=False):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            norm_layer(dim),
        ]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv_block(x)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return nn.BatchNorm2d
    elif norm_type == 'instance':
        return nn.InstanceNorm2d
    else:
        raise NotImplementedError(f'norm layer [{norm_type}] not found')

class ResnetGenerator(nn.Module):
    """Generator: input_nc=3 (OCT+μ_t+mask) → output_nc=3 (H&E)"""
    def __init__(self, input_nc=3, output_nc=3, ngf=64,
                 norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf * 2),
            nn.ReLU(True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            norm_layer(ngf * 4),
            nn.ReLU(True),
        )
        self.blocks = nn.ModuleList([
            ResnetBlock(ngf * 4, norm_layer, use_dropout)
            for _ in range(n_blocks)
        ])
        decoder = [
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.decoder = nn.Sequential(*decoder)
        self.nce_layer_specs = [
            ('initial', ngf), ('down1', ngf * 2), ('down2', ngf * 4),
            ('blocks3', ngf * 4), ('blocks6', ngf * 4),
        ]

    def encode(self, x):
        feats = {}
        h = self.initial(x)
        feats['initial'] = h
        h = self.down1(h)
        feats['down1'] = h
        h = self.down2(h)
        feats['down2'] = h
        target_names = [s[0] for s in self.nce_layer_specs]
        for i, block in enumerate(self.blocks):
            h = block(h)
            key = f'blocks{i}'
            if key in target_names:
                feats[key] = h
        return [feats[name] for name, _ in self.nce_layer_specs]

    def forward(self, x):
        h = self.initial(x)
        h = self.down1(h)
        h = self.down2(h)
        for block in self.blocks:
            h = block(h)
        return self.decoder(h)

class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

class PatchNCELoss(nn.Module):
    def __init__(self, nce_T=0.07, num_patches=256):
        super().__init__()
        self.nce_T = nce_T
        self.num_patches = num_patches
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, feat_q, feat_k):
        B, C, H, W = feat_q.shape
        feat_q = feat_q.permute(0, 2, 3, 1).reshape(B, H * W, C)
        feat_k = feat_k.permute(0, 2, 3, 1).reshape(B, H * W, C)
        n = min(self.num_patches, feat_q.shape[1])
        idx_q = torch.randperm(feat_q.shape[1], device=feat_q.device)[:n]
        idx_k = torch.randperm(feat_k.shape[1], device=feat_k.device)[:n]
        feat_q = feat_q[:, idx_q, :]
        feat_k = feat_k[:, idx_k, :]
        feat_q = F.normalize(feat_q, dim=-1)
        feat_k = F.normalize(feat_k, dim=-1)
        logits = torch.bmm(feat_q, feat_k.permute(0, 2, 1)) / self.nce_T
        logits = logits.reshape(-1, n)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return self.cross_entropy_loss(logits, labels).mean()