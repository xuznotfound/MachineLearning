#python train_anime_dcgan_interval.py --batch-size 256 --workers 32 --epochs 500 --save-every 50

import argparse
import os
import random
from pathlib import Path
from typing import List

import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image

# -----------------------------
# Dataset utilities
# -----------------------------

class ImageFileDataset(Dataset):
    """Loads all images under a directory into a flat dataset."""

    def __init__(self, root: Path, transform: transforms.Compose):
        self.root = root
        self.transform = transform
        self.paths: List[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            self.paths.extend(root.rglob(ext))
        self.paths = sorted(self.paths)
        if not self.paths:
            raise RuntimeError(f"No image files found under {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img)


def resolve_dataset_dir(download_dir: Path) -> Path:
    """Pick a sensible directory containing images inside the download."""
    if (download_dir / "images").is_dir():
        return download_dir / "images"
    return download_dir


# -----------------------------
# Model definitions (DCGAN 64x64)
# -----------------------------


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, nz: int, ngf: int, nc: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, nc: int, ndf: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1)


# -----------------------------
# Training
# -----------------------------


def train(args):
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print(f"Using seed: {args.seed}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print("Downloading dataset via kagglehub...")
    download_path = Path(kagglehub.dataset_download("splcher/animefacedataset"))
    dataset_dir = resolve_dataset_dir(download_path)
    print(f"Dataset directory: {dataset_dir}")

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImageFileDataset(dataset_dir, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    netG = Generator(args.nz, args.ngf, args.nc).to(device)
    netD = Discriminator(args.nc, args.ndf).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(args.sample_count, args.nz, 1, 1, device=device)
    real_label = 1.0
    fake_label = 0.0

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting training...")
    total_iters = 0
    for epoch in range(args.epochs):
        for i, real in enumerate(dataloader):
            netD.zero_grad()
            real = real.to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, device=device)
            output = netD(real)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output_fake = netD(fake.detach())
            errD_fake = criterion(output_fake, label)
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if total_iters % args.log_interval == 0:
                print(
                    f"[Epoch {epoch+1}/{args.epochs}] [Batch {i+1}/{len(dataloader)}] "
                    f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                    f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}"
                )

            total_iters += 1

        should_save = (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs
        if should_save:
            with torch.no_grad():
                fake_samples = netG(fixed_noise).detach().cpu()
            grid = make_grid(fake_samples, padding=2, normalize=True)
            sample_path = output_dir / f"samples_epoch_{epoch+1:03d}.png"
            save_image(grid, sample_path)

            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
            torch.save({
                "epoch": epoch + 1,
                "netG": netG.state_dict(),
                "netD": netD.state_dict(),
                "optimizerG": optimizerG.state_dict(),
                "optimizerD": optimizerD.state_dict(),
                "args": vars(args),
            }, checkpoint_path)
            print(f"Saved checkpoint and sample at epoch {epoch+1}")

    final_gen_path = output_dir / "generator_final.pt"
    torch.save(netG.state_dict(), final_gen_path)
    print(f"Training complete. Final generator saved to {final_gen_path}")


# -----------------------------
# CLI
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="DCGAN training on Anime Face dataset (interval checkpoints)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8), help="Dataloader workers")
    parser.add_argument("--image-size", type=int, default=64, help="Image size (pixels)")
    parser.add_argument("--nz", type=int, default=100, help="Latent vector size")
    parser.add_argument("--ngf", type=int, default=64, help="Generator feature map size")
    parser.add_argument("--ndf", type=int, default=64, help="Discriminator feature map size")
    parser.add_argument("--nc", type=int, default=3, help="Number of image channels")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save checkpoints and samples")
    parser.add_argument("--seed", type=int, default=None, help="Manual seed")
    parser.add_argument("--log-interval", type=int, default=100, help="Iterations between log prints")
    parser.add_argument("--sample-count", type=int, default=64, help="Number of images in fixed noise grid")
    parser.add_argument("--save-every", type=int, default=20, help="Save samples and checkpoints every N epochs")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
