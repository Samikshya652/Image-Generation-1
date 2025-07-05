import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# -------------------------
# 1. Dataset
# -------------------------
class PairedDataset(Dataset):
    def _init_(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        self.transform = transform

    def _len_(self):
        return len(self.files)

    def _getitem_(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(path).convert("RGB")
        w, h = image.size
        w2 = int(w / 2)
        input_image = image.crop((0, 0, w2, h))
        target_image = image.crop((w2, 0, w, h))

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

# -------------------------
# 2. U-Net Generator
# -------------------------
class UNetBlock(nn.Module):
    def _init_(self, in_c, out_c, down=True, act='relu', use_dropout=False):
        super()._init_()
        if down:
            self.conv = nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False)

        self.bn = nn.BatchNorm2d(out_c)
        self.activation = nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return self.dropout(x)

class Generator(nn.Module):
    def _init_(self):
        super()._init_()
        self.down1 = UNetBlock(3, 64, down=True, act='leaky')
        self.down2 = UNetBlock(64, 128, down=True, act='leaky')
        self.down3 = UNetBlock(128, 256, down=True, act='leaky')
        self.down4 = UNetBlock(256, 512, down=True, act='leaky')
        self.down5 = UNetBlock(512, 512, down=True, act='leaky')
        self.down6 = UNetBlock(512, 512, down=True, act='leaky')
        self.down7 = UNetBlock(512, 512, down=True, act='leaky')
        self.down8 = UNetBlock(512, 512, down=True, act='leaky')

        self.up1 = UNetBlock(512, 512, down=False, use_dropout=True)
        self.up2 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.up3 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.up4 = UNetBlock(1024, 512, down=False)
        self.up5 = UNetBlock(1024, 256, down=False)
        self.up6 = UNetBlock(512, 128, down=False)
        self.up7 = UNetBlock(256, 64, down=False)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        return self.final(torch.cat([u7, d1], 1))

# -------------------------
# 3. PatchGAN Discriminator
# -------------------------
class Discriminator(nn.Module):
    def _init_(self):
        super()._init_()
        def block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, stride, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2)
            )

        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),  # input + target
            nn.LeakyReLU(0.2),
            block(64, 128),
            block(128, 256),
            block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.model(x)

# -------------------------
# 4. Training
# -------------------------
def train():
    # Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    epochs = 100
    lr = 2e-4
    img_size = 256
    l1_lambda = 100

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Dataset
    dataset = PairedDataset("data/facades/train", transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Models
    gen = Generator().to(device)
    disc = Discriminator().to(device)

    # Losses & optimizers
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    os.makedirs("output", exist_ok=True)

    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        for idx, (input_img, target_img) in enumerate(loop):
            input_img, target_img = input_img.to(device), target_img.to(device)

            # Train Discriminator
            fake_img = gen(input_img)
            real_pred = disc(input_img, target_img)
            fake_pred = disc(input_img, fake_img.detach())

            loss_disc = 0.5 * (bce(real_pred, torch.ones_like(real_pred)) +
                               bce(fake_pred, torch.zeros_like(fake_pred)))

            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator
            fake_pred = disc(input_img, fake_img)
            loss_gan = bce(fake_pred, torch.ones_like(fake_pred))
            loss_l1 = l1(fake_img, target_img)
            loss_gen = loss_gan + l1_lambda * loss_l1

            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(D_loss=loss_disc.item(), G_loss=loss_gen.item())

        # Save output sample
        save_image(fake_img * 0.5 + 0.5, f"output/fake_{epoch+1}.png")
        save_image(target_img * 0.5 + 0.5, f"output/real_{epoch+1}.png")

if _name_ == "_main_":
    train()