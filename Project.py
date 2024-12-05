{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNekP5s7gSapCA6jL5A8uDy"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
        "id": "M5qqe6lUP8BU"
      },
      "outputs": [],
      "source": [
        "# Import necessary libraries\n",
        "import os\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "from torch.utils.data import DataLoader, Dataset\n",
        "from torchvision import transforms, utils\n",
        "from PIL import Image\n",
        "from pathlib import Path\n",
        "import matplotlib.pyplot as plt\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# ==============================\n",
        "# Dataset Class\n",
        "# ==============================\n",
        "\n",
        "class SketchToImageDataset(Dataset):\n",
        "    def __init__(self, root, transform=None):\n",
        "        \"\"\"Dataset loader for paired sketch-to-image data.\"\"\"\n",
        "        self.root = root\n",
        "        self.transform = transform\n",
        "        self.image_paths = sorted(list(Path(root).glob(\"*.jpg\")))\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.image_paths)\n",
        "\n",
        "    def __getitem__(self, index):\n",
        "        image = Image.open(self.image_paths[index]).convert(\"RGB\")\n",
        "        width, height = image.size\n",
        "\n",
        "        # Assuming the sketch is on the left half and the real image on the right\n",
        "        sketch = image.crop((0, 0, width // 2, height))\n",
        "        target = image.crop((width // 2, 0, width, height))\n",
        "\n",
        "        if self.transform:\n",
        "            sketch = self.transform(sketch)\n",
        "            target = self.transform(target)\n",
        "\n",
        "        return sketch, target\n"
      ],
      "metadata": {
        "id": "7xbGRJ9dQCwd"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# ==============================\n",
        "# Generator (U-Net Architecture)\n",
        "# ==============================\n",
        "\n",
        "class GeneratorUNet(nn.Module):\n",
        "    def __init__(self, in_channels=3, out_channels=3):\n",
        "        super(GeneratorUNet, self).__init__()\n",
        "\n",
        "        def down_block(in_feat, out_feat, normalize=True):\n",
        "            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1, bias=False)]\n",
        "            if normalize:\n",
        "                layers.append(nn.BatchNorm2d(out_feat))\n",
        "            layers.append(nn.LeakyReLU(0.2))\n",
        "            return nn.Sequential(*layers)\n",
        "\n",
        "        def up_block(in_feat, out_feat, dropout=0.0):\n",
        "            layers = [\n",
        "                nn.ConvTranspose2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1, bias=False),\n",
        "                nn.BatchNorm2d(out_feat),\n",
        "                nn.ReLU(inplace=True)\n",
        "            ]\n",
        "            if dropout:\n",
        "                layers.append(nn.Dropout(dropout))\n",
        "            return nn.Sequential(*layers)\n",
        "\n",
        "        # Downsampling layers\n",
        "        self.down1 = down_block(in_channels, 64, normalize=False)\n",
        "        self.down2 = down_block(64, 128)\n",
        "        self.down3 = down_block(128, 256)\n",
        "        self.down4 = down_block(256, 512)\n",
        "        self.down5 = down_block(512, 512)\n",
        "        self.down6 = down_block(512, 512)\n",
        "        self.down7 = down_block(512, 512)\n",
        "        self.down8 = down_block(512, 512, normalize=False)\n",
        "\n",
        "        # Upsampling layers\n",
        "        self.up1 = up_block(512, 512, dropout=0.5)\n",
        "        self.up2 = up_block(1024, 512, dropout=0.5)\n",
        "        self.up3 = up_block(1024, 512, dropout=0.5)\n",
        "        self.up4 = up_block(1024, 512)\n",
        "        self.up5 = up_block(1024, 256)\n",
        "        self.up6 = up_block(512, 128)\n",
        "        self.up7 = up_block(256, 64)\n",
        "        self.up8 = nn.Sequential(\n",
        "            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),\n",
        "            nn.Tanh()\n",
        "        )\n",
        "\n",
        "    def forward(self, x):\n",
        "        d1 = self.down1(x)\n",
        "        d2 = self.down2(d1)\n",
        "        d3 = self.down3(d2)\n",
        "        d4 = self.down4(d3)\n",
        "        d5 = self.down5(d4)\n",
        "        d6 = self.down6(d5)\n",
        "        d7 = self.down7(d6)\n",
        "        d8 = self.down8(d7)\n",
        "\n",
        "        u1 = self.up1(d8)\n",
        "        u2 = self.up2(torch.cat([u1, d7], 1))\n",
        "        u3 = self.up3(torch.cat([u2, d6], 1))\n",
        "        u4 = self.up4(torch.cat([u3, d5], 1))\n",
        "        u5 = self.up5(torch.cat([u4, d4], 1))\n",
        "        u6 = self.up6(torch.cat([u5, d3], 1))\n",
        "        u7 = self.up7(torch.cat([u6, d2], 1))\n",
        "        u8 = self.up8(torch.cat([u7, d1], 1))\n",
        "        return u8\n"
      ],
      "metadata": {
        "id": "w8B3QNjiQHCv"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# ==============================\n",
        "# Discriminator (PatchGAN)\n",
        "# ==============================\n",
        "\n",
        "class Discriminator(nn.Module):\n",
        "    def __init__(self, in_channels=6):\n",
        "        super(Discriminator, self).__init__()\n",
        "\n",
        "        def disc_block(in_feat, out_feat, normalize=True):\n",
        "            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)]\n",
        "            if normalize:\n",
        "                layers.append(nn.BatchNorm2d(out_feat))\n",
        "            layers.append(nn.LeakyReLU(0.2, inplace=True))\n",
        "            return nn.Sequential(*layers)\n",
        "\n",
        "        self.model = nn.Sequential(\n",
        "            disc_block(in_channels, 64, normalize=False),\n",
        "            disc_block(64, 128),\n",
        "            disc_block(128, 256),\n",
        "            disc_block(256, 512),\n",
        "            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)\n",
        "        )\n",
        "\n",
        "    def forward(self, img_A, img_B):\n",
        "        x = torch.cat([img_A, img_B], 1)\n",
        "        return self.model(x)\n"
      ],
      "metadata": {
        "id": "vZ8weBABQJQD"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# ==============================\n",
        "# Training Functionality\n",
        "# =============================="
      ],
      "metadata": {
        "id": "z7PeqwWnQLgP"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Initialize models, loss functions, optimizers\n",
        "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
        "generator = GeneratorUNet().to(device)\n",
        "discriminator = Discriminator().to(device)\n",
        "\n",
        "criterion_GAN = nn.BCEWithLogitsLoss()\n",
        "criterion_pixelwise = nn.L1Loss()\n",
        "\n",
        "lambda_pixel = 100\n",
        "\n",
        "optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))\n",
        "optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))"
      ],
      "metadata": {
        "id": "NliaQaLiTzFt"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Training loop\n",
        "\n",
        "def train(dataloader, epochs=200):\n",
        "    for epoch in range(epochs):\n",
        "        for i, (sketch, real_image) in enumerate(dataloader):\n",
        "            sketch, real_image = sketch.to(device), real_image.to(device)\n",
        "\n",
        "            # Train Discriminator\n",
        "            optimizer_D.zero_grad()\n",
        "            real_preds = discriminator(sketch, real_image)\n",
        "            real_targets = torch.ones_like(real_preds, device=device)\n",
        "            loss_real = criterion_GAN(real_preds, real_targets)\n",
        "\n",
        "            fake_image = generator(sketch)\n",
        "            fake_preds = discriminator(sketch, fake_image.detach())\n",
        "            fake_targets = torch.zeros_like(fake_preds, device=device)\n",
        "            loss_fake = criterion_GAN(fake_preds, fake_targets)\n",
        "\n",
        "            loss_D = (loss_real + loss_fake) / 2\n",
        "            loss_D.backward()\n",
        "            optimizer_D.step()\n",
        "\n",
        "             # Train Generator\n",
        "            optimizer_G.zero_grad()\n",
        "            preds = discriminator(sketch, fake_image)\n",
        "            targets = torch.ones_like(preds, device=device)\n",
        "            loss_GAN = criterion_GAN(preds, targets)\n",
        "\n",
        "            loss_pixel = criterion_pixelwise(fake_image, real_image)\n",
        "\n",
        "            loss_G = loss_GAN + lambda_pixel * loss_pixel\n",
        "            loss_G.backward()\n",
        "            optimizer_G.step()\n"
      ],
      "metadata": {
        "id": "y-SSfsFOT0VE"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Print progress\n",
        "            if i % 100 == 0:\n",
        "                print(f\"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}]: \"\n",
        "                      f\"Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 110
        },
        "id": "e8LTgWsFT4By",
        "outputId": "244461fe-872d-4488-8e69-fda31fe423d1"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "error",
          "ename": "IndentationError",
          "evalue": "unexpected indent (<ipython-input-11-2c0ec0455cf7>, line 2)",
          "traceback": [
            "\u001b[0;36m  File \u001b[0;32m\"<ipython-input-11-2c0ec0455cf7>\"\u001b[0;36m, line \u001b[0;32m2\u001b[0m\n\u001b[0;31m    if i % 100 == 0:\u001b[0m\n\u001b[0m    ^\u001b[0m\n\u001b[0;31mIndentationError\u001b[0m\u001b[0;31m:\u001b[0m unexpected indent\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Save model checkpoints\n",
        "        torch.save(generator.state_dict(), f\"generator_epoch_{epoch}.pth\")\n",
        "        torch.save(discriminator.state_dict(), f\"discriminator_epoch_{epoch}.pth\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 110
        },
        "id": "wMQv_jRYUAic",
        "outputId": "cf7265c7-4f6a-41fa-97ec-cd36327196a3"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "error",
          "ename": "IndentationError",
          "evalue": "unexpected indent (<ipython-input-12-c6c880650641>, line 2)",
          "traceback": [
            "\u001b[0;36m  File \u001b[0;32m\"<ipython-input-12-c6c880650641>\"\u001b[0;36m, line \u001b[0;32m2\u001b[0m\n\u001b[0;31m    torch.save(generator.state_dict(), f\"generator_epoch_{epoch}.pth\")\u001b[0m\n\u001b[0m    ^\u001b[0m\n\u001b[0;31mIndentationError\u001b[0m\u001b[0;31m:\u001b[0m unexpected indent\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# ==============================\n",
        "# Visualization\n",
        "# =============================="
      ],
      "metadata": {
        "id": "-Xet1HyfUDPC"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def visualize(generator, dataloader, device=\"cuda\"):\n",
        "    \"\"\"Visualize results from the generator using a batch of sketches from the dataloader.\"\"\"\n",
        "    generator.eval()\n",
        "    with torch.no_grad():\n",
        "        # Fetch a batch of sample sketches\n",
        "        sample_sketch, sample_target = next(iter(dataloader))\n",
        "        sample_sketch, sample_target = sample_sketch.to(device), sample_target.to(device)\n",
        "\n",
        "        # Generate images from sketches\n",
        "        generated_images = generator(sample_sketch).cpu()\n",
        "\n",
        "        # Unnormalize images for visualization\n",
        "        def unnormalize(img):\n",
        "            img = img * 0.5 + 0.5  # Convert [-1, 1] to [0, 1]\n",
        "            return img\n",
        "\n",
        "        # Prepare a grid of sketches, generated images, and real images\n",
        "        sample_sketch = unnormalize(sample_sketch.cpu())\n",
        "        sample_target = unnormalize(sample_target.cpu())\n",
        "        generated_images = unnormalize(generated_images)\n",
        "\n",
        "        n_samples = min(5, sample_sketch.size(0))  # Show up to 5 samples\n",
        "        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))\n",
        "        for i in range(n_samples):\n",
        "            axes[i, 0].imshow(sample_sketch[i].permute(1, 2, 0))\n",
        "            axes[i, 0].set_title(\"Sketch\")\n",
        "            axes[i, 0].axis(\"off\")\n",
        "\n",
        "            axes[i, 1].imshow(generated_images[i].permute(1, 2, 0))\n",
        "            axes[i, 1].set_title(\"Generated Image\")\n",
        "            axes[i, 1].axis(\"off\")\n",
        "\n",
        "            axes[i, 2].imshow(sample_target[i].permute(1, 2, 0))\n",
        "            axes[i, 2].set_title(\"Real Image\")\n",
        "            axes[i, 2].axis(\"off\")\n",
        "\n",
        "        plt.tight_layout()\n",
        "        plt.show()\n",
        "\n",
        "# Example usage:\n",
        "# visualize(generator, test_dataloader, device)"
      ],
      "metadata": {
        "id": "aDs9m_Z_UF48"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "pHoNGohuUKPd"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}