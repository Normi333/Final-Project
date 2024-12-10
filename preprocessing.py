from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import torch

# ==============================
# Dataset Class with Preprocessing
# ==============================

class SketchToImageDataset(Dataset):
    def __init__(self, root, transform=None):
        """Dataset loader for paired sketch-to-image data."""
        self.root = root
        self.transform = transform
        self.image_paths = sorted(list(Path(root).glob("*.jpg")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        width, height = image.size

        # Assuming the sketch is on the left half and the real image on the right
        sketch = image.crop((0, 0, width // 2, height))
        target = image.crop((width // 2, 0, width, height))

        if self.transform:
            sketch = self.transform(sketch)
            target = self.transform(target)

        return sketch, target


# ==============================
# Define Transformations
# ==============================

transform = transforms.Compose([
    transforms.Resize((256, 512)),  # Resize to ensure consistent size (256x512 for paired images)
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# ==============================
# Dataset and DataLoader Initialization
# ==============================

# Define dataset path
dataset_path = "path_to_your_dataset/paired_images"  # Replace with your actual path

# Initialize dataset
dataset = SketchToImageDataset(root=dataset_path, transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))  # 80% training
val_size = len(dataset) - train_size  # 20% validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# ==============================
# Visualize Samples (Optional)
# ==============================

def show_samples(dataloader):
    data_iter = iter(dataloader)
    sketches, targets = next(data_iter)

    fig, axes = plt.subplots(2, len(sketches), figsize=(15, 5))
    for i in range(len(sketches)):
        sketch = (sketches[i].permute(1, 2, 0).numpy() + 1) / 2  # De-normalize and permute to HWC
        target = (targets[i].permute(1, 2, 0).numpy() + 1) / 2
        axes[0, i].imshow(sketch)
        axes[0, i].axis("off")
        axes[1, i].imshow(target)
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.show()

# Visualize training samples
show_samples(train_loader)
