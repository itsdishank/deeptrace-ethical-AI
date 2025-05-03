import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from code.spatial_cnn import SpatialModel

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--epochs", type=int, default=5)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SpatialModel().to(device)

train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

for epoch in range(args.epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(images).view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "spatial_model.pth")
print("Training Complete. Model saved.")
