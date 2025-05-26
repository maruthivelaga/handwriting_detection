# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Paths
DATA_DIR = 'dataset'
os.makedirs("models", exist_ok=True)
MODEL_PATH = 'models/handwriting_quality_model.pth'

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dataset & DataLoader
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Updated to avoid deprecation warning
model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes: neat, medium, messy
model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

# Training Loop
EPOCHS = 10
for epoch in range(EPOCHS):
    running_loss = 0.0
    model.train()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(dataloader):.4f}")

# Save model
torch.save(model.state_dict(), os.path.join("models", "handwriting_quality_model.pth"))
print("Model trained and saved.")
print("✅ Model trained and saved at:", MODEL_PATH)