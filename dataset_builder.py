import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from attacks.fgsm_attack import fgsm_attack  # your FGSM function
import os

# ------------------------------
# Device and folders
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epsilons = [0.1, 0.15, 0.2, 0.25]
max_images_per_epsilon = 1000

for eps in epsilons:
    os.makedirs(f"adversarial_dataset/eps_{eps}/images", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ------------------------------
# SimpleCNN definition
# ------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# ------------------------------
# Load MNIST dataset
# ------------------------------
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ------------------------------
# Train the model
# ------------------------------
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
epochs = 3

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "mnist_cnn.pth")
print("Model trained and saved!")

# ------------------------------
# Generate adversarial dataset for multiple epsilons
# ------------------------------
model.eval()

for eps in epsilons:

    print(f"\nGenerating adversarial images for epsilon = {eps}")
    count = 0

    for images, labels in test_loader:

        images, labels = images.to(device), labels.to(device)

        # Enable gradient for FGSM
        images.requires_grad = True

        outputs = model(images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        data_grad = images.grad.data

        adv_images = fgsm_attack(images, eps, data_grad)

        batch_size = images.size(0)

        for i in range(batch_size):

            if count >= max_images_per_epsilon:
                break

            torch.save(images[i].cpu(), f"adversarial_dataset/eps_{eps}/images/clean_{count}.pt")
            torch.save(adv_images[i].cpu(), f"adversarial_dataset/eps_{eps}/images/adv_{count}.pt")

            count += 1

        if count >= max_images_per_epsilon:
            break

    print(f"Completed epsilon = {eps}, total images saved: {count}")