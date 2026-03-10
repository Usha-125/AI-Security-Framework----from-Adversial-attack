import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn import SimpleCNN
from utils import get_data_loaders

def train():

    device = torch.device("cpu")

    train_loader, test_loader = get_data_loaders()

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "mnist_cnn.pth")
    print("Model saved!")

if __name__ == "__main__":
    train()