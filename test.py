import torch
from models.cnn import SimpleCNN
from utils import get_data_loaders

def test():

    device = torch.device("cpu")

    _, test_loader = get_data_loaders()

    model = SimpleCNN()
    model.load_state_dict(torch.load("mnist_cnn.pth"))
    model.to(device)

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    test()