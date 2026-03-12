import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cnn import SimpleCNN
from attacks.fgsm_attack import fgsm_attack


# device
device = torch.device("cpu")

# attack strength
epsilon = 0.25

# transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# dataset
test_dataset = datasets.MNIST(
    './data',
    train=False,
    transform=transform,
    download=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=True
)

# load model
model = SimpleCNN()

# ✅ FIXED PATH
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))

model.to(device)
model.eval()

correct = 0
total = 0

for image, label in test_loader:

    image, label = image.to(device), label.to(device)

    # enable gradient for attack
    image.requires_grad = True

    # forward pass
    output = model(image)

    loss = torch.nn.functional.cross_entropy(output, label)

    # clear gradients
    model.zero_grad()

    # backprop
    loss.backward()

    # collect gradient
    data_grad = image.grad.data

    # FGSM attack
    perturbed_image = fgsm_attack(image, epsilon, data_grad)

    # re-classify perturbed image
    output = model(perturbed_image)

    final_pred = output.max(1, keepdim=True)[1]

    if final_pred.item() == label.item():
        correct += 1

    total += 1

accuracy = correct / total

print("Accuracy under FGSM attack:", accuracy)