import torch
import os
from torch.utils.data import Dataset, DataLoader
from detector_model import DetectorCNN

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdvDataset(Dataset):

    def __init__(self, root):

        self.samples = []

        # Iterate through epsilon folders
        for eps_folder in sorted(os.listdir(root)):

            eps_path = os.path.join(root, eps_folder)

            if not os.path.isdir(eps_path):
                continue

            image_path = os.path.join(eps_path, "images")

            if not os.path.exists(image_path):
                continue

            for file in os.listdir(image_path):

                full_path = os.path.join(image_path, file)

                if "clean" in file:
                    label = 0
                else:
                    label = 1

                self.samples.append((full_path, label))

        print("Total samples loaded:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        file, label = self.samples[idx]

        img = torch.load(file)

        # Ensure float tensor
        img = img.float()

        return img, torch.tensor(label, dtype=torch.long)


# Load dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "adversarial_dataset")

dataset = AdvDataset(DATASET_PATH)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

# Initialize model
model = DetectorCNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

criterion = torch.nn.CrossEntropyLoss()

epochs = 5


# Training loop
for epoch in range(epochs):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Loss: {total_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("-------------------------")


# Save trained model
torch.save(model.state_dict(), "detector_model.pth")

print("Detector model saved successfully.")