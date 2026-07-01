import torch
import torch.nn as nn
import torch.optim as optim



from dataset import get_dataloaders
from model import get_model

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(device)

train_loader, val_loader, test_loader = get_dataloaders()
model = get_model()
model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)
num_epochs = 10
best_accuracy = 0

for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)}")

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f}")

    # ---------------- Validation ---------------- #

    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total

    print(
        f"Validation Loss: {val_loss:.4f} | "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(model.state_dict(), "models/best_model.pth")

        print("Best model saved!")

torch.save(model.state_dict(), "models/best_model.pth")

print("Best model saved!")