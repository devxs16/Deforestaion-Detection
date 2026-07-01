import torch
from PIL import Image
from torchvision import transforms

from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model()

model.load_state_dict(
    torch.load("models/best_model.pth", map_location=device)
)

model.to(device)
model.eval()

class_names = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

image = Image.open("test.jpg").convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

image = image.to(device)

with torch.no_grad():

    outputs = model(image)

    _, predicted = torch.max(outputs,1)

print(class_names[predicted.item()])