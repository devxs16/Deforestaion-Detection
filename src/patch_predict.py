import torch
from PIL import Image
from torchvision import transforms

from src.model import get_model

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
def predict_landcover(image_path):
    image = Image.open(image_path).convert("RGB")

    width, height = image.size
    print(width, height)
    PATCH_SIZE = 64

    patches = []

    for y in range(0, height, PATCH_SIZE):

        row_predictions = []

        for x in range(0, width, PATCH_SIZE):

            patch = image.crop(
            (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
            )

            patch = transform(patch)

            patch = patch.unsqueeze(0)

            patch = patch.to(device)

            with torch.no_grad():

                outputs = model(patch)

                probabilities = torch.softmax(outputs, dim=1)

                confidence, predicted = torch.max(probabilities, 1)

            row_predictions.append({
                "x": x,
                "y": y,
                "class": class_names[predicted.item()],
                "confidence": confidence.item()
            })
    
        patches.extend(row_predictions)
    return patches

if __name__ == "__main__":
    patches = predict_landcover("satellite.jpg")

    print(f"Total patches: {len(patches)}")
    print(patches[:5])
