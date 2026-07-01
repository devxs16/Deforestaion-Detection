import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_model():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)


    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    return model  