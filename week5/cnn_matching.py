import zipfile
import os
import torch
import time
import copy
import numpy as np
import torchvision.models as models

from torchvision import transforms
from torch import nn, optim
from torchvision.datasets import ImageFolder
from numpy import dot
from numpy.linalg import norm
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def get_finetuned_model():
    model_ft, input_size = initialize_model("resnet", num_classes=125, feature_extract=False, use_pretrained=True)
    model_ft.load_state_dict(torch.load("resnet18.pth"))
    return model_ft

def extract_features(x, model_ft):
    ### strip the last layer
    feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1])
    ### check this works
    output = feature_extractor(x) # output now has the features corresponding to input x
    return output.cpu().detach().numpy().reshape((512))

def cos_similarity(a, b):
    a = a / np.sqrt(np.sum(a**2))
    b = b / np.sqrt(np.sum(b**2))
    return 1 - dot(a, b)/(norm(a)*norm(b))

def euclidean_distance(a, b):
    a = a / np.sqrt(np.sum(a**2))
    b = b / np.sqrt(np.sum(b**2))
    return norm(a-b)


def dist_matching(img1, img2):
    img_transforms = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)
    img1 = img_transforms(img1).unsqueeze(0).to(device)
    img2 = img_transforms(img2).unsqueeze(0).to(device)

    model = get_finetuned_model()
    features_img1 = extract_features(img1, model)
    features_img2 = extract_features(img2, model)
    dist = cos_similarity(features_img1, features_img2)
    return dist

