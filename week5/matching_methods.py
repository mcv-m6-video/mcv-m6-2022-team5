from dis import dis
import zipfile
import os
import torch
import cv2
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
    print("loading state dict")
    model_ft.load_state_dict(torch.load("/home/group05/mcv-m6-2022-team5/week5/resnet18.pth"))
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


def dist_matching_cnn(model, img1, img2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_transforms = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)
    img1 = img_transforms(img1).unsqueeze(0).to(device)
    img2 = img_transforms(img2).unsqueeze(0).to(device)
    model = model.to(device)

    features_img1 = extract_features(img1, model)
    features_img2 = extract_features(img2, model)
    dist = euclidean_distance(features_img1, features_img2)
    return dist


def change_color_space(img, space):
    if space == "RGB":
        return img

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if space == "HSV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif space == "LAB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    return img

def dist_matching_histogram(img1, img2, space):
    img1 = change_color_space(img1, space)
    img2 = change_color_space(img2, space)

    h_1 = cv2.calcHist([img1], [0, 1, 2], None, [25,25,25], [0, 255, 0, 255, 0, 255])
    h_1 = cv2.normalize(h_1, h_1).flatten()
    h_2 = cv2.calcHist([img2], [0, 1, 2], None, [25,25,25], [0, 255, 0, 255, 0, 255])
    h_2 = cv2.normalize(h_2, h_2).flatten()

    # cv2.HISTCMP_INTERSECT
    dist = cv2.compareHist(h_1, h_2, cv2.HISTCMP_BHATTACHARYYA)

    return dist

def dist_matching_sift(img1, img2):
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1,descriptors_2,k=2)

    true_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            true_matches.append([m, n])
    
    dist = 1/len(true_matches)

    return dist



    
    
    

