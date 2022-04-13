import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class TripletMIT_split(Dataset):

    def __init__(self, mit_split_dataset, split, transform=None):
        self.dataset = mit_split_dataset
        self.n_samples = len(self.dataset)
        self.train = split == 'train'
        self.transform = transform

        if self.train:
            self.train_labels = self.dataset.targets
            self.train_data = self.dataset.samples
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(np.asarray(self.train_labels) == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.dataset.targets
            self.test_data = self.dataset.samples
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(np.asarray(self.test_labels)  == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - {self.test_labels[i]})
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - {label1}))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.open(img1[0])
        img2 = Image.open(img2[0])
        img3 = Image.open(img3[0])

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return self.n_samples # if you want to subsample for speed

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class EmbeddingNet_V3(nn.Module):
    def __init__(self, backbone, model_id):
        super(EmbeddingNet_V3, self).__init__()

        basemodel = torch.hub.load('pytorch/vision:v0.10.0', backbone, pretrained=True)

        self.model_id = model_id
        self.base_resnet = torch.nn.Sequential(*(list(basemodel.children())[:-1]))

        fc_layer = nn.Sequential(nn.Linear(2048, 256),
                                 nn.PReLU(),
                                 nn.Linear(256, 256),
                                 nn.PReLU(),
                                 nn.Linear(256, 2)
                                 )
        self.fc = fc_layer if "fc" in self.model_id else None

    def forward(self, x):
        output = self.base_resnet(x).squeeze()
        if "fc" in self.model_id:
            output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

def extract_embeddings(dataloader, model, size):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), size))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            images = images.to(device)
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

def plot_embeddings(embeddings, targets, legend_cls, colors, xlim=None, ylim=None, n_classes=8):
    plt.figure(figsize=(15,15))
    for i in range(n_classes):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(legend_cls)

def getDistances(comparisonMethod, baseImageHistograms, queryImageHistogram):
    # loop over the index
    results = {}
    for path,label,hist in baseImageHistograms:
        # compute the distance between the two histograms
        # using the method and update the results dictionary
        query = cv2.UMat(np.array(queryImageHistogram, dtype=np.float32))
        histBase = cv2.UMat(np.array(hist, dtype=np.float32))
        distance = cv2.compareHist(query, histBase, comparisonMethod)

        results[path] = (label, distance)
    return results