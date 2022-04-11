import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.optim as optim

import os
import wandb
wandb.init(project="M6-Metric-Learning", entity="fantastic5")

from metric_learning_utils import *
from metric_learning_trainer import fit

cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import warnings
warnings.filterwarnings('ignore')

TRIAN_DATA_PATH = "aicity_cars_dataset/train"
TEST_DATA_PATH = "aicity_cars_dataset/test"

batch_size = 32
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

def load_split_train_test(TRIAN_DATA_PATH, TEST_DATA_PATH):
    train_transforms = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                       ])

    train_data = ImageFolder(TRIAN_DATA_PATH,       
                    transform=train_transforms)
    test_data = ImageFolder(TEST_DATA_PATH,
                    transform=train_transforms)

    # Prepare triplet dataset
    triplet_train_dataset = TripletMIT_split(train_data, split='train', transform=train_transforms) # Returns triplet of images and target same/different
    triplet_test_dataset = TripletMIT_split(test_data, split='test', transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader, triplet_train_loader, triplet_test_loader

train_loader, test_loader, triplet_train_loader, triplet_test_loader = load_split_train_test(TRIAN_DATA_PATH, TEST_DATA_PATH)

num_classes = len(train_loader.dataset.classes)

# Set up the network and training parameters

save_path = 'aicity_cars_metric_learning_results/metric_learning_ResNet34.pth'

## test different backbones
embedding_net = EmbeddingNet_V3('resnet34', '1')

model = TripletNet(embedding_net)

if cuda:
    model.cuda()

if not os.path.exists(save_path):
    if cuda:
        model.cuda()
        print('Cuda!!!')

    margin = 1.
    loss_fn = nn.TripletMarginLoss(margin=0.5)
    lr = 1e-3
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, 3, gamma=0.1, last_epoch=-1)
    n_epochs = 25
    log_interval = 10

    ## Training !!!
    wandb.config = {
        "learning_rate": lr,
        "epochs": n_epochs,
        "batch_size": batch_size
    }


    fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
    torch.save(model.state_dict(), save_path)
else:
    print('Loading model...')
    model.load_state_dict(torch.load(save_path))
    model.to(device)