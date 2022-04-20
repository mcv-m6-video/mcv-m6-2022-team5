import torch
from metric_learning_utils import *
from torchvision import transforms

cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import warnings
warnings.filterwarnings('ignore')
import pandas as pd

def init_global_track_ids_df():
    track_ids = []
    detection_frames = []
    global_track_ids_df = pd.DataFrame()

def get_metric_learning_model():
    save_path = 'aicity_cars_metric_learning_results/metric_learning_ResNet18_sgd_TripletMarginLoss.pth'
    ## test different backbones
    embedding_net = EmbeddingNet_V3('resnet18', '0')
    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()
        
    model.load_state_dict(torch.load(save_path))
    model=model.to(device)

    return model
    
def get_metric_distance(model, img1, img2):
    img_transforms = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                       ])

    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)
    img1 = img_transforms(img1).unsqueeze(0).to(device)
    img2 = img_transforms(img2).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        img1, img2, _ = model(img1, img2, img2)

    img1.data.cpu().numpy()
    img2.data.cpu().numpy()

    ## evaluate distenaces
    distance = getDistances(cv2.HISTCMP_BHATTACHARYYA, [(0,0,img1.data.cpu().numpy())], img2.data.cpu().numpy())

    return distance

## create and add new global track id
def add_global_track_ids(global_track_ids_df, detection):


## add frames to exisiting track
def update_global_track_id(global_track_ids_df, detection, track_id):


## perform matching with exisiting ids
def get_global_track_id(detection):

    return track_id

def get_global_track_id_df():
    return global_track_ids_df