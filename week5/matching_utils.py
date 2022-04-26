import torch
from metric_learning_utils import *
from torchvision import transforms

cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import warnings
warnings.filterwarnings('ignore')
import pandas as pd

def init_global_track_ids_df():
    data = []
    global_track_ids_df = pd.DataFrame(data, columns=['track_ids', 'detection_frames'])
    return global_track_ids_df

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

def get_detected_box_image(img, bbox):
    img = np.asarray(img)

    for index, b in enumerate(bbox):
        if b < 0:
            bbox[index] = 0

    if img is not None:
        crop = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        return crop

    print('Failed to read!!')
    return None

## create and add new global track id
def add_global_track_ids(global_track_ids_df, img, detection):
    crop = get_detected_box_image(img, detection['bbox'])
    row = {'track_ids': detection['track'], 'detection_frames': [crop]}
    global_track_ids_df = global_track_ids_df.append(row,ignore_index=True)

    return global_track_ids_df

## add frames to exisiting track
def update_global_track_id(global_track_ids_df, img, detection):
    crop = get_detected_box_image(img, detection['bbox'])

    index = global_track_ids_df.index[global_track_ids_df['track_ids'] == detection['track']].tolist()[0]

    detection_frames_list = global_track_ids_df.at[index, 'detection_frames']
    detection_frames_list.append(crop)
    global_track_ids_df.at[index, 'detection_frames'] = detection_frames_list

    return global_track_ids_df
