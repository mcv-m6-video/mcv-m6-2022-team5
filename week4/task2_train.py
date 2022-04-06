import os
import gc
import bz2
import pickle
import _pickle as cPickle
import torch
import cv2
import numpy as np
from PIL import Image
from VehicleDetection import *
from itertools import chain
from tqdm import tqdm
# from pqdm.processes import pqdm


# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2_dataset_loader import *

def extract_video(path, div_frames, skip):
    vidcap = cv2.VideoCapture(path)
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    # Read Half the frames 
    for _ in range(num_frames//div_frames):
        for i in range(skip):
            frame = vidcap.read()[1]
            if i == 0:
                frames.append(frame.astype(np.float32)) # Reduce soze
    return iter(frames) # Iterator

def readDetections(path):
  #Generates detection dictionary where the frame number is the key and the values are the info of the corresponding detection/s
  
    with open(path) as f:
        lines = f.readlines()

    detections = {}
    for line in lines:
        data = line.split(',')
        if data[0] in detections:
            detections[data[0]].append(VehicleDetection(int(data[0]), int(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6])))
        else:
            detections[data[0]] = [VehicleDetection(int(data[0]), int(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6]))]

    return detections


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.cuda.empty_cache()
# PATHS
DATASET = "../datasets/aic19-track1-mtmc-train/train/"
SEQUENCES = [DATASET+seq+"/" for seq in os.listdir(DATASET)]
CAMERAS = [[seq+cam+"/" for cam in os.listdir(seq)]for seq in SEQUENCES]
SEQUENCES = [seq.replace(DATASET, "").replace("/", "") for seq in SEQUENCES]
CAMERAS = dict(zip(SEQUENCES, CAMERAS))

# DEFINE SPLITS
train = ["S01", "S04"]
test = ["S03"]

# Model Parameters
selected_model = 'COCO-Detection/retinanet_R_101_FPN_3x.yaml'

seq_data = []

# For each training seq move through cameras and extact even frames and even gt
for i, seq in enumerate(train):
    for j, cam in tqdm(enumerate(CAMERAS[seq]), total = len(CAMERAS[seq]), desc = f"Processing {seq}..."):
        data = {}
        data["div"] = 1
        data["base_path"] = cam + "frames/" # To Save Frames
        data["gt_detected"] = readDetections(cam + "gt/gt.txt")
        data["gt_detected"] = {key:data["gt_detected"][key] for key in data["gt_detected"].keys() if int(key) % data["div"] == 0}
        data["frames"] = extract_video(cam + "vdo.avi", 10,data["div"])
        seq_data.append(data)


DatasetCatalog.clear()
DatasetCatalog.register("AICity_train" , lambda d=seq_data: get_AICity_dicts_big(d))
MetadataCatalog.get("AICity_train").set(thing_classes=["car"])
AICity_metadata = MetadataCatalog.get("AICity_train")

gc.collect()


# Training
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(selected_model))
cfg.DATASETS.TRAIN = ("AICity_train",)
#cfg.DATASETS.VAL = ('AICity_valid',)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(selected_model)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 1e-3
cfg.SOLVER.MAX_ITER = 500
cfg.SOLVER.STEPS = [] # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.BACKBONE.FREEZE_AT = 1

cfg.OUTPUT_DIR = "./results_train_seq01-04"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# Inference and evaluation

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

# Evaluation based on COCO metrics

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

seq_data_val = []

# For each training seq move through cameras and extact even frames and even gt
for i, seq in enumerate(test):
    for j, cam in tqdm(enumerate(CAMERAS[seq]), total = len(CAMERAS[seq]), desc = f"Processing {seq}..."):
        data = {}
        data["div"] = 1
        data["base_path"] = cam + "frames/" # To Save Frames
        data["gt_detected"] = readDetections(cam + "gt/gt.txt")
        data["gt_detected"] = {key:data["gt_detected"][key] for key in data["gt_detected"].keys() if int(key) % data["div"] == 0}
        data["frames"] = extract_video(cam + "vdo.avi", 10,data["div"])
        seq_data.append(data)

DatasetCatalog.register("AICity_valid" , lambda d=seq_data_val: get_AICity_dicts_big(d))
MetadataCatalog.get("AICity_valid").set(thing_classes=["car"])

evaluator = COCOEvaluator("AICity_valid", output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "AICity_valid")
print(inference_on_dataset(predictor.model, val_loader, evaluator))