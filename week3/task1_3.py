from asyncio import selector_events
import enum
from importlib.resources import path
from load_utils  import *
from os import  listdir

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import some common libraries
import numpy as np
import os, argparse, cv2
import pickle

# Import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from PIL import Image
from sklearn.model_selection import KFold

from detectron2_dataset_loader import *
from detectron2.engine import DefaultTrainer
# ------------------------------------------------------------

def split_data(data, k=5):
    split = KFold(n_splits=k)
    split.get_n_splits(data)
    train_folds, val_folds = [], []

    for train_index, val_index in split.split(data):
        train, val = data[train_index], data[val_index]
        train_folds.append(train)
        val_folds.append(val)

    return train_folds, val_folds


# Preparing the custom dataset
def parse_args():
    parser = argparse.ArgumentParser(description= 'Arguments to run the inference script')
    parser.add_argument('-g', '--ground', default='../../m6_dataset/ai_challenge_s03_c010-full_annotation.xml', type=str)
    parser.add_argument('-p', '--img_path', default='../../m6_dataset/dataset/', type=str)
    parser.add_argument('-o', '--out_path', default='./results', type=str, help='Relative path to output folder')
    parser.add_argument('-m', '--model', default='COCO-Detection/retinanet_R_101_FPN_3x.yaml', type=str, help='Detectron2 Model')
    parser.add_argument('-d', '--detections', default='model_detections', type=str, help='Name of the file to save the detections')
    parser.add_argument('-lr', '--learn', default=0.0001, type=float, help='learning rate')

    return parser.parse_args()

args = parse_args()
selected_model = args.model
path_train_imgs = args.img_path

gt_detected = readDetectionsXML(args.ground)
all_frames = listdir(path_train_imgs)
train_frames = all_frames[0:len(all_frames)//4]
test_frames = all_frames[len(all_frames)//4:]

k_folds = 5
tr, val = split_data(np.array(train_frames), k_folds)

SAVEPATH = './AICity_m6_dict'

for k in range(k_folds):
    for d, data in [('train', tr[k]), ('valid', val[k])]:
        DatasetCatalog.register(f"Fold_{k}_AICity_{d}", lambda d=data: get_AICity_dicts(gt_detected,d,path_train_imgs))
        MetadataCatalog.get(f"Fold_{k}_AICity_{d}" + d).set(thing_classes=["car"])
    AICity_metadata = MetadataCatalog.get(f"Fold_{k}_AICity_{d}")


    # ------------------------------------------------------------

    # Training

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(selected_model))
    cfg.DATASETS.TRAIN = (f"Fold_{k}_AICity_train",)
    cfg.DATASETS.VAL = (f"Fold_{k}_AICity_valid",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(selected_model)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = args.learn
    cfg.SOLVER.MAX_ITER = 2000
    cfg.SOLVER.STEPS = [] # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 # (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.OUTPUT_DIR = args.out_path

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    trainer.train()

# ------------------------------------------------------------

# Inference and evaluation

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
predictor = DefaultPredictor(cfg)
# Evaluation based on COCO metrics

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

DatasetCatalog.register("AICity_test", lambda d=test_frames: get_AICity_dicts(gt_detected,d,path_train_imgs))
MetadataCatalog.get("AICity_test").set(thing_classes=["car"])

evaluator = COCOEvaluator("AICity_test", output_dir=args.out_path)
val_loader = build_detection_test_loader(cfg, "AICity_test")
print(inference_on_dataset(predictor.model, val_loader, evaluator))